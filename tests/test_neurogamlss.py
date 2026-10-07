"""Tests of NeuroGAMLSS on synthetic data: python -m pytest tests"""
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.io import savemat
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neurogamlss as NG  # noqa: E402


def simulate_shash(n=1500, p=40, seed=2):
    """SHASHo2 data with age-dependent mu and sigma and constant nu, tau per feature."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 85, n)
    male = rng.integers(0, 2, n).astype(float)
    site = rng.integers(0, 2, n)
    u = (age - 20) / 65
    mu = 1 - 0.5 * u[:, None] * rng.uniform(0.5, 1.5, p) + 0.05 * male[:, None]
    sig = 0.1 * (1 + 0.4 * u)[:, None]
    nu, tau = rng.uniform(-0.8, 0.8, p), rng.uniform(0.6, 1.4, p)
    z = np.sinh((np.arcsinh(rng.standard_normal((n, p))) + nu) / tau)
    return mu + sig * tau * z, age, male, site, nu, tau


def simulate_gg(n=1500, p=20, seed=3):
    """Generalized gamma data: theta (y/mu)^nu ~ Gamma(theta) with theta = 1/(sigma nu)^2."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 85, n)
    male = rng.integers(0, 2, n).astype(float)
    site = np.zeros(n, int)
    mu = np.exp(1 + 0.3 * (age - 20) / 65)[:, None] * np.ones(p)
    sig, nu = 0.2, rng.uniform(-1, 2, p)
    theta = 1 / (sig * nu) ** 2
    g = rng.gamma(theta, 1 / theta, size=(n, p))
    return mu * g ** (1 / nu), age, male, site, nu


def test_shash_derivatives():
    rng = np.random.default_rng(1)
    n = 300
    y = 2 * rng.normal(size=n)
    args = [0.5 * rng.normal(size=n), 0.3 * rng.normal(size=n),
            0.5 * rng.normal(size=n), 0.3 * rng.normal(size=n)]
    ll, d1, d2 = NG.shash_terms(y, *args)
    h = 1e-6
    for i in range(4):
        up, dn = list(args), list(args)
        up[i], dn[i] = args[i] + h, args[i] - h
        lu, d1u, _ = NG.shash_terms(y, *up, order=1)
        ld, d1d, _ = NG.shash_terms(y, *dn, order=1)
        assert np.allclose((lu - ld) / (2 * h), d1[i], rtol=1e-5, atol=1e-6)
        for j in range(4):
            assert np.allclose((d1u[j] - d1d[j]) / (2 * h), d2[(min(i, j), max(i, j))],
                               rtol=1e-4, atol=1e-5)


def test_shash_normal_special_case():
    y = np.linspace(-3, 3, 13)
    ll, _, _ = NG.shash_terms(y, 0.0, 0.0, 0.0, 0.0, order=0)
    assert np.allclose(ll, norm.logpdf(y))
    assert np.allclose(NG.shash_score(y, 0.0, 0.0, 0.0, 0.0), y)


@pytest.mark.parametrize("mu,sig,nu", [(2.0, 0.3, 1.0), (1.0, 0.2, -0.8), (5.0, 0.1, 1e-6),
                                       (3.0, 0.25, 0.0), (0.5, 0.5, 2.5)])
def test_gg_density_and_scores(mu, sig, nu):
    def f(x):          # density of log y
        y = np.exp(x)
        return np.exp(NG.gg_logpdf(np.array([y]), np.log(mu), np.log(sig), np.array([nu]))[0]) * y
    lo, hi = np.log(mu) - 30 * sig, np.log(mu) + 30 * sig
    assert abs(quad(f, lo, hi, limit=500)[0] - 1) < 1e-6
    for y in (0.6 * mu, mu, 1.4 * mu):
        cdf = quad(f, lo, np.log(y), limit=500)[0]
        z = NG.gg_score(np.array([y]), np.log(mu), np.log(sig), np.array([nu]))[0]
        assert abs(norm.cdf(z) - cdf) < 1e-5


def test_gg_smooth_through_lognormal():
    y = np.array([1.3])
    v = [NG.gg_logpdf(y, 0.0, np.log(0.2), np.array([x]))[0] for x in (-1e-3, -1e-6, 0, 1e-6, 1e-3)]
    assert abs(v[2] - norm.logpdf(np.log(1.3), 0, 0.2) + np.log(1.3)) < 1e-12   # log-normal at 0
    assert abs(v[2] - 0.5 * (v[1] + v[3])) < 1e-9                               # no kink at 0
    assert abs((v[3] - v[1]) / 2e-6 - (v[4] - v[0]) / 2e-3) < 1e-3              # same slope


def test_shash_fit_recovers_parameters():
    Y, age, male, site, nu, tau = simulate_shash()
    m = NG.NormativeModel(5, 3, family='shash', shape_prior=3.0).fit(Y, age, male, site)
    assert np.all(m.shape_converged)
    assert np.corrcoef(m.kappa[0], nu)[0, 1] > 0.95
    assert np.corrcoef(m.lam[0], np.log(tau))[0, 1] > 0.9
    z = m.zscores(Y, age, male, site)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
    assert np.median(np.abs(np.mean(z ** 3, axis=0))) < 0.15


def test_shash_fit_is_stationary():
    Y, age, male, site, _, _ = simulate_shash(n=800, p=5)
    m = NG.NormativeModel(5, 3, family='shash').fit(Y, age, male, site)
    X, W = m._design(age, male, site)
    rng = np.random.default_rng(5)

    def objective(b, t, k, lam, j):
        ll = NG.shash_terms(Y[:, j], X @ b, W @ t, np.full(len(age), k), np.full(len(age), lam),
                            order=0)[0].sum()
        return ll - 0.5 * (k ** 2 + lam ** 2) / m.shape_prior ** 2
    for j in range(5):
        best = objective(m.beta[:, j], m.theta[:, j], m.kappa[0, j], m.lam[0, j], j)
        for _ in range(20):
            e = 1e-3 * rng.normal(size=4)
            assert objective(m.beta[:, j] + e[0] * m.beta[:, j].std(), m.theta[:, j] + e[1],
                             m.kappa[0, j] + e[2], m.lam[0, j] + e[3], j) <= best + 1e-6


def test_tau_bound():
    rng = np.random.default_rng(4)
    n = 1000
    age = rng.uniform(20, 80, n)
    Y = rng.uniform(-1, 1, (n, 3))                 # very light tails
    m = NG.NormativeModel(5, 3, family='shash', tau_max=1.5).fit(Y, age, np.zeros(n), None)
    assert np.all(np.exp(m.lam[0]) < 1.5 * 1.02)


def test_gg_fit():
    Y, age, male, site, nu = simulate_gg()
    m = NG.NormativeModel(5, 3, family='gg', shape_prior=3.0).fit(Y, age, male, site)
    assert np.all(m.shape_converged)
    assert np.corrcoef(m.kappa[0], nu)[0, 1] > 0.9
    z = m.zscores(Y, age, male, site)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05


def test_age_dependent_shape_selected_where_needed():
    rng = np.random.default_rng(6)
    n, p = 3000, 6
    age = rng.uniform(20, 85, n)
    nu = np.where(np.arange(p) < 3, 1.5 * (age[:, None] - 52) / 33, 0.3)   # 3 with age-dependent skew
    z = np.sinh(np.arcsinh(rng.standard_normal((n, p))) + nu)
    m = NG.NormativeModel(5, 3, family='shash', shape_df=2).fit(z, age, np.zeros(n), None)
    assert m.shape_age[:3].all() and not m.shape_age[3:].any()


def test_q_statistics_calibrated():
    rng = np.random.default_rng(7)
    Z, age = rng.standard_normal((2000, 300)), rng.uniform(20, 80, 2000)
    q = NG.q_statistics(Z, age, 10)
    assert np.all(q['df'] == 10)
    rate = np.mean(q['p'] < 0.05, axis=1)
    assert np.all((rate > 0.01) & (rate < 0.10))


def test_diagnostics_shash_versus_normal(tmp_path):
    Y, age, male, site, _, _ = simulate_shash(n=3000, p=60, seed=5)
    d = NG.Data(Y.astype(np.float32), age, male, site, 'sim_8mm_X.mat', res='8')
    rates = {}
    for fam in ('shash', 'normal'):
        sets = [NG.ModelSet(NG.data_info(d), None, NG.VoxelModel(family=fam).fit(d))]
        prefix = str(tmp_path / fam)
        NG.run_diagnostics(sets, [d], prefix)
        with open(prefix + '_diagnostics.csv') as f:
            rates[fam] = [float(v) for v in f.read().splitlines()[1].split(',')[2:6]]
    assert max(rates['shash']) < 15                    # Q1-Q4, nominal 5%
    assert rates['normal'][2] > 50                     # skewness misfit of the normal model


@pytest.mark.parametrize("family", ['shash', 'gg', 'normal'])
@pytest.mark.parametrize("ext", ['.npz', '.mat'])
def test_save_load_roundtrip(tmp_path, family, ext):
    if family == 'gg':
        Y, age, male, site, _ = simulate_gg(n=400, p=8)
    else:
        Y, age, male, site, _, _ = simulate_shash(n=400, p=8)
    d = NG.Data(Y.astype(np.float32), age, male, site, 'sim_8mm_X.mat', res='8')
    vm = NG.VoxelModel(family=family, shape_df=2 if family != 'normal' else 0).fit(d)
    sets = [NG.ModelSet(NG.data_info(d), None, vm)]
    path = str(tmp_path / f'model{ext}')
    NG.save_models(path, sets, NG.model_description([d], dict(cov=None), 'test', dict(family=family)))
    loaded, desc = NG.load_models(path)
    assert desc['settings']['voxel_model']['family'] == family
    z1, z2 = vm.zmaps(d)['Z'], loaded[0].voxel.zmaps(d)['Z']
    assert np.array_equal(np.isnan(z1), np.isnan(z2)) and np.nanmax(np.abs(z1 - z2)) == 0


def test_command_line(tmp_path):
    Y, age, male, site, _, _ = simulate_shash(n=600, p=30)
    files = {}
    for name, sel in (('TR', slice(0, 400)), ('TE', slice(400, 600))):
        files[name] = str(tmp_path / f's4rp1_8mm_{name}_CAT12.9.mat')
        savemat(files[name], dict(Y=Y[sel].astype(np.float32), age=age[sel][:, None],
                                  male=male[sel][:, None]))
    model, out = str(tmp_path / 'norm.npz'), str(tmp_path / 'res')
    NG.main(['--normative-only', '--train', files['TR'], '--save-model', model,
             '--diagnostics', '--out', out])
    assert os.path.isfile(out + '_diagnostics.csv')
    NG.main(['--normative-only', '--model', model, '--test', files['TE'], '--adjust', '1:100',
             '--out', out])
    assert os.path.isfile(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat')


def test_convergence_saved(tmp_path):
    Y, age, male, site, _, _ = simulate_shash(n=400, p=8)
    Y[:, 0] = np.where(np.arange(400) < 3, 50.0, Y[:, 0])     # extreme outliers in feature 0
    d = NG.Data(Y.astype(np.float32), age, male, site, 'sim_8mm_X.mat', res='8')
    vm = NG.VoxelModel(family='shash').fit(d)
    vm.model.converged[1] = False                             # flag one feature
    path = str(tmp_path / 'model.mat')
    NG.save_models(path, [NG.ModelSet(NG.data_info(d), None, vm)],
                   NG.model_description([d], dict(cov=None), 'test'))
    m = NG.load_models(path)[0][0].voxel.model
    assert np.array_equal(m.converged, vm.model.converged) and not m.converged[1]
    assert np.array_equal(m.shape_converged, m.converged)
    conv = vm.zmaps(d)['converged']
    assert conv.shape == (8,) and conv[1] == 0


def simulate_trajectories(n=900, p=60, wiggly=False, seed=11):
    """Normal data whose mean depends on age linearly or with several bends."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(10, 80, n)
    male = rng.integers(0, 2, n).astype(float)
    site = rng.integers(0, 2, n)
    u = (age - 10) / 70
    shape = np.sin(3 * np.pi * u) if wiggly else u
    Y = shape[:, None] * rng.uniform(0.5, 1.5, p) + 0.1 * site[:, None] + 0.3 * rng.standard_normal((n, p))
    return NG.Data(Y.astype(np.float32), age, male, site, 'sim_8mm_X.mat', res='8')


def test_select_df_normative_follows_the_trajectory():
    lin = NG.select_df_normative(simulate_trajectories(), max_features=60)
    wig = NG.select_df_normative(simulate_trajectories(wiggly=True), max_features=60)
    assert lin[0] <= 3 and wig[0] >= 5
    assert lin[2]['selected'] and len(lin[2]['stages']) == 2
    fixed = NG.select_df_normative(simulate_trajectories(), df_mu=4, max_features=60)
    assert fixed[0] == 4 and fixed[2]['stages'][0]['df_mu'] == [4] * len(NG.DF_GRID_VOXEL['sigma'])


def simulate_brain(n=500, p=300, seed=12):
    """Features with age effects of different shapes, for NDMBrainAge."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    male = rng.integers(0, 2, n).astype(float)
    u = (age - 20) / 60
    load = rng.standard_normal((3, p))
    Y = (np.column_stack([u, u ** 2, np.sqrt(u)]) @ load + 0.5 * rng.standard_normal((n, p)))
    return NG.Data(Y.astype(np.float32), age, male, np.zeros(n, int), 'sim_8mm_X.mat', res='8')


def test_with_df_equals_a_new_fit():
    d = simulate_brain()
    est = NG.NDMBrainAge(pca=10, mask_threshold=0).fit(d)
    a = est.with_df(d, 2, 1).predict(d)
    b = NG.NDMBrainAge(df_mu=2, df_sigma=1, pca=10, mask_threshold=0).fit(d).predict(d)
    assert np.allclose(a['age'], b['age']) and np.allclose(a['sd'], b['sd'], equal_nan=True)


def test_select_df_brainage():
    d = simulate_brain()
    dm, ds, rec = NG.select_df_brainage(d, dict(pca=10, mask_threshold=0))
    assert dm in NG.DF_GRID_BRAINAGE['mu'] and ds in NG.DF_GRID_BRAINAGE['sigma']
    first, last = rec['stages']
    assert first['df_mu'] == list(NG.DF_GRID_BRAINAGE['mu']) and set(last['df_mu']) == {dm}
    assert last['mae'][last['df_sigma'].index(ds)] == min(last['mae'])
    assert NG.select_df_brainage(d, dict(pca=10, mask_threshold=0), 2, 3)[2]['selected'] is False


def test_command_line_stores_the_chosen_df(tmp_path):
    import json
    from scipy.io import loadmat
    d = simulate_trajectories(n=500, p=40)
    files = {}
    for name, sel in (('TR', slice(0, 350)), ('TE', slice(350, 500))):
        files[name] = str(tmp_path / f's4rp1_8mm_{name}_CAT12.9.mat')
        savemat(files[name], dict(Y=d.Y[sel], age=d.age[sel][:, None], male=d.male[sel][:, None]))
    model, out = str(tmp_path / 'norm.npz'), str(tmp_path / 'res')
    NG.main(['--normative-only', '--family', 'normal', '--train', files['TR'],
             '--save-model', model, '--diagnostics', '--out', out])
    with open(str(tmp_path / 'norm.json')) as f:
        rec = json.load(f)['models'][0]['df_voxel']
    assert rec['selected'] and rec['df_mu'] in NG.DF_GRID_VOXEL['mu']
    loaded = NG.load_models(model)[0][0]
    assert loaded.info['df_voxel']['df_mu'] == rec['df_mu'] == loaded.voxel.model.df_mu
    NG.main(['--normative-only', '--model', model, '--test', files['TE'], '--out', out])
    z = loadmat(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat', squeeze_me=True)['NDMzmap']
    assert int(z['df_mu']) == rec['df_mu'] and int(z['df_sigma']) == rec['df_sigma']
    NG.main(['--normative-only', '--df-mu', '4', '--df-sigma', '2', '--train', files['TR'],
             '--save-model', model, '--out', out])
    assert NG.load_models(model)[0][0].voxel.model.df_mu == 4


def masked_data(surface=False, n=300, seed=13):
    """Positive data with near-empty features 0-4, a constant feature 5 and a NaN in feature 6."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    male = rng.integers(0, 2, n).astype(float)
    scale = 2.5 if surface else 0.5
    Y = scale * (1 + 0.2 * rng.standard_normal((n, 40))) - 0.002 * (age[:, None] - 50)
    Y[:, :5] = np.abs(0.02 * scale * rng.standard_normal((n, 5)))        # near zero
    Y[:, 5] = scale
    Y[3, 6] = np.nan
    name = 's12.mesh.thickness_X_CAT12.9.mat' if surface else 's4rp1_8mm_X_CAT12.9.mat'
    return NG.Data(Y.astype(np.float32), age, male, np.zeros(n, int), name,
                   is_surf=surface, res=None if surface else '8')


def test_feature_mask_rules():
    vol, surf = masked_data(), masked_data(surface=True)
    keep, rec = NG.feature_mask(vol)                      # default 0.05, absolute
    assert rec['spec'] == '0.05' and rec['threshold'] == 0.05
    assert not keep[:7].any() and keep[7:].all() and rec['n_used'] == 33
    assert rec['n_without_variance'] == 2 and rec['n_below_threshold'] == 5
    keep, rec = NG.feature_mask(surf)                     # default 5% of the median of the means
    assert rec['spec'] == '5%' and np.isclose(rec['threshold'], 0.05 * rec['median_of_means'])
    assert not keep[:7].any() and keep[7:].all()
    keep, rec = NG.feature_mask(vol, 0)                   # only finite, not constant
    assert keep[:5].all() and not keep[5:7].any() and rec['threshold'] == 0
    neg = NG.Data(-vol.Y, vol.age, vol.male, vol.site, vol.name)
    assert NG.feature_mask(neg, '5%')[1]['threshold'] == 0  # no positive median: not used
    for bad in ('-1', 'x', '5 %%'):
        with pytest.raises(ValueError):
            NG.parse_mask_threshold(bad)


@pytest.mark.parametrize("ext", ['.npz', '.mat'])
def test_mask_in_models(tmp_path, ext):
    d = masked_data()
    vm = NG.VoxelModel(family='normal').fit(d)
    Z = vm.zmaps(d)['Z']
    assert np.all(np.isnan(Z[:, :7])) and np.all(np.isfinite(Z[:, 7:]))
    est = NG.NDMBrainAge(pca=5).fit(d)
    assert set(est.parts[0].cols) == set(range(7, 40))
    path = str(tmp_path / f'model{ext}')
    NG.save_models(path, [NG.ModelSet(dict(NG.data_info(d), mask=vm.mask_info), est, vm)],
                   NG.model_description([d], dict(cov=None), 'test'))
    s = NG.load_models(path)[0][0]
    assert s.voxel.mask_info['n_used'] == 33 and s.brainage.mask_info['threshold'] == 0.05
    assert np.array_equal(np.isnan(s.voxel.zmaps(d)['Z']), np.isnan(Z))


def test_command_line_mask(tmp_path):
    import json
    d = masked_data(n=200)
    f = str(tmp_path / 's4rp1_8mm_TR_CAT12.9.mat')
    savemat(f, dict(Y=d.Y, age=d.age[:, None], male=d.male[:, None]))
    model = str(tmp_path / 'norm.npz')
    NG.main(['--normative-only', '--family', 'normal', '--df-mu', '3', '--df-sigma', '2',
             '--mask-threshold', '50%', '--train', f, '--save-model', model,
             '--out', str(tmp_path / 'r')])
    with open(str(tmp_path / 'norm.json')) as fh:
        desc = json.load(fh)
    assert desc['settings']['mask_threshold'] == '50%'
    assert desc['models'][0]['mask']['spec'] == '50%' and desc['models'][0]['mask']['n_used'] == 33
    with pytest.raises(SystemExit):
        NG.main(['--normative-only', '--mask-threshold', 'abc', '--train', f, '--save-model', model])


def test_command_line_suggests_joining_sites(tmp_path, capsys):
    d = masked_data(n=120)
    files = []
    for name, sel in (('A', slice(0, 70)), ('B', slice(70, 120))):
        files.append(str(tmp_path / f's4rp1_8mm_{name}_CAT12.9.mat'))
        savemat(files[-1], dict(Y=d.Y[sel], age=d.age[sel][:, None], male=d.male[sel][:, None]))
    with pytest.raises(SystemExit):
        NG.main(['--normative-only', '--train', *files, '--save-model', str(tmp_path / 'm.npz')])
    err = capsys.readouterr().err
    assert "has 50 subjects" in err and "--train " + '+'.join(files) in err


def test_tilde_in_joined_files(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    d = masked_data(n=60)
    for name, sel in (('A', slice(0, 30)), ('B', slice(30, 60))):
        savemat(str(tmp_path / f's4rp1_8mm_{name}_CAT12.9.mat'),
                dict(Y=d.Y[sel], age=d.age[sel][:, None], male=d.male[sel][:, None]))
        np.savetxt(str(tmp_path / f'cov_{name}.txt'), d.age[sel][:, None])
    joined = NG.load_data('~/s4rp1_8mm_A_CAT12.9.mat+~/s4rp1_8mm_B_CAT12.9.mat')
    assert joined.n == 60 and np.array_equal(np.unique(joined.site), [0, 1])
    assert NG.read_table('~/cov_A.txt+~/cov_B.txt')[1].shape == (60, 1)


@pytest.mark.parametrize("family", ['shash', 'normal', 'gg'])
def test_results_do_not_depend_on_threads(family):
    if family == 'gg':
        Y, age, male, site, _ = simulate_gg(n=300, p=60)
    else:
        Y, age, male, site, _, _ = simulate_shash(n=300, p=60)
    fits = []
    for jobs in (1, 4):
        m = NG.NormativeModel(4, 2, family=family, jobs=jobs)
        m.fit(Y, age, male, site)
        fits.append(m)
    for key in ('beta', 'theta', 'kappa', 'lam', 'converged'):
        a, b = getattr(fits[0], key), getattr(fits[1], key)
        assert (a is None and b is None) or np.array_equal(a, b)


def test_fit_chunks_in_threads():
    Y, age, male, site, _, _ = simulate_shash(n=400, p=50)
    X = np.column_stack([np.ones(400), age])
    W = X.copy()
    one = NG.fit_location_scale(Y, X, W, max_elements=400 * 7, jobs=1)
    many = NG.fit_location_scale(Y, X, W, max_elements=400 * 7, jobs=3)    # 8 chunks
    assert all(np.array_equal(a, b) for a, b in zip(one, many))


def write_brain(tmp_path, name, d, sel, age):
    path = str(tmp_path / f's4rp{name}_CAT12.9.mat')
    savemat(path, dict(Y=np.abs(d.Y[sel]) + 0.5, age=np.asarray(age, float)[:, None],
                       male=d.male[sel][:, None]))
    return path


def test_ages_of_zero_or_nan(tmp_path):
    from scipy.io import loadmat
    d = simulate_brain(n=400, p=120)
    age = d.age.copy()
    age[[0, 1, 300, 301]] = 0
    age[[2, 302]] = np.nan
    tr = write_brain(tmp_path, '1_8mm_TR', d, slice(0, 250), age[:250])
    te = write_brain(tmp_path, '1_8mm_TE', d, slice(250, 400), age[250:])
    assert np.isnan(NG.load_data(tr).age[:3]).all()                 # 0 and NaN are unknown
    out = str(tmp_path / 'r')
    NG.main(['--train', tr, '--test', te, '--adjust', '1:80', '--zmaps', '--pca', '10',
             '--df-mu', '3', '--df-sigma', '2', '--no-gpr', '--out', out])
    r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
    z = loadmat(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat', squeeze_me=True,
                struct_as_record=False)['NDMzmap']
    bad = [50, 51, 52]
    assert np.isnan(r.BrainAGE[bad]).all() and np.isfinite(r.PredictedAge[bad]).all()
    assert np.isfinite(np.delete(r.BrainAGE, bad)).all() and np.isnan(z.Z[bad]).all()
    assert r.ind_control.size == 77 and not np.isin(bad, r.ind_control - 1).any()   # 3 of 1:80 unknown


def test_samples_without_valid_ages(tmp_path):
    from scipy.io import loadmat
    d = simulate_brain(n=300, p=120)
    files = [write_brain(tmp_path, f'{t}_8mm_TR', d, slice(0, 200), d.age[:200]) for t in (1, 2)]
    tests = [write_brain(tmp_path, f'{t}_8mm_NA', d, slice(200, 300), np.zeros(100)) for t in (1, 2)]
    out = str(tmp_path / 'r')
    NG.main(['--train', *files, '--test', *tests, '--pca', '10', '--df-mu', '3', '--df-sigma',
             '2', '--no-gpr', '--out', out])                       # brain age without ages
    r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
    assert np.allclose(r.weights, 0.5) and np.isfinite(r.PredictedAge_ensemble).all()
    with pytest.raises(SystemExit):                                  # controls without ages
        NG.main(['--train', *files, '--test', *tests, '--adjust', '1:10', '--pca', '10',
                 '--no-gpr', '--out', out])
    few = write_brain(tmp_path, '1_8mm_FEW', d, slice(0, 20), np.r_[d.age[:5], np.zeros(15)])
    with pytest.raises(SystemExit):                                  # too few training ages
        NG.main(['--normative-only', '--train', few, '--save-model', out + '.npz'])
