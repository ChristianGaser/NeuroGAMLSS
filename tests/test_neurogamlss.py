"""Tests of NeuroGAMLSS on synthetic data: python -m pytest tests"""
import os
import sys

import numpy as np
from dataclasses import replace
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


def test_spline_boundary_knots():
    rng = np.random.default_rng(8)
    age = rng.uniform(20, 80, 500)
    b = NG.NaturalSpline(age, 4)
    lo, hi = np.quantile(age, NG.SPLINE_BOUNDARY)
    assert b.df == 4 and np.isclose(b.lo, lo) and np.isclose(b.hi, hi)
    for x in (np.linspace(0, lo, 5), np.linspace(hi, 120, 5)):      # linear outside
        assert np.allclose(np.diff(b(x), 2, axis=0), 0, atol=1e-9)
    old = NG.NaturalSpline(np.append(age, 110), 4)                  # one very old subject
    assert abs(old.hi - b.hi) < 0.5 and np.allclose(old.knots, b.knots, atol=0.01)
    x = np.r_[np.zeros(97), np.ones(3)]                             # rare binary covariate
    c = NG.NaturalSpline(x, 3)
    assert c.df == 1 and np.allclose(c(x)[:, 0], x)


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


def test_skewed_fits_converge_quickly():
    """Strongly skewed features, whose Hessian is indefinite on the way from the normal
    start and some of whose tau end at the bound, converge within 30 Newton steps (the
    outer product of the scores, used before, needed up to 60) to the maximum of a
    fit with a much smaller tolerance."""
    rng = np.random.default_rng(7)
    n, p = 1000, 20
    age = rng.uniform(20, 85, n)
    male = rng.integers(0, 2, n).astype(float)
    u = (age - 20) / 65
    nu, tau = rng.uniform(1.5, 3.0, p), rng.uniform(0.5, 2.5, p)
    z = np.sinh((np.arcsinh(rng.standard_normal((n, p))) + nu) / tau)
    Y = 0.5 - 0.3 * u[:, None] + 0.05 * (1 + 0.5 * u)[:, None] * tau * z
    m = NG.NormativeModel(4, 2).fit(Y, age, male)
    X, W = m._design(age, male)
    c, s = Y.mean(axis=0), Y.std(axis=0)            # standardized as in _fit_shape
    b0 = m.beta / s
    b0[0] -= c / s
    t0 = m.theta.copy()
    t0[0] -= np.log(s)
    one = np.ones((n, 1))
    Ys = (Y - c) / s
    args = (Ys, [X, W, one, one], NG.shash_terms, [b0, t0, np.zeros((1, p)), np.zeros((1, p))],
            (2, 3), 1.0, (3, np.log(NG.TAU_MIN), np.log(2.0)))
    fast, _, conv = NG.fit_distribution(*args, max_iter=30, jobs=1)
    ref, _, conv_ref = NG.fit_distribution(*args, tol=1e-13, jobs=1)
    assert np.all(conv) and np.all(conv_ref)
    assert np.any(np.exp(fast[3][0]) > 1.99)               # bound reached
    z1, z2 = [NG.shash_score(Ys, X @ b[0], W @ b[1], b[2], b[3]) for b in (fast, ref)]
    assert np.max(np.abs(z1 - z2)) < 1e-3


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


@pytest.mark.parametrize("n,p,rank", [(300, 40, 5), (300, 40, 40), (60, 150, 5)])
def test_residual_correlation_is_the_low_rank_model(n, p, rank):
    """z R^-1 z' from D and A is that of the explicit R = V diag(lam) V' + diag(psi), whether
    the eigenvectors come from the p x p matrix (p <= n) or from the n x n Gram matrix."""
    rng = np.random.default_rng(16)
    Z = rng.standard_normal((n, 4)) @ rng.standard_normal((4, p)) + rng.standard_normal((n, p))
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    corr = NG.ResidualCorrelation(rank, 0.01).fit(Z)
    lam, V = np.linalg.eigh(Z.T @ Z / n)
    k = min(rank, n - 1, p)
    lam, V = lam[::-1][:k], V[:, ::-1][:, :k]
    psi = np.maximum(1 - (V ** 2) @ lam, 0.01)
    R = (V * lam) @ V.T + np.diag(psi)
    z = rng.standard_normal((20, p))
    explicit = np.sum((z @ np.linalg.inv(R)) * z, axis=1)
    woodbury = (z ** 2) @ corr.D - np.sum((z @ corr.A) ** 2, axis=1)
    assert np.allclose(corr.D, 1 / psi) and np.allclose(woodbury, explicit, rtol=1e-8)


def test_standardized_pca_ignores_the_scale_of_voxels():
    d = simulate_brain()
    scaled = NG.Data((d.Y * np.geomspace(0.1, 10, d.Y.shape[1])).astype(np.float32), d.age,
                     d.male, d.site, d.name, res='8')
    kw = dict(df_mu=3, df_sigma=1, pca=10, mask_threshold=0)
    est = NG.NDMBrainAge(**kw).fit(d)
    a = est.predict(d)['age']
    assert np.allclose(NG.NDMBrainAge(**kw).fit(scaled).predict(scaled)['age'], a, atol=1e-3)
    raw = NG.NDMBrainAge(standardize=False, **kw)
    b = raw.fit(d).predict(d)['age']
    assert np.max(np.abs(raw.fit(scaled).predict(scaled)['age'] - b)) > 0.01
    loaded = NG.NDMBrainAge.from_state(est.get_state())
    assert loaded.standardize and np.array_equal(loaded.predict(d)['age'], a)


def test_brainloss_measures_loss_beyond_the_brain_age():
    d = simulate_brain(n=600)
    hit = np.zeros(d.n, dtype=bool)
    hit[np.random.default_rng(14).choice(d.n, 120, replace=False)] = True
    Y = d.Y.copy()
    Y[hit] -= 0.3                                       # less tissue, unrelated to age
    loss = NG.Data(Y, d.age, d.male, d.site, d.name, res='8')
    tr, te = np.arange(400), np.arange(400, 600)
    est = NG.NDMBrainAge(df_mu=3, df_sigma=1, pca=10, mask_threshold=0, brainloss=True)
    out = est.fit(d.subset(tr)).predict(loss.subset(te))
    h, bl = hit[te], out['brainloss']
    assert bl[h].mean() > bl[~h].mean() + 3 and abs(bl[~h].mean()) < 0.5
    ba = out['age'] - d.age[te]
    assert abs(ba[h].mean() - ba[~h].mean()) < 1         # the brain age hardly changes
    loaded = NG.NDMBrainAge.from_state(est.get_state())
    assert loaded.brainloss and np.array_equal(loaded.predict(loss.subset(te))['brainloss'], bl)


def test_brainloss_age_includes_the_older_looking_brain():
    """At chronological age, the tissue that an older-looking brain lacks counts as loss;
    at the brain age it does not, and the two differ by about b * BrainAGE."""
    rng = np.random.default_rng(15)
    n, p = 700, 200
    age = np.r_[rng.uniform(20, 80, 500), rng.uniform(25, 60, 200)]
    older = np.arange(n) >= 600                         # brains that look 15 years older
    u = (age + 15 * older - 20) / 60
    load = rng.standard_normal((3, p))
    Y = (np.column_stack([u, u ** 2, np.sqrt(u)]) @ load - u[:, None]   # tissue declines
         + 0.5 * rng.standard_normal((n, p)))
    d = NG.Data(Y.astype(np.float32), age, rng.integers(0, 2, n).astype(float),
                np.zeros(n, int), 'sim_8mm_X.mat', res='8')
    tr, te = np.arange(500), np.arange(500, 700)
    est = NG.NDMBrainAge(df_mu=3, df_sigma=1, pca=10, mask_threshold=0, brainloss=True)
    out = est.fit(d.subset(tr)).predict(d.subset(te))
    o, ba = older[te], out['age'] - age[te]
    bl, bla = out['brainloss'], out['brainloss_age']
    assert ba[o].mean() > 10 and abs(ba[~o].mean()) < 2
    assert abs(bl[o].mean() - bl[~o].mean()) < 0.5      # nothing beyond the brain age
    assert bla[o].mean() > bla[~o].mean() + 3           # but less tissue than at their age
    assert np.corrcoef(bla - bl, ba)[0, 1] > 0.9
    unknown = d.subset(te[:5])
    unknown.age = np.full(5, np.nan)
    res = est.predict(unknown)
    assert np.isnan(res['brainloss_age']).all() and np.isfinite(res['brainloss']).all()
    cv = NG.cross_validate([d.subset(tr)], kfold=3, gpr=False, df_mu=3, df_sigma=1, pca=10,
                           mask_threshold=0, brainloss=True)
    assert cv['BrainLOSS_age'].shape == (500, 1) and np.isfinite(cv['BrainLOSS_age']).all()


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
    assert desc['settings']['spline_boundary'] == list(NG.SPLINE_BOUNDARY)
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


def test_cells_give_the_same_fit(monkeypatch):
    """Subjects with the same rows of the designs (whole-year ages) are pooled into
    cells.  The fit equals the one over the subjects, also for data far from 0, and
    does not depend on the threads."""
    Y, age, male, site, _, _ = simulate_shash(n=1000, p=30)
    age = np.round(age)
    Y = Y + 1000                             # sums of squares around a large mean
    m = NG.NormativeModel(4, 2)
    site = m._setup(age, male, site)
    X, W = m._design(age, male, site)
    assert len(NG._cells(X, W)[1]) < 0.5 * len(age)
    one = NG.fit_location_scale(Y, X, W, max_elements=1000 * 7, jobs=1)
    many = NG.fit_location_scale(Y, X, W, max_elements=1000 * 7, jobs=3)    # 5 chunks
    assert all(np.array_equal(a, b) for a, b in zip(one, many))
    monkeypatch.setattr(NG, 'CELL_SHARE', 0.0)                  # every subject on its own
    assert NG._cells(X, W) is None
    ref = NG.fit_location_scale(Y, X, W, max_elements=1000 * 7, jobs=1)
    assert np.all(one[2]) and np.all(ref[2])
    assert np.max(np.abs(X @ (one[0] - ref[0])) / np.exp(W @ ref[1])) < 1e-5
    assert np.max(np.abs(W @ (one[1] - ref[1]))) < 1e-5


def test_intercept_only_shape_blocks():
    """The predictors of shape blocks with the intercept alone are rows that broadcast.
    The fit equals the one from a design with an additional zero column, also where
    the bound of tau is reached."""
    Y, age, male, site, _, _ = simulate_shash(n=800, p=12)
    m = NG.NormativeModel(4, 2).fit(Y, age, male, site)
    X, W = m._design(age, male, site)
    one = np.ones((len(age), 1))
    bound = (3, np.log(NG.TAU_MIN), np.log(1.1))         # tau of the simulation up to 1.4
    fits = [NG.fit_distribution(Y, [X, W, V, V], NG.shash_terms,
                                [m.beta, m.theta] + [np.zeros((V.shape[1], Y.shape[1]))] * 2,
                                (2, 3), 1.0, bound, jobs=1)
            for V in (one, np.column_stack([one, np.zeros(len(age))]))]
    (c1, l1, v1), (c2, l2, v2) = fits
    assert np.all(v1) and np.all(v2)
    assert np.any(c1[3][0] > np.log(1.1))                # bound reached
    assert all(np.allclose(a, b[:len(a)], rtol=0, atol=1e-6) for a, b in zip(c1, c2))
    assert np.all(c2[2][1] == 0) and np.all(c2[3][1] == 0)
    assert np.allclose(l1, l2, rtol=0, atol=1e-5)


def simulate_site_variance(n=1200, p=20, sd=(0.2, 0.5, 0.1), seed=21):
    """Normal data whose mean and noise SD differ between three sites; the log SD also
    rises by 0.2 per unit of a covariate x."""
    rng = np.random.default_rng(seed)
    site = rng.integers(0, len(sd), n)
    age = rng.uniform(20, 80, n)
    male = rng.integers(0, 2, n).astype(float)
    x = rng.standard_normal(n)
    noise = (np.asarray(sd)[site] * np.exp(0.2 * x))[:, None] * rng.standard_normal((n, p))
    return 1 - 0.01 * (age - 20)[:, None] + 0.1 * site[:, None] + noise, age, male, site, x[:, None]


@pytest.mark.parametrize("family", ['normal', 'shash'])
def test_site_sd(family):
    Y, age, male, site, C = simulate_site_variance()
    cov = NG.Covariates(('x',), (), ('x',))
    old = NG.NormativeModel(2, 1, cov, family=family).fit(Y, age, male, site, C=C)
    new = NG.NormativeModel(2, 1, cov, family=family, site_sd=True).fit(Y, age, male, site, C=C)
    assert old._design(age, male, site, C)[1].shape[1] == 4     # intercept, age, sex, x
    assert new._design(age, male, site, C)[1].shape[1] == 6     # and two site dummies

    def spread(m):
        z = m.zscores(Y, age, male, site, C)
        return np.array([np.std(z[site == s]) for s in range(3)])
    assert np.all(np.abs(spread(new) - 1) < 0.1)                 # calibrated at every site
    s_old = spread(old)
    assert s_old[1] > 1.2 and s_old[2] < 0.8                     # pooled SD
    if family == 'normal':
        lw = new.cov_effects(C)[1]                               # covariates stay the last columns
        slope = np.linalg.lstsq(C - np.median(C), lw, rcond=None)[0]
        assert abs(np.mean(slope) - 0.2) < 0.03
        sd_ref = new.mu_sigma(age, male, C=np.full_like(C, np.median(C)))[1]
        props = np.bincount(site) / site.size                    # reference: size-weighted mean
        assert abs(np.mean(np.log(sd_ref)) - props @ np.log([0.2, 0.5, 0.1])) < 0.03
    loaded = NG.NormativeModel.from_state(new.get_state())
    assert loaded.site_sd and np.array_equal(loaded.zscores(Y, age, male, site, C),
                                             new.zscores(Y, age, male, site, C))
    state = old.get_state()
    del state['site_sd']                                         # models saved before site_sd
    assert not NG.NormativeModel.from_state(state).site_sd


def test_command_line_site_sd(tmp_path):
    import json
    Y, age, male, site, _ = simulate_site_variance(n=600, p=30)
    files = []
    for s in range(3):
        files.append(str(tmp_path / f's4rp1_8mm_S{s}_CAT12.9.mat'))
        savemat(files[-1], dict(Y=Y[site == s].astype(np.float32), age=age[site == s][:, None],
                                male=male[site == s][:, None]))
    model, out = str(tmp_path / 'norm.npz'), str(tmp_path / 'r')
    NG.main(['--train', '+'.join(files), '--site-sd', '--zmaps', '--family', 'normal', '--pca',
             '10', '--df-mu', '2', '--df-sigma', '1', '--brainloss', '--kfold', '0',
             '--save-model', model, '--out', out])
    with open(str(tmp_path / 'norm.json')) as f:
        assert json.load(f)['settings']['site_sd'] is True
    s = NG.load_models(model)[0][0]
    assert s.voxel.model.site_sd and s.brainage.site_sd
    assert s.brainage.parts[0].model.site_sd and s.brainage.loss_model.site_sd
    with pytest.raises(SystemExit):                              # saved models keep their design
        NG.main(['--normative-only', '--model', model, '--site-sd', '--test', files[0]])


def simulate_test_site(n_ctrl=150, n_pat=100, p=40, seed=31):
    """Test sample whose data depend on age, on an image quality measure q and, for the
    patients, on a disease effect of -0.3; q is worse with age and in the patients."""
    rng = np.random.default_rng(seed)
    n = n_ctrl + n_pat
    age = rng.uniform(20, 80, n)
    patient = np.arange(n) >= n_ctrl
    q = rng.standard_normal(n) + 0.02 * (age - 50) + patient
    male = rng.integers(0, 2, n).astype(float)
    Y = ((1 - 0.01 * (age - 20))[:, None] - 0.2 * q[:, None] * rng.uniform(0.5, 1.5, p)
         - 0.3 * patient[:, None] + 0.1 * rng.standard_normal((n, p)))
    d = NG.Data(Y.astype(np.float32), age, male, np.zeros(n, int), 'sim_8mm_T.mat', res='8')
    return d, q[:, None], patient


def effects(Y, *x):
    """Least-squares effects of the variables x on every column of Y (len(x), p)."""
    D = np.column_stack([np.ones(len(x[0]))] + list(x))
    return np.linalg.lstsq(D, np.asarray(Y, np.float64), rcond=None)[0][1:]


def test_remove_nuisance_keeps_age_and_disease():
    d, q, patient = simulate_test_site()
    ctrl = np.flatnonzero(~patient)
    out, rec = NG.remove_nuisance(d, q, ctrl, names=['IQR'])
    assert rec['names'] == ['IQR'] and rec['n_controls'] == 150 and rec['explained'] > 0.3
    e_q = effects(out.Y, d.age, patient, q[:, 0])[2]
    assert np.all(np.abs(e_q) < 0.03)                           # quality effect removed
    e_age, e_pat = effects(out.Y, d.age, patient)               # group comparison without q
    assert abs(e_pat.mean() + 0.3) < 0.02                       # disease effect kept
    assert abs(e_age.mean() + 0.01) < 0.001                     # age effect kept
    # estimated in all subjects, the quality effect absorbs part of the disease effect
    e_pat_all = effects(NG.remove_nuisance(d, q, np.arange(d.n))[0].Y, d.age, patient)[1]
    assert e_pat_all.mean() > -0.27
    q[0] = np.nan                                               # missing: no correction
    out = NG.remove_nuisance(d, q, ctrl)[0]
    assert np.allclose(out.Y[0], d.Y[0], atol=1e-5)
    with pytest.raises(ValueError):                             # constant in the controls
        NG.remove_nuisance(d, np.where(patient, 1.0, 0.0)[:, None], ctrl)


def test_command_line_test_nuisance(tmp_path):
    from scipy.io import loadmat
    d, q, patient = simulate_test_site(n_ctrl=150, n_pat=60, p=30)
    rng = np.random.default_rng(32)
    age = rng.uniform(20, 80, 300)
    Ytr = (1 - 0.01 * (age - 20))[:, None] + 0.15 * rng.standard_normal((300, 30))
    tr, te = str(tmp_path / 's4rp1_8mm_TR_CAT12.9.mat'), str(tmp_path / 's4rp1_8mm_TE_CAT12.9.mat')
    savemat(tr, dict(Y=Ytr.astype(np.float32), age=age[:, None]))
    savemat(te, dict(Y=d.Y, age=d.age[:, None]))
    table, out = str(tmp_path / 'iqm.csv'), str(tmp_path / 'r')
    np.savetxt(table, np.column_stack([q, d.age]), delimiter=',', header='IQR,age', comments='')
    base = ['--normative-only', '--family', 'normal', '--df-mu', '2', '--df-sigma', '1',
            '--train', tr, '--test', te, '--adjust', '1:150', '--out', out]
    mean_z = []
    for extra in ([], ['--test-nuisance', table, '--nuisance-cols', 'IQR']):
        NG.main(base + extra)
        z = loadmat(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat', squeeze_me=True,
                    struct_as_record=False)['NDMzmap']
        mean_z.append(np.nanmean(z.Z, axis=1))
    assert str(z.nuisance) == 'IQR' and 0 < z.nuisance_explained < 1
    ctrl = ~patient
    assert np.corrcoef(mean_z[0][ctrl], q[ctrl, 0])[0, 1] < -0.5
    assert abs(np.corrcoef(mean_z[1][ctrl], q[ctrl, 0])[0, 1]) < 0.1
    assert mean_z[1][patient].mean() < -1                      # the disease effect remains
    NG.main(base[3:] + ['--zmaps', '--pca', '5', '--no-gpr', '--test-nuisance', table,
                        '--nuisance-cols', 'IQR'])            # brain age and z-maps
    r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
    assert str(r.nuisance) == 'IQR' and 0 < float(r.nuisance_explained) < 1
    with pytest.raises(SystemExit):                             # needs controls
        NG.main([a for a in base if a not in ('--adjust', '1:150')] + ['--test-nuisance', table])
    trcov = str(tmp_path / 'train_iqm.csv')
    np.savetxt(trcov, rng.standard_normal((300, 1)), header='IQR', comments='')
    with pytest.raises(SystemExit):                             # also a covariate of the models
        NG.main(base + ['--train-cov', trcov, '--test-cov', table, '--test-nuisance', table,
                        '--nuisance-cols', 'IQR'])


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
             '--df-mu', '3', '--df-sigma', '2', '--brainloss', '--no-gpr', '--out', out])
    r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
    z = loadmat(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat', squeeze_me=True,
                struct_as_record=False)['NDMzmap']
    bad = [50, 51, 52]
    assert np.isnan(r.BrainAGE[bad]).all() and np.isfinite(r.PredictedAge[bad]).all()
    assert np.isfinite(np.delete(r.BrainAGE, bad)).all() and np.isnan(z.Z[bad]).all()
    assert r.ind_control.size == 77 and not np.isin(bad, r.ind_control - 1).any()   # 3 of 1:80 unknown
    assert np.isfinite(r.BrainLOSS).all()                # at the brain age: also without age
    assert abs(np.mean(r.BrainLOSS[r.ind_control - 1])) < 1e-9   # standardized with controls
    assert np.isnan(r.BrainLOSS_age[bad]).all()          # at chronological age: needs the age
    assert np.isfinite(np.delete(r.BrainLOSS_age, bad)).all()
    assert abs(np.mean(r.BrainLOSS_age[r.ind_control - 1])) < 1e-9
    with open(out + '.csv') as f:
        header = f.readline().strip().split(',')
    assert 'BrainLOSS_age_ensemble' in header and 'BrainLOSS_age_s4rp1_8mm_TE_CAT12.9' in header
    with pytest.raises(SystemExit):                      # BrainLOSS needs brain age models
        NG.main(['--normative-only', '--brainloss', '--train', tr, '--save-model', out + '.npz'])


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


def test_volume_files_with_and_without_ind(tmp_path):
    """Older volume files of cat_io_data2mat.m store no voxel indices: models of files with
    ind apply to files without ind, but not to files with other indices."""
    d = simulate_brain(n=300, p=40)
    ind = 7 * np.arange(1, 41, dtype=float)[:, None]

    def write(name, sel, index):
        path = str(tmp_path / f's4rp1_8mm_{name}_CAT12.9.mat')
        savemat(path, dict(Y=np.abs(d.Y[sel]) + 0.5, age=d.age[sel][:, None],
                           male=d.male[sel][:, None], ind=index))
        return path
    model, out = str(tmp_path / 'm.npz'), str(tmp_path / 'r')
    NG.main(['--normative-only', '--train', write('TR', slice(0, 200), ind), '--family',
             'normal', '--df-mu', '2', '--df-sigma', '1', '--save-model', model])
    old = write('OLD', slice(200, 300), np.zeros((0, 1)))
    NG.main(['--normative-only', '--model', model, '--test', old, '--out', out])
    assert os.path.isfile(out + '_zmaps_s4rp1_8mm_OLD_CAT12.9.mat')
    with pytest.raises(ValueError):
        NG.main(['--normative-only', '--model', model, '--test',
                 write('NEW', slice(200, 300), ind + 1), '--out', out])


def test_mask_nonconverged(tmp_path, monkeypatch):
    """--mask-nonconverged sets exactly the non-converged features to NaN and leaves
    the others unchanged; without it the z-scores of all features are kept."""
    from scipy.io import loadmat
    Y, age, male, site, _, _ = simulate_shash(n=500, p=30)
    files = {}
    for name, sel in (('TR', slice(0, 350)), ('TE', slice(350, 500))):
        files[name] = str(tmp_path / f's4rp1_8mm_{name}_CAT12.9.mat')
        savemat(files[name], dict(Y=Y[sel].astype(np.float32), age=age[sel][:, None],
                                  male=male[sel][:, None]))
    # too few iterations for the shape fit, so that some features do not converge
    d = list(NG.fit_distribution.__defaults__)
    d[3] = 5                       # 18 of 30 features do not converge with this seed
    monkeypatch.setattr(NG.fit_distribution, '__defaults__', tuple(d))
    model = str(tmp_path / 'norm.npz')
    NG.main(['--normative-only', '--df-mu', '3', '--df-sigma', '2', '--train', files['TR'],
             '--save-model', model, '--out', str(tmp_path / 'tr')])
    monkeypatch.undo()
    out = {}
    for key, extra in (('keep', []), ('mask', ['--mask-nonconverged'])):
        NG.main(['--normative-only', '--model', model, '--test', files['TE'],
                 '--out', str(tmp_path / key)] + extra)
        out[key] = loadmat(str(tmp_path / f'{key}_zmaps_s4rp1_8mm_TE_CAT12.9.mat'),
                           squeeze_me=True)['NDMzmap']
    field = lambda r, k: np.asarray(r[k].item() if r[k].dtype == object else r[k], float)
    conv = field(out['keep'], 'converged')
    zk, zm = field(out['keep'], 'Z'), field(out['mask'], 'Z')
    nc = conv == 0
    assert nc.any() and (conv == 1).any()
    assert np.all(np.isfinite(zk[:, nc]))
    assert np.all(np.isnan(zm[:, nc]))
    assert np.array_equal(zk[:, conv == 1], zm[:, conv == 1])


def test_ensemble_weights_without_adjust(tmp_path):
    """Without --adjust the test subjects are not known to be controls: the ensembles in
    the outputs use equal weights.  With --adjust the gls weights come from the controls."""
    from scipy.io import loadmat
    d = simulate_brain(n=300, p=120)
    noise = np.random.default_rng(5).standard_normal(d.Y.shape)        # second, noisier model
    ds = [d, NG.Data((d.Y + 0.5 * noise).astype(np.float32), d.age, d.male, d.site,
                     'sim_8mm_X.mat', res='8')]                        # of the same subjects
    files = [write_brain(tmp_path, f'{t}_8mm_TR', d, slice(0, 200), ds[0].age[:200])
             for t, d in zip((1, 2), ds)]
    tests = [write_brain(tmp_path, f'{t}_8mm_TE', d, slice(200, 300), ds[0].age[200:])
             for t, d in zip((1, 2), ds)]
    base = ['--train', *files, '--test', *tests, '--pca', '10', '--df-mu', '3', '--df-sigma', '2']
    out = str(tmp_path / 'noadj')
    NG.main(base + ['--out', out])
    r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
    assert r.ensemble_method == 'mean' and np.allclose(r.weights, 0.5)
    assert np.allclose(r.BrainAGE_ensemble, np.mean(r.BrainAGE, axis=1))
    assert np.allclose(r.BrainAGE_GPR_ensemble, np.mean(r.BrainAGE_GPR, axis=1))
    out = str(tmp_path / 'adj')
    NG.main(base + ['--adjust', '1:60', '--out', out])
    r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
    assert r.ensemble_method == 'gls' and not np.allclose(r.weights, 0.5)


def _two_models(n=200, p=60):
    d = simulate_brain(n=n, p=p)
    noise = np.random.default_rng(5).standard_normal(d.Y.shape)
    return [d, NG.Data((d.Y + 0.5 * noise).astype(np.float32), d.age, d.male, d.site,
                       'sim_8mm_Y.mat', res='8')]


def test_kfold_gpr_trend_and_ensemble_are_crossfitted():
    """--kfold: the GPR trend and the ensemble weights of every fold come from the other
    folds (in-sample, r(BrainAGE_GPR, age) was zero by construction)."""
    ds = _two_models()
    cv = NG.cross_validate(ds, kfold=4, gpr=True, df_mu=3, df_sigma=1, pca=10, mask_threshold=0)
    age, fold = ds[0].age, cv['fold'] - 1
    for j in range(2):
        assert abs(np.corrcoef(cv['BrainAGE_GPR'][:, j], age)[0, 1]) > 1e-6
    P = cv['PredictedAge']
    for f in range(4):
        te, tr = fold == f, fold != f
        w = NG.ensemble_weights(P[tr], age[tr], 'gls')
        assert np.allclose(cv['PredictedAge_ensemble'][te], P[te] @ w)


def test_nuisance_share_crossfitted():
    """The in-sample share of a null nuisance correction is positive with few controls,
    the cross-fitted reduction is not; a real effect is found by both."""
    d = simulate_brain(n=200, p=80)
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((d.n, 4))
    rec = NG.remove_nuisance(d, Q, np.arange(20))[1]
    assert rec['explained'] > 0.05 and rec['explained_crossfit'] < 0
    Y = d.Y + (1.5 * Q[:, :1] * rng.standard_normal((1, d.Y.shape[1]))).astype(np.float32)
    rec = NG.remove_nuisance(replace(d, Y=Y), Q, np.arange(100))[1]
    assert rec['explained_crossfit'] > 0.2


def test_crossfitted_accuracy_in_controls(tmp_path):
    from scipy.io import loadmat
    ds = _two_models(n=300)
    files = [write_brain(tmp_path, f'{t}_8mm_TR', d, slice(0, 200), ds[0].age[:200])
             for t, d in zip((1, 2), ds)]
    tests = [write_brain(tmp_path, f'{t}_8mm_TE', d, slice(200, 300), ds[0].age[200:])
             for t, d in zip((1, 2), ds)]
    base = ['--train', *files, '--test', *tests, '--adjust', '1:40', '--pca', '10',
            '--df-mu', '3', '--df-sigma', '2']
    for corr in ('none', 'offset', 'adapt'):
        out = str(tmp_path / corr)
        NG.main(base + ['--correction', corr, '--out', out])
        r = loadmat(out + '.mat', squeeze_me=True, struct_as_record=False)['NDM']
        c = r.ind_control - 1
        others = np.setdiff1d(np.arange(100), c)
        assert np.isfinite(r.BrainAGE_ensemble_crossfit[c]).all()
        assert np.isnan(r.BrainAGE_ensemble_crossfit[others]).all()
        if corr == 'none':      # no correction: single models need no fit, so equal
            assert np.allclose(r.BrainAGE_crossfit[c], r.BrainAGE[c])
    NG.main(base + ['--crossfit-folds', '0', '--out', str(tmp_path / 'off')])
    r = loadmat(str(tmp_path / 'off.mat'), squeeze_me=True, struct_as_record=False)['NDM']
    assert 'BrainAGE_crossfit' not in r._fieldnames


def test_controls_zmaps_leave_one_out():
    """The z-map of a control equals its z-map after adaptation to the other controls;
    patients are unchanged; --controls-in-sample restores mean 0 / SD 1."""
    d = simulate_brain(n=120, p=40)
    vm = NG.VoxelModel(df_mu=3, df_sigma=2, family='normal').fit(d.subset(np.arange(80)))
    te = d.subset(np.arange(80, 120))
    ctrl = np.arange(15)
    z = vm.zmaps(te, ctrl)['Z']
    for i in (0, 7):
        zi = vm.zmaps(te, np.delete(ctrl, i))['Z']
        assert np.allclose(z[i], zi[i], atol=1e-4, equal_nan=True)
    zin = vm.zmaps(te, ctrl, loo_controls=False)['Z']
    assert np.allclose(z[15:], zin[15:], equal_nan=True)
    v = np.isfinite(zin[ctrl]).all(axis=0)
    assert np.allclose(np.mean(zin[ctrl][:, v], axis=0), 0, atol=1e-5)
    assert np.allclose(np.std(zin[ctrl][:, v], axis=0, ddof=1), 1, atol=1e-4)
    assert not np.allclose(np.mean(z[ctrl][:, v], axis=0), 0, atol=1e-3)
