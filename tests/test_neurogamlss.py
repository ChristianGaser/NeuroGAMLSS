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
