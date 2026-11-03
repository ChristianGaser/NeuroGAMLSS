"""Simulated neuroGAMLSS-TestData and a run of scripts/harmonization_comparison.py on it.

simulate() writes the layout of neuroGAMLSS-TestData (TrainingData, TestData and their
tables) with small samples and 200 voxels.  Gray matter is simulated as

    base + age + sex + TIV + subject (biology) + site location + IQM effects + group
         + site scale x scan noise,

so the technical part (sites, IQMs, motion) is known.  The test runs all parts of the
comparison and checks that the conditions recover what was simulated: IQM effects and
motion removed held out (and not with permuted IQMs), site scales equalized by the
location-scale model, the AD effect kept and the pMCI-sMCI difference unchanged.  In
mode insample, it checks the optimism: higher log densities than held out, variance
removed by permuted IQMs, and IQM models across ten subjects that look good in-sample
but make held-out subjects worse.

    python tests/test_harmonization_comparison.py <folder>

simulates into <folder>/data, runs the comparison into <folder>/results and prints the
report.  Needs ComCat (COMCAT_DIR or ~/GitHub/ComCat); the test is skipped without it.
"""

import os
import sys

import numpy as np
import pytest
from scipy.io import savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

DIM = (5, 8, 5)
P = int(np.prod(DIM))
IQM_MEAN = np.array([0.30, 2.0, 2.2, 1.8, 0.30])          # ICR NCR res_ECR res_RMS contrastr
IQM_SD = np.array([0.05, 0.25, 0.30, 0.10, 0.04])


class Sim:
    def __init__(self, seed=0):
        self.rng = rng = np.random.default_rng(seed)
        self.base = rng.uniform(0.35, 0.6, P)
        self.age1 = rng.normal(-0.0025, 0.0007, P)
        self.age2 = rng.normal(-2e-5, 1e-5, P)
        self.sex = rng.normal(0.004, 0.003, P)
        self.tiv = rng.normal(-0.01, 0.004, P)
        self.bio = rng.uniform(0.02, 0.035, P)
        self.tech = rng.uniform(0.006, 0.012, P)
        # effects of the standardized IQMs ICR, NCR, res_ECR, contrastr per voxel
        self.B = np.column_stack([rng.normal(m, 0.003, P) for m in (-0.002, -0.006, -0.008,
                                                                    0.003)])
        self.ad = np.where(rng.random(P) < 0.4, rng.normal(-0.03, 0.008, P), 0.0)
        self.asd = np.where(rng.random(P) < 0.3, -0.006, 0.0)

    def sites(self, k, loc=0.012, scale=0.25):
        return (self.rng.normal(0, loc, (k, P)), np.exp(self.rng.normal(0, scale, k)),
                self.rng.normal(0, 0.5, (k, 5)))

    def iqms(self, n, shift=None, extra=None, spread=1.0, z=None):
        """Raw IQM table (n, 7): ICR NCR res_ECR res_RMS contrastr SIQR IQR, from the
        standardized deviations z (n, 5; default random)."""
        z = self.rng.normal(0, spread, (n, 5)) if z is None else np.array(z, float)
        if shift is not None:
            z = z + shift
        if extra is not None:
            z = z + extra
        q = IQM_MEAN + z * IQM_SD
        ncr, rms, ecr = q[:, 1], q[:, 3], q[:, 2]
        iqr = np.mean(np.column_stack([ncr, rms]) ** 8, axis=1) ** (1 / 8)
        siqr = np.mean(np.column_stack([ncr, rms, ecr]) ** 4, axis=1) ** (1 / 4)
        return np.column_stack([q, siqr, iqr])

    def gm(self, age, male, tiv, Q, subj, site_loc, site_scale, effect=None):
        """Gray matter (n, P) of scans with subject codes subj (biology shared)."""
        rng = self.rng
        n = len(age)
        units = np.unique(subj)
        u = dict(zip(units, rng.normal(0, 1, (len(units), P)) * self.bio))
        zq = (Q[:, [0, 1, 2, 4]] - IQM_MEAN[[0, 1, 2, 4]]) / IQM_SD[[0, 1, 2, 4]]
        a = age - 50
        y = (self.base + np.outer(a, self.age1) + np.outer(a ** 2, self.age2)
             + np.outer(male, self.sex) + np.outer((tiv - 1500) / 100, self.tiv)
             + np.array([u[s] for s in subj]) + site_loc + zq @ self.B.T
             + 0.004 * np.tanh(zq[:, [1]]) * np.sign(self.B[:, 1])
             + rng.normal(0, 1, (n, P)) * self.tech * site_scale[:, None])
        if effect is not None:
            y = y + effect
        return np.clip(y, 0.01, 0.99).astype(np.float32)


def _mat(path, Y, age, male=None, names=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dict(Y=Y, age=np.asarray(age, float)[:, None],
             ind=np.arange(1, P + 1, dtype=float)[:, None], dim=np.array([DIM], float))
    if male is not None:
        d['male'] = np.asarray(male, float)[:, None]
    if names is not None:
        d['sample'] = dict(data=np.array(names, dtype=object))
    savemat(path, d)


def _txt(path, values, fmt='%.6g'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.asarray(values), fmt=fmt)


def simulate(root, seed=0):
    """Training and test data of neuroGAMLSS-TestData with small samples."""
    s = Sim(seed)
    rng = s.rng
    tr = os.path.join(root, 'TrainingData')
    tsd = os.path.join(root, 'TestData')
    train = (('CamCan651', 'CamCan', 651, 150, 18, 88), ('IXI547', 'IXI', 547, 120, 20, 86),
             ('NKIe2213', 'NKIe', 2213, 220, 6, 85), ('OASIS3_549', 'OASIS3', 549, 120, 42, 95),
             ('SALD494', 'SALD', 494, 110, 19, 80), ('NIH879', 'NIH', 879, 160, 5, 22))
    for key, folder, nom, n, lo, hi in train:
        loc, _, qs = s.sites(1, scale=0)
        age, male = rng.uniform(lo, hi, n), rng.integers(0, 2, n).astype(float)
        tiv = rng.normal(1450, 110, n) + 130 * male
        Q = s.iqms(n, qs[0])
        Y = s.gm(age, male, tiv, Q, np.arange(n), loc[0], np.ones(n))
        _mat(os.path.join(tr, f's6mwp1_4mm_{key}_CAT12.9.mat'), Y, age, male)
        _txt(os.path.join(tr, folder, 'tables', f'TIV{nom}.txt'), tiv)
        _txt(os.path.join(tr, folder, 'tables', f'IQM{nom}.txt'), Q)

    def write_test(key, folder, nom, age, male, tiv, Q, subj, site, effect=None, names=None,
             male_in_mat=True, extra=None):
        k = int(site.max()) + 1
        loc, scale, _ = s.sites(k)
        Y = s.gm(age, male, tiv, Q, subj, loc[site], scale[site], effect)
        _mat(os.path.join(tsd, f's6mwp1_4mm_{key}_CAT12.9.mat'), Y, age,
             male if male_in_mat else None, names)
        d = os.path.join(tsd, folder)
        _txt(os.path.join(d, f'TIV{nom}.txt'), tiv)
        _txt(os.path.join(d, f'IQM{nom}.txt'), Q)
        for name, values, fmt in (extra or []):
            _txt(os.path.join(d, name), values, fmt)

    # Tohoku: one subject, 40 acquisitions with widely varying quality, one site
    n = 40
    Q = s.iqms(n, spread=1.6)
    write_test('Tohoku121', 'Tohoku', 121, np.full(n, 35.0), np.ones(n), 1480 + rng.normal(0, 8, n),
         Q, np.zeros(n, int), np.zeros(n, int), male_in_mat=False,
         extra=[('male121.txt', np.ones(n), '%d')])
    # Buchert: one subject on 12 scanners, 4 scans each
    site = np.repeat(np.arange(12), 4)
    _, _, qs = s.sites(12)
    Q = s.iqms(site.size, qs[site])
    write_test('Buchert531', 'Buchert', 531, rng.uniform(49, 51, site.size), np.ones(site.size),
         1550 + rng.normal(0, 8, site.size), Q, np.zeros(site.size, int), site,
         male_in_mat=False, extra=[('male531.txt', np.ones(site.size), '%d'),
                                   ('scannerID531.txt', 1001 + site, '%d')])
    # MR-ART: 30 subjects without, with medium and with strong motion, one scanner
    subj = np.repeat(np.arange(30), 3)
    motion = np.tile([1, 2, 3], 30)
    age = np.repeat(rng.uniform(20, 70, 30), 3)
    male = np.repeat(rng.integers(0, 2, 30), 3).astype(float)
    tiv = np.repeat(rng.normal(1450, 110, 30), 3) + 130 * male + rng.normal(0, 8, subj.size)
    m = (motion - 1)[:, None] * np.array([0.3, 1.0, 1.2, 0.0, -0.2])
    Q = s.iqms(subj.size, extra=m)
    tags = {1: 'standard', 2: 'headmotion1', 3: 'headmotion2'}
    names = [f'mwp1sub-{100000 + i}_acq-{tags[v]}_T1w' for i, v in zip(subj, motion)]
    write_test('MR-ART436', 'MR-ART/tables', 436, age, male, tiv, Q, subj, np.zeros(subj.size, int),
         names=names, extra=[('motion436.txt', motion, '%d')])
    # ON-Harmony: 6 subjects on 4 scanners
    subj = np.repeat(np.arange(6), 4)
    site = np.tile(np.arange(4), 6)
    _, _, qs = s.sites(4)
    age = np.repeat(rng.uniform(24, 48, 6), 4)
    male = np.repeat(rng.integers(0, 2, 6), 4).astype(float)
    tiv = np.repeat(rng.normal(1450, 110, 6), 4) + 130 * male + rng.normal(0, 8, subj.size)
    names = [f'mwp1sub-{i:03d}_ses-{c}_T1w' for i, c in zip(subj, site)]
    write_test('ON-Harmony80', 'ON-Harmony/tables', 80, age, male, tiv, s.iqms(subj.size, qs[site]),
         subj, site, names=names, extra=[('scanner80.txt', site + 1, '%d')])
    # ABIDE: 120 children on 6 scanners, autism (1) and typically developing (2)
    n = 120
    site = np.arange(n) % 6
    _, _, qs = s.sites(6)
    grp = np.where(rng.random(n) < 0.5, 1, 2)
    age = rng.uniform(10, 18, n)
    male = (rng.random(n) < 0.85).astype(float)
    tiv = rng.normal(1450, 110, n) + 130 * male
    asd = (grp == 1)[:, None] * np.array([0.0, 0.3, 0.3, 0.0, 0.0])
    write_test('ABIDE437', 'ABIDE', 437, age, male, tiv, s.iqms(n, qs[site], asd), np.arange(n), site,
         effect=np.outer(grp == 1, s.asd),
         extra=[('Scanner437.txt', np.array([f'SITE_{c}' for c in site]), '%s'),
                ('DX_group437.txt', grp, '%d')])
    # ADNI: 160 subjects of 11 sites; AD 1, pMCI 2, sMCI 3, CN 4, uMCI 5; site 10 without
    # controls (AD and uMCI only); AD with worse image quality; pMCI and sMCI with the same
    # mean IQMs (no chance difference), so that the IQM correction must not change their
    # difference
    n = 160
    site = np.arange(n) % 11
    grp = rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.16, 0.22, 0.32, 0.08])
    grp[site == 10] = np.where(grp[site == 10] == 1, 1, 5)
    _, _, qs = s.sites(11)
    age = rng.uniform(55, 90, n)
    male = rng.integers(0, 2, n).astype(float)
    tiv = rng.normal(1450, 110, n) + 130 * male
    sev = np.select([grp == 1, grp == 2, grp == 3], [1.0, 0.6, 0.2], 0.0)
    worse = (grp == 1)[:, None] * np.array([0.5, 0.5, 0.5, 0.0, 0.0])
    dev = rng.normal(0, 1, (n, 5))
    full = dev + qs[site]
    target = full[np.isin(grp, (2, 3))].mean(axis=0)
    for g in (2, 3):
        dev[grp == g] += target - full[grp == g].mean(axis=0)
    names = [f'mwp1ADNI_{100 + c:03d}_S_{1000 + i:04d}_MR_MPRAGE' for i, c in enumerate(site)]
    write_test('200AD_404MCI_231Normal', 'ADNI', 835, age, male, tiv,
               s.iqms(n, qs[site], worse, z=dev), np.arange(n), site,
               effect=np.outer(sev, s.ad), names=names,
               extra=[('groups_sc-1.5T_1-AD_2-pMCI_3-sMCI_4-No_5-unclearMCI.txt', grp, '%d')])
    return root


def _comcat_available():
    try:
        import comcat_zmaps as CZ
        CZ.import_comcat()
        return True
    except ImportError:
        return False


def run(folder, quiet=False):
    import harmonization_comparison as HC
    data, out = os.path.join(folder, 'data'), os.path.join(folder, 'results')
    simulate(data)
    argv = ['--root', data, '--out', out, '--part', 'all', '--jobs', '2']
    if quiet:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            HC.main(argv)
    else:
        HC.main(argv)
    import csv
    with open(os.path.join(out, 'comparison.csv')) as f:
        return {(r['sample'], r['condition'], r['metric'], r['mode']): float(r['value'])
                for r in csv.DictReader(f)}


@pytest.mark.skipif(not _comcat_available(), reason="ComCat not found (COMCAT_DIR)")
def test_harmonization_comparison(tmp_path):
    v = run(str(tmp_path), quiet=True)

    def g(sample, cond, metric, mode='crossfit'):
        return v[(sample, cond, metric, mode)]

    samples = {k[0] for k in v}
    assert samples == {'Tohoku', 'Buchert', 'MR-ART', 'ON-Harmony', 'ABIDE', 'ADNI'}
    assert all(np.isfinite(val) for k, val in v.items() if k[2] == 'log_density')
    # IQM effects removed held out in the single subject, not with permuted IQMs
    w0 = g('Tohoku', 'gamlss', 'within_sd')
    for c in ('gamlss_test_nuisance', 'comcat', 'comcat_z', 'gamlss_iqm_model'):
        assert g('Tohoku', c, 'within_sd') < 0.8 * w0, c
    for c in ('gamlss_test_nuisance_perm', 'comcat_perm', 'comcat_z_perm'):
        assert g('Tohoku', c, 'within_sd') > 0.95 * w0, c
    # motion effect removed, not with permuted IQMs
    m0 = abs(g('MR-ART', 'gamlss', 'motion_strong'))
    for c in ('gamlss_test_nuisance', 'comcat', 'comcat_z', 'comcat_subj', 'comcat_z_subj'):
        assert abs(g('MR-ART', c, 'motion_strong')) < 0.5 * m0, c
    assert abs(g('MR-ART', 'comcat_z_perm', 'motion_strong')) > 0.8 * m0
    # site scales: the location-scale model equalizes them, location only does not
    for s in ('Buchert', 'ABIDE'):
        assert g(s, 'comcat', 'site_scale_rms') < g(s, 'comcat_loc', 'site_scale_rms'), s
        assert g(s, 'comcat_z', 'site_scale_rms') < g(s, 'comcat_z_loc', 'site_scale_rms'), s
    # site offsets: adapted in every condition
    for s in ('Buchert', 'ABIDE', 'ADNI'):
        assert g(s, 'gamlss', 'site_offset_rms') < 0.3, s
    # AD effect kept; the IQM correction removes the quality part of AD-CN (mean difference
    # of the location-adapted z), not the difference of pMCI and sMCI, whose IQMs agree
    d0 = g('ADNI', 'gamlss', 'diff_AD_CN')
    m0 = g('ADNI', 'gamlss', 'diff_pMCI_sMCI')
    for c in ('comcat', 'comcat_z', 'comcat_dx', 'comcat_z_dx', 'gamlss_test_nuisance'):
        assert 0.3 * d0 > g('ADNI', c, 'diff_AD_CN') > 0.8 * d0, c     # both negative
        ratio = g('ADNI', c, 'diff_pMCI_sMCI') / m0
        assert 0.75 < ratio < 1.25, (c, ratio)
    # calibration of the held-out controls, without and with ComCat (location and scale,
    # adaptation from cross-fitted values)
    for s in ('ABIDE', 'ADNI'):
        for c in ('gamlss', 'comcat', 'comcat_z', 'comcat_dx'):
            assert 0.85 < g(s, c, 'z_sd') < 1.2, (s, c)
    # in-sample: optimistic for every correction fitted on the test sample
    for s in samples:
        for c in ('gamlss_test_nuisance', 'comcat', 'comcat_z'):
            assert g(s, c, 'log_density', 'insample') > g(s, c, 'log_density'), (s, c)
    # permuted IQMs remove variance in-sample only (fitted noise)
    w1 = g('Tohoku', 'gamlss', 'within_sd', 'insample')
    for c in ('gamlss_test_nuisance_perm', 'comcat_perm', 'comcat_z_perm'):
        assert g('Tohoku', c, 'within_sd', 'insample') < 0.97 * w1, c
    # IQMs across ten subjects: in-sample hides that held-out subjects get worse
    for c in ('comcat', 'comcat_z'):
        assert (g('ON-Harmony', c, 'within_sd', 'insample')
                < 0.5 * g('ON-Harmony', c, 'within_sd')), c


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else 'harmonization_simulation'
    run(folder)
    with open(os.path.join(folder, 'results', 'comparison.md')) as f:
        print(f.read())
