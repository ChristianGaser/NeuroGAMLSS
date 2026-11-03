"""scripts/comcat_data.py: ComCat on the test data of mat-files, then neurogamlss.py.

Uses the simulated neuroGAMLSS-TestData of test_harmonization_comparison.py.  The ADNI
sample (11 sites, one without controls, AD with worse image quality, pMCI and sMCI with
the same IQMs) is harmonized with the diagnosis preserved and scored by normative models
trained on raw data; a v7.3 (HDF5) mat-file must give the same harmonized data as v5.
Skipped without ComCat (COMCAT_DIR or ~/GitHub/ComCat).
"""

import glob
import json
import os
import sys

import numpy as np
import pytest
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import neurogamlss as NG  # noqa: E402
import test_harmonization_comparison as THC  # noqa: E402

pytestmark = pytest.mark.skipif(not THC._comcat_available(),
                                reason="ComCat not found (COMCAT_DIR)")

ADNI = 's6mwp1_4mm_200AD_404MCI_231Normal_CAT12.9.mat'


@pytest.fixture(scope='module')
def adni(tmp_path_factory):
    """Simulated data, the ADNI sample harmonized by comcat_data.py, and its inputs."""
    import comcat_data as CD
    root = str(tmp_path_factory.mktemp('comcat_data'))
    data = THC.simulate(os.path.join(root, 'data'))
    tdir = os.path.join(data, 'TestData', 'ADNI')
    mat = os.path.join(data, 'TestData', ADNI)
    d = NG.load_data(mat)
    names = [str(x) for x in np.ravel(loadmat(mat, squeeze_me=True,
                                              struct_as_record=False)['sample'].data)]
    site_file = os.path.join(root, 'site.txt')
    np.savetxt(site_file, [n.split('_')[1] for n in names], fmt='%s')
    group_file = os.path.join(tdir, 'groups_sc-1.5T_1-AD_2-pMCI_3-sMCI_4-No_5-unclearMCI.txt')
    group = np.loadtxt(group_file).astype(int)
    adjust = ','.join(str(i + 1) for i in np.flatnonzero(group == 4))
    out = os.path.join(root, 'comcat')
    args = ['--test', mat, '--test-site', site_file, '--adjust', adjust,
            '--nuisance', os.path.join(tdir, 'IQM835.txt'), '--nuisance-cols', '1,2,3,5',
            '--preserve', os.path.join(tdir, 'TIV835.txt'), '--preserve-group', group_file]
    CD.main(args + ['--out-dir', out])
    return dict(root=root, data=data, mat=mat, d=d, site_file=site_file, group=group,
                adjust=adjust, args=args, out=os.path.join(out, ADNI))


def _site_r2(Y, site, X):
    """Mean over features of the partial R2 of the sites after the columns X."""
    S = (site[:, None] == np.unique(site)[None, 1:]).astype(float)
    r0 = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    X1 = np.column_stack([X, S])
    r1 = Y - X1 @ np.linalg.lstsq(X1, Y, rcond=None)[0]
    return float(np.mean(1 - np.sum(r1 ** 2, axis=0) / np.sum(r0 ** 2, axis=0)))


def test_output_file(adni):
    """Same name and fields as the input, Y harmonized, design in the JSON file."""
    a, b = loadmat(adni['mat']), loadmat(adni['out'])
    assert set(a) - {'__header__'} == set(b) - {'__header__'}
    for k in ('age', 'male', 'ind', 'dim'):
        assert np.array_equal(a[k], b[k]), k
    assert a['Y'].shape == b['Y'].shape and a['Y'].dtype == b['Y'].dtype
    assert np.all(np.isfinite(b['Y'])) and not np.allclose(a['Y'], b['Y'])
    with open(os.path.splitext(adni['out'])[0] + '.json') as f:
        info = json.load(f)
    assert 'groups preserved' in info['harmonization']
    assert info['nuisance'] == ['1', '2', '3', '5'] and info['preserved'] == ['age', 'sex', '1']
    assert set(info['fold']) == {0, 1, 2, 3, 4}       # all subjects cross-fitted


def test_sites_removed_groups_kept(adni):
    d, group = adni['d'], adni['group']
    H = NG.load_data(adni['out']).Y.astype(np.float64)
    Y = d.Y.astype(np.float64)
    site = np.loadtxt(adni['site_file'], dtype=str)
    tiv = np.loadtxt(os.path.join(adni['data'], 'TestData', 'ADNI', 'TIV835.txt'))
    dummies = (group[:, None] == np.array([1, 2, 3, 5])[None]).astype(float)
    X = np.column_stack([np.ones(d.n), d.age, d.male, tiv, dummies])
    assert _site_r2(H, site, X) < 0.5 * _site_r2(Y, site, X)

    def effects(Z):
        """Mean over voxels of AD - CN and pMCI - sMCI, adjusted for age (the groups
        differ in age by chance, and the simulated age effect is large)."""
        b = np.linalg.lstsq(np.column_stack([X, d.age ** 2]), Z, rcond=None)[0]
        return b[4].mean(), (b[5] - b[6]).mean()
    true_ad = THC.Sim(0).ad.mean()                     # simulate() uses seed 0
    (ad0, mci0), (ad1, mci1) = effects(Y), effects(H)
    # raw: AD effect plus its worse image quality; harmonized: the AD effect; pMCI and
    # sMCI have the same IQMs, so their difference stays
    assert ad0 < 1.5 * true_ad < 0                     # both negative
    assert 0.75 < ad1 / true_ad < 1.25
    assert 0.85 < mci1 / mci0 < 1.15


def test_same_as_harmonize(adni):
    """The file holds what comcat_zmaps.harmonize gives for the same design."""
    import comcat_zmaps as CZ
    d, group = adni['d'], adni['group']
    tdir = os.path.join(adni['data'], 'TestData', 'ADNI')
    Q = np.loadtxt(os.path.join(tdir, 'IQM835.txt'))[:, [0, 1, 2, 4]]
    tiv = np.loadtxt(os.path.join(tdir, 'TIV835.txt'))
    code = NG.read_sites(adni['site_file'], d.n)[0]
    H = CZ.harmonize(CZ.import_comcat(), d.Y, np.flatnonzero(group == 4), code, Q,
                     np.column_stack([d.age, d.male, tiv]), group=group)[0]
    np.testing.assert_allclose(NG.load_data(adni['out']).Y, H, rtol=1e-6, atol=1e-6)


def test_neurogamlss_on_harmonized(adni):
    """neurogamlss.py scores the harmonized file with normative models of raw data."""
    root, data = adni['root'], adni['data']
    tr = os.path.join(data, 'TrainingData')
    keys = (('CamCan651', 'CamCan', 651), ('IXI547', 'IXI', 547), ('SALD494', 'SALD', 494))
    model = os.path.join(root, 'model.npz')
    NG.main(['--normative-only', '--family', 'normal', '--df-mu', '3', '--df-sigma', '1',
             '--train', '+'.join(os.path.join(tr, f's6mwp1_4mm_{k}_CAT12.9.mat')
                                 for k, _, _ in keys),
             '--train-cov', '+'.join(os.path.join(tr, f, 'tables', f'TIV{n}.txt')
                                     for _, f, n in keys),
             '--age-range', '18', '100', '--save-model', model, '--out',
             os.path.join(root, 'model')])
    prefix = os.path.join(root, 'z')
    NG.main(['--normative-only', '--model', model, '--test', adni['out'], '--test-cov',
             os.path.join(data, 'TestData', 'ADNI', 'TIV835.txt'), '--test-site',
             adni['site_file'], '--adjust', adni['adjust'], '--out', prefix])
    Z = loadmat(glob.glob(prefix + '_zmaps_*.mat')[0], squeeze_me=True,
                struct_as_record=False)['NDMzmap'].Z
    z = Z[adni['group'] == 4]
    assert np.all(np.isfinite(Z))
    assert 0.85 < np.median(np.std(z, axis=0, ddof=1)) < 1.2


def _write_v73(path, m):
    """MATLAB v7.3 mat-file (HDF5 with the MATLAB header) with the numeric arrays of m,
    stored transposed as MATLAB does."""
    import h5py
    with h5py.File(path, 'w', userblock_size=512) as h:
        for k, v in m.items():
            h.create_dataset(k, data=np.asarray(v).T)
    head = b'MATLAB 7.3 MAT-file, Platform: test, HDF5 schema 1.00 .'.ljust(116)
    with open(path, 'r+b') as f:
        f.write(head + b' ' * 8 + b'\x00\x02IM')


def test_v73_same_as_v5(adni, tmp_path):
    import comcat_data as CD
    import h5py
    a = loadmat(adni['mat'])
    v73 = os.path.join(str(tmp_path), ADNI)
    _write_v73(v73, {k: a[k] for k in ('Y', 'age', 'male', 'ind', 'dim')})
    out = os.path.join(str(tmp_path), 'comcat')
    args = list(adni['args'])
    args[args.index('--test') + 1] = v73
    CD.main(args + ['--out-dir', out])
    with h5py.File(v73, 'r') as h0, h5py.File(os.path.join(out, ADNI), 'r') as h1:
        assert set(h0) == set(h1)
        for k in ('age', 'male', 'ind', 'dim'):
            assert np.array_equal(h0[k][()], h1[k][()]), k
        assert h1['Y'].shape == h0['Y'].shape and h1['Y'].dtype == h0['Y'].dtype
    np.testing.assert_array_equal(NG.load_data(os.path.join(out, ADNI)).Y,
                                  NG.load_data(adni['out']).Y)


def test_subject_folds(adni, tmp_path):
    """MR-ART: with --subject, the scans of a subject stay in one fold, with the effects
    estimated across subjects or, with --within-subjects, within subjects."""
    import comcat_data as CD
    import comcat_zmaps as CZ
    mat = os.path.join(adni['data'], 'TestData', 's6mwp1_4mm_MR-ART436_CAT12.9.mat')
    names = [str(x) for x in np.ravel(loadmat(mat, squeeze_me=True,
                                              struct_as_record=False)['sample'].data)]
    subject = np.array([nm.split('_')[0][4:] for nm in names])        # sub-<id>
    subj_file = os.path.join(str(tmp_path), 'subject.txt')
    np.savetxt(subj_file, subject, fmt='%s')
    tdir = os.path.join(adni['data'], 'TestData', 'MR-ART', 'tables')
    args = ['--test', mat, '--nuisance', os.path.join(tdir, 'IQM436.txt'), '--nuisance-cols',
            '1,2,3,5', '--preserve', os.path.join(tdir, 'TIV436.txt'), '--subject', subj_file]
    out = {}
    for tag, extra in (('across', []), ('within', ['--within-subjects'])):
        CD.main(args + extra + ['--out-dir', os.path.join(str(tmp_path), tag)])
        base = os.path.join(str(tmp_path), tag, os.path.basename(mat))
        with open(os.path.splitext(base)[0] + '.json') as f:
            fold = np.array(json.load(f)['fold'])
        assert all(np.unique(fold[subject == s]).size == 1 for s in np.unique(subject)), tag
        out[tag] = NG.load_data(base).Y
    d = NG.load_data(mat)
    tiv = np.loadtxt(os.path.join(tdir, 'TIV436.txt'))
    Q = np.loadtxt(os.path.join(tdir, 'IQM436.txt'))[:, [0, 1, 2, 4]]
    CC = CZ.import_comcat()
    P = np.column_stack([d.age, d.male, tiv])
    for tag, terms in (('across', False), ('within', True)):
        H = CZ.harmonize(CC, d.Y, np.arange(d.n), None, Q, P, subject, subject_terms=terms)[0]
        np.testing.assert_allclose(out[tag], H, rtol=1e-6, atol=1e-6, err_msg=tag)
    assert not np.allclose(out['across'], out['within'])


def test_refuses_to_overwrite_input(adni):
    import comcat_data as CD
    with pytest.raises(SystemExit):
        CD.main(adni['args'] + ['--out-dir', os.path.dirname(adni['mat'])])
