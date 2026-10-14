#!/usr/bin/env python3
"""Image quality measures (IQMs) and site effects on the log SD in NeuroGAMLSS.

Comparisons on CAT12 gray matter data:

heldout  5-fold cross-validation in the lifespan sample (CamCAN, IXI, OASIS-3, SALD;
         folds stratified by site and age): held-out log-likelihood per subject of
         normal and SHASH models of 1,000 random voxels at the subjects' own sites,
         relative to the model without IQMs and with site in the mean only.  Sites
         are the four samples or, with --scanner-sites, their scanners.
test     models of the lifespan sample applied, without any adaptation, to four
         samples in which biology is fixed and image quality varies: ON-Harmony (10
         subjects on 6 scanners), Tohoku (one subject, 121 acquisitions), Buchert
         (one subject on 116 scanners) and MR-ART (148 subjects scanned without and
         with head motion).  NormBrainAGE of four models (equal-weight ensemble) and
         SHASH z-maps of one model.
testonly the same test samples with models without IQMs; the IQM effects are
         estimated within every test sample instead and removed from the data, by
         --test-nuisance of neurogamlss.py or by ComCat harmonization of the test
         data only, on all scans (in-sample) or on one half of the subjects (scans
         of a single subject) and removed from the other (split).
cv       10-fold cross-validation of NormBrainAGE in the lifespan sample (four
         models): mean absolute error and the brain age offsets of the samples.

The IQMs are the CAT12 quality ratings ICR, NCR, res_ECR, res_RMS, contrastr, SIQR
and IQR (lower is better), read from the tables of get_IQM.m.  The conditions
compare no IQMs, the IQR alone, four basic ratings and all seven, in the mean and
log SD, and site effects on the log SD (--site-sd of neurogamlss.py).  The test
samples contain no sex, so sex is left out of all models.

    python scripts/iqm_comparison.py --part all --out iqm_results

Writes <out>_heldout.csv (<out>_heldout_scanner.csv), <out>_test.csv,
<out>_test.npz (BrainAGE of every model and z-map summaries per scan),
<out>_testonly.csv and <out>_cv.csv.
"""

import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neurogamlss as NG  # noqa: E402

IQM_NAMES = ('ICR', 'NCR', 'res_ECR', 'res_RMS', 'contrastr', 'SIQR', 'IQR')
TRAIN = ('CamCan651', 'IXI547', 'OASIS3_549', 'SALD494')
# the normative tables list CamCan651, IXI547, NKIe629, OASIS3_549 and SALD494 in
# this order; these rows are the training samples
TRAIN_TABLE = 'normativeSample/tables/IQMrall2870.txt'
SITE_TABLE = 'normativeSample/tables/site2870.txt'
TRAIN_ROWS = np.r_[0:1198, 1827:2870]
TESTS = {'ON-Harmony80': 'ON-Harmony/tables/IQMrall80.txt',
         'Tohoku121': 'Tohoku/tables/IQMrall121.txt',
         'Buchert531': 'Buchert/tables/IQMrall531.txt',
         'MRART436': 'MR-ART/tables/IQMrall436.txt'}
MOTION = ('standard', 'headmotion1', 'headmotion2')      # MR-ART: none, medium, strong

BASIC = ('ICR', 'NCR', 'res_ECR', 'contrastr')
# name: (IQMs, in the mean, in the log SD, site in the log SD)
CONDITIONS = {
    'base': ((), False, False, False),
    'site_sd': ((), False, False, True),
    'IQR': (('IQR',), True, True, False),
    'basic_mean': (BASIC, True, False, False),
    'basic': (BASIC, True, True, False),
    'basic_site_sd': (BASIC, True, True, True),
    'all_mean': (IQM_NAMES, True, False, False),
    'all': (IQM_NAMES, True, True, False),
    'all_site_sd': (IQM_NAMES, True, True, True),
}


def covariates(cond):
    names, mean, sd, _ = CONDITIONS[cond]
    if not names:
        return None
    return NG.Covariates(tuple(names), tuple(names) if mean else (), tuple(names) if sd else ())


def with_cov(d, T, cond):
    """Data d with the IQMs (table T, columns IQM_NAMES) of the condition."""
    cov = covariates(cond)
    if cov is None:
        return NG.replace(d, C=None, cov_names=())
    return NG.replace(d, C=T[:, [IQM_NAMES.index(nm) for nm in cov.names]],
                      cov_names=cov.names)


def load_train(args, model, scanner_sites=False):
    """Lifespan training data of one model without sex, and its IQM table."""
    d = NG.load_data('+'.join(os.path.join(args.data_dir, f'{model}_{s}_CAT12.9.mat')
                              for s in TRAIN))
    T = np.loadtxt(os.path.join(args.iqm_dir, TRAIN_TABLE))[TRAIN_ROWS]
    age = np.loadtxt(os.path.join(args.iqm_dir, 'normativeSample/tables/age2870.txt'))[TRAIN_ROWS]
    if len(T) != d.n or not np.allclose(age, d.age):
        raise ValueError(f"{model}: the IQM table does not match the training data.")
    d = NG.replace(d, male=np.zeros_like(d.male))
    if scanner_sites:
        site = np.loadtxt(os.path.join(args.iqm_dir, SITE_TABLE))[TRAIN_ROWS].astype(int)
        for s, n in zip(*np.unique(site, return_counts=True)):
            if n < 20:           # scanners of OASIS-3 with few subjects: its largest scanner
                same = np.unique(site[d.site == d.site[site == s][0]])
                site[site == s] = max(same, key=lambda v: np.sum(site == v))
        d = NG.replace(d, site=np.unique(site, return_inverse=True)[1])
    return d, T


def scan_groups(path, sample):
    """Subject of every scan and, for MR-ART, its motion level (0, 1, 2)."""
    from scipy.io import loadmat
    mat = loadmat(path, variable_names=['sample'], squeeze_me=True, struct_as_record=False)
    files = [os.path.basename(str(f)).strip() for f in np.atleast_1d(mat['sample'].data)]
    if sample in ('Tohoku121', 'Buchert531'):           # one person
        subject = np.zeros(len(files), int)
    else:
        ids = [re.search(r'sub-([^_]+)_', f).group(1) for f in files]
        subject = np.unique(ids, return_inverse=True)[1]
    motion = None
    if sample == 'MRART436':
        motion = np.array([MOTION.index(re.search(r'acq-([^_]+)_', f).group(1)) for f in files])
    return subject, motion


def within_sd(X, subject):
    """Pooled within-subject SD of the rows of X, (n,) or (n, p)."""
    X = np.asarray(X, np.float64)
    ss, dof = 0.0, 0
    for s in np.unique(subject):
        x = X[subject == s]
        if len(x) > 1:
            ss = ss + np.sum((x - x.mean(axis=0)) ** 2, axis=0)
            dof += len(x) - 1
    return np.sqrt(ss / dof)


def paired(X, subject, motion, level):
    """Mean and SE over subjects of X at a motion level minus X without motion."""
    diff = []
    for s in np.unique(subject):
        a = np.flatnonzero((subject == s) & (motion == 0))
        b = np.flatnonzero((subject == s) & (motion == level))
        if a.size and b.size:
            diff.append(np.asarray(X[b[0]], np.float64) - X[a[0]])
    diff = np.array(diff)
    return diff.mean(axis=0), diff.std(axis=0, ddof=1) / np.sqrt(len(diff))


def loglik(m, Y, age, male, site, C):
    """Log-likelihood (n, p_valid) of the valid columns Y at the subjects' own sites."""
    X, W = m._design(age, male, site, C)
    a, s = X @ m.beta, W @ m.theta
    if m.family == 'normal':
        return -s - 0.5 * ((Y - a) * np.exp(-s)) ** 2 - 0.5 * NG.LOG2PI
    V = m._shape_design(age)
    return NG.shash_terms(Y, a, s, V @ m.kappa, V @ m.lam, order=0)[0]


def run_heldout(args):
    rows = []
    for model in args.heldout_models:
        d0, T = load_train(args, model, args.scanner_sites)
        ok = np.flatnonzero(NG.feature_mask(d0)[0])
        cols = np.sort(np.random.default_rng(0).choice(ok, min(args.voxels, ok.size),
                                                       replace=False))
        Y = np.asarray(d0.Y[:, cols], np.float64)
        fold = NG._df_folds(d0, 5, 0)
        dm, ds, _ = NG.select_df_normative(d0)          # df of the model without IQMs
        print(f"{model}: {d0.n} subjects, {np.unique(d0.site).size} sites, df {dm}/{ds}",
              flush=True)
        LL = {}
        for cond in args.conditions:
            d = with_cov(d0, T, cond)
            cov, site_sd = covariates(cond), CONDITIONS[cond][3]
            for family in ('normal', 'shash'):
                t0 = time.time()
                ll = np.full(Y.shape, np.nan)
                for f in range(5):
                    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                    Ct = (lambda idx: None) if d.C is None else (lambda idx: d.C[idx])
                    m = NG.NormativeModel(dm, ds, cov, family=family, site_sd=site_sd).fit(
                        Y[tr], d.age[tr], d.male[tr], d.site[tr], C=Ct(tr))
                    ll[np.ix_(te, m.valid)] = loglik(m, Y[te][:, m.valid], d.age[te],
                                                     d.male[te], d.site[te], Ct(te))
                LL[cond, family] = ll
                print(f"  {cond:12s} {family:6s} {time.time() - t0:5.0f}s", flush=True)
        for family in ('normal', 'shash'):
            base = LL['base', family]
            for cond in args.conditions:
                gain = np.nanmean(LL[cond, family] - base, axis=0)        # per voxel
                row = dict(model=model, sites='scanner' if args.scanner_sites else 'sample',
                           family=family, condition=cond,
                           gain_mnat=1000 * np.nanmedian(gain),
                           pct_voxels_better=100 * np.mean(gain > 0))
                for s in np.unique(d0.site):
                    g = np.nanmean(LL[cond, family][d0.site == s] - base[d0.site == s], axis=0)
                    row[f'gain_mnat_site{s}'] = 1000 * np.nanmedian(g)
                rows.append(row)
    write_csv(rows, args.out + ('_heldout_scanner.csv' if args.scanner_sites else '_heldout.csv'))
    print_rows(rows, ('model', 'family', 'condition', 'gain_mnat', 'pct_voxels_better'))


def load_tests(args):
    """Test samples: data of every model (without sex), IQM table, subject of every
    scan and the MR-ART motion levels."""
    tests = {}
    for t, table in TESTS.items():
        T = np.loadtxt(os.path.join(args.iqm_dir, table))
        ds = []
        for model in args.models:
            d = NG.load_data(os.path.join(args.data_dir, f'{model}_{t}_CAT12.9.mat'))
            ds.append(NG.replace(d, male=np.zeros_like(d.male)))
        if len(T) != ds[0].n:
            raise ValueError(f"{t}: {len(T)} IQM rows for {ds[0].n} scans.")
        path = os.path.join(args.data_dir, f'{args.models[0]}_{t}_CAT12.9.mat')
        tests[t] = (ds, T) + scan_groups(path, t)
    return tests


def test_row(cond, t, P, age, Z, subject, motion, models):
    """Metrics of the predicted ages P (n, models) and z-maps Z (n, voxels) of a test
    sample."""
    BA = P - age[:, None]
    ba = P.mean(axis=1) - age                                       # equal weights
    row = dict(condition=cond, test=t, n=len(age),
               MAE=np.mean(np.abs(ba)), mean_BA=np.mean(ba),
               MAE_offset=np.mean(np.abs(ba - np.median(ba))),
               within_SD_BA=float(within_sd(ba, subject)),
               within_SD_z=float(np.median(within_sd(Z, subject))),
               pct_z258=100 * np.mean(np.abs(Z) > 2.58),
               pct_z329=100 * np.mean(np.abs(Z) > 3.29))
    for j, model in enumerate(models):
        row[f'MAE_{model}'] = np.mean(np.abs(BA[:, j]))
        row[f'within_SD_BA_{model}'] = float(within_sd(BA[:, j], subject))
    if motion is not None:
        for level in (1, 2):
            mb, sb = paired(ba, subject, motion, level)
            mz, _ = paired(Z, subject, motion, level)
            row[f'dBA_{MOTION[level]}'], row[f'dBA_{MOTION[level]}_se'] = mb, sb
            row[f'dz_{MOTION[level]}'] = float(np.mean(mz))
            row[f'abs_dz_{MOTION[level]}'] = float(np.mean(np.abs(mz)))
        for level in range(3):
            extreme = np.abs(Z[motion == level]) > 2.58
            row[f'pct_z258_{MOTION[level]}'] = 100 * np.mean(extreme)
    return row


def print_tests(rows):
    for t in TESTS:
        print(f"\n{t}")
        cols = ['condition', 'MAE', 'mean_BA', 'MAE_offset', 'within_SD_BA', 'within_SD_z',
                'pct_z258']
        if t == 'MRART436':
            cols += ['dBA_headmotion1', 'dBA_headmotion2', 'dz_headmotion2',
                     'pct_z258_standard', 'pct_z258_headmotion2']
        print_rows([r for r in rows if r['test'] == t], cols)


def fit_conditions(args, train, cond):
    """Brain age models of all models and the z-map model of one condition."""
    t0 = time.time()
    kw = dict(df_mu=NG.DF_AUTO, df_sigma=NG.DF_AUTO, pca=100, cov=covariates(cond),
              site_sd=CONDITIONS[cond][3])
    tr = [with_cov(*train[model], cond) for model in args.models]
    sets = NG.fit_models(tr, kw, brainage=True, voxel=False)
    zi = args.models.index(args.zmap_model)
    zset = NG.fit_models([tr[zi]], kw, brainage=False, voxel=True)[0]
    print(f"{cond}: models fitted in {time.time() - t0:.0f}s", flush=True)
    return sets, zset, zi


def run_test(args):
    train = {model: load_train(args, model) for model in args.models}
    tests = load_tests(args)
    rows, store = [], {}
    for cond in args.conditions:
        sets, zset, zi = fit_conditions(args, train, cond)
        m = zset.voxel.model
        for t, (ds, T, subject, motion) in tests.items():
            te = [with_cov(d, T, cond) for d in ds]
            out = NG.apply_models(sets, te, None, 'none', 'mean', gpr=False)
            Z = zset.voxel.zmaps(te[zi])['Z'][:, m.valid].astype(np.float64)
            row = test_row(cond, t, out['PredictedAge'], te[0].age, Z, subject, motion,
                           args.models)
            row['df'] = ' '.join(f"{s.brainage.df_mu}/{s.brainage.df_sigma}" for s in sets)
            row['df_z'] = f"{m.df_mu}/{m.df_sigma}"
            rows.append(row)
            store[f'{cond}/{t}/BrainAGE'] = out['BrainAGE']
            store[f'{cond}/{t}/mean_z'] = Z.mean(axis=1)
            store[f'{cond}/{t}/within_SD_z'] = within_sd(Z, subject)
        write_csv(rows, args.out + '_test.csv')              # after every condition
    np.savez_compressed(args.out + '_test.npz', **store)
    print_tests(rows)


def halves(subject, seed=0):
    """Random halves (0, 1) of the subjects, or of the scans of a single subject."""
    rng = np.random.default_rng(seed)
    u = np.unique(subject)
    if u.size > 1:
        h = np.zeros(u.size, int)
        h[rng.permutation(u.size)[:u.size // 2]] = 1
        return h[np.searchsorted(u, subject)]
    h = np.zeros(len(subject), int)
    h[rng.permutation(len(subject))[:len(subject) // 2]] = 1
    return h


# test-only corrections: name: (method, IQMs, residual_delta of ComCat)
TESTONLY = {
    'nuisance_basic': ('neurogamlss', BASIC, False),
    'nuisance_all': ('neurogamlss', IQM_NAMES, False),
    'comcat_all': ('comcat', IQM_NAMES, False),
    'comcat_all_resdelta': ('comcat', IQM_NAMES, True),
}


def correct(method, d, Q, several, est, app, residual_delta=False):
    """Data of the scans app of the test sample d with the effects of the IQMs Q (n, k)
    removed, estimated on the scans est (the same scans: in-sample).

    neurogamlss : remove_nuisance (--test-nuisance) with est as controls
    comcat      : ComCat harmonization as of the test data only: all scans one site,
                  the IQMs as nuisance (B-splines, gam_df chosen by ComCat) and age
                  preserved if the sample has several subjects; split halves through
                  comcat_from_training with spline bounds covering all scans
    """
    if method == 'neurogamlss':
        return NG.remove_nuisance(d, Q, est)[0].Y[app]
    import comcat as CC
    Y = np.asarray(d.Y, np.float64)
    pres = d.age[:, None] if several else None

    def sub(idx):
        return None if pres is None else pres[idx]
    if np.array_equal(est, app):
        Yh = CC.comcat(Y[est].T, np.ones(est.size, int), Q[est], sub(est),
                       residual_delta=residual_delta)[0]
    else:
        bounds = [(float(Q[:, j].min()), float(Q[:, j].max())) for j in range(Q.shape[1])]
        E = CC.comcat(Y[est].T, np.ones(est.size, int), Q[est], sub(est), return_estimates=True,
                      smooth_term_bounds=bounds, residual_delta=residual_delta)[4]
        Yh = CC.comcat_from_training(Y[app].T, np.ones(app.size, int), Q[app], sub(app),
                                     estimates=E)
    return Yh.T.astype(np.float32)


def run_testonly(args):
    """Models without IQMs applied to test data whose IQM effects were removed within
    the test sample (NeuroGAMLSS --test-nuisance or ComCat), in-sample and split-half."""
    sys.path.insert(0, os.path.expanduser(args.comcat_dir))
    train = {model: load_train(args, model) for model in args.models}
    tests = load_tests(args)
    sets, zset, zi = fit_conditions(args, train, 'base')
    m = zset.voxel.model
    rows = []
    for t, (ds, T, subject, motion) in tests.items():
        age = ds[0].age
        several = np.unique(subject).size > 1
        half = halves(subject)
        everything = np.arange(len(age))
        Z0 = zset.voxel.zmaps(ds[zi])['Z'][:, m.valid].astype(np.float64)
        P0 = NG.apply_models(sets, ds, None, 'none', 'mean', gpr=False)['PredictedAge']
        rows.append(test_row('base', t, P0, age, Z0, subject, motion, args.models))
        for name, (method, iqms, rd) in TESTONLY.items():
            Q = T[:, [IQM_NAMES.index(nm) for nm in iqms]]
            for mode in ('insample', 'split'):
                t0 = time.time()
                parts = ([(everything, everything)] if mode == 'insample' else
                         [(np.flatnonzero(half != h), np.flatnonzero(half == h)) for h in (0, 1)])
                te = []
                for d in ds:
                    Y = np.empty(d.Y.shape, np.float32)
                    for est, app in parts:
                        Y[app] = correct(method, d, Q, several, est, app, rd)
                    te.append(NG.replace(d, Y=Y))
                P = NG.apply_models(sets, te, None, 'none', 'mean', gpr=False)['PredictedAge']
                if method == 'comcat':
                    Z = zset.voxel.zmaps(te[zi])['Z'][:, m.valid].astype(np.float64)
                else:                     # --test-nuisance: z-maps corrected on the z scale
                    Z = np.empty_like(Z0)
                    for est, app in parts:
                        nz = dict(values=Q, ctrl=est, names=list(iqms))
                        Z[app] = zset.voxel.zmaps(ds[zi], nuisance=nz)['Z'][app][:, m.valid]
                rows.append(test_row(f'{name}_{mode}', t, P, age, Z, subject, motion,
                                     args.models))
                print(f"{t} {name} {mode}: {time.time() - t0:.0f}s", flush=True)
            write_csv(rows, args.out + '_testonly.csv')
    print_tests(rows)


def run_cv(args):
    train = {model: load_train(args, model) for model in args.models}
    rows = []
    for cond in args.conditions:
        datas = [with_cov(*train[model], cond) for model in args.models]
        kw = dict(df_mu=NG.DF_AUTO, df_sigma=NG.DF_AUTO, pca=100, cov=covariates(cond),
                  site_sd=CONDITIONS[cond][3])
        out = NG.cross_validate(datas, 10, gpr=False, **kw)
        age, site = datas[0].age, datas[0].site
        ba = out['PredictedAge'].mean(axis=1) - age               # equal weights
        offsets = [np.mean(ba[site == s]) for s in np.unique(site)]
        row = dict(condition=cond, MAE=np.mean(np.abs(ba)),
                   MAE_gls=np.mean(np.abs(out['BrainAGE_ensemble'])),
                   r_BA_age=np.corrcoef(ba, age)[0, 1], SD_site_offsets=np.std(offsets))
        for s, off in zip(np.unique(site), offsets):
            row[f'mean_BA_{TRAIN[s]}'] = off
        for j, model in enumerate(args.models):
            row[f'MAE_{model}'] = np.mean(np.abs(out['BrainAGE'][:, j]))
        rows.append(row)
        write_csv(rows, args.out + '_cv.csv')
    print_rows(rows, ['condition', 'MAE', 'MAE_gls', 'r_BA_age', 'SD_site_offsets']
               + [f'mean_BA_{s}' for s in TRAIN])


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.4f}' if isinstance(r.get(k), (float, np.floating))
                             else str(r.get(k, '')) for k in keys) + '\n')
    print(f"Saved {path}")


def print_rows(rows, cols):
    print('  '.join(f'{c:>14s}' for c in cols))
    for r in rows:
        print('  '.join(f'{r[c]:14.3f}' if isinstance(r[c], (float, np.floating))
                        else f'{str(r[c]):>14s}' for c in cols))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part', choices=['heldout', 'test', 'testonly', 'cv', 'all'],
                   default='all')
    p.add_argument('--data-dir', default='~/Dropbox/BrainAGE',
                   help="mat-files of BA_data2mat.m (<model>_<sample>_CAT12.9.mat)")
    p.add_argument('--iqm-dir', default='~/Dropbox/Harmonization-Oxford',
                   help="IQM, age and site tables of the samples")
    p.add_argument('--models', nargs='+',
                   default=['s6rp1_4mm', 's6rp1_8mm', 's12rp1_4mm', 's12rp1_8mm'],
                   help="brain age models (test)")
    p.add_argument('--zmap-model', default='s6rp1_8mm', help="model of the z-maps (test)")
    p.add_argument('--heldout-models', nargs='+', default=['s6rp1_8mm'],
                   help="models of the held-out comparison")
    p.add_argument('--voxels', type=int, default=1000,
                   help="heldout: number of random voxels (default 1000)")
    p.add_argument('--scanner-sites', action='store_true',
                   help="heldout: scanners as sites instead of samples")
    p.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument('--comcat-dir', default='~/GitHub/ComCat',
                   help="folder of comcat.py (testonly)")
    p.add_argument('--out', default='iqm_comparison')
    args = p.parse_args(argv)
    args.data_dir = os.path.expanduser(args.data_dir)
    args.iqm_dir = os.path.expanduser(args.iqm_dir)
    if 'base' not in args.conditions:
        args.conditions.insert(0, 'base')
    if args.part in ('heldout', 'all'):
        run_heldout(args)
    if args.part in ('test', 'all'):
        run_test(args)
    if args.part in ('testonly', 'all'):
        run_testonly(args)
    if args.part in ('cv', 'all'):
        run_cv(args)


if __name__ == '__main__':
    main()
