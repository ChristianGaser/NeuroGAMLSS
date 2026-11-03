#!/usr/bin/env python3
"""Sites and image quality in NeuroGAMLSS z-maps: ComCat on the data, or terms of the GAMLSS.

Compares, on the samples of neuroGAMLSS-TestData, where the effects of the test sites
and of image quality measures (IQMs) are handled.  Approach 1, the main one, is the same
as for BrainAGE: ComCat harmonizes the test data (scripts/comcat_data.py), and the
normative models, trained on raw data, score them.  Its alternative applies ComCat to
the normal scores instead (scripts/comcat_zmaps.py).  Approach 2 leaves the data
unchanged and handles sites and IQMs in the normative models and their adaptation to the
test sites (neurogamlss.py).

All conditions share the normative models (s6mwp1 4 mm, SHASH, TIV in the mean, the
training samples as sites) and the adaptation to the test sites (offsets shrunk toward
the mean of all controls, one common scale).  Two modes (--modes):

  crossfit  every estimate from the test sample (adaptation, IQM effects, ComCat) is
            cross-fitted: the controls are split into folds (stratified by site, all
            scans of a subject in one fold), every fold is scored with estimates from
            the other folds, and the patients with estimates from all controls.  The
            adaptation is estimated from the training controls' corrected values, which
            are themselves cross-fitted within the training folds (--inner-folds):
            corrected in-sample, they would be too close to the fitted model, and the
            common scale too small.
  insample  every estimate is fitted on all controls (on all subjects for the *_dx
            conditions) and applied to all scans, the scored controls included; the
            adaptation uses the controls' in-sample corrected values.  Optimistic by
            construction: the difference to crossfit shows how much of a gain is fitted
            noise.

Conditions
  gamlss                training sites in the mean, test sites adapted; no IQMs (the
                        reference of the report)
  approach 1: ComCat on the test data, then the adaptation
  comcat                sites (location and scale, EB, df-consistent site variances after
                        the IQM effects) and the four IQMs (P-splines, REML) removed, age,
                        sex and TIV preserved, fitted on the controls (comcat_data.py)
  combat                ComBat on the test data (location and scale, empirical Bayes,
                        ComBat's site variances) [several sites]
  alternative to approach 1: ComCat on the normal scores, then the adaptation
  comcat_z              as comcat (comcat_zmaps.py)
  approach 2: terms of the GAMLSS
  gamlss_test_nuisance  + IQM effects estimated in the controls within the sites and
                        removed from the normal scores (--test-nuisance)
  gamlss_iqm_model      + ICR, NCR, res_ECR and contrastr in the mean of the normative
                        models (--train-cov, --cov-mean)
Variants of comcat and comcat_z
  comcat_loc, comcat_z_loc         location only [several sites]
  comcat_subj, comcat_z_subj       effects within subjects [repeated scans of several
                                   subjects]
  comcat_dx, comcat_z_dx           fitted on all subjects with the diagnosis preserved,
                                   every subject cross-fitted [samples with patients]
  comcat_iqm, comcat_z_iqm         IQMs only; the test sites only by the adaptation
                                   [several sites]
  comcat_iqm_dx, comcat_z_iqm_dx   the same, diagnosis preserved (ADNI design)
Single-subject samples (Tohoku, Buchert) preserve nothing in ComCat: their age and TIV
vary only with the scanning date or technically.
  *_perm                           IQMs permuted across scans (control for flexibility)

Metrics (controls, held out in mode crossfit; final z unless stated; median over voxels)
  log_density       mean log density of the data of the controls, nats per subject and
                    voxel: comparable across all conditions (proper score)
  z_sd, z_mean      SD and mean of the controls' z (1 and 0 if calibrated)
  beyond_2.58       share of |z| > 2.58, pooled (1% if calibrated)
  site_offset_rms   RMS of the deviations of the site means of the controls' corrected
                    scores before the adaptation (which removes them), sampling noise
                    subtracted: the site differences a correction leaves [several sites]
  site_scale_rms    RMS of the log SD of the controls' z per site, sampling noise
                    subtracted [several sites]
  within_sd         within-subject SD of the location-adapted z (before the common
                    scale) [repeated scans]
  between_sd        SD of the subjects' means of the same [several subjects]
  motion_<level>    MR-ART: mean over voxels of the mean paired difference of the
                    location-adapted z to the scan without motion, and of its absolute
                    value (_abs)
  t_<a>_<b>, d_..   median |t| (Welch) and median Cohen's d of group a vs group b
  diff_<a>_<b>      mean over voxels of the difference of the groups' mean location-
                    adapted z (units of the normative SD); unlike t and d, it does not
                    grow when a correction removes noise

Parts
  models   normative models of the adult and child training samples with TIV, and with
           TIV and the IQMs, in the mean (saved in --out; existing files are reused)
  compare  all conditions for the test samples, in both modes -> <out>/comparison.csv
  report   <out>/comparison.md (one table per sample and mode)

    python scripts/harmonization_comparison.py --root ~/Dropbox/neuroGAMLSS-TestData \\
        --out harmonization --part all

or scripts/run_harmonization_comparison.sh (all parts with logs).  ComCat is imported from
--comcat-dir, COMCAT_DIR or ~/GitHub/ComCat.  tests/test_harmonization_comparison.py runs
the comparison on simulated data with the layout of neuroGAMLSS-TestData.
"""

import argparse
import csv
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import neurogamlss as NG  # noqa: E402
import comcat_zmaps as CZ  # noqa: E402

FEATURE = 's6mwp1_4mm'
IQM4 = ('ICR', 'NCR', 'res_ECR', 'contrastr')
IQM_COLUMNS = (0, 1, 2, 4)   # CAT12 IQM tables: ICR NCR res_ECR res_RMS contrastr SIQR IQR

# training samples: (key of the mat-file, folder of the tables, n), age range
TRAIN = {
    'adult': dict(samples=(('CamCan651', 'CamCan', 651), ('IXI547', 'IXI', 547),
                           ('NKIe2213', 'NKIe', 2213), ('OASIS3_549', 'OASIS3', 549),
                           ('SALD494', 'SALD', 494)), age=(18, 100)),
    'child': dict(samples=(('NIH879', 'NIH', 879), ('NKIe2213', 'NKIe', 2213)), age=(5, 30)),
}

# test samples: mat-file key, training sample, folder and n of the tables (TIV<n>.txt,
# IQM<n>.txt), sex table if the mat-file has none, sites (table or pattern in the
# file names), subjects ('one' or pattern), MR-ART motion, groups, folds
TEST = {
    'Tohoku': dict(file='Tohoku121', train='adult', tables='Tohoku', n=121,
                   male='male121.txt', subject='one', folds=10),
    'Buchert': dict(file='Buchert531', train='adult', tables='Buchert', n=531,
                    male='male531.txt', site_table='scannerID531.txt', subject='one',
                    folds=3),
    'MR-ART': dict(file='MR-ART436', train='adult', tables='MR-ART/tables', n=436,
                   motion='motion436.txt', subject=r'sub-([A-Za-z0-9]+)', folds=5),
    'ON-Harmony': dict(file='ON-Harmony80', train='adult', tables='ON-Harmony/tables', n=80,
                       male='male80.txt', site_table='scanner80.txt',
                       subject=r'sub-([A-Za-z0-9]+)', folds=10),
    'ABIDE': dict(file='ABIDE437', train='child', tables='ABIDE', n=437,
                  site_table='Scanner437.txt', group='DX_group437.txt', control=2,
                  groups={1: 'ASD', 2: 'TD'}, contrasts=(('ASD', 'TD'),), folds=5),
    'ADNI': dict(file='200AD_404MCI_231Normal', train='adult', tables='ADNI', n=835,
                 site_pattern=r'_(\d{3})_S_\d{4}',
                 group='groups_sc-1.5T_1-AD_2-pMCI_3-sMCI_4-No_5-unclearMCI.txt', control=4,
                 groups={1: 'AD', 2: 'pMCI', 3: 'sMCI', 4: 'CN', 5: 'uMCI'},
                 contrasts=(('AD', 'CN'), ('pMCI', 'sMCI')), folds=5),
}

CONDITIONS = {   # name: (label, spec); the order of the report
    'gamlss': ("GAMLSS: sites only", dict(model=0)),
    # approach 1: ComCat on the data (comcat_data.py), as for BrainAGE
    'comcat': ("ComCat on the data", dict(level='data')),
    'comcat_loc': ("ComCat on the data, location only", dict(level='data', loc=True,
                                                             sites=True)),
    'comcat_subj': ("ComCat on the data, within subjects", dict(level='data', subj=True)),
    'comcat_dx': ("ComCat on the data, diagnosis preserved", dict(level='data', dx=True)),
    'comcat_iqm': ("ComCat on the data, IQMs only", dict(level='data', nosites=True,
                                                         sites=True)),
    'comcat_iqm_dx': ("ComCat on the data, IQMs only, diagnosis preserved",
                      dict(level='data', nosites=True, sites=True, dx=True)),
    'combat': ("ComBat on the data", dict(level='data', iqm=False, sites=True, cc=True)),
    # alternative: ComCat on the normal scores (comcat_zmaps.py)
    'comcat_z': ("ComCat on the normal scores", dict(level='z')),
    'comcat_z_loc': ("ComCat on the normal scores, location only", dict(level='z', loc=True,
                                                                        sites=True)),
    'comcat_z_subj': ("ComCat on the normal scores, within subjects", dict(level='z',
                                                                           subj=True)),
    'comcat_z_dx': ("ComCat on the normal scores, diagnosis preserved", dict(level='z',
                                                                             dx=True)),
    'comcat_z_iqm': ("ComCat on the normal scores, IQMs only", dict(level='z', nosites=True,
                                                                    sites=True)),
    'comcat_z_iqm_dx': ("ComCat on the normal scores, IQMs only, diagnosis preserved",
                        dict(level='z', nosites=True, sites=True, dx=True)),
    # approach 2: terms of the GAMLSS
    'gamlss_test_nuisance': ("GAMLSS: IQMs in the test controls", dict(level='nuisance')),
    'gamlss_iqm_model': ("GAMLSS: IQMs in the normative model", dict(model=1)),
    'gamlss_test_nuisance_perm': ("GAMLSS: permuted IQMs in the test controls",
                                  dict(level='nuisance', perm=True)),
    'comcat_perm': ("ComCat on the data, permuted IQMs", dict(level='data', perm=True)),
    'comcat_z_perm': ("ComCat on the normal scores, permuted IQMs", dict(level='z',
                                                                         perm=True)),
}


def model_path(out, train, cov):
    return os.path.join(out, f'model_{train}_{cov}.npz')


# ---------------------------------------------------------------------------
# Normative models
# ---------------------------------------------------------------------------

def _column(path, n=None, dtype=float):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    v = np.loadtxt(path, dtype=dtype, ndmin=1)
    if n is not None and v.shape[0] != n:
        raise ValueError(f"{path}: {v.shape[0]} rows, expected {n}.")
    return v


def _iqm(path, n=None):
    v = np.loadtxt(os.path.expanduser(path), ndmin=2)
    if v.shape[1] < 5:
        raise ValueError(f"{path}: expected the 7 columns of a CAT12 IQM table.")
    if n is not None and v.shape[0] != n:
        raise ValueError(f"{path}: {v.shape[0]} rows, expected {n}.")
    return v[:, IQM_COLUMNS]


def _write_table(path, header, columns):
    """Table of the columns, with a header line unless header is None."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        if header is not None:
            f.write(' '.join(header) + '\n')
        for row in np.column_stack(columns):
            f.write(' '.join('NaN' if not np.isfinite(v) else f'{v:.6g}' for v in row) + '\n')


def fit_models(args, trains):
    """Normative models with TIV (gamlss) and with TIV and the IQMs (gamlss_iqm_model)
    in the mean, fitted to the same training subjects (those with all IQMs)."""
    for train in trains:
        for cov in ('tiv', 'tiv_iqm'):
            path = model_path(args.out, train, cov)
            if os.path.exists(path):
                print(f"{path} exists and is used")
                continue
            files, tables = [], []
            for key, folder, n in TRAIN[train]['samples']:
                files.append(os.path.join(args.root, 'TrainingData',
                                          f'{FEATURE}_{key}_CAT12.9.mat'))
                tdir = os.path.join(args.root, 'TrainingData', folder, 'tables')
                tiv = _column(os.path.join(tdir, f'TIV{n}.txt'))
                iqm = _iqm(os.path.join(tdir, f'IQM{n}.txt'), tiv.size)
                tiv = np.where(np.all(np.isfinite(iqm), axis=1), tiv, np.nan)
                tab = os.path.join(args.out, 'tables', f'train_{key}_{cov}.txt')
                if cov == 'tiv':           # no header, as TIV<n>.txt of the test samples
                    _write_table(tab, None, [tiv])
                else:
                    _write_table(tab, ['TIV', *IQM4], [tiv, iqm])
                tables.append(tab)
            lo, hi = TRAIN[train]['age']
            argv = ['--normative-only', '--train', '+'.join(files), '--train-cov',
                    '+'.join(tables), '--age-range', str(lo), str(hi), '--save-model', path,
                    '--out', os.path.splitext(path)[0]]
            if args.jobs:
                argv += ['--jobs', str(args.jobs)]
            argv += args.ng_options.split() if args.ng_options else []
            print(f"== fitting {path}: neurogamlss.py {' '.join(argv)}", flush=True)
            NG.main(argv)


def load_model(path):
    sets, desc = NG.load_models(path)
    if len(sets) != 1 or sets[0].voxel is None:
        raise ValueError(f"{path}: expected the voxel-wise models of one input.")
    cov = NG.Covariates.from_state(desc['covariates']) if desc.get('covariates') else None
    return SimpleNamespace(vm=sets[0].voxel, cov=cov, uses_sex=NG._uses_sex(sets[0]))


# ---------------------------------------------------------------------------
# Test samples
# ---------------------------------------------------------------------------

def load_sample(args, name, rng):
    """Test data and tables of a sample, restricted to the subjects with a valid age,
    TIV and IQMs."""
    cfg = TEST[name]
    path = os.path.join(args.root, 'TestData', f"{FEATURE}_{cfg['file']}_CAT12.9.mat")
    d = NG.load_data(path)
    n = d.n
    tdir = os.path.join(args.root, 'TestData', cfg['tables'])
    tiv = _column(os.path.join(tdir, f"TIV{cfg['n']}.txt"), n)
    iqm = _iqm(os.path.join(tdir, f"IQM{cfg['n']}.txt"), n)
    male = d.male if d.has_male else None
    if male is None and cfg.get('male') and os.path.isfile(os.path.join(tdir, cfg['male'])):
        male = _column(os.path.join(tdir, cfg['male']), n)
    names = None
    if cfg.get('site_pattern') or str(cfg.get('subject', 'one')) != 'one':
        names = CZ.read_names(path)
        if names is None or len(names) != n:
            raise ValueError(f"{path}: needs the file names of the subjects (sample.data).")
    site = np.zeros(n, dtype=object)
    if cfg.get('site_table'):
        site = _column(os.path.join(tdir, cfg['site_table']), n, str)
    elif cfg.get('site_pattern'):
        site = CZ.labels_from_names(names, cfg['site_pattern'])
    subject = units = None
    if cfg.get('subject') == 'one':
        subject = np.zeros(n, dtype=np.int64)
    elif cfg.get('subject'):
        subject = units = CZ.labels_from_names(names, cfg['subject'])
    motion = _column(os.path.join(tdir, cfg['motion']), n).astype(int) if cfg.get('motion') \
        else None
    group = _column(os.path.join(tdir, cfg['group']), n).astype(int) if cfg.get('group') \
        else None

    ok = np.isfinite(d.age) & (d.age > 0) & np.isfinite(tiv) & np.all(np.isfinite(iqm), axis=1)
    if male is not None:
        ok &= np.isfinite(male)
    rows = np.flatnonzero(ok)
    if rows.size < n:
        print(f"  {n - rows.size} subject(s) without valid age, TIV or IQMs are left out")

    def sub(a):
        return None if a is None else np.asarray(a)[rows]

    _, code = np.unique(sub(site).astype(str), return_inverse=True)
    S = SimpleNamespace(name=name, cfg=cfg, Y=d.Y[rows], age=d.age[rows], male=sub(male),
                        tiv=tiv[rows], iqm=iqm[rows], code=code.astype(np.int64),
                        k=int(code.max()) + 1, subject=sub(subject), units=sub(units),
                        motion=sub(motion), group=sub(group))
    S.n = rows.size
    S.ctrl = (np.ones(S.n, dtype=bool) if group is None else S.group == cfg['control'])
    S.repeated = S.subject is not None and np.unique(S.subject).size < S.n
    S.several_subjects = S.repeated and np.unique(S.subject).size > 1
    S.iqm_perm = S.iqm[rng.permutation(S.n)]
    S.seed, S.inner = args.seed, int(args.inner_folds)
    import comcat as CC
    k = int(args.folds or cfg['folds'])
    fc = CC._make_folds(S.code, S.units, S.ctrl, k, args.seed)
    fp = (CC._make_folds(S.code, S.units, ~S.ctrl, k, args.seed + 1) if (~S.ctrl).any()
          else np.full(S.n, -1))
    S.fold = np.where(S.ctrl, fc, fp)
    return S


def covariate_matrix(S, model):
    """Columns of the model's covariates: TIV ('TIV', or '1' for a table without header)
    and the IQMs."""
    if model.cov is None:
        return None
    cols = dict(TIV=S.tiv, **{'1': S.tiv}, **{q: S.iqm[:, j] for j, q in enumerate(IQM4)})
    missing = [nm for nm in model.cov.names if nm not in cols]
    if missing:
        raise ValueError(f"unknown covariate(s) of the model: {', '.join(missing)}")
    return np.column_stack([cols[nm] for nm in model.cov.names])


def prepare(S, models, args, rng):
    """Normal scores and log densities of all subjects under the models with TIV (Z0, L0)
    and with TIV and IQMs (Z1, L1) at the reference site, on the voxels valid in both."""
    m0, m1 = models
    for m in models:
        if m.vm.model.n_features != S.Y.shape[1]:
            raise ValueError(f"{S.name}: {S.Y.shape[1]} features, the models "
                             f"{m.vm.model.n_features}.")
        if m.uses_sex and S.male is None:
            raise ValueError(f"{S.name}: the models use sex, but there is no sex: add "
                             f"{S.cfg.get('male', 'a male table')} (1 = male).")
    male = S.male if S.male is not None else np.zeros(S.n)
    S.male_used = male
    v0, v1 = m0.vm.model.valid, m1.vm.model.valid
    feat = np.intersect1d(v0, v1)
    if args.features and feat.size > args.features:
        feat = np.sort(rng.choice(feat, args.features, replace=False))
    R = SimpleNamespace(feat=feat, pos0=np.searchsorted(v0, feat),
                        pos1=np.searchsorted(v1, feat), C0=covariate_matrix(S, m0),
                        C1=covariate_matrix(S, m1), vm0=m0.vm)
    Z0, L0 = CZ.reference_scores(m0.vm, S.Y, S.age, male, R.C0, density=True)
    Z1, L1 = CZ.reference_scores(m1.vm, S.Y, S.age, male, R.C1, density=True)
    R.Z0, R.L0 = Z0[:, R.pos0], L0[:, R.pos0]
    R.Z1, R.L1 = Z1[:, R.pos1], L1[:, R.pos1]
    keep = np.all(np.isfinite(R.Z0) & np.isfinite(R.Z1) & np.isfinite(R.L0)
                  & np.isfinite(R.L1), axis=0)
    if not keep.all():
        print(f"  {int(np.sum(~keep))} voxel(s) with non-finite scores are left out")
        for a in ('feat', 'pos0', 'pos1'):
            setattr(R, a, getattr(R, a)[keep])
        R.Z0, R.L0, R.Z1, R.L1 = R.Z0[:, keep], R.L0[:, keep], R.Z1[:, keep], R.L1[:, keep]
    R.Ydata = np.asarray(S.Y[:, R.feat], np.float64)
    # one subject: age and TIV vary only with the scanning date or technically
    preserve = [] if S.cfg.get('subject') == 'one' else [S.age, male, S.tiv]
    R.preserve = np.column_stack(preserve) if preserve else None
    if S.group is not None:
        g = S.group
        dummies = [(g == lv).astype(float) for lv in np.unique(g[~S.ctrl])
                   if lv not in set(g[S.ctrl])]
        R.preserve_dx = np.column_stack(preserve + dummies) if dummies else R.preserve
    return R


def data_scores(S, R, rows, H):
    """Normal scores and log densities of the rows with the harmonized data H
    (rows, n_feat) under the model with TIV."""
    Y = np.array(S.Y[rows], dtype=np.float64)
    Y[:, R.feat] = H
    Z, L = CZ.reference_scores(R.vm0, Y, S.age[rows], S.male_used[rows],
                               None if R.C0 is None else R.C0[rows], density=True)
    return Z[:, R.pos0], L[:, R.pos0]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def applicable(S, name):
    spec = CONDITIONS[name][1]
    if spec.get('sites') and S.k < 2:
        return False
    if spec.get('subj') and not S.several_subjects:
        return False
    if spec.get('dx') and S.group is None:
        return False
    return True


def passes(S, dx=False, mode='crossfit'):
    """(fit, adapt, apply) index arrays: the rows that the correction is fitted on, the
    controls of the adaptation and the rows scored.  Mode insample: one pass with the
    correction fitted on all controls (all subjects with dx), the adaptation on all
    controls and every scan scored."""
    if mode == 'insample':
        c = np.flatnonzero(S.ctrl)
        yield (np.arange(S.n) if dx else c), c, np.arange(S.n)
        return
    labels = np.unique(S.fold[S.fold >= 0])
    if dx:
        for f in labels:
            te = np.flatnonzero(S.fold == f)
            yield np.flatnonzero(S.fold != f), np.flatnonzero(S.ctrl & (S.fold != f)), te
        return
    for f in labels:
        te = np.flatnonzero(S.ctrl & (S.fold == f))
        if te.size:
            tr = np.flatnonzero(S.ctrl & (S.fold != f))
            yield tr, tr, te
    if (~S.ctrl).any():
        c = np.flatnonzero(S.ctrl)
        yield c, c, np.flatnonzero(~S.ctrl)


def corrector(CC, S, R, spec):
    """correct(fit, rows) -> the corrected values of rows (normal scores, or data for
    ComCat on the data) and their derivative with respect to the uncorrected values,
    with the correction fitted on the rows fit."""
    level = spec['level']
    Q = S.iqm_perm if spec.get('perm') else S.iqm
    Q = Q if spec.get('iqm', True) else None
    P = R.preserve_dx if spec.get('dx') else R.preserve
    subject = S.subject if spec.get('subj') else None
    batch = None if spec.get('nosites') else S.code
    opts = CZ.comcat_options(spec.get('loc', False), spec.get('cc', False))

    def correct(fit, rows):
        if level == 'nuisance':
            N, Pm, _, _ = NG.nuisance_design(Q, S.age, S.male_used, fit, 1, list(IQM4),
                                             S.code if S.k > 1 else None)
            return R.Z0[rows] - N[rows] @ (Pm @ R.Z0[fit]), np.ones((rows.size, 1))
        X = R.Z0 if level == 'z' else R.Ydata
        _, H, D = CZ.comcat_fit_apply(CC, X, fit, rows, batch, Q, P, subject, slope=True,
                                      **opts)
        return H, D
    return correct


def inner_crossfit(CC, S, correct, fit, rows):
    """Corrected values of rows (a subset of fit), each from a correction fitted on the
    fit rows of the other inner folds (folds over subjects, stratified by site), so
    that the adaptation is estimated from held-out values as for new subjects."""
    units = None if S.units is None else S.units[fit]
    fold = CC._make_folds(S.code[fit], units, np.ones(fit.size, dtype=bool), S.inner,
                          S.seed + 7)
    out = None
    for f in np.unique(fold):
        te, tr = fit[fold == f], fit[fold != f]
        tgt = te[np.isin(te, rows)]
        if tgt.size:
            H = correct(tr, tgt)[0]
            if out is None:
                out = np.full((rows.size, H.shape[1]), np.nan)
            out[np.searchsorted(rows, tgt)] = H
    return out


def run_condition(CC, S, R, name, mode='crossfit'):
    """Final z (n, p), location-adapted z before the common scale (n, p), for the
    controls the log density of their data (n, p), and the corrected scores before the
    adaptation (n, p).

    In every pass, the correction is fitted on the fit rows and applied to the scored
    rows; the adaptation (offsets and common scale) is estimated from the controls'
    corrected values, which are cross-fitted within the fit rows (inner folds), as
    harmonize_zscores of comcat_zmaps.py does it for new data.  In mode insample, the
    correction and the adaptation are fitted on the rows they are applied to."""
    spec = CONDITIONS[name][1]
    level = spec.get('level')
    correct = corrector(CC, S, R, spec) if level else None
    shape = R.Z0.shape
    Zf, Zl, LD, Zr = (np.full(shape, np.nan, dtype=np.float32) for _ in range(4))
    for fit, ad, ap in passes(S, spec.get('dx', False), mode):
        slope = 1.0                     # derivative of the harmonization (ComCat L/S)
        if level is None:
            Zr, Lr = (R.Z1, R.L1) if spec.get('model') == 1 else (R.Z0, R.L0)
            Za, Zp, Zpre, Lp = Zr[ad], Zr[ap], Zr[ap], Lr[ap]
        else:
            Hp, slope = correct(fit, ap)
            if mode == 'insample':
                Ha = Hp[np.searchsorted(ap, ad)]        # ad is a subset of ap = all rows
            else:
                Ha = inner_crossfit(CC, S, correct, fit, ad)
            if level == 'data':
                Za = data_scores(S, R, ad, Ha)[0]
                Zp, Lp = data_scores(S, R, ap, Hp)
                Zpre = Zp
            else:
                Za, Zp, Zpre, Lp = Ha, Hp, R.Z0[ap], R.L0[ap]
        loc, fin, r = CZ.adapt(Za, S.code[ad], Zp, S.code[ap], S.k)
        Zl[ap], Zf[ap], Zr[ap] = loc, fin, Zp
        held = S.ctrl[ap]
        # log density of the data: log phi(z) + log |dz/dy|, with dz/dy = slope / r *
        # f(y_c) / phi(z_c) for the normal score z_c of the (harmonized) data y_c
        with np.errstate(invalid='ignore', divide='ignore'):
            ls = 0.0 if np.isscalar(slope) else np.log(np.broadcast_to(slope, fin.shape)[held])
            LD[ap[held]] = (-0.5 * fin[held] ** 2 - np.log(r) + Lp[held]
                            + 0.5 * Zpre[held] ** 2 + ls)
    return Zf, Zl, LD, Zr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _med(x):
    return float(np.nanmedian(x)) if np.any(np.isfinite(x)) else np.nan


def site_spread(z, code, k):
    """RMS of the site offsets and of the log SDs of the controls' z per site, with the
    sampling noise subtracted (sites with at least 3 controls), weighted by size."""
    n = np.sum(np.all(np.isfinite(z), axis=1))
    m = np.nanmean(z, axis=0)
    ssw, dof = 0.0, 0
    for s in range(k):
        x = z[code == s]
        if len(x) > 1:
            ssw = ssw + np.nansum((x - np.nanmean(x, axis=0)) ** 2, axis=0)
            dof += len(x) - 1
    v = ssw / max(dof, 1)
    off, sc, w = [], [], []
    for s in range(k):
        x = z[code == s]
        ns = len(x)
        if ns < 3:
            continue
        dev = (np.nanmean(x, axis=0) - m) ** 2 - v * (1 / ns - 1 / max(n, 1))
        lsd = np.log(np.nanstd(x, axis=0, ddof=1) / np.sqrt(v))
        off.append(np.nanmean(dev))
        sc.append(np.nanmean(lsd ** 2) - 1 / (2 * (ns - 1)))
        w.append(ns)
    if not w:
        return np.nan, np.nan
    return (float(np.sqrt(max(0.0, np.average(off, weights=w)))),
            float(np.sqrt(max(0.0, np.average(sc, weights=w)))))


def within_between(X, subject):
    """Pooled within-subject SD (p,) and SD of the subjects' means (p,)."""
    ss, dof, means = 0.0, 0, []
    for s in np.unique(subject):
        x = np.asarray(X[subject == s], np.float64)
        means.append(np.nanmean(x, axis=0))
        if len(x) > 1:
            ss = ss + np.nansum((x - np.nanmean(x, axis=0)) ** 2, axis=0)
            dof += len(x) - 1
    within = np.sqrt(ss / max(dof, 1))
    between = np.nanstd(np.array(means), axis=0, ddof=1) if len(means) > 1 else None
    return within, between


def paired(X, subject, motion, level, base):
    """Mean over subjects (p,) of X at a motion level minus X without motion."""
    diff = []
    for s in np.unique(subject):
        a = np.flatnonzero((subject == s) & (motion == base))
        b = np.flatnonzero((subject == s) & (motion == level))
        if a.size and b.size:
            diff.append(np.asarray(X[b[0]], np.float64) - X[a[0]])
    return np.mean(diff, axis=0) if diff else None


def welch(a, b):
    """Welch t (p,) and Cohen's d (p,) of the rows of a vs b."""
    ma, mb = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    va, vb = np.nanvar(a, axis=0, ddof=1), np.nanvar(b, axis=0, ddof=1)
    na, nb = len(a), len(b)
    t = (ma - mb) / np.sqrt(va / na + vb / nb)
    d = (ma - mb) / np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return t, d


def metrics(S, Zf, Zl, LD, Zr):
    z = Zf[S.ctrl].astype(np.float64)
    fin = np.isfinite(z)
    out = dict(log_density=float(np.nanmean(LD[S.ctrl])),
               z_sd=_med(np.nanstd(z, axis=0, ddof=1)),
               z_mean=_med(np.nanmean(z, axis=0)),
               **{'beyond_2.58': float(np.mean(np.abs(z[fin]) > 2.58))})
    if S.k > 1:
        out['site_scale_rms'] = site_spread(z, S.code[S.ctrl], S.k)[1]
        out['site_offset_rms'] = site_spread(Zr[S.ctrl].astype(np.float64), S.code[S.ctrl],
                                             S.k)[0]
    if S.repeated:
        w, b = within_between(Zl, S.subject)
        out['within_sd'] = _med(w)
        if b is not None:
            out['between_sd'] = _med(b)
    if S.motion is not None:
        levels = np.unique(S.motion)
        for lv, tag in zip(levels[1:], ('medium', 'strong')):
            dm = paired(Zl, S.subject, S.motion, lv, levels[0])
            if dm is not None:
                out[f'motion_{tag}'] = float(np.nanmean(dm))
                out[f'motion_{tag}_abs'] = float(np.nanmean(np.abs(dm)))
    if S.group is not None:
        inv = {v: k for k, v in S.cfg['groups'].items()}
        for a, b in S.cfg['contrasts']:
            ga, gb = S.group == inv[a], S.group == inv[b]
            if ga.sum() > 1 and gb.sum() > 1:
                t, dd = welch(Zf[ga].astype(np.float64), Zf[gb].astype(np.float64))
                out[f't_{a}_{b}'] = _med(np.abs(t))
                out[f'd_{a}_{b}'] = _med(dd)
                # difference of the location-adapted z (units of the normative SD), not
                # changed by the removal of noise, unlike t and d
                out[f'diff_{a}_{b}'] = float(np.nanmean(np.nanmean(Zl[ga], axis=0)
                                                        - np.nanmean(Zl[gb], axis=0)))
    return out


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def compare(args):
    CC = CZ.import_comcat(args.comcat_dir)
    rows = []
    path = os.path.join(args.out, 'comparison.csv')
    for name in args.samples:
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        print(f"== {name}", flush=True)
        S = load_sample(args, name, rng)
        train = S.cfg['train']
        models = [load_model(model_path(args.out, train, c)) for c in ('tiv', 'tiv_iqm')]
        R = prepare(S, models, args, rng)
        print(f"  {S.n} scans, {int(S.ctrl.sum())} controls, {S.k} site(s), "
              f"{R.feat.size} voxels, {len(np.unique(S.fold[S.fold >= 0]))} folds",
              flush=True)
        names = [c for c in args.conditions if applicable(S, c)]
        for mode in args.modes:
            for c in names:
                t1 = time.time()
                Zf, Zl, LD, Zr = run_condition(CC, S, R, c, mode)
                m = metrics(S, Zf, Zl, LD, Zr)
                m.update(n_scans=S.n, n_controls=int(S.ctrl.sum()),
                         n_voxels=int(R.feat.size), seconds=round(time.time() - t1, 1))
                rows += [dict(sample=name, mode=mode, condition=c, metric=k, value=v)
                         for k, v in m.items()]
                print(f"  {mode:8s} {c:26s} " + '  '.join(
                    f"{k} {v:.3f}" for k, v in m.items()
                    if k not in ('n_scans', 'n_controls', 'n_voxels')), flush=True)
        write_csv(rows, path)
        print(f"  {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {path}")


def read_csv(path):
    """Rows of comparison.csv; files without the column mode (earlier versions) were
    cross-fitted."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['mode'] = r.get('mode') or 'crossfit'
    return rows


def write_csv(rows, path):
    """Write rows, keeping the rows of other samples and modes of an existing file."""
    old = []
    if os.path.exists(path):
        done = {(r['sample'], r['mode']) for r in rows}
        old = [r for r in read_csv(path) if (r['sample'], r['mode']) not in done]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample', 'mode', 'condition', 'metric', 'value'])
        w.writeheader()
        w.writerows(old + rows)


REPORT_COLUMNS = (   # metric, header, format
    ('log_density', 'Δ log density (10⁻³ nat)', 'rel'),
    ('z_sd', 'z SD', '.3f'),
    ('beyond_2.58', 'z beyond ±2.58 (%)', 'pct'),
    ('site_offset_rms', 'site offset before adaptation', '.3f'),
    ('site_scale_rms', 'site log-scale', '.3f'),
    ('within_sd', 'within-subject SD', '.3f'),
    ('between_sd', 'between-subject SD', '.3f'),
    ('motion_strong', 'Δz strong motion', '+.3f'),
    ('motion_strong_abs', 'Δz strong motion, absolute', '.3f'),
)


MODE_TITLES = {'crossfit': 'held out (cross-fitted)', 'insample': 'in-sample'}


def report(args):
    path = os.path.join(args.out, 'comparison.csv')
    rows = read_csv(path)
    val = {(r['sample'], r['mode'], r['condition'], r['metric']): float(r['value'])
           for r in rows}
    lines = ["# Sites and image quality in NeuroGAMLSS z-maps", "",
             "Approach 1, the main one: ComCat on the test data (`comcat*`, `combat`), as "
             "for BrainAGE; its alternative: ComCat on the normal scores (`comcat_z*`); "
             "approach 2: sites and IQMs as terms of the GAMLSS (`gamlss*`).", "",
             "Controls held out (cross-fitted) or in-sample (every estimate fitted on the "
             "scans it is applied to; optimistic); Δ log density relative to `gamlss` of "
             "the same mode, in thousandths of a nat per subject and voxel (higher is "
             "better); t: median |t| relative to `gamlss`; mean Δz: difference of the "
             "groups' mean location-adapted z (normative SD units, not changed by removing "
             "noise).  See the docstring of harmonization_comparison.py.", ""]
    modes = [m for m in MODE_TITLES if any(r['mode'] == m for r in rows)]
    for sample in dict.fromkeys(r['sample'] for r in rows):
        lines += [f"## {sample}", ""]
        for mode in modes:
            conds = [c for c in CONDITIONS if (sample, mode, c, 'z_sd') in val]
            if not conds:
                continue
            metr = {r['metric'] for r in rows if r['sample'] == sample and r['mode'] == mode}
            cols = [c for c in REPORT_COLUMNS if c[0] in metr]
            tcols = sorted(m for m in metr if m.startswith('t_'))
            dcols = ['diff_' + m[2:] for m in tcols if 'diff_' + m[2:] in metr]
            head = (['Condition'] + [c[1] for c in cols]
                    + [f"t {m[2:].replace('_', '-')}" for m in tcols]
                    + [f"mean Δz {m[5:].replace('_', '-')}" for m in dcols])
            lines += [f"### {sample}, {MODE_TITLES[mode]}", "",
                      '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
            for c in conds:
                cells = [CONDITIONS[c][0] + f" (`{c}`)"]
                for m, _, fmt in cols:
                    v = val.get((sample, mode, c, m), np.nan)
                    if fmt == 'rel':
                        v0 = val.get((sample, mode, 'gamlss', m), np.nan)
                        cells.append(f"{1000 * (v - v0):+.1f}")
                    elif fmt == 'pct':
                        cells.append(f"{100 * v:.2f}")
                    else:
                        cells.append(format(v, fmt) if np.isfinite(v) else '-')
                for m in tcols:
                    v = val.get((sample, mode, c, m), np.nan)
                    v0 = val.get((sample, mode, 'gamlss', m), np.nan)
                    cells.append(f"{v / v0:.3f}" if np.isfinite(v) and v0 > 0 else '-')
                for m in dcols:
                    v = val.get((sample, mode, c, m), np.nan)
                    cells.append(f"{v:+.3f}" if np.isfinite(v) else '-')
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines.append('')
    out = os.path.join(args.out, 'comparison.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='~/Dropbox/neuroGAMLSS-TestData',
                   help="folder with TrainingData and TestData")
    p.add_argument('--out', default='harmonization', help="output folder")
    p.add_argument('--part', choices=['models', 'compare', 'report', 'all'], default='all')
    p.add_argument('--samples', default=','.join(TEST),
                   help=f"comma-separated test samples (default: {','.join(TEST)})")
    p.add_argument('--conditions', default=','.join(CONDITIONS),
                   help="comma-separated conditions (default: all)")
    p.add_argument('--modes', default='crossfit,insample',
                   help="crossfit (held out), insample (fitted and evaluated on the same "
                        "scans) or both (default: crossfit,insample)")
    p.add_argument('--folds', type=int, default=0,
                   help="folds of the cross-fit (default: per sample, see TEST)")
    p.add_argument('--inner-folds', type=int, default=5,
                   help="inner folds: the controls' values for the adaptation are "
                        "cross-fitted within every training part (default 5)")
    p.add_argument('--features', type=int, default=0,
                   help="random subset of this many voxels (default: all)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--jobs', type=int, default=0, help="threads of neurogamlss.py")
    p.add_argument('--ng-options', default='',
                   help="further options for fitting the models, e.g. "
                        "--ng-options=--site-random (with '=')")
    p.add_argument('--comcat-dir', help="folder of comcat.py")
    a = p.parse_args(argv)
    a.root, a.out = os.path.expanduser(a.root), os.path.expanduser(a.out)
    a.samples = [s.strip() for s in a.samples.split(',') if s.strip()]
    a.conditions = [c.strip() for c in a.conditions.split(',') if c.strip()]
    a.modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    for m in a.modes:
        if m not in MODE_TITLES:
            p.error(f"unknown mode {m} ({', '.join(MODE_TITLES)})")
    for s in a.samples:
        if s not in TEST:
            p.error(f"unknown sample {s} ({', '.join(TEST)})")
    for c in a.conditions:
        if c not in CONDITIONS:
            p.error(f"unknown condition {c} ({', '.join(CONDITIONS)})")
    if 'gamlss' not in a.conditions:
        a.conditions.insert(0, 'gamlss')            # the reference of the report
    os.makedirs(a.out, exist_ok=True)
    if a.part in ('models', 'all'):
        CZ.import_comcat(a.comcat_dir)               # fail early if ComCat is missing
        fit_models(a, list(dict.fromkeys(TEST[s]['train'] for s in a.samples)))
    if a.part in ('compare', 'all'):
        compare(a)
    if a.part in ('report', 'all'):
        report(a)


if __name__ == '__main__':
    main()
