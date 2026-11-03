#!/usr/bin/env python3
"""ComCat harmonization of the test data in mat-files, before NeuroGAMLSS or BrainAGE.

The main approach for ComCat before normative models, the same as before BrainAGE:
ComCat (github.com/ChristianGaser/ComCat) harmonizes the data of the test sample (Y of
BA_data2mat mat-files), and the models that turn the data into scores, the normative
GAMLSS of neurogamlss.py or the GPR of BrainAGE, stay trained on raw data.  ComCat
removes the effects of the test sites (location and scale with empirical Bayes and the
df-consistent site variances of ComCat, which do not shrink small sites; location only
with one site or with subject terms) and of nuisance variables such as the image quality
measures ICR, NCR, res_ECR and contrastr (P-splines whose smoothness REML estimates).
Age, sex and the covariates of --preserve (e.g. TIV, if the normative models use it)
are preserved.

The effects are estimated in the controls (--adjust, default all subjects), so that a
disease effect cannot enter them; every control is harmonized by a model fitted on the
controls of the other folds (folds over subjects, stratified by site), all other
subjects by the model of all controls.  With --preserve-group, all subjects are used
and their groups are preserved instead (few controls per site); with --subject, the
effects are estimated within subjects (repeated scans).

Every input is written with its name and all its fields to --out-dir, with Y
harmonized, and a JSON file with the design next to it.  The harmonized file is then
the test data, with the usual adaptation with the controls:

    python scripts/comcat_data.py --test TestData/s6mwp1_4mm_ABIDE437_CAT12.9.mat \\
        --test-site TestData/ABIDE/Scanner437.txt --adjust "$TD" \\
        --nuisance TestData/ABIDE/IQM437.txt --nuisance-cols 1,2,3,5 \\
        --preserve TestData/ABIDE/TIV437.txt --out-dir comcat
    python neurogamlss.py --normative-only --model model_child_tiv.npz \\
        --test comcat/s6mwp1_4mm_ABIDE437_CAT12.9.mat --test-cov TestData/ABIDE/TIV437.txt \\
        --test-site TestData/ABIDE/Scanner437.txt --adjust "$TD" --out abide_comcat

Several mat-files of the same subjects (e.g. the four feature sets of BrainAGE) are
harmonized with the same design and folds.  Subjects without a valid age or without
values of the nuisance or preserved covariates are not harmonized and get NaN.  Columns
1, 2, 3 and 5 of a CAT12 IQM table (no header) are ICR, NCR, res_ECR and contrastr.
The alternative, ComCat on the normal scores, is comcat_zmaps.py.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import neurogamlss as NG  # noqa: E402
import comcat_zmaps as CZ  # noqa: E402


def _oriented(Y, shape):
    """Y (n_subjects, n_features) in the orientation of a stored Y of this (MATLAB)
    shape: subjects in rows unless only the columns match them, as load_data reads it."""
    return Y.T if shape[0] != Y.shape[0] and shape[1] == Y.shape[0] else Y


def write_mat(src, dst, Y):
    """Copy of the mat-file src at dst with Y (n_subjects, n_features) replaced, in the
    orientation and type of src; all other fields are kept.  A v7.3 file (HDF5, arrays
    stored transposed) is copied and its Y overwritten, a v5 file is written again."""
    from scipy.io import loadmat, savemat
    try:
        m = loadmat(src)
    except NotImplementedError:
        import h5py
        shutil.copyfile(src, dst)
        with h5py.File(dst, 'r+') as h:
            ds = h['Y']
            ds[...] = _oriented(Y, ds.shape[::-1]).T.astype(ds.dtype)
        return
    Y0 = m['Y']
    m = {k: v for k, v in m.items() if not k.startswith('__')}
    m['Y'] = _oriented(Y, Y0.shape).astype(Y0.dtype)
    savemat(dst, m, do_compression=True)


def _table(path, spec, n, label):
    """Columns spec (names or 1-based numbers; default all numeric) of a table with one
    row per subject, and their names."""
    names, values, numeric = NG.read_table(path)
    if len(values) != n:
        raise ValueError(f"{label} has {len(values)} rows for {n} test subjects.")
    cols = (NG._parse_columns(spec, names, numeric) if spec else
            [nm for j, nm in enumerate(names) if numeric[j]])
    return values[:, [names.index(nm) for nm in cols]], cols


def _comcat_version(CC):
    """Short git commit of the ComCat repository, if available."""
    try:
        out = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(CC.__file__)),
                              'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test', required=True, nargs='+', help="mat-file(s) of the same test "
                   "subjects, e.g. the feature sets of BrainAGE")
    p.add_argument('--test-male', help="sex of the test subjects (1 = male), if the "
                   "mat-files contain none")
    p.add_argument('--test-site', help="site of every test subject, one label per line "
                   "(default: one site)")
    p.add_argument('--adjust', help="1-based indices of the controls, on which the effects "
                   "are estimated, e.g. 1:108,150 (default: all subjects)")
    p.add_argument('--nuisance', help="table of nuisance variables, e.g. image quality "
                   "measures, one row per subject")
    p.add_argument('--nuisance-cols', help="columns of --nuisance: names or 1-based numbers "
                   "(default: all numeric columns)")
    p.add_argument('--preserve', help="table of further covariates to preserve besides age "
                   "and sex, e.g. TIV if the normative models use it")
    p.add_argument('--preserve-cols', help="columns of --preserve (default: all numeric "
                   "columns)")
    p.add_argument('--preserve-group', help="group of every test subject (e.g. diagnosis): "
                   "fit ComCat on all subjects with the groups preserved instead of on the "
                   "controls only")
    p.add_argument('--subject', help="subject of every scan (repeated scans): site and "
                   "nuisance effects are estimated within subjects")
    p.add_argument('--location-only', action='store_true',
                   help="no site variances (ComCat mean_only)")
    p.add_argument('--combat-compatible', action='store_true',
                   help="site variances of ComBat (ComCat combat_compatible), which shrink "
                        "the data of small sites; default: df-consistent site variances")
    p.add_argument('--folds', type=int, default=5, help="folds of the cross-fit (default 5)")
    p.add_argument('--insample', action='store_true',
                   help="no cross-fit: ComCat fitted on all controls (all subjects with "
                        "--preserve-group) and applied to all (optimistic; for comparison)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--comcat-dir', help="folder of comcat.py (default: COMCAT_DIR, "
                   "~/GitHub/ComCat)")
    p.add_argument('--out-dir', required=True, help="folder of the harmonized mat-files, "
                   "which keep the names of the inputs")
    a = p.parse_args(argv)
    if not a.nuisance and not a.test_site:
        p.error("Nothing to remove: give --nuisance, --test-site or both.")
    CC = CZ.import_comcat(a.comcat_dir)

    datas = [NG.load_data(t) for t in a.test]
    msg = NG._subjects_mismatch(datas, a.test, '--test')
    if msg:
        p.error(msg)
    d = datas[0]
    n = d.n
    male = d.male if d.has_male else None
    if a.test_male:
        male = CZ._read_column(a.test_male, n, '--test-male')
    site, site_names = np.zeros(n, dtype=np.int64), ('1',)
    if a.test_site:
        site, site_names = NG.read_sites(a.test_site, n)
    ctrl = np.arange(n) if not a.adjust else NG._parse_index(a.adjust, n)
    ok = np.isfinite(d.age) & (d.age > 0)
    Q = C = None
    qcols = ccols = []
    if a.nuisance:
        Q, qcols = _table(a.nuisance, a.nuisance_cols, n, '--nuisance')
        ok &= np.all(np.isfinite(Q), axis=1)
        print(f"Nuisance variables: {', '.join(qcols)}")
    if a.preserve:
        C, ccols = _table(a.preserve, a.preserve_cols, n, '--preserve')
        ok &= np.all(np.isfinite(C), axis=1)
    if male is None:
        print("The test data contain no sex (--test-male): sex is not preserved.")
    else:
        ok &= np.isfinite(male)
    group = None if not a.preserve_group else CZ._read_column(a.preserve_group, n,
                                                               '--preserve-group', str)
    subject = None if not a.subject else CZ._read_column(a.subject, n, '--subject', str)

    rows = np.flatnonzero(ok)
    if rows.size < n:
        print(f"{n - rows.size} subject(s) without valid age, nuisance or preserved values "
              "are not harmonized (NaN).")
    pos = {r: i for i, r in enumerate(rows)}
    ctrl_r = np.array([pos[c] for c in ctrl if c in pos], dtype=np.int64)
    if not ctrl_r.size:
        p.error("None of the controls has valid values.")
    preserve = np.column_stack([d.age[rows]] + ([male[rows]] if male is not None else [])
                               + ([C[rows]] if C is not None else []))
    _, code = np.unique(site[rows], return_inverse=True)
    k = int(code.max()) + 1
    kept = ['age'] + (['sex'] if male is not None else []) + ccols
    parts = []
    if k > 1:                       # with subject terms, ComCat corrects the location only
        scale = ('location' if a.location_only or subject is not None else
                 'location and scale, ComBat site variances' if a.combat_compatible else
                 'location and scale')
        parts.append(f"sites (empirical Bayes, {scale})")
    if Q is not None:
        parts.append(f"nuisance {', '.join(qcols)}")
    parts.append(f"preserved: {', '.join(kept)}")
    design = ('ComCat on the data: ' + ', '.join(parts)
              + ('; fitted on all subjects, groups preserved' if group is not None
                 else '; fitted on the controls')
              + ('; in-sample (no cross-fit)' if a.insample
                 else f'; cross-fitted ({a.folds} folds)')
              + ('; within subjects' if subject is not None else ''))
    print(f"{rows.size} subjects, {ctrl_r.size} controls, {k} site(s): {design}")

    os.makedirs(os.path.expanduser(a.out_dir), exist_ok=True)
    for path, dd in zip(a.test, datas):
        src = os.path.expanduser(path)
        dst = os.path.join(os.path.expanduser(a.out_dir), os.path.basename(src))
        if os.path.exists(dst) and os.path.samefile(src, dst):
            p.error(f"--out-dir would overwrite the input {src}.")
        t0 = time.time()
        H, fold = CZ.harmonize(CC, dd.Y[rows], ctrl_r, code, None if Q is None else Q[rows],
                               preserve, None if subject is None else subject[rows],
                               None if group is None else group[rows], a.folds, a.seed,
                               a.location_only, a.combat_compatible, a.insample)
        Y = np.full(dd.Y.shape, np.nan, dtype=np.float32)
        Y[rows] = H
        write_mat(src, dst, Y)
        folds = [None] * n
        for r, f in zip(rows, fold):
            folds[r] = int(f)
        info = dict(input=os.path.abspath(src), output=os.path.abspath(dst),
                    harmonization=design, n_subjects=n, n_harmonized=int(rows.size),
                    n_controls=int(ctrl_r.size), sites=[str(s) for s in site_names],
                    nuisance=qcols, preserved=kept,
                    options=CZ.comcat_options(a.location_only, a.combat_compatible),
                    comcat=_comcat_version(CC), seed=a.seed,
                    created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    command=['comcat_data.py'] + list(sys.argv[1:] if argv is None else argv),
                    fold=folds, fold_note="fold of every subject in the cross-fit "
                    "(0-based; -1: harmonized by the model of all fit subjects, or "
                    "in-sample; null: not harmonized)")
        with open(os.path.splitext(dst)[0] + '.json', 'w') as f:
            json.dump(info, f, indent=1)
        print(f"{dst}: {dd.Y.shape[1]} features, {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
