#!/usr/bin/env python3
"""ComCat harmonization of the normal scores (z-maps) of NeuroGAMLSS.

The alternative to harmonizing the test data themselves (scripts/comcat_data.py, the
main approach, as for BrainAGE): the voxel/vertex-wise normative models give every test
subject its normal scores at the reference site, without adaptation.  ComCat
(github.com/ChristianGaser/ComCat) then removes from these scores the effects of the
test sites (location and scale with empirical Bayes, as ComBat, but with df-consistent
site variances from the residuals after the nuisance effects, which do not shrink small
sites) and of nuisance variables such as the image quality measures ICR, NCR,
res_ECR and contrastr (penalized splines whose smoothness REML estimates).  Age, sex
and the covariates of the normative models are preserved.

The effects are estimated in the control subjects (--adjust), so that a disease effect
cannot enter them; every control is harmonized by a model fitted on the controls of
the other folds.  With --preserve-group, all subjects are used and their groups are
preserved instead, for samples with few controls per site.  Finally the scores are
adapted with the controls as by neurogamlss.py (offsets per site shrunk toward the
mean of all controls and one common scale; the controls' own scores leave-one-out),
so that the controls' z-scores have mean 0 and SD 1.

Removing the effects from the normal scores instead of the data matters for skewed
(SHASH) models, in which a shift of the data is not a constant shift of the score.

    python scripts/comcat_zmaps.py --model adni_norm_s6mwp1.npz \\
        --test TestData/s6mwp1_4mm_200AD_404MCI_231Normal_CAT12.9.mat \\
        --test-cov TestData/ADNI/TIV835.txt --test-site adni_s6mwp1_site.txt \\
        --adjust "$adjust" --nuisance TestData/ADNI/IQM835.txt --nuisance-cols 1,2,3,5 \\
        --out adni_zcomcat

writes adni_zcomcat_zmaps_<test>.mat (struct NDMzmap, as neurogamlss.py), which
mat2image.py converts to images.  Columns 1, 2, 3 and 5 of a CAT12 IQM table (no
header) are ICR, NCR, res_ECR and contrastr.

The functions are also used by comcat_data.py and harmonization_comparison.py.
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neurogamlss as NG  # noqa: E402

LOG2PI = np.log(2 * np.pi)


def import_comcat(path=None):
    """The comcat module, from path, COMCAT_DIR, ~/GitHub/ComCat or the Python path."""
    for d in (path, os.environ.get('COMCAT_DIR'), '~/GitHub/ComCat'):
        if d and os.path.isfile(os.path.join(os.path.expanduser(d), 'comcat.py')):
            sys.path.insert(0, os.path.expanduser(d))
            break
    import comcat
    import inspect
    if 'combat_compatible' not in inspect.signature(comcat.comcat).parameters:
        raise ImportError("ComCat of 2026-11-02 or later is needed (df-consistent site "
                          "variances, option combat_compatible).")
    return comcat


def comcat_options(location_only=False, combat_compatible=False):
    """Options of comcat(): P-splines with REML, empirical Bayes for the sites and, unless
    location_only, their scale step with the df-consistent site variances of ComCat
    (residual df n_i minus leverage, after the nuisance effects), which do not shrink the
    residuals of small sites.  combat_compatible: the site variances of ComBat instead
    (from the residuals).  With one site, comcat() corrects the location only in any
    case."""
    opts = dict(nuisance_model='pspline', eb=True, mean_only=bool(location_only))
    if combat_compatible:
        opts.update(combat_compatible=True, residual_delta=True)
    return opts


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_names(path):
    """File names of the subjects of a BA_data2mat mat-file (field sample.data), or None."""
    path = os.path.expanduser(path)
    try:
        from scipy.io import loadmat
        m = loadmat(path, variable_names=['sample'], squeeze_me=True,
                    struct_as_record=False)
        if 'sample' not in m:
            return None
        data = getattr(m['sample'], 'data', None)
        return None if data is None else [str(x) for x in np.atleast_1d(data)]
    except NotImplementedError:                 # v7.3: HDF5, cell of char matrices
        import h5py
        with h5py.File(path, 'r') as h:
            if 'sample' not in h or 'data' not in h['sample']:
                return None
            out = []
            for ref in np.ravel(h['sample/data'][()]):
                a = np.asarray(h[ref][()])
                if a.dtype.kind == 'O':
                    out += [''.join(map(chr, np.ravel(h[r][()]))).rstrip() for r in a.ravel()]
                else:
                    out += [''.join(map(chr, row)).rstrip() for row in a.T]
            return out


def labels_from_names(names, pattern):
    """The first group of the regular expression pattern in every name."""
    out = []
    for nm in names:
        m = re.search(pattern, nm)
        if m is None:
            raise ValueError(f"{nm}: no match of {pattern!r}.")
        out.append(m.group(1))
    return np.array(out)


# ---------------------------------------------------------------------------
# Normal scores and log densities at the reference site
# ---------------------------------------------------------------------------

def reference_model(vm):
    """Copy of the normative model of a VoxelModel without site adaptation."""
    if vm.warper is not None:
        raise ValueError("Models with a warp (brain age models) are not supported.")
    m = vm.model.copy()
    m.offset[:], m.scale[:] = 0, 1
    return m


def log_density(m, Yv, age, male, C=None):
    """Log density (n, p_valid) of the valid columns Yv under the model m at the
    reference site (normal and SHASH families)."""
    X, W = m._design(np.atleast_1d(age), np.atleast_1d(male), None, C)
    a, s = X @ m.beta, W @ m.theta
    Yv = np.asarray(Yv, np.float64)
    if m.family == 'normal':
        return -s - 0.5 * ((Yv - a) * np.exp(-s)) ** 2 - 0.5 * LOG2PI
    if m.family != 'shash':
        raise ValueError(f"family {m.family}: log densities only for normal and shash.")
    V = m._shape_design(np.atleast_1d(age))
    return NG.shash_terms(Yv, a, s, V @ m.kappa, V @ m.lam, order=0)[0]


def reference_scores(vm, Y, age, male, C=None, density=False, chunk=256):
    """Normal scores (n, p_valid) of the rows of Y (n, n_features) at the reference site,
    and with density also their log densities; float64."""
    m = reference_model(vm)
    n = Y.shape[0]
    Z = np.empty((n, m.valid.size))
    L = np.empty((n, m.valid.size)) if density else None
    for c0 in range(0, n, chunk):
        sl = slice(c0, c0 + chunk)
        Cs = None if C is None else C[sl]
        Z[sl] = m.zscores(Y[sl], age[sl], male[sl], C=Cs)
        if density:
            L[sl] = log_density(m, np.asarray(Y[sl])[:, m.valid], age[sl], male[sl], Cs)
    return (Z, L) if density else Z


# ---------------------------------------------------------------------------
# ComCat
# ---------------------------------------------------------------------------

def _columns(P, rows):
    """Columns of P (n, k) that vary within rows (None if none)."""
    if P is None:
        return None
    P = np.asarray(P, np.float64).reshape(len(P), -1)
    keep = [j for j in range(P.shape[1]) if np.ptp(P[rows, j]) > 0]
    return P[:, keep] if keep else None


def comcat_fit_apply(CC, X, fit, apply, batch=None, nuisance=None, preserve=None,
                     subject=None, slope=False, **opts):
    """ComCat fitted on the rows fit of X (n, p) and applied to the rows apply.

    Returns the harmonized rows fit (in-sample) and apply (comcat_from_training; sites
    without fit rows keep their site effect).  batch: site codes (n,) or None; one
    site among the fit rows means no site correction.  Preserved columns that do not
    vary in the fit rows are left out.  Columns of X with non-finite values in these
    rows are returned unchanged.  With slope, also the derivative of the harmonized
    apply rows with respect to X (the transformation is affine: 1 / sqrt(delta) of
    the site, 1 with location only)."""
    fit, apply = np.asarray(fit), np.asarray(apply)
    rows = np.concatenate([fit, apply])
    cols = np.flatnonzero(np.all(np.isfinite(X[rows]), axis=0))
    Hf = np.array(X[fit], dtype=np.float64)
    Ha = np.array(X[apply], dtype=np.float64)
    b = None if batch is None or np.unique(batch[fit]).size < 2 else np.asarray(batch)
    P = _columns(preserve, fit)

    def sub(a, r):
        return None if a is None else np.asarray(a)[r]

    D = np.ones(Ha.shape) if slope else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        res = CC.comcat(X[np.ix_(fit, cols)].T, sub(b, fit), sub(nuisance, fit), sub(P, fit),
                        subject=sub(subject, fit), return_estimates=True, **opts)
        Hf[:, cols] = res[0].T
        if apply.size:
            bb = np.ones(apply.size, dtype=int) if b is None else b[apply]

            def to(Xa):
                return CC.comcat_from_training(Xa.T, bb, sub(nuisance, apply), sub(P, apply),
                                               estimates=res[4], unseen_batch='mean').T
            Ha[:, cols] = to(X[np.ix_(apply, cols)])
            if slope:
                D[:, cols] = to(X[np.ix_(apply, cols)] + 1.0) - Ha[:, cols]
    return (Hf, Ha, D) if slope else (Hf, Ha)


def adapt(Zc_adapt, code_adapt, Zc_apply, code_apply, k):
    """Site adaptation as neurogamlss.py: offsets per site shrunk toward the mean of the
    adaptation rows and one common scale.  Returns the location-adapted (unscaled) and
    the final scores of the apply rows, and the scale (p,)."""
    O, r, _ = NG.site_adaptation(Zc_adapt, code_adapt, k)
    loc = Zc_apply - O[code_apply]
    return loc, loc / r, r


def harmonize(CC, X, ctrl, site=None, nuisance=None, preserve=None, subject=None,
              group=None, folds=5, seed=0, location_only=False, combat_compatible=False,
              insample=False, subject_terms=False):
    """ComCat on X (n, p), the data of the test subjects (comcat_data.py) or their
    normal scores (harmonize_zscores), with the effects estimated on the controls ctrl
    (indices).

    The controls are harmonized cross-fitted (folds over subjects, stratified by
    site), all others by the model of all controls.  With group (n,) labels, ComCat
    is fitted on all subjects instead, with indicators of every group not among the
    controls preserved, and every subject is harmonized cross-fitted.  subject (n,)
    labels of repeated scans keep the scans of a subject in one fold and, with
    subject_terms, estimate the effects within subjects.  insample: fitted on all
    controls (all subjects with group) and applied to every subject, the fitted ones
    included (fold labels -1); for comparison only, because the harmonized values of
    the fitted subjects are optimistic.  Returns the harmonized X (n, p) and the fold
    labels (n,)."""
    n = X.shape[0]
    is_ctrl = np.zeros(n, dtype=bool)
    is_ctrl[np.asarray(ctrl)] = True
    code = np.zeros(n, dtype=np.int64) if site is None else np.asarray(site, np.int64)
    P = None if preserve is None else np.asarray(preserve, np.float64).reshape(n, -1)
    fit = is_ctrl
    if group is not None:
        g = np.asarray(group)
        dummies = [(g == lv).astype(float) for lv in np.unique(g[~is_ctrl])
                   if lv not in set(g[is_ctrl])]
        if dummies:
            P = np.column_stack(([P] if P is not None else []) + dummies)
        fit = np.ones(n, dtype=bool)
    opts = comcat_options(location_only, combat_compatible)
    terms = subject if subject_terms else None
    if insample:
        H = comcat_fit_apply(CC, X, np.flatnonzero(fit), np.arange(n), code, nuisance, P,
                             terms, **opts)[1]
        return H, np.full(n, -1)
    fold = CC._make_folds(code, subject, fit, int(folds), seed)
    H = np.full(X.shape, np.nan)
    for f in np.unique(fold[fold >= 0]):
        tr, te = np.flatnonzero(fit & (fold != f)), np.flatnonzero(fold == f)
        H[te] = comcat_fit_apply(CC, X, tr, te, code, nuisance, P, terms, **opts)[1]
    rest = np.flatnonzero(~fit)
    if rest.size:
        H[rest] = comcat_fit_apply(CC, X, np.flatnonzero(fit), rest, code, nuisance, P,
                                   terms, **opts)[1]
    return H, fold


def harmonize_zscores(CC, Z, ctrl, site=None, nuisance=None, preserve=None, subject=None,
                      group=None, folds=5, seed=0, location_only=False,
                      combat_compatible=False, insample=False, subject_terms=False):
    """ComCat on normal scores Z (n, p) (harmonize) and adaptation with the controls
    ctrl (indices).  Returns the final z-scores (n, p): (z - offset of the site) /
    common scale, with the controls' adaptation leave-one-out, and the fold labels.

    insample: ComCat fitted on all controls (all subjects with group) and applied to
    every subject, the fitted ones included, and the controls adapted with all
    controls including themselves (fold labels -1).  For comparison with the
    cross-fitted scores only: in-sample scores of the controls are optimistic."""
    n = Z.shape[0]
    ctrl = np.asarray(ctrl)
    is_ctrl = np.zeros(n, dtype=bool)
    is_ctrl[ctrl] = True
    code = np.zeros(n, dtype=np.int64) if site is None else np.asarray(site, np.int64)
    k = int(code.max()) + 1
    H, fold = harmonize(CC, Z, ctrl, site, nuisance, preserve, subject, group, folds, seed,
                        location_only, combat_compatible, insample, subject_terms)
    if insample:
        return adapt(H[ctrl], code[ctrl], H, code, k)[1], fold
    out = np.full(Z.shape, np.nan)
    others = np.flatnonzero(~is_ctrl)
    if others.size:
        out[others] = adapt(H[ctrl], code[ctrl], H[others], code[others], k)[1]
    out[ctrl] = NG._site_loo_scores(H[ctrl], H[ctrl], code[ctrl], k)
    return out, fold


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _read_column(path, n, label, dtype=float):
    values = np.loadtxt(os.path.expanduser(path), dtype=dtype, ndmin=1)
    if values.shape[0] != n:
        raise ValueError(f"{label} has {values.shape[0]} rows for {n} test subjects.")
    return values


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help="normative models of neurogamlss.py "
                   "(--save-model) with voxel/vertex-wise models")
    p.add_argument('--test', required=True, help="test mat-file")
    p.add_argument('--test-cov', help="covariate table of the test subjects (as for "
                   "neurogamlss.py), needed if the models use covariates")
    p.add_argument('--test-male', help="sex of the test subjects (1 = male), if the "
                   "mat-file contains none")
    p.add_argument('--test-site', help="site of every test subject, one label per line")
    p.add_argument('--adjust', required=True, help="1-based indices of the test controls, "
                   "e.g. 1:108,150")
    p.add_argument('--nuisance', help="table of nuisance variables, e.g. image quality "
                   "measures, one row per subject")
    p.add_argument('--nuisance-cols', help="columns of --nuisance: names or 1-based numbers "
                   "(default: all numeric columns)")
    p.add_argument('--preserve-group', help="group of every test subject (e.g. diagnosis): "
                   "fit ComCat on all subjects with the groups preserved instead of on the "
                   "controls only")
    p.add_argument('--subject', help="subject of every scan (repeated scans): the scans of a "
                   "subject stay in one fold; the effects are estimated across subjects")
    p.add_argument('--within-subjects', action='store_true',
                   help="with --subject: site and nuisance effects estimated within "
                        "subjects (subject terms, location only), e.g. for few subjects")
    p.add_argument('--location-only', action='store_true',
                   help="no site variances (ComCat mean_only)")
    p.add_argument('--combat-compatible', action='store_true',
                   help="site variances of ComBat (ComCat combat_compatible), which shrink "
                        "the scores of small sites; default: df-consistent site variances")
    p.add_argument('--folds', type=int, default=5, help="folds of the cross-fit (default 5)")
    p.add_argument('--insample', action='store_true',
                   help="no cross-fit: ComCat and the adaptation fitted on all controls "
                        "(all subjects with --preserve-group) and applied to all, the "
                        "controls' own scores in-sample (optimistic; for comparison)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--comcat-dir', help="folder of comcat.py (default: COMCAT_DIR, "
                   "~/GitHub/ComCat)")
    p.add_argument('--out', default='comcat_zmaps', help="output prefix")
    a = p.parse_args(argv)
    CC = import_comcat(a.comcat_dir)

    sets, desc = NG.load_models(os.path.expanduser(a.model))
    if len(sets) != 1 or sets[0].voxel is None:
        p.error(f"{a.model} must contain the voxel/vertex-wise models of one input.")
    vm = sets[0].voxel
    d = NG.load_data(a.test)
    if vm.model.n_features != d.Y.shape[1]:
        p.error(f"{a.test} has {d.Y.shape[1]} features, the models {vm.model.n_features}.")
    male = d.male
    if not d.has_male:
        if a.test_male:
            male = _read_column(a.test_male, d.n, '--test-male')
        elif NG._uses_sex(sets[0]):
            p.error("The models use sex, but the test data contain none: give --test-male.")
    C, cov_cols = None, []
    if desc.get('covariates'):
        cov = NG.Covariates.from_state(desc['covariates'])
        if not a.test_cov:
            p.error(f"The models use covariates ({', '.join(cov.names)}): give --test-cov.")
        names, values, _ = NG.read_table(a.test_cov)
        C = NG.covariate_matrix(names, values, cov, d.n, a.test_cov)
        cov_cols = list(cov.names)
    ok = np.isfinite(d.age) & (d.age > 0)
    if C is not None:
        ok &= np.all(np.isfinite(C), axis=1)
    site, site_names = np.zeros(d.n, dtype=np.int64), ('1',)
    if a.test_site:
        site, site_names = NG.read_sites(a.test_site, d.n)
    ctrl = NG._parse_index(a.adjust, d.n)
    Q = None
    if a.nuisance:
        names, values, numeric = NG.read_table(a.nuisance)
        if len(values) != d.n:
            p.error(f"--nuisance has {len(values)} rows for {d.n} test subjects.")
        cols = (NG._parse_columns(a.nuisance_cols, names, numeric) if a.nuisance_cols else
                [nm for j, nm in enumerate(names) if numeric[j]])
        Q = values[:, [names.index(nm) for nm in cols]]
        ok &= np.all(np.isfinite(Q), axis=1)
        print(f"Nuisance variables: {', '.join(cols)}")
    group = None if not a.preserve_group else _read_column(a.preserve_group, d.n,
                                                            '--preserve-group', str)
    subject = None if not a.subject else _read_column(a.subject, d.n, '--subject', str)
    if a.within_subjects and subject is None:
        p.error("--within-subjects needs --subject.")
    terms = subject is not None and a.within_subjects          # effects within subjects

    rows = np.flatnonzero(ok)
    if rows.size < d.n:
        print(f"{d.n - rows.size} subject(s) without valid age, covariates or nuisance "
              "values get no z-scores.")
    pos = {r: i for i, r in enumerate(rows)}
    ctrl_r = np.array([pos[c] for c in ctrl if c in pos])
    preserve = [d.age[rows]] + [male[rows]] + ([C[rows]] if C is not None else [])
    Z = reference_scores(vm, d.Y[rows], d.age[rows], male[rows],
                         None if C is None else C[rows])
    fin = np.all(np.isfinite(Z), axis=0)
    _, code = np.unique(site[rows], return_inverse=True)
    Zh = np.full(Z.shape, np.nan)
    Zh[:, fin], _ = harmonize_zscores(
        CC, Z[:, fin], ctrl_r, code, None if Q is None else Q[rows],
        np.column_stack(preserve), None if subject is None else subject[rows],
        None if group is None else group[rows], a.folds, a.seed, a.location_only,
        a.combat_compatible, a.insample, terms)

    full = np.full(d.Y.shape, np.nan, dtype=np.float32)
    full[np.ix_(rows, vm.model.valid)] = Zh
    test_name = os.path.splitext(os.path.basename(a.test))[0]
    out = dict(Z=full, converged=NG._full_map(vm.model.converged[None].astype(float),
                                              vm.model)[0],
               model=d.name, family=vm.model.family, ind=d.ind, dim=d.dim, age=d.age,
               male=male, df_mu=vm.model.df_mu, df_sigma=vm.model.df_sigma,
               site=site + 1, site_names=np.array(site_names, dtype=object),
               ind_control=ctrl + 1,
               harmonization=('ComCat on the normal scores: sites (empirical Bayes, '
                              + ('location' if a.location_only or terms
                                 else 'location and scale')
                              + (', ComBat site variances' if a.combat_compatible
                                 and not a.location_only and not terms else '')
                              + ')' + (', nuisance ' + ', '.join(cols) if Q is not None else '')
                              + ', preserved: age, sex'
                              + (', ' + ', '.join(cov_cols) if cov_cols else '')
                              + ('; fitted on all subjects, groups preserved' if group is not None
                                 else '; fitted on the controls')
                              + ('; in-sample (no cross-fit)' if a.insample
                                 else '; cross-fitted')
                              + ('; within subjects' if terms else
                                 '; across subjects, folds over subjects'
                                 if subject is not None else '')))
    from scipy.io import savemat
    path = f'{a.out}_zmaps_{test_name}.mat'
    savemat(path, {'NDMzmap': {k: (np.zeros(0) if v is None else v) for k, v in out.items()}},
            do_compression=True)
    zc = Zh[ctrl_r]
    print(f"{path}: {rows.size} subjects, {ctrl_r.size} controls, {int(fin.sum())} "
          f"features; controls' z: SD {np.nanmedian(np.nanstd(zc, axis=0)):.3f} (median), "
          f"{100 * np.nanmean(np.abs(zc) > 2.58):.2f}% beyond +-2.58")


if __name__ == '__main__':
    main()
