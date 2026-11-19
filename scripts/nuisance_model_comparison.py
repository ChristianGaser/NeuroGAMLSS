"""Terms of the nuisance variables of --test-nuisance, held out: linear terms, penalized
splines (--nuisance-model pspline) and unpenalized natural splines (the former
--nuisance-df, removed).

The input are z-maps without adaptation and without nuisance correction
(neurogamlss.py --normative-only --correction none) of scans whose biology is fixed or
known: repeated scans of one subject (Tohoku, Buchert) or of several subjects
(--subject: MR-ART).  The nuisance effects are estimated in some scans and removed from
the others, on the z scale as in VoxelModel.zmaps (nuisance_design, nuisance_solve):

--sizes   random parts of that many scans (with --subject: subjects, with all their
          scans) estimate the effects, all other scans are scored; mean over --draws
          parts.  The table gives the SD of z over the held-out scans (with --subject:
          the root mean within-subject variance) relative to no correction: below 1 the
          correction helps, above 1 it adds variance.
--site    several test sites: the effects are estimated within the sites in --folds
          folds stratified by site, and the site offsets of the other folds are removed
          afterwards (site_adaptation, without its scale).  The table gives the SD of z
          over all scans, each held out, mean over --draws fold assignments.

    python scripts/nuisance_model_comparison.py --zmaps Tohoku_zmaps_X.mat \\
        --nuisance TestData/Tohoku/IQM121.txt --nuisance-cols 1,2,3,5 --sizes 20,40,80
    python scripts/nuisance_model_comparison.py --zmaps MR-ART_zmaps_X.mat \\
        --nuisance TestData/MR-ART/tables/IQM436.txt --nuisance-cols 1,2,3,5 \\
        --subject TestData/MR-ART/tables/subject436.txt --sizes 10,20,40,80,118
    python scripts/nuisance_model_comparison.py --zmaps Buchert_zmaps_X.mat \\
        --nuisance TestData/Buchert/IQM531.txt --nuisance-cols 1,2,3,5 \\
        --site TestData/Buchert/scannerID531.txt --folds 3

--rows restricts the scans (e.g. the controls of ABIDE).  The results are in
docs/models.md (Image quality measures).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neurogamlss as NG  # noqa: E402

MODELS = ('linear', 'ns2', 'ns3', 'ns4', 'pspline')


def labels(path, n):
    with open(path) as f:
        lab = np.array([line.strip() for line in f if line.strip()])
    if lab.size != n:
        raise SystemExit(f"{path}: {lab.size} labels for {n} scans.")
    return np.unique(lab, return_inverse=True)[1]


def nuisance_part(Z, Q, age, male, fit, model, site=None):
    """Nuisance part (n, p) of all scans from the scans fit, relative to their mean,
    and the effective df of the nuisance terms; None if the scans are too few."""
    kind = model if model in NG.NUISANCE_MODELS else 'linear'
    try:
        N, D, pen, used, _ = NG.nuisance_design(Q, age, male, fit, kind, None, site)
    except ValueError:
        return None, np.nan
    if kind != model:                   # unpenalized natural splines with df = model[2:]
        cols = [int(name) - 1 for name in used]
        bases = [NG.NaturalSpline(Q[fit, j], int(model[2:])) for j in cols]
        base = D[:, :D.shape[1] - N.shape[1]]
        N = np.hstack([b(Q[:, j]) - b(Q[fit, j]).mean(axis=0) for j, b in zip(cols, bases)])
        D = np.hstack([base, N[fit]])
        pen = np.zeros(D.shape[1], dtype=bool)
        if fit.size <= D.shape[1]:
            return None, np.nan
    P, _, edf = NG.nuisance_solve(D, pen, N.shape[1], [Z[fit]])
    return N @ (P @ Z[fit]), edf


def within_sd(Z, subject):
    """Root of the mean within-subject variance, per feature."""
    lev, code = np.unique(subject, return_inverse=True)
    ssw = sum(((Z[code == s] - Z[code == s].mean(axis=0)) ** 2).sum(axis=0)
              for s in range(lev.size))
    return np.sqrt(ssw / (len(subject) - lev.size))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--zmaps', required=True, help="z-maps (struct NDMzmap) of neurogamlss.py "
                                                  "--normative-only --correction none")
    p.add_argument('--nuisance', required=True, help="table of nuisance variables, one row "
                                                     "per scan, without header")
    p.add_argument('--nuisance-cols', help="1-based columns, e.g. 1,2,3,5 (default: all)")
    p.add_argument('--subject', help="subject of every scan, one label per line")
    p.add_argument('--site', help="site of every scan, one label per line")
    p.add_argument('--rows', help="1-based indices of the scans to use, e.g. 1:108")
    p.add_argument('--sizes', default='20,40,80', help="scans (subjects) that estimate the "
                                                       "effects (default 20,40,80)")
    p.add_argument('--folds', type=int, default=5, help="folds with --site (default 5)")
    p.add_argument('--draws', type=int, default=20, help="random parts or fold assignments")
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args(argv)
    from scipy.io import loadmat
    m = loadmat(a.zmaps, squeeze_me=True, struct_as_record=False)['NDMzmap']
    Z = np.asarray(m.Z, dtype=np.float64)
    n = Z.shape[0]
    age, male = np.asarray(m.age, dtype=np.float64), np.asarray(m.male, dtype=np.float64)
    Q = np.loadtxt(a.nuisance, ndmin=2)
    if a.nuisance_cols:
        Q = Q[:, [int(c) - 1 for c in a.nuisance_cols.split(',')]]
    subject = labels(a.subject, n) if a.subject else None
    site = labels(a.site, n) if a.site else None
    rows = NG._parse_index(a.rows, n) if a.rows else np.arange(n)
    Z, age, male, Q = Z[rows], age[rows], male[rows], Q[rows]
    subject, site = (None if x is None else x[rows] for x in (subject, site))
    Z = Z[:, np.all(np.isfinite(Z), axis=0)]
    n = Z.shape[0]
    rng = np.random.default_rng(a.seed)
    head = ''.join(f'{model:>9s}' for model in MODELS) + '  P-spline edf'
    if site is not None:
        code, k = np.unique(site, return_inverse=True)[1], np.unique(site).size
        res, edfs = {model: [] for model in (None,) + MODELS}, []
        for _ in range(a.draws):
            fold = NG._site_folds(code, a.folds, rng)
            for model in res:
                out = np.empty_like(Z)
                for f in range(a.folds):
                    fit, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                    R, edf = (0.0, 0.0) if model is None else nuisance_part(
                        Z, Q, age, male, fit, model, code)
                    if R is None:
                        out[:] = np.nan
                        break
                    W = Z - R
                    offsets = NG.site_adaptation(W[fit], code[fit], k)[0]
                    out[te] = W[te] - offsets[code[te]] + W[fit].mean(axis=0)
                    edfs += [edf] if model == 'pspline' else []
                res[model].append(np.mean(np.std(out, axis=0, ddof=1)))
        print(f"SD of z, held out in {a.folds} folds, nuisance effects within {k} sites, "
              f"site offsets removed ({n} scans, {a.draws} fold assignments)")
        print(f"{'offsets only':>13s}" + head)
        print(f"{np.mean(res[None]):13.3f}" + ''.join(f'{np.mean(res[mo]):9.3f}' for mo in MODELS)
              + f'  {np.mean(edfs):.1f} of {Q.shape[1] * (NG.NUISANCE_BASIS - 1)}')
        return
    unit = np.arange(n) if subject is None else subject
    units = np.unique(unit)
    what = 'scans' if subject is None else 'subjects'
    print(f"Held-out {'SD' if subject is None else 'within-subject SD'} of z relative to no "
          f"correction ({n} scans, {a.draws} random parts)")
    print(f'{what + " in fit":>16s}' + head)
    for size in (int(s) for s in a.sizes.split(',')):
        res, edfs = {model: [] for model in MODELS}, []
        for _ in range(a.draws):
            part = rng.permutation(units)
            fit = np.flatnonzero(np.isin(unit, part[:size]))
            te = np.flatnonzero(np.isin(unit, part[size:]))
            sd = ((lambda X: np.sqrt(np.mean(within_sd(X, subject[te]) ** 2)))
                  if subject is not None else (lambda X: np.mean(np.std(X, axis=0, ddof=1))))
            ref = sd(Z[te])
            for model in MODELS:
                R, edf = nuisance_part(Z, Q, age, male, fit, model)
                res[model].append(np.nan if R is None else sd(Z[te] - R[te]) / ref)
                edfs += [edf] if model == 'pspline' else []
        print(f'{size:16d}' + ''.join(f'{np.mean(res[model]):9.3f}' for model in MODELS)
              + f'  {np.mean(edfs):.1f} of {Q.shape[1] * (NG.NUISANCE_BASIS - 1)}')


if __name__ == '__main__':
    main()
