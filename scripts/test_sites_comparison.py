#!/usr/bin/env python3
"""Test data from several sites in NeuroGAMLSS: adaptation per site.

ABIDE437 (10-18 years) has 14 scanners (Scanner437.txt: SITE_ID, with UM_1 and UM_2
apart) with 5 to 52 typically developing (TD) controls each.  Normative models of
NIH879 and NKIe2213 aged 5-30 years (gray matter, s6rp1 8 mm, SHASH z-maps and
NormBrainAGE with the defaults of neurogamlss.py) are applied to it, and every
adaptation is estimated in TD controls and evaluated in other TD controls:

zmaps     5 folds stratified by scanner, 5 seeds: log density of the held-out
          controls' normal scores under the adaptation of the other folds, relative
          to one site for the whole sample (thousandths of a nat per subject and
          voxel), their share beyond +-2.58 and the median SD.  Adaptations: one
          site; separate runs per scanner (mean and SD per scanner); fixed offsets
          per scanner with a common SD; offsets shrunk toward the mean of all
          controls with a common scale (site_adaptation, with n - 1 or n - edf
          degrees of freedom); and with the IQMs ICR, NCR, res_ECR and contrastr
          (--test-nuisance) estimated within or across the scanners.
controls  the same with k = 3, 5, 8 and 12 controls per scanner (30 draws; the other
          controls of every scanner are held out).
brainage  the cross-fitted accuracy of the controls that apply_models reports, with
          one site and with the scanners as sites, for the corrections offset, adapt
          and trend; and BrainAGE offsets per scanner from k controls per scanner, as
          medians (correct_age) or shrunk as the z-maps.
scanners  the size of the scanner effects in the controls and the share of the IQM
          variance between the scanners.

    python scripts/test_sites_comparison.py --data ~/Dropbox/BrainAGE \\
        --tables ~/Dropbox/Harmonization-Oxford/ABIDE/tables --out sites

Writes <out>_zmaps.csv, <out>_controls.csv and <out>_brainage.csv; the models are
fitted once and saved as <out>_models.npz (--model reuses them).
"""

import argparse
import contextlib
import io
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neurogamlss as NG  # noqa: E402

IQM_COLS, IQM_NAMES = [0, 1, 2, 4], ['ICR', 'NCR', 'res_ECR', 'contrastr']
LOG2PI = np.log(2 * np.pi)


def load(args):
    d = lambda name: os.path.join(os.path.expanduser(args.data), f's6rp1_8mm_{name}_CAT12.9.mat')
    model = args.model or args.out + '_models.npz'
    if not os.path.isfile(model):
        NG.main(['--zmaps', '--no-gpr', '--train', d('NIH879') + '+' + d('NKIe2213'),
                 '--age-range', '5', '30', '--save-model', model, '--out', args.out + '_fit'])
    sets = NG.load_models(model)[0]
    tab = lambda name: os.path.join(os.path.expanduser(args.tables), name)
    test = NG.load_data(d('ABIDE437'))
    scanner = np.loadtxt(tab('Scanner437.txt')).astype(int) - 1
    td = np.flatnonzero(np.loadtxt(tab('DX_group437.txt')).astype(int) == 2)
    iqm = np.loadtxt(tab('IQMrall437.txt'))[:, IQM_COLS]
    return sets, replace(test, site=scanner, site_names=()), td, iqm


def one_site(Z, ctrl, code):
    o, r, _ = NG.site_adaptation(Z[ctrl], np.zeros(ctrl.size, int), 1)
    return np.tile(o, (len(Z), 1)), r


def separate(Z, ctrl, code):
    """Separate runs per scanner: mean and SD of its own controls."""
    m, r = one_site(Z, ctrl, code)
    m, r = m.copy(), np.tile(r, (len(Z), 1))
    for s in np.unique(code):
        c = ctrl[code[ctrl] == s]
        if c.size >= 2:
            m[code == s], r[code == s] = Z[c].mean(0), Z[c].std(0, ddof=1)
    return m, r


def fixed(Z, ctrl, code):
    """Fixed offsets: the mean of every scanner's controls, common SD (n - k df)."""
    k = code.max() + 1
    n, mean, ssw = NG._site_sums(Z[ctrl], code[ctrl], k)
    has = n > 0
    o = np.where(has, mean, (n * mean).sum(0) / n.sum(0))
    return o[code], np.sqrt(ssw.sum(0) / (n.sum(0) - has.sum(0)))


def shrunk(Z, ctrl, code, df_fit=True):
    """Shrunk offsets and a common scale (site_adaptation); df_fit=False divides the
    residual sum of squares by n - 1 instead of n - edf."""
    k = code.max() + 1
    o, r, lam = NG.site_adaptation(Z[ctrl], code[ctrl], k)
    if not df_fit:
        n = NG._site_sums(Z[ctrl], code[ctrl], k)[0]
        N = n.sum(0)
        r = r * np.sqrt((N - 1 - (lam * (1 - n / N)).sum(0)) / (N - 1))
    return o[code], r


def with_iqm(adapt, within):
    """Nuisance correction of the IQMs (VoxelModel.zmaps), within or across scanners,
    followed by the adaptation adapt."""
    def f(Z, ctrl, code, iqm=None, age=None, male=None):
        N, P, _, _ = NG.nuisance_design(iqm, age, male, ctrl, 1, IQM_NAMES,
                                        code if within else None)
        R = N @ (P @ Z[ctrl])
        m, r = adapt(Z - R, ctrl, code)
        return m + R, r
    return f


STRATEGIES = {
    'one site': one_site,
    'separate runs per scanner': separate,
    'fixed offsets per scanner, common SD': fixed,
    'shrunk offsets, common scale (n - 1 df)': lambda Z, c, s: shrunk(Z, c, s, False),
    'shrunk offsets, common scale (n - edf df)': shrunk,
    'one site + IQMs': with_iqm(one_site, False),
    'shrunk offsets + IQMs within scanners': with_iqm(shrunk, True),
    'shrunk offsets + IQMs across scanners': with_iqm(shrunk, False),
}


def scores(name, Z, tr, te, code, extra):
    f = STRATEGIES[name]
    m, r = f(Z, tr, code, **extra) if 'IQM' in name else f(Z, tr, code)
    r = np.broadcast_to(r, Z.shape)
    z = (Z[te] - m[te]) / r[te]
    return -0.5 * z ** 2 - np.log(r[te]) - 0.5 * LOG2PI, z


def zmaps(Z, td, code, extra, out, seeds=5, k=5):
    rows = {nm: [] for nm in STRATEGIES}
    for seed in range(seeds):
        fold = NG._site_folds(code[td], k, np.random.default_rng(100 + seed))
        L = {nm: np.full((td.size, Z.shape[1]), np.nan) for nm in STRATEGIES}
        A = {nm: np.full((td.size, Z.shape[1]), np.nan) for nm in STRATEGIES}
        for j in range(k):
            for nm in STRATEGIES:
                L[nm][fold == j], A[nm][fold == j] = scores(
                    nm, Z, td[fold != j], td[fold == j], code, extra)
        for nm in STRATEGIES:
            rows[nm].append((1000 * np.mean(L[nm] - L['one site']),
                             100 * np.mean(np.abs(A[nm]) > 2.58),
                             np.median(A[nm].std(0, ddof=1))))
    print("\nz-maps, held-out controls (5 folds by scanner, 5 seeds): log density relative to "
          "one site (1/1000 nat), share beyond +-2.58, median SD")
    with open(out + '_zmaps.csv', 'w') as f:
        f.write('strategy,dlogdensity,beyond258,sd\n')
        for nm, v in rows.items():
            a = np.mean(v, axis=0)
            print(f"  {nm:42s} {a[0]:+7.1f}  {a[1]:5.2f}%  {a[2]:.3f}")
            f.write(f'{nm},{a[0]:.2f},{a[1]:.3f},{a[2]:.4f}\n')


def describe(Z, td, code, bag, iqm):
    """Size of the scanner effects: RMS over voxels of the offset of every scanner's
    controls from the mean of all controls, minus its sampling noise; the median
    BrainAGE of its controls; and the share of the IQM variance between scanners."""
    print("\nScanners: TD controls, RMS offset of their normal scores (minus sampling noise), "
          "median shrinkage factor of the offset with all controls, median BrainAGE")
    grand = Z[td].mean(0)
    lam = NG.site_adaptation(Z[td], code[td], code.max() + 1)[2]
    for s in np.unique(code[td]):
        c = td[code[td] == s]
        ms = np.mean((Z[c].mean(0) - grand) ** 2 - Z[c].var(0, ddof=1) / c.size)
        print(f"  {s + 1:2d}: {c.size:3d}  {np.sqrt(max(ms, 0)):.2f}  {np.median(lam[s]):.2f}  "
              f"{np.median(bag[c]):+5.1f}")
    share = []
    for j in range(iqm.shape[1]):
        q, c = iqm[td, j], code[td]
        share.append(sum((c == s).sum() * (q[c == s].mean() - q.mean()) ** 2
                         for s in np.unique(c)) / ((q - q.mean()) ** 2).sum())
    print("Share of the variance of the IQMs between scanners: "
          + ', '.join(f'{nm} {v:.2f}' for nm, v in zip(IQM_NAMES, share)))


def controls(Z, td, code, out, draws=30):
    names = ['one site', 'separate runs per scanner', 'fixed offsets per scanner, common SD',
             'shrunk offsets, common scale (n - 1 df)',
             'shrunk offsets, common scale (n - edf df)']
    rng = np.random.default_rng(5)
    print("\nk controls per scanner (30 draws; the others held out): log density relative to "
          "one site (1/1000 nat), share beyond +-2.58, for " + '; '.join(names))
    with open(out + '_controls.csv', 'w') as f:
        f.write('k,' + ','.join(f'{nm} {m}' for nm in names for m in ('dlogdensity', 'beyond258'))
                + '\n')
        for kk in (3, 5, 8, 12):
            acc = {nm: [] for nm in names}
            for _ in range(draws):
                tr, te = [], []
                for s in np.unique(code[td]):
                    c = rng.permutation(td[code[td] == s])
                    tr += list(c[:min(kk, c.size - 2)])
                    te += list(c[min(kk, c.size - 2):])
                tr, te = np.array(tr), np.array(te)
                base = scores('one site', Z, tr, te, code, {})[0]
                for nm in names:
                    L, z = scores(nm, Z, tr, te, code, {})
                    acc[nm].append((1000 * np.mean(L - base), 100 * np.mean(np.abs(z) > 2.58)))
            a = {nm: np.mean(v, axis=0) for nm, v in acc.items()}
            print(f"  {kk:2d}: " + '  '.join(f"{v[0]:+7.1f} ({v[1]:.2f}%)" for v in a.values()))
            f.write(f'{kk},' + ','.join(f'{v[0]:.2f},{v[1]:.3f}' for v in a.values()) + '\n')


def shrunk_offsets(bag, tr, site):
    """BrainAGE corrected with offsets per site shrunk as the z-maps (site_adaptation of
    the controls' BrainAGE)."""
    o = NG.site_adaptation(bag[tr, None], site[tr], site.max() + 1)[0][:, 0]
    return bag - o[site]


def brainage(sets, test, td, out, draws=30):
    rows = []
    print("\nBrainAGE, cross-fitted MAE of the controls (apply_models, 5 folds)")
    for corr in ('offset', 'adapt', 'trend'):
        mae = []
        for t in (replace(test, site=np.zeros(test.n, int)), test):
            with contextlib.redirect_stdout(io.StringIO()):
                res = NG.apply_models(sets, [t], td, corr, gpr=False)
            mae.append(np.mean(np.abs(res['BrainAGE_crossfit'][td, 0])))
        rows.append((corr, *mae))
        print(f"  {corr:7s} one site {mae[0]:.2f}  scanners {mae[1]:.2f}")
    bag = sets[0].brainage.predict(test)['age'] - test.age
    rng = np.random.default_rng(11)
    print("BrainAGE offsets from k controls per scanner (30 draws): MAE of the others, one "
          "site / median per scanner / shrunk per scanner")
    for kk in (3, 5, 8, 12):
        acc = []
        for _ in range(draws):
            tr, te = [], []
            for s in np.unique(test.site[td]):
                c = rng.permutation(td[test.site[td] == s])
                tr += list(c[:min(kk, c.size - 2)])
                te += list(c[min(kk, c.size - 2):])
            tr, te = np.array(tr), np.array(te)
            acc.append([np.mean(np.abs(NG.correct_age(bag, 0 * bag, tr, 'offset', s)[te]))
                        for s in (None, test.site)]
                       + [np.mean(np.abs(shrunk_offsets(bag, tr, test.site)[te]))])
        a = np.mean(acc, axis=0)
        rows.append((f'offset, {kk} controls per scanner', *a))
        print(f"  {kk:2d}: {a[0]:.2f} / {a[1]:.2f} / {a[2]:.2f}")
    with open(out + '_brainage.csv', 'w') as f:
        f.write('correction,mae_one_site,mae_scanners,mae_scanners_shrunk\n')
        f.writelines(','.join([r[0]] + [f'{v:.3f}' for v in r[1:]]
                              + [''] * (3 - len(r[1:]))) + '\n' for r in rows)
    return bag


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', required=True, help="folder of the BrainAGE mat-files")
    p.add_argument('--tables', required=True, help="folder of the ABIDE tables")
    p.add_argument('--model', help="saved models of NIH879 and NKIe2213 (fitted if missing)")
    p.add_argument('--out', default='sites')
    args = p.parse_args()
    sets, test, td, iqm = load(args)
    vm = sets[0].voxel
    Z = vm.zmaps(test)['Z'][:, vm.model.valid].astype(np.float64)   # unadapted
    Z = Z[:, np.all(np.isfinite(Z), axis=0)]
    code = test.site
    zmaps(Z, td, code, dict(iqm=iqm, age=test.age, male=test.male), args.out)
    controls(Z, td, code, args.out)
    bag = brainage(sets, test, td, args.out)
    describe(Z, td, code, bag, iqm)


if __name__ == '__main__':
    main()
