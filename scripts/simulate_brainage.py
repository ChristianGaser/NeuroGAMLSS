#!/usr/bin/env python3
"""
simulate_brainage.py - age bias of NormBrainAGE on simulated data with known true age.

Every subject's data are drawn at its chronological age, so the true BrainAGE of
every subject is 0, and any dependence of the estimated BrainAGE on age is a bias of
the method.  The data pass through the whole pipeline of neurogamlss.py: mask,
standardization, PCA, choice of the spline df, normative models of the component
scores, residual correlation and maximum likelihood on the age grid.

Simulated data
--------------
p "voxels" with values like tissue densities.  The mean depends on age through three
smooth spatial patterns with different trajectories; subjects differ by 600 latent
spatial factors with a power-law spectrum, whose size grows with age by 15%, a sex
effect and white noise.  The sizes are chosen so that the errors resemble those of
real data (about 5 and 6 years with 1,000 training subjects):

  gm   trajectories that change at every age (linear decline, accelerated decline
       in old age, fast early change): age information everywhere
  wm   an inverted U with its maximum at 46 years, a late decline and a linear
       trend: little age information in mid-adulthood

The data follow the model that NormBrainAGE assumes (normal, with a mean and a
standard deviation that depend on age), except that the trajectories are not
splines.  The simulation therefore shows the bias of the estimate with finite
training samples, not the effects of non-normal data or of sites.

Design
------
Training samples of several sizes (--sizes) and age distributions (--distributions)
on 18 to 88 years:

  uniform  equal numbers at all ages
  young    most subjects young, few old (beta 1.2, 3)
  old      most subjects old, few young (beta 3, 1.2)
  bimodal  young and old adults, few in mid-adulthood
  middle   mid- to late adulthood, few at both ends (normal, 55 +- 10 years)

Every run evaluates three sets of subjects:

  test       an independent sample with uniform ages, the same for all training age
             distributions, so that differences come from the training sample only
  oof        the training subjects out of fold (10-fold cross-validation; PCA, models
             and residual correlation from the other folds, the df chosen once on all)
  insample   the training subjects with the models fitted to them

and these methods:

  ndm_loglik, ndm_mae   NormBrainAGE with the df chosen by the held-out log-likelihood
             or by the mean absolute error of the brain age (no correction)
  ndm_bc     ndm_loglik minus its bias at the brain age, by parametric bootstrap
  gpr_raw    replica of the GPR BrainAGE (gpr_baseline) without correction
  gpr_trend  with the linear trend correction: for the test sample estimated in one
             half of it (its "controls") and applied to the other half, out of fold
             from the out-of-fold estimates of the other folds (as cross_validate)

Reported per method and set: the mean BrainAGE (bias) and the slope of BrainAGE on age
within the deciles of the reference (uniform) age distribution, the overall slope and
the mean absolute error; for NormBrainAGE also the coverage of the 95% likelihood-
ratio and Laplace intervals, the share of estimates at the ends of the age grid, the
df chosen by both criteria and the bias that the parametric bootstrap predicts.

Usage
-----
    python scripts/simulate_brainage.py --out sim                 # full design, slow
    python scripts/simulate_brainage.py --sizes 300 --reps 2 --distributions uniform young

Outputs: <out>.json (all runs and their means), <out>.md (tables) and, with
matplotlib, <out>_bias.png (bias by age for every training age distribution).
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neurogamlss as NG  # noqa: E402

AGE_RANGE = (18.0, 88.0)
SCENARIOS = ('gm', 'wm')
DISTRIBUTIONS = ('uniform', 'young', 'old', 'bimodal', 'middle')
SETS = ('test', 'oof', 'insample')
METHODS = ('ndm_loglik', 'ndm_mae', 'ndm_bc', 'gpr_raw', 'gpr_trend')
N_BINS = 10


def draw_ages(dist, n, rng):
    """n ages from the training age distribution dist on AGE_RANGE."""
    lo, hi = AGE_RANGE
    if dist == 'uniform':
        u = rng.uniform(0, 1, n)
    elif dist == 'young':
        u = rng.beta(1.2, 3, n)
    elif dist == 'old':
        u = rng.beta(3, 1.2, n)
    elif dist == 'bimodal':
        u = np.where(rng.uniform(size=n) < 0.5, rng.beta(2, 8, n), rng.beta(8, 2, n))
    elif dist == 'middle':
        u = np.empty(0)
        while u.size < n:                       # normal, truncated to the age range
            x = (rng.normal(55, 10, 2 * n) - lo) / (hi - lo)
            u = np.r_[u, x[(x >= 0) & (x <= 1)]]
        u = u[:n]
    else:
        raise ValueError(f"Unknown age distribution {dist!r}.")
    return lo + (hi - lo) * u


class World:
    """Population of the simulation: spatial patterns of the age effects, of the latent
    factors and of sex, drawn once (seed), from which samples are drawn at given ages."""

    def __init__(self, scenario='gm', p=3000, n_factors=600, decay=1.5, factor_sd=0.03,
                 noise_sd=0.01, age_sd=None, het=0.15, wm_linear=1.0, seed=0):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}.")
        rng = np.random.default_rng(seed)
        self.scenario, self.p = scenario, p
        if age_sd is None:                  # errors of about 5 (gm) and 6 (wm) years
            age_sd = 0.015 if scenario == 'gm' else 0.02
        x = np.arange(p) / p
        # latent factors: Fourier modes with random phases, finer with their number
        freq = 1 + np.arange(n_factors) // 2
        self.L = np.sqrt(2) * np.cos(2 * np.pi * freq[:, None] * x + rng.uniform(0, 2 * np.pi, (n_factors, 1)))
        self.lam = factor_sd * (1 + np.arange(n_factors)) ** (-decay / 2)       # SDs
        # age and sex patterns: smooth mixtures of the first modes
        mix = rng.standard_normal((4, 40)) / np.sqrt(40)
        self.W = age_sd * mix[:3] @ self.L[:40]
        self.sex = 0.005 * mix[3] @ self.L[:40]
        self.noise_sd, self.het, self.wm_linear = noise_sd, het, wm_linear

    def trajectories(self, age):
        """Age functions (n, 3) of the three patterns."""
        u = (np.asarray(age, dtype=np.float64) - AGE_RANGE[0]) / (AGE_RANGE[1] - AGE_RANGE[0])
        if self.scenario == 'gm':
            return np.column_stack([-2.0 * u, -4.0 * np.maximum(u - 0.55, 0) ** 2,
                                    1.5 * np.exp(-4 * u)])
        return np.column_stack([-5.0 * (u - 0.4) ** 2, -5.0 * np.maximum(u - 0.6, 0) ** 2,
                                -self.wm_linear * u])

    def sample(self, age, rng, name='s4rp1_8mm_SIM_CAT12.9.mat'):
        """Data of subjects of the given ages (true BrainAGE 0)."""
        n = len(age)
        male = rng.integers(0, 2, n).astype(np.float64)
        u = (age - AGE_RANGE[0]) / (AGE_RANGE[1] - AGE_RANGE[0])
        T = rng.standard_normal((n, len(self.lam))) * self.lam * (1 + self.het * u[:, None])
        Y = 0.5 + self.trajectories(age) @ self.W + T @ self.L + male[:, None] * self.sex
        Y += self.noise_sd * rng.standard_normal((n, self.p))
        return NG.Data(Y.astype(np.float32), np.asarray(age, dtype=np.float64), male,
                       np.zeros(n, int), name, res='8')


def summarize(err, age, edges):
    """Age bias of the BrainAGE values err: mean and slope on age within the age bins
    (NaN with fewer than 20 subjects), overall slope and mean absolute error."""
    ok = np.isfinite(err)
    err, age = err[ok], age[ok]
    which = np.clip(np.searchsorted(edges, age, side='right') - 1, 0, len(edges) - 2)
    bias, slope = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = which == b
        if sel.sum() >= 20:
            bias[b] = err[sel].mean()
            slope[b] = np.polyfit(age[sel], err[sel], 1)[0]
    return dict(bias=bias, slope=slope, slope_all=float(np.polyfit(age, err, 1)[0]),
                mae=float(np.mean(np.abs(err))), mean=float(np.mean(err)))


def folds_by_age(age, k, rng):
    """Age-stratified folds as in cross_validate."""
    order = np.lexsort((rng.random(len(age)), age))
    fold = np.empty(len(age), dtype=int)
    fold[order] = np.arange(len(age)) % k
    return fold


def crossfit_trend(pred, age, fold):
    """BrainAGE after a linear trend correction of every fold estimated in the other
    folds."""
    out = np.full(len(age), np.nan)
    for f in np.unique(fold):
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        out[te] = NG.trend_correct(pred, age, tr)[te]
    return out


def run(world, dist, n, seed, args):
    """One training sample: errors of all methods on the three sets, summarized."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(*AGE_RANGE, N_BINS + 1)
    train = world.sample(draw_ages(dist, n, rng), rng)
    test = world.sample(rng.uniform(*AGE_RANGE, args.n_test), rng)
    kw = dict(pca=args.pca)
    fold = folds_by_age(train.age, args.kfold, rng)
    half = rng.permutation(test.n) % 2            # halves of the test sample
    res = dict(dist=dist, n=n, seed=seed, scenario=world.scenario)
    err = {s: {} for s in SETS}
    ests = {}
    for crit in ('loglik', 'mae'):
        dm, ds, rec = NG.select_df_brainage(train, kw, criterion=crit)
        res[f'df_{crit}'] = [dm, ds]
        res[f'cv_{crit}'] = {key: rec['stages'][-1][key][rec['stages'][-1]['df_sigma'].index(ds)]
                             for key in ('loglik', 'mae', 'slope', 'bias_young', 'bias_old')}
        if (dm, ds) not in ests:
            est = NG.NDMBrainAge(df_mu=dm, df_sigma=ds, **kw).fit(train)
            out = est.predict(test)
            oof = np.full(train.n, np.nan)
            for f in range(args.kfold):
                tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                oof[te] = NG.NDMBrainAge(df_mu=dm, df_sigma=ds, **kw).fit(
                    train.subset(tr)).predict(train.subset(te))['age']
            ests[(dm, ds)] = dict(est=est, out=out, oof=oof, ins=est.predict(train)['age'])
        e = ests[(dm, ds)]
        err['test'][f'ndm_{crit}'] = e['out']['age'] - test.age
        err['oof'][f'ndm_{crit}'] = e['oof'] - train.age
        err['insample'][f'ndm_{crit}'] = e['ins'] - train.age
    e = ests[tuple(res['df_loglik'])]
    est, out = e['est'], e['out']
    res.update(
        cover_lr=float(np.mean((out['ci_low'] <= test.age) & (test.age <= out['ci_high']))),
        cover_laplace=float(np.nanmean(np.abs(out['age'] - test.age) < 1.96 * out['sd'])),
        at_bound=float(np.mean(out['at_bound'])), ci_open=float(np.mean(out['ci_open'])),
        deviation=float(np.mean(out['deviation_age'])))
    # age information and bias predicted by the model at the centres of the age bins
    centres = (edges[:-1] + edges[1:]) / 2
    cur = est.age_curves(centres, n_rep=args.bias_replicates, seed=seed)
    res['info_se'] = np.mean(cur['se'], axis=1)
    if args.bias_replicates:
        res['boot_bias'], res['boot_sd'] = np.mean(cur['bias'], axis=1), np.mean(cur['sd'], axis=1)
        res['boot_cover'] = np.mean(cur['cover'], axis=1)
        fine = est.age_curves(n_rep=args.bias_replicates, seed=seed)     # along the grid
        bias = np.full(test.n, np.nan)
        for c, m in enumerate(fine['male']):
            sel = test.male == m if fine['male'].size > 1 else np.ones(test.n, bool)
            bias[sel] = np.interp(out['age'][sel], fine['age'], fine['bias'][:, c])
        err['test']['ndm_bc'] = out['age'] - bias - test.age
    # GPR replica
    g_test = NG.gpr_baseline(train.Y, train.age, test.Y)
    err['test']['gpr_raw'] = g_test - test.age
    err['test']['gpr_trend'] = crossfit_trend(g_test, test.age, half)
    g_oof = np.full(train.n, np.nan)
    for f in range(args.kfold):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        g_oof[te] = NG.gpr_baseline(train.Y[tr], train.age[tr], train.Y[te])
    err['oof']['gpr_raw'] = g_oof - train.age
    err['oof']['gpr_trend'] = crossfit_trend(g_oof, train.age, fold)
    g_ins = NG.gpr_baseline(train.Y, train.age, train.Y)
    err['insample']['gpr_raw'] = g_ins - train.age
    err['insample']['gpr_trend'] = NG.trend_correct(g_ins, train.age, np.arange(train.n))
    ages = dict(test=test.age, oof=train.age, insample=train.age)
    res['summary'] = {s: {m: summarize(v, ages[s], edges) for m, v in err[s].items()}
                      for s in SETS}
    return res


def aggregate(runs):
    """Mean and standard error over the runs of one condition."""
    def stats(values):
        A = np.array(values, dtype=np.float64)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)      # age bins without subjects
            k = np.sum(np.isfinite(A), axis=0)
            mean = np.nanmean(A, axis=0)
            se = np.nanstd(A, axis=0, ddof=1) / np.sqrt(k) if len(A) > 1 else np.full_like(mean, np.nan)
        return dict(mean=mean, se=se)
    out = dict(reps=len(runs), summary={})
    for s in SETS:
        out['summary'][s] = {}
        for m in runs[0]['summary'][s]:
            out['summary'][s][m] = {key: stats([r['summary'][s][m][key] for r in runs])
                                    for key in ('bias', 'slope', 'slope_all', 'mae', 'mean')}
    for key in ('cover_lr', 'cover_laplace', 'at_bound', 'ci_open', 'deviation', 'info_se',
                'boot_bias', 'boot_sd', 'boot_cover'):
        if key in runs[0]:
            out[key] = stats([r[key] for r in runs])
    for crit in ('loglik', 'mae'):
        dfs = [tuple(r[f'df_{crit}']) for r in runs]
        out[f'df_{crit}'] = {f'{a}/{b}': dfs.count((a, b)) for a, b in sorted(set(dfs))}
        out[f'cv_{crit}'] = {key: stats([r[f'cv_{crit}'][key] for r in runs])
                             for key in runs[0][f'cv_{crit}']}
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [None if not np.isfinite(v) else float(v) for v in obj.ravel()] \
            if obj.ndim == 1 else _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _cell(st, i=None, digits=2):
    m = st['mean'] if i is None else st['mean'][i]
    s = st['se'] if i is None else st['se'][i]
    if not np.isfinite(m):
        return ''
    return f"{m:+.{digits}f}" + (f" ± {s:.{digits}f}" if np.isfinite(s) else '')


def report(conditions, edges):
    """Markdown tables of the aggregated conditions (dict key -> aggregate)."""
    bins = [f"{a:.0f}–{b:.0f}" for a, b in zip(edges[:-1], edges[1:])]
    lines = []
    for key, agg in conditions.items():
        scenario, dist, n = key
        lines += [f"### {scenario}, training ages {dist}, n = {n} ({agg['reps']} samples)", '']
        lines += [f"df of mu/sigma chosen by the log-likelihood: {agg['df_loglik']}; by the MAE: "
                  f"{agg['df_mae']}.  Test sample: coverage of the 95% intervals "
                  f"{agg['cover_lr']['mean']:.3f} (likelihood ratio), "
                  f"{agg['cover_laplace']['mean']:.3f} (Laplace); "
                  f"{100 * agg['at_bound']['mean']:.1f}% of the estimates at an end of the grid, "
                  f"{100 * agg['ci_open']['mean']:.1f}% of the intervals open.", '']
        for s in SETS:
            lines += [f"**{s}**: mean BrainAGE [years] by age (true value 0), mean ± SE over the "
                      "samples", '',
                      '| Method | ' + ' | '.join(bins) + ' | Slope | MAE |',
                      '|---|' + '---|' * (len(bins) + 2)]
            for m, st in agg['summary'][s].items():
                lines.append(f"| {m} | " + ' | '.join(_cell(st['bias'], i) for i in range(len(bins)))
                             + f" | {_cell(st['slope_all'], digits=3)} | {st['mae']['mean']:.2f} |")
            lines.append('')
        s = 'test'
        lines += ["**test**: slope of BrainAGE on age within the age deciles [years per year]", '',
                  '| Method | ' + ' | '.join(bins) + ' |', '|---|' + '---|' * len(bins)]
        for m, st in agg['summary'][s].items():
            lines.append(f"| {m} | " + ' | '.join(_cell(st['slope'], i) for i in range(len(bins)))
                         + ' |')
        lines.append('')
        if 'boot_bias' in agg:
            lines += ["**Model-based curves** of ndm_loglik at the centres of the age deciles: "
                      "standard error from the Fisher information, and bias, SD and coverage of "
                      "the likelihood-ratio intervals by parametric bootstrap", '',
                      '| | ' + ' | '.join(bins) + ' |', '|---|' + '---|' * len(bins)]
            for label, k2, fmt in (('SE from I(a)', 'info_se', '.2f'), ('Bootstrap SD', 'boot_sd', '.2f'),
                                   ('Bootstrap bias', 'boot_bias', '+.2f'),
                                   ('Bootstrap coverage', 'boot_cover', '.3f')):
                lines.append(f"| {label} | " + ' | '.join(format(v, fmt) for v in agg[k2]['mean'])
                             + ' |')
            lines.append('')
    return '\n'.join(lines)


def figure(conditions, edges, path, n):
    """Bias by age of the test sample for every scenario and training age distribution."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    keys = [k for k in conditions if k[2] == n]
    scenarios = sorted({k[0] for k in keys})
    dists = [d for d in DISTRIBUTIONS if any(k[1] == d for k in keys)]
    if not keys:
        return False
    x = (edges[:-1] + edges[1:]) / 2
    style = dict(ndm_loglik=('#1f6fb4', '-', 'NormBrainAGE'),
                 ndm_bc=('#1f6fb4', ':', 'NormBrainAGE, bias removed'),
                 gpr_trend=('#d1622b', '-', 'GPR, trend corrected'),
                 gpr_raw=('#d1622b', '--', 'GPR, uncorrected'))
    fig, axes = plt.subplots(len(scenarios), len(dists), figsize=(3.1 * len(dists), 2.9 * len(scenarios)),
                             sharex=True, sharey=True, squeeze=False)
    for i, sc in enumerate(scenarios):
        for j, dist in enumerate(dists):
            ax = axes[i, j]
            agg = conditions.get((sc, dist, n))
            ax.axhline(0, color='0.5', lw=0.8)
            if agg is not None:
                for m, (col, ls, label) in style.items():
                    st = agg['summary']['test'].get(m)
                    if st is not None:
                        ax.errorbar(x, st['bias']['mean'], yerr=st['bias']['se'], color=col, ls=ls,
                                    lw=1.5, marker='o', ms=3, label=label)
            ax.set_title(f"{sc.upper()}, training ages: {dist}", fontsize=9)
            if i == len(scenarios) - 1:
                ax.set_xlabel('Age [years]')
            if j == 0:
                ax.set_ylabel('Mean BrainAGE [years]')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Simulated independent test sample (true BrainAGE 0), training n = {n}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--scenarios', nargs='+', default=list(SCENARIOS), choices=SCENARIOS)
    p.add_argument('--distributions', nargs='+', default=list(DISTRIBUTIONS),
                   choices=DISTRIBUTIONS, help="training age distributions")
    p.add_argument('--sizes', nargs='+', type=int, default=[1000],
                   help="training sample sizes (default 1000)")
    p.add_argument('--reps', type=int, default=10, help="training samples per condition")
    p.add_argument('--n-test', type=int, default=4000, help="subjects of the test sample")
    p.add_argument('--features', type=int, default=3000, help="number of voxels")
    p.add_argument('--pca', type=int, default=100, help="principal components (--pca)")
    p.add_argument('--kfold', type=int, default=10, help="folds of the out-of-fold estimates")
    p.add_argument('--bias-replicates', type=int, default=200,
                   help="replicates of the parametric bootstrap of the bias (0 = off)")
    p.add_argument('--noise-sd', type=float, default=0.01, help="SD of the white noise")
    p.add_argument('--age-sd', type=float, default=None,
                   help="size of the age patterns (default 0.015 for gm, 0.02 for wm)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='simulate_brainage', help="output prefix")
    a = p.parse_args(argv)
    edges = np.linspace(*AGE_RANGE, N_BINS + 1)
    runs, conditions = [], {}
    t0 = time.time()
    for sc in a.scenarios:
        world = World(sc, p=a.features, noise_sd=a.noise_sd, age_sd=a.age_sd, seed=a.seed)
        for n in a.sizes:
            for dist in a.distributions:
                cond = [run(world, dist, n, a.seed + 1000 * r + n, a) for r in range(a.reps)]
                runs += cond
                agg = conditions[(sc, dist, n)] = aggregate(cond)
                t = agg['summary']['test']
                print(f"{sc} {dist:8s} n={n:5d}: slope NDM {t['ndm_loglik']['slope_all']['mean']:+.3f}"
                      f" (df by MAE {t['ndm_mae']['slope_all']['mean']:+.3f}), GPR trend "
                      f"{t['gpr_trend']['slope_all']['mean']:+.3f}, raw "
                      f"{t['gpr_raw']['slope_all']['mean']:+.3f}; MAE NDM "
                      f"{t['ndm_loglik']['mae']['mean']:.2f}, GPR {t['gpr_trend']['mae']['mean']:.2f}"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
    with open(a.out + '.json', 'w') as f:
        json.dump(_jsonable(dict(settings=vars(a), age_bins=edges, runs=runs,
                                 conditions={'/'.join(map(str, k)): v
                                             for k, v in conditions.items()})), f)
    with open(a.out + '.md', 'w') as f:
        f.write(report(conditions, edges) + '\n')
    saved = [a.out + '.json', a.out + '.md']
    if figure(conditions, edges, a.out + '_bias.png', a.sizes[len(a.sizes) // 2]):
        saved.append(a.out + '_bias.png')
    print(f"Saved {', '.join(saved)}; total time {time.time() - t0:.0f}s")
    return conditions


if __name__ == '__main__':
    main()
