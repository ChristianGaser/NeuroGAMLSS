#!/usr/bin/env python3
"""
NDMScheme.py - Figure of the NDM brain age (NormBrainAGE in neurogamlss.py) and
of its difference to the conventional use of normative models.

  a  pipeline: both approaches share the normative models; the conventional
     approach computes z-scores at chronological age, NDM estimates the age at
     which the data are most likely
  b  normative model of one feature with the z-score of one subject at its
     chronological age and at its brain age
  c  log-likelihood of age of this subject
  d  schematic of the decomposition of the deviation in feature space: the
     conventional deviation mixes the shift along the aging trajectory (BrainAGE)
     with the non-aging deviation

b and c are simulated: 12 features whose mean and SD change with age, and one
subject (chronological age 45 years) whose data were drawn at age 58 years with
non-aging deviations in two features.  The brain age is estimated as in
neurogamlss.py (grid search with parabolic refinement and Laplace standard
error), here with R = I.  d uses two features with equal, age-independent SD and
R = I, where the non-aging deviation is orthogonal to the normative trajectory.

    python3 figures/NDMScheme.py      # writes figures/NDMScheme.png and .pdf
"""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon

logging.getLogger('fontTools').setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))

# colours: chronological age / conventional, brain age / NDM, non-aging deviation
ORANGE, BLUE, AQUA = '#eb6834', '#2a78d6', '#1baf7a'
INK, INK2, MUTED, AXIS = '#0b0b0b', '#52514e', '#898781', '#c3c2b7'
FILL_SHARED, FILL_NDM, FILL_CONV = '#f4f4f1', '#eaf2fc', '#fdeee8'
FS = 6.5                                   # annotations (pt)
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]   # text over data

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})


def num(x, fmt='.1f'):
    """Number with a typographic minus sign."""
    return format(x, fmt).replace('-', '−')


# ---------------------------------------------------------------------------
# simulated normative models of 12 features (age 20..90, SD units at age 20)
# ---------------------------------------------------------------------------
K = np.array([9, 8, 7, 6, 5, 4, 3.5, 3, 2.5, 2, 1.5, 1.0])        # age effect
SHAPE = np.array([0.8, -0.3, 0.5, 0.2, 0.7, 0.0, 0.4, 0.6, -0.2, 0.3, 0.1, 0.5])
S0 = np.array([1, 0.75, 1, 0.9, 1, 1.1, 1, 0.85, 1, 1, 0.9, 1])  # SD at age 20
AGE_CHRON, AGE_TRUE = 45.0, 58.0
EPS = np.array([1.3, -1.4, 0.3, -0.4, -2.4, 0.6, -0.2, 0.8, 2.1, -0.5, 0.2, -0.7])
ALL = np.arange(12)


def mu(a, j):
    u = (np.asarray(a, float) - 20) / 70
    return -K[j] * ((1.8 - SHAPE[j]) * u + SHAPE[j] * u ** 2)


def sig(a, j):
    u = (np.asarray(a, float) - 20) / 70
    return S0[j] * (1 + 0.4 * u)


def loglik(y, grid):
    """log p(y | a) on the age grid with independent features (R = I)."""
    m, s = mu(grid[:, None], ALL), sig(grid[:, None], ALL)
    return -0.5 * np.sum(((y - m) / s) ** 2, axis=1) - np.sum(np.log(s), axis=1)


y_sub = mu(AGE_TRUE, ALL) + sig(AGE_TRUE, ALL) * EPS
grid = np.arange(15, 95.001, 0.25)
ll = loglik(y_sub, grid)
g = np.argmax(ll)                          # as neurogamlss._argmax_refine
curv = ll[g - 1] - 2 * ll[g] + ll[g + 1]
a_hat = grid[g] + 0.5 * (ll[g - 1] - ll[g + 1]) / curv * 0.25
se = 0.25 / np.sqrt(-curv)
fine = np.arange(15, 95.001, 0.05)
ll_fine = loglik(y_sub, fine)
ll_max = loglik(y_sub, np.array([a_hat]))[0]
z_chron = (y_sub - mu(AGE_CHRON, ALL)) / sig(AGE_CHRON, ALL)
z_hat = (y_sub - mu(a_hat, ALL)) / sig(a_hat, ALL)
print(f"brain age {a_hat:.2f} +- {se:.2f} (chronological {AGE_CHRON:.0f}, true {AGE_TRUE:.0f})")
print("z at chronological age:", np.round(z_chron, 2), int(np.sum(np.abs(z_chron) > 1.96)), "extreme")
print("z at brain age:        ", np.round(z_hat, 2), int(np.sum(np.abs(z_hat) > 1.96)), "extreme")

# ---------------------------------------------------------------------------
# figure layout (mm)
# ---------------------------------------------------------------------------
MM = 1 / 25.4
FIG_W, FIG_H = 180.0, 126.0
A_W, A_H, A_TOP = 0.95 * FIG_W, 47.0, 3.0
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
axa = fig.add_axes([0.035, 1 - (A_TOP + A_H) / FIG_H, 0.95, A_H / FIG_H])
gs = fig.add_gridspec(1, 3, left=0.068, right=0.99, top=0.505, bottom=0.095,
                      wspace=0.36, width_ratios=[1, 1, 1])
axb, axc, axe = (fig.add_subplot(gs[0, i]) for i in range(3))
to_fig = fig.transFigure.inverted()


def letter(x, y, s):
    fig.text(x, y, s, fontsize=10, fontweight='bold', ha='left', va='baseline', color=INK)


def title(ax, text, s, x_letter):
    """Left-aligned title with the panel letter on the same baseline."""
    ax.text(0, 1.04, text, transform=ax.transAxes, ha='left', va='baseline',
            fontsize=7.5, color=INK)
    letter(x_letter, to_fig.transform(ax.transAxes.transform((0, 1.04)))[1], s)


# --- a: pipeline ---------------------------------------------------------------
ax = axa
ax.set_xlim(0, A_W)
ax.set_ylim(0, A_H)
ax.axis('off')
STYLE = {'shared': (FILL_SHARED, MUTED, 0.7), 'ndm': (FILL_NDM, BLUE, 0.9),
         'conv': (FILL_CONV, ORANGE, 0.9)}


def box(x, y, w, h, text, kind):
    fc, ec, lw = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.3,
            color=INK, linespacing=1.4, zorder=3)


def path(points, color, lw=0.8):
    p = np.asarray(points, float)
    if len(p) > 2:
        ax.plot(p[:-1, 0], p[:-1, 1], color=color, lw=lw, zorder=1)
    ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle='-|>', mutation_scale=6, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=1))


def header(x, y, text):
    ax.text(x, y, text, ha='left', va='baseline', fontsize=6.8, color=INK, fontweight='bold')


yc, yn, yv = 21.0, 35.5, 8.0              # centres: shared, NDM and conventional rows
box(0, yc - 7, 28, 14, "Training sample\n$y_j$: PCA scores of\nvoxels or vertices,\n"
    "with age, sex, site", 'shared')
path([(28, yc), (32, yc)], INK2)
box(32, yc - 8, 33, 16, "Normative model\nper feature $j$\n"
    "$y_j \\sim \\mathrm{N}(\\mu_j(a),\\,\\sigma_j^2(a))$\nsplines of age $a$; sex; site", 'shared')
header(0, yc + 9.6, "Shared by both approaches")
ax.plot([65, 68], [yc, yc], color=INK2, lw=0.8, zorder=1)
path([(68, yc), (68, yn), (71, yn)], BLUE)
path([(68, yc), (68, yv), (71, yv)], ORANGE)

hn = 14
box(71, yn - hn / 2, 27, hn, "Residual correlation $R$\nof the training\nz-scores", 'ndm')
path([(98, yn), (102, yn)], BLUE)
box(102, yn - hn / 2, 33, hn, "Likelihood of the subject's\ndata at candidate age $a$\n"
    "$\\ell(a) = \\log\\mathrm{N}(z(a);\\,0,\\,R)$\n$-\\ \\Sigma_j \\log\\sigma_j(a)$", 'ndm')
path([(135, yn), (139, yn)], BLUE)
box(139, yn - hn / 2, 32, hn, "Brain age $\\hat a$ = argmax $\\ell(a)$\n± SE (curvature)\n"
    "BrainAGE = $\\hat a$ − age\nnon-aging deviation $z(\\hat a)$", 'ndm')
y_head = yn + hn / 2 + 1.6
header(71, y_head, "NDM: age is estimated from the data")

hc = 13
box(71, yv - hc / 2, 64, hc, "z-scores at chronological age\n"
    "$z_j = (y_j - \\mu_j(\\mathrm{age}))\\,/\\,\\sigma_j(\\mathrm{age})$\n"
    "feature by feature; no age estimate", 'conv')
path([(135, yv), (139, yv)], ORANGE)
box(139, yv - hc / 2, 32, hc, "Deviation map\nextreme deviations,\ne.g. $|z_j| > 1.96$", 'conv')
header(71, yv + hc / 2 + 1.6, "Conventional normative modeling: age is given")
letter(0.008, to_fig.transform(ax.transData.transform((0, y_head)))[1], 'a')

x_left = 0.008
x_mid = axc.get_position().x0 - 0.062
x_right = axe.get_position().x0 - 0.035

# --- b: normative model of one feature -------------------------------------------
ax, j = axb, 3
rng = np.random.default_rng(7)
a_tr = rng.uniform(20, 88, 240)
y_tr = mu(a_tr, j) + sig(a_tr, j) * rng.standard_normal(a_tr.size)
ag = np.linspace(18, 92, 300)
ax.fill_between(ag, mu(ag, j) - 1.96 * sig(ag, j), mu(ag, j) + 1.96 * sig(ag, j),
                color=MUTED, alpha=0.15, lw=0, zorder=1)
ax.scatter(a_tr, y_tr, s=5, color=AXIS, lw=0, zorder=2)
ax.plot(ag, mu(ag, j), color=INK, lw=1.1, zorder=3)
yj, mj = y_sub[j], mu(AGE_CHRON, j)
ylo, yhi = -14.5, 4.5
ax.set_xlim(18, 92)
ax.set_ylim(ylo, yhi)
ax.vlines([AGE_CHRON], ylo, yj, color=ORANGE, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.vlines([a_hat], ylo, yj, color=BLUE, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.annotate('', xy=(AGE_CHRON, yj), xytext=(AGE_CHRON, mj), zorder=4,
            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.0, mutation_scale=7,
                            shrinkA=0, shrinkB=3))
ax.annotate('', xy=(a_hat, yj), xytext=(AGE_CHRON, yj), zorder=4,
            arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.0, mutation_scale=7,
                            shrinkA=3, shrinkB=3))
ax.scatter([AGE_CHRON], [yj], s=34, color=ORANGE, edgecolor='white', lw=0.8, zorder=5)
ax.scatter([a_hat], [yj], s=34, color=BLUE, edgecolor='white', lw=0.8, zorder=5)
ax.text(AGE_CHRON - 1.8, (yj + mj) / 2, f"$z_4$(age) = {num(z_chron[j])}",
        ha='right', va='center', color=INK, fontsize=FS, path_effects=HALO)
ax.text((AGE_CHRON + a_hat) / 2, yj - 0.5, f"$z_4(\\hat a)$\n= {num(z_hat[j])}",
        ha='center', va='top', color=INK, fontsize=FS, linespacing=1.2,
        path_effects=HALO)
ax.text(AGE_CHRON, ylo + 0.4, "age", ha='center', va='bottom', color=INK, fontsize=FS)
ax.text(a_hat, ylo + 0.4, "$\\hat a$", ha='center', va='bottom', color=INK, fontsize=FS)
ax.legend(handles=[
    Line2D([], [], color=AXIS, marker='o', ms=2.6, lw=0, label='training sample'),
    Line2D([], [], color=INK, lw=1.1, label='mean $\\mu_4(a)$'),
    Patch(fc=MUTED, alpha=0.15, label='$\\mu_4(a)$ ± 1.96 $\\sigma_4(a)$')],
    loc='upper right', frameon=False, handlelength=1.6, handletextpad=0.5,
    borderaxespad=0.3, labelspacing=0.35)
ax.set_xlabel("age $a$ (years)")
ax.set_ylabel("feature $y_4$ (a.u.)")
title(ax, "Normative model of feature 4", 'b', x_left)

# --- c: likelihood of age ------------------------------------------------------------
ax = axc
sel = (fine >= 25) & (fine <= 85)
rel = ll_fine - ll_max
l_chron = loglik(y_sub, np.array([AGE_CHRON]))[0] - ll_max
ylo, yhi = -26, 3
ax.plot(fine[sel], rel[sel], color=BLUE, lw=1.5, zorder=3)
ax.vlines(AGE_CHRON, ylo, l_chron, color=ORANGE, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.vlines(a_hat, ylo, 0, color=BLUE, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.scatter([AGE_CHRON], [l_chron], s=34, color=ORANGE, edgecolor='white', lw=0.8, zorder=5)
ax.errorbar([a_hat], [0], xerr=[se], fmt='o', color=BLUE, ms=5.5, mec='white', mew=0.8,
            capsize=2, elinewidth=1, zorder=5)
ax.text(a_hat + 7.0, -0.3, f"$\\hat a$ ± SE =\n{a_hat:.1f} ± {se:.1f} y", ha='left',
        va='top', color=INK, fontsize=FS, linespacing=1.2)
ax.text(AGE_CHRON - 1.5, l_chron, "chronological\nage", ha='right', va='center',
        color=INK, fontsize=FS, linespacing=1.2)
ya = ylo + 3.0
ax.annotate('', xy=(a_hat, ya), xytext=(AGE_CHRON, ya), zorder=4,
            arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.8, mutation_scale=6,
                            shrinkA=0, shrinkB=0))
ax.text((AGE_CHRON + a_hat) / 2, ya + 0.8, f"BrainAGE\n+{a_hat - AGE_CHRON:.1f} y",
        ha='center', va='bottom', color=INK, fontsize=FS, linespacing=1.2)
ax.set_xlim(25, 85)
ax.set_ylim(ylo, yhi)
ax.set_xlabel("candidate age $a$ (years)")
ax.set_ylabel("log-likelihood $\\ell(a) - \\ell(\\hat a)$")
title(ax, "Likelihood of the subject's age", 'c', x_mid)

# --- d: schematic decomposition in feature space -------------------------------------------
ax = axe


def traj(a):
    u = (np.asarray(a, float) - 20) / 70
    return np.stack([9 * u, -u - 5 * u ** 2], axis=-1)


def unit_tn(a):
    u = (float(a) - 20) / 70
    t = np.array([9.0, -1 - 10 * u])
    t /= np.linalg.norm(t)
    return t, np.array([-t[1], t[0]])      # tangent, outer normal


E_AGE, E_HAT, E_DEV = 45.0, 59.0, 1.8
t_hat, n_hat = unit_tn(E_HAT)
p_age, p_hat = traj(E_AGE), traj(E_HAT)
p_sub = p_hat + E_DEV * n_hat
ae = np.linspace(20, 90, 7001)
assert abs(ae[np.argmin(np.sum((traj(ae) - p_sub) ** 2, axis=1))] - E_HAT) < 0.05
P = traj(ae)
ax.plot(P[:, 0], P[:, 1], color=INK, lw=1.1, zorder=2)
decades = np.arange(20, 91, 10)
ax.scatter(*traj(decades).T, s=7, color=INK, lw=0, zorder=2)    # every 10 years
for a0 in (30, 70):
    _, n0 = unit_tn(a0)
    q = traj(a0) + 0.3 * n0
    ax.text(q[0], q[1], f"{a0} y", ha='left', va='bottom', color=INK2, fontsize=6)
q = traj(76) - 0.3 * unit_tn(76)[1]
ax.text(q[0], q[1], "$\\mu(a)$", ha='right', va='top', color=INK, fontsize=FS)
seg = traj(np.linspace(E_AGE, E_HAT - 0.9, 100))
ax.plot(seg[:, 0], seg[:, 1], color=BLUE, lw=2.6, zorder=3, solid_capstyle='butt')
ax.annotate('', xy=p_hat, xytext=seg[-6], zorder=3,
            arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.0, mutation_scale=9,
                            shrinkA=0, shrinkB=4))
ax.annotate('', xy=p_sub, xytext=p_age, zorder=4,
            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2, mutation_scale=7,
                            shrinkA=4, shrinkB=4))
ax.annotate('', xy=p_sub, xytext=p_hat, zorder=4,
            arrowprops=dict(arrowstyle='-|>', color=AQUA, lw=1.2, mutation_scale=7,
                            shrinkA=4, shrinkB=4))
c = 0.22                                   # right-angle mark
ax.add_patch(Polygon([p_hat - c * t_hat, p_hat - c * t_hat + c * n_hat, p_hat + c * n_hat],
                     closed=False, fill=False, ec=INK2, lw=0.6, zorder=3))
for p, col in ((p_age, ORANGE), (p_hat, BLUE), (p_sub, INK)):
    ax.scatter(*p, s=36, color=col, edgecolor='white', lw=0.8, zorder=6)
_, n_age = unit_tn(E_AGE)
ax.text(*(p_age - 0.3 * n_age), "$\\mu$(age)", ha='right', va='top', color=INK, fontsize=FS)
ax.text(*(p_hat - 0.36 * n_hat), "$\\mu(\\hat a)$", ha='center', va='top', color=INK,
        fontsize=FS)
ax.text(*(p_sub + np.array([0.2, 0.0])), "subject $y$", ha='left', va='center',
        color=INK, fontsize=FS)
m = (p_age + p_sub) / 2
ax.text(m[0] - 0.1, m[1] + 0.3, "total deviation\n(conventional z)", ha='center',
        va='bottom', color=INK, fontsize=FS, linespacing=1.2)
q = traj(53) - 0.5 * unit_tn(53)[1]
ax.text(q[0], q[1], "aging: BrainAGE", ha='right', va='top', color=INK, fontsize=FS)
m = (p_hat + p_sub) / 2 + 0.3 * t_hat
ax.text(m[0], m[1], "non-aging\ndeviation", ha='left', va='center', color=INK, fontsize=FS,
        linespacing=1.2)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlim(0.3, 8.4)
ax.set_ylim(-4.6, 0.9)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("feature 1 (SD units)")
ax.set_ylabel("feature 2 (SD units)")
title(ax, "Decomposition of the deviation", 'd', x_right)

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'NDMScheme.{ext}'))
print("saved figures/NDMScheme.png and figures/NDMScheme.pdf")
