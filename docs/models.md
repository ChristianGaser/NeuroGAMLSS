# Statistical models

This page describes the models, the fitting algorithms and the diagnostics of NeuroGAMLSS as implemented in [neurogamlss.py](../neurogamlss.py). The [README](../README.md) explains how to run them, and [NDM_methods.md](../NDM_methods.md) describes NormBrainAGE.

## Model

Every voxel or vertex gets its own GAMLSS (Rigby and Stasinopoulos, 2005). For subject $`i`$ with age $`a_i`$, sex $`s_i`$, site $`c_i`$ and covariates $`\mathbf{x}_i`$,

```math
y_i \sim \mathcal{D}(\mu_i, \sigma_i, \nu_i, \tau_i),
```

```math
\begin{aligned}
g(\mu_i) &= \beta_0 + \mathbf{b}_\mu(a_i)^\top \boldsymbol\beta + \beta_s s_i + \gamma_{c_i} + \mathbf{h}(\mathbf{x}_i)^\top \boldsymbol\beta_x,
\\ \log\, \sigma_i &= \theta_0 + \mathbf{b}_\sigma(a_i)^\top \boldsymbol\theta + \theta_s s_i \;[+\, \delta_{c_i}] + \mathbf{h}(\mathbf{x}_i)^\top \boldsymbol\theta_x,
\\ \nu_i &= \kappa_0 \;[+\, \mathbf{b}_\nu(a_i)^\top \boldsymbol\kappa],
\\ \log\, \tau_i &= \lambda_0 \;[+\, \mathbf{b}_\nu(a_i)^\top \boldsymbol\lambda].
\end{aligned}
```

- **Link of the location.** $`g`$ is the identity for the normal and SHASH families and the logarithm for GG.
- **Age.** $`\mathbf{b}_\mu`$, $`\mathbf{b}_\sigma`$ and $`\mathbf{b}_\nu`$ are natural cubic spline bases without intercept (Hastie et al., 2009). The degrees of freedom of $`\mathbf{b}_\mu`$ and $`\mathbf{b}_\sigma`$ are chosen for every training sample, see [Flexibility of the age curves](#flexibility-of-the-age-curves), and those of $`\mathbf{b}_\nu`$ are given by `--shape-df`. Their boundary knots lie at the 5th and 95th percentiles of the training ages and their inner knots at equally spaced quantiles between them, see [Boundary knots](#boundary-knots). The splines are linear below and above the boundary knots, so the models extrapolate linearly. With `--penalized`, $`\mathbf{b}_\mu`$ and $`\mathbf{b}_\sigma`$ have 10 and 6 df, and a penalty on their curvature sets the flexibility for every voxel, see [Penalized splines](#penalized-splines).
- **Sex** enters location and log scale as a main effect. It is left out when the training sample has only one sex or the test data contain no sex.
- **Site** enters the location as a fixed effect, with $`\gamma_1 = 0`$. With `--site-sd`, it also enters the log scale, with $`\delta_1 = 0`$ (term in square brackets). Predictions for new data use the size-weighted mean of the site effects, as the standardized mean of ComBat, for the log scale as well. With `--site-random`, the effects of all sites are random, $`\gamma_c \sim \mathcal{N}(0, \omega_\gamma^2)`$ and $`\delta_c \sim \mathcal{N}(0, \omega_\delta^2)`$, with variances estimated for every voxel, and new data are predicted at the population mean, $`\gamma = \delta = 0`$, see [Random site effects](#random-site-effects).
- **Covariates** $`\mathbf{h}(\mathbf{x})`$ are natural splines with `--cov-df` degrees of freedom and knots placed as for age, linear by default, centred at their training median. A covariate is dropped from a predictor when it is constant or collinear with the other columns of that predictor.
- **Shape.** The shape parameters $`\nu`$ and $`\tau`$ are constant by default. The terms in square brackets are added with `--shape-df` where BIC prefers them, see [Age-dependent shape](#age-dependent-shape).

## Voxels and vertices that are modelled

A voxel or vertex $`j`$ is modelled if its training values are finite and not constant and if their mean reaches a threshold,

```math
\bar y_j = \frac{1}{n} \sum_i y_{ij} \ge t .
```

`--mask-threshold` gives $`t`$ as a number, an absolute value, or as a percentage of the median of $`\bar y_j`$ over all voxels or vertices. The defaults are $`t = 0.05`$ for volume data, a tissue density, and 5% of the median for surface data, whose measures have different units. A threshold of 0 keeps all voxels and vertices with finite, varying values. If the median is not positive, as for signed measures, a percentage is not used.

- **Why.** In near-empty voxels most subjects have values close to zero, a point mass that no continuous distribution describes. Their z-scores mean little, because a tiny absolute difference becomes a large z, and flexible curves fail there out of sample.
- **Why the mean.** A rule on the minimum, like the absolute threshold masking of SPM, would drop border voxels in which older subjects lose tissue, which is the effect of interest. On surfaces, it would also drop half of the sulc vertices, which are zero or negative on gyral crowns.
- **Use.** The mask is computed from the training data when the models are fitted, also within the folds of cross-validation and of the choice of the df. It is stored with the models, and the JSON description records the threshold and the number of voxels or vertices used. Z-maps and brain age use the same mask. Masked features are NaN in the z-maps and left out of the principal component analysis. Test subjects are not masked individually, because their low values are data.

In 8 mm data of the 2,241 lifespan subjects, the voxels below 0.05 contained nearly all voxels with problems, although they are few:

| Voxels with a mean below 0.05 | Gray matter | White matter |
|---|---|---|
| Share of all voxels | 5% | 14% |
| Share of the SHASH fits that did not converge | 96% | 94% |
| Share of the voxels whose flexible SD curves failed out of sample | 97% | 67% |

On the surface data of NKI-Rockland, the default of 5% removed no vertex of thickness, area, depth, fractal dimension, gyrification and toroGI, and 7.5% of the sulc vertices. The mask changed the mean absolute error of the default brain age by at most 0.05 years. It made the voxel-wise variant more accurate, because there every voxel enters the likelihood.

## Families

### Normal

$`y \sim \mathcal{N}(\mu, \sigma^2)`$, and the z-score is $`z = (y - \mu)/\sigma`$.

### SHASH

The sinh-arcsinh distribution of Jones and Pewsey (2009) in the SHASHo2 parameterization of gamlss.dist (Rigby et al., 2019). With

```math
z = \frac{y - \mu}{\sigma \tau}, \qquad t = \tau\, \mathrm{asinh}(z) - \nu, \qquad r = \sinh(t),
```

$`r`$ follows a standard normal distribution, and the log-density is

```math
\log\, f(y) = \log\, \cosh\, t - \log\, \sigma - \tfrac{1}{2}\log(1 + z^2) - \tfrac{1}{2} r^2 - \tfrac{1}{2}\log\, 2\pi .
```

- **Special case.** $`\nu = 0`$ and $`\tau = 1`$ give the normal distribution with mean $`\mu`$ and standard deviation $`\sigma`$.
- **Skewness.** $`\nu \gt 0`$ skews the distribution to the right and $`\nu \lt 0`$ to the left.
- **Tails.** $`\tau \lt 1`$ gives heavier tails and $`\tau \gt 1`$ lighter tails than the normal distribution.
- **Scale.** Near the centre, $`y - \mu \approx \sigma r`$ for every $`\tau`$ when $`\nu = 0`$. $`\sigma`$ therefore keeps the meaning of a scale while $`\tau`$ changes the tails. $`\mu`$ and $`\sigma`$ are still not the mean and standard deviation unless $`\nu = 0`$ and $`\tau = 1`$.
- **Normal score.** The z-score is $`r`$ itself, the standard normal quantile of the fitted cumulative distribution function.

### Generalized gamma

The GG family of gamlss.dist, which Brain Charts used (Bethlehem et al., 2022), for $`y \gt 0`$. With $`\theta = 1/(\sigma\nu)^2`$ and $`w = (y/\mu)^\nu`$, $`\theta w`$ follows a gamma distribution with shape $`\theta`$ and scale 1, and

```math
f(y) = \frac{|\nu|\, \theta^\theta\, w^\theta\, e^{-\theta w}}{\Gamma(\theta)\, y}.
```

As $`\nu \to 0`$, GG becomes the log-normal distribution with $`\log\, y \sim \mathcal{N}(\log\, \mu, \sigma^2)`$. The density is evaluated in a form that is smooth through this limit. With $`L = \log\, y - \log\, \mu`$,

```math
\log\, f(y) = -\log\, \sigma - \tfrac{1}{2}\log\, 2\pi - \log\, y + S(\theta) - \frac{L^2}{\sigma^2} E(\nu L),
```

where $`S(\theta) = \theta\log\, \theta - \theta - \log\, \Gamma(\theta) - \tfrac{1}{2}\log\, \theta + \tfrac{1}{2}\log\, 2\pi`$ is the remainder of Stirling's formula, which tends to 0, and $`E(x) = (e^x - 1 - x)/x^2`$, which tends to $`\tfrac{1}{2}`$. Both are computed with series expansions where the direct formulas lose precision.

The normal score is $`\Phi^{-1}`$ of the regularized incomplete gamma function $`P(\theta, \theta w)`$ for $`\nu \gt 0`$, and of $`1 - P(\theta, \theta w)`$ for $`\nu \lt 0`$. The tail with the smaller probability is used to avoid cancellation. For $`\theta \gt 10^4`$ the Wilson–Hilferty approximation is used, and at $`\nu = 0`$ the log-normal score $`L/\sigma`$.

## Estimation

### Normal family: RS algorithm

The normal model is fitted by the RS algorithm of GAMLSS, ported from ComBatLS (Gardner et al., 2024). Starting from ordinary least squares and a constant standard deviation, it alternates two updates:

1. a weighted least-squares update of the location coefficients with weights $`\sigma^{-2}`$;
2. a Fisher scoring step for the coefficients of $`\log\, \sigma`$, the least-squares regression of $`(z^2 - 1)/2`$ on the design of $`\log\, \sigma`$. The step is halved, up to 20 times, while the deviance increases.

Iterations stop when the location changes by less than $`10^{-6}\sigma`$ and the coefficients of $`\log\, \sigma`$ by less than $`10^{-6}`$, or after 2,000 iterations. All voxels are updated together with batched linear algebra, in chunks of voxels that run in parallel threads.

Subjects with the same rows of both designs, such as the same age, sex and site without continuous covariates, have the same $`\mu`$ and $`\sigma`$. If there are at most 0.7 of these cells per subject (`CELL_SHARE`), both updates use the mean of $`y`$ and the sum of squares around it in every cell instead of the single subjects. The sum of squared residuals of a cell of $`m`$ subjects is then $`\sum_i (y_i - \bar y)^2 + m (\bar y - \mu)^2`$, which avoids the cancellation of $`\sum_i y_i^2 - 2 \mu \sum_i y_i + m \mu^2`$ for data far from 0. The fit is the same up to the convergence tolerance.

### SHASH and GG: penalized Newton steps

The models with shape parameters maximize, for every voxel, the penalized log-likelihood

```math
\ell_p = \sum_i \log\, f(y_i) - \frac{1}{2 s^2} \lVert \boldsymbol\kappa_{\mathrm{all}} \rVert^2 - \frac{1}{2 s^2} \lVert \boldsymbol\lambda_{\mathrm{all}} \rVert^2 - \frac{\rho}{2} \sum_i [ (\log\, \tau_i - \log\, \tau_{\max})_+^2 + (\log\, \tau_{\min} - \log\, \tau_i)_+^2 ].
```

- **Priors.** $`\boldsymbol\kappa_{\mathrm{all}}`$ and $`\boldsymbol\lambda_{\mathrm{all}}`$ are all coefficients of $`\nu`$ and $`\log\, \tau`$. Their normal priors with standard deviation $`s`$ (`--shape-prior`, default 1) shrink the shape toward the normal distribution, or toward the log-normal distribution for GG.
- **Bound of τ.** A quadratic penalty with $`\rho = 100`$ keeps $`\tau`$ between $`\tau_{\min} = 0.2`$ and $`\tau_{\max}`$ (`--tau-max`, default 2). SHASH with $`\tau \gt 1`$ has lighter tails than the normal distribution, and its normal scores grow like $`z^\tau`$ far from the centre. Without the bound, values outside the training range, which are common in patients, can get extreme z-scores.
- **Start.** SHASH starts from the fitted normal model with $`\nu = 0`$ and $`\tau = 1`$, which is the same distribution. GG starts from the normal model of $`\log\, y`$, its log-normal limit. The data of each voxel are standardized for SHASH and divided by their geometric mean for GG, and the coefficients are transformed back after the fit.

The coefficients are updated by Newton steps for all voxels at once:

- **Derivatives.** The first and second derivatives of $`\log\, f`$ with respect to the linear predictors are exact for SHASH. For GG, they are central differences with step $`10^{-4}`$. The gradient and the Hessian of every voxel are assembled from these derivatives with one matrix product per pair of predictors.
- **Indefinite Hessian.** Where the negative Hessian plus the prior is not positive definite, its eigenvalues are replaced by their absolute values, at least $`10^{-8}`$ times its largest diagonal element (a modified Newton step; Nocedal and Wright, 2006, section 3.4). This gives an ascent direction that keeps the curvature of the likelihood. The outer product of the scores (the BHHH approximation, the squared-score approximation that gamlss.dist uses for the expected second derivatives of SHASH) also gives an ascent direction, but it needed several times more steps in strongly skewed voxels (see below).
- **Steps across the bound of τ.** The penalty of the bound is steep, with curvature $`\rho n`$ for constant $`\tau`$, and the Newton step at a point inside the bound does not see it. For constant $`\tau`$, a step that moves $`\log\, \tau`$ across a bound is therefore replaced by the Newton step of the quadratic model that includes the penalty. This model is exact beyond the bound, and its step ends between the bound and the plain step. Age-dependent $`\tau`$ keeps the plain step (see below).
- **Backtracking.** A step is shortened, up to 30 times, until the penalized log-likelihood does not decrease, as in the autostep of gamlss. Instead of halving, the new length is the maximum of the quadratic through the current penalized log-likelihood, its slope along the step and the rejected value, kept within 0.1 to 0.5 of the last length. Every trial step is evaluated with its derivatives, so that the gradient and the Hessian of an accepted step are ready for the next iteration without a second evaluation.
- **Convergence.** A voxel stops when half its Newton decrement falls below $`10^{-9}`$ per subject. Voxels that reach 500 iterations, or whose step cannot be improved by halving, keep their best fit and are flagged as not converged.
- **Parallel chunks.** The voxels are processed in chunks of about 500,000 values, which run in parallel threads (`--jobs`, by default half the CPU count). NumPy releases Python's global lock in its array operations, so the threads use several cores. The chunks do not depend on the number of threads, so neither do the results.
- **Arithmetic.** One exponential of $`t`$ gives $`\sinh\, t`$, $`\cosh\, t`$, $`\tanh\, t`$ and $`\cosh\, 2t`$, and $`\mathrm{asinh}\, z`$ is computed as $`\mathrm{log1p}(|z| + z^2/(1 + \sqrt{1 + z^2}))`$ with the sign of $`z`$, which is accurate also for small $`|z|`$. Exponentials and logarithms take most of the time, so the log-density uses only two more: the exponential for $`1/(\sigma\tau)`$ and one logarithm of $`\cosh(t)/\sqrt{1 + z^2}`$. Constant $`\nu`$ and $`\tau`$ enter as one value per voxel rather than one per subject.
- **Mixed algorithm.** The optimizer can start with iterations that ignore the cross-derivatives between predictors, as the RS phase of the mixed algorithm of gamlss (`rs_iter` of `fit_distribution`). This was slower without better fits for SHASH and is off by default.

In 8 mm gray and white matter of 2,241 subjects without a mask, with the df chosen by cross-validation (2 and 2), 0.3% and 0.4% of the voxels did not converge. These voxels contain almost no tissue, with a median density of 0.001, and only a few subjects have nonzero values. Their raw skewness is about 10, and most of them have $`\tau`$ between 0.2 and 0.4. With the default mask, all 3,546 gray matter voxels and all 3,242 white matter voxels converged.

The modified Newton step and the steps across the bound replaced the BHHH approximation and plain step halving. In gray matter at 4 mm of the lifespan sample (df 5 and 3), 14% of the voxels had an indefinite Hessian at some iteration. These are strongly skewed voxels, with a median $`|\nu|`$ of 1.3. With BHHH, each needed on average 19.5 iterations and 34 halvings. Another 9% of the voxels end at $`\tau_{\max}`$, and each needed 10 iterations and 20 halvings. Every trial step costs a full evaluation of the derivatives, so these 23% of the voxels caused 72% of the work. The comparison used the same models, the same voxels and four threads on an Apple M2, alternating the two versions:

| SHASH fit, lifespan sample, df chosen by cross-validation | BHHH, halving | Modified Newton | Faster |
|---|---|---|---|
| Gray matter 8 mm, 3,546 voxels | 7.9 s, 1 not converged | 3.0 s, 0 | 2.6× |
| White matter 8 mm, 3,242 voxels | 12.6 s, 1 not converged | 3.6 s, 0 | 3.5× |
| Gray matter 4 mm, 28,222 voxels | 58 s, 5 not converged | 22 s, 0 | 2.7× |
| White matter 4 mm, 25,918 voxels | 89 s, 7 not converged | 26 s, 0 | 3.4× |
| Thickness of NKI-Rockland, 59,380 vertices, df 5 and 2 | 16 s, 0 not converged | 16 s, 0 | 1.0× |

Evaluations of the derivatives per voxel fell from 15 to 7 in gray matter at 4 mm (df 5 and 3) and from 27 to 9 in white matter (df 3 and 2). Where both versions converged, the penalized log-likelihood agreed within $`2.2 \cdot 10^{-6}`$, the convergence tolerance, and the normal scores within 0.001. The voxels that had not converged before reached a higher penalized log-likelihood, by 0.5 to 88, and their normal scores changed by up to 0.6. Cortical thickness has few strongly skewed vertices and gained nothing. Without the mask, both versions stop at the iteration limit in the same near-empty voxels, and their parameters there are arbitrary. With age-dependent shape (`--shape-df 2`), the fit of 8 mm gray matter took 6 s instead of 16 to 19 s, the BIC chose the same voxels, and no voxel reached a lower maximum. The steps across the bound are used only for constant shape. With age-dependent $`\tau`$, whose penalty per subject is weaker, they kept $`\tau`$ inside the bound on the way and led 54 of these 3,546 voxels to lower local maxima, by up to 30.

### Many sites: arrowhead layout

Two site columns are never nonzero in the same row. With at least 8 site columns (`SITE_ARROW`), fixed or random, both fits therefore order the coefficients as the dense ones (intercepts, age splines, sex, covariates and shape) followed by the effects of every site, one for the location and one for the log scale with `--site-sd`. The Hessian then has an arrowhead form: a dense block, its coupling with the sites, and one block of size 1 or 2 per site.

- **Assembly.** The Hessian is built from products of the dense columns and from sums per site, instead of products of all pairs of columns, whose number grows with the square of the number of sites.
- **Linear algebra.** Solves, the modified Newton step (absolute eigenvalues of every site block and then of the Schur complement of the dense block) and the traces for the smoothing parameters go through the site blocks and the Schur complement. The full matrix is never formed.
- **Results.** The fits agree with those of dense matrices within the convergence tolerance, with normal scores within 0.001.

SHASH fits of 1,000 gray matter voxels of the 3,916 subjects of the [random-effects comparison](#random-site-effects), with its 20 sites and with the larger sites split into 57 and 104 groups, on four threads of an Apple M2:

| Sites | Fixed effects, dense | Fixed effects, arrowhead | Random location, dense | Random location, arrowhead | Random location and scale, dense | Random location and scale, arrowhead |
|---|---|---|---|---|---|---|
| 1 | 1.7 s | | | | | |
| 20 | 3.4 s | 2.3 s | 5.1 s | 4.7 s | 8.9 s | 8.2 s |
| 57 | 6.7 s | 2.9 s | 29.3 s | 9.0 s | 70.6 s | 19.0 s |
| 104 | 10.0 s | 3.5 s | 150.7 s | 13.5 s | 820 s | 32.5 s |

Below 8 site columns the matrices are small, and the dense layout is kept.

### Age-dependent shape

With `--shape-df k`, every voxel is first fitted with constant shape and then with $`\nu`$ and $`\log\, \tau`$ as natural splines of age with $`k`$ degrees of freedom, starting from the constant fit. A voxel keeps the age-dependent shape if the Bayesian information criterion prefers it,

```math
2(\ell_2 - \ell_1) > m k \log\, n,
```

where $`\ell_1`$ and $`\ell_2`$ are the unpenalized log-likelihoods of the two fits, $`m`$ is the number of shape parameters (2 for SHASH, 1 for GG) and $`n`$ is the number of subjects. In gray and white matter at 8 mm with $`k = 3`$, BIC chose age-dependent shape for 8% of the voxels.

## Flexibility of the age curves

By default, the degrees of freedom (df) of the age splines of $`\mu`$ and $`\sigma`$ are chosen for every training sample and model by 5-fold cross-validation. The folds are stratified by site and age: the subjects of every site are sorted by age and dealt to the folds in turn. The same df apply to all voxels and to all component scores.

- **Voxel-wise models.** Normal models of 1,000 random voxels are fitted to the training folds, and every setting is scored by the median over voxels of its held-out log-likelihood per subject, relative to the mean of the settings compared. The df of $`\mu`$ are searched first, from 2 to 12 with 3 for $`\sigma`$, and then those of $`\sigma`$, from 1 to 5. The normal model keeps the search fast, and for SHASH the same df were best in tests.
- **NormBrainAGE.** Every setting is scored by the mean absolute error of the held-out global brain age after removing its median. The df of $`\mu`$ are searched from 1 to 8 with 3 for $`\sigma`$, then those of $`\sigma`$ from 1 to 4. The principal components of every fold are computed once and reused for all settings, and the warp is left out during the search. With `--pca 0`, the voxel-wise model is shared with the z-maps, and its df are chosen as for them.

Given numbers in `--df-mu` or `--df-sigma` are not searched. The choice, its criterion and the scores of all settings tried are stored with the models (`df_voxel` and `df_brain_age` in the JSON description). With `--kfold`, the df are chosen once on all subjects before the folds, which makes the cross-validated errors slightly optimistic.

### Evidence

Three comparisons on 8 mm gray and white matter of the 2,241 lifespan subjects motivated this design. The first used 5-fold cross-validation within the sample with SHASH models. Gains are the held-out log-likelihood per subject and voxel relative to the former fixed 5 and 3 df, in thousandths of a nat:

| SHASH, choice of the df | Gray matter | White matter |
|---|---|---|
| 3 and 2 df for all voxels | 1.12 | 0.85 |
| Per voxel by the Akaike information criterion | 0.92 | 0.65 |
| Per voxel by the Bayesian information criterion | 1.07 | 0.84 |
| Per voxel by cross-validation | 1.00 | 0.73 |
| Best df of every voxel, an upper bound | 1.90 | 1.59 |

A choice voxel by voxel, which penalized splines with automatic smoothing such as `pb()` in gamlss would make, was no better than one setting for all voxels. The z-scores changed by about 0.05 standard deviations between these settings, which is small compared with the change between the normal model and SHASH. For the normal model, very flexible standard-deviation curves failed badly out of sample in a few near-empty voxels. This dominated the mean log-likelihood, so the criterion uses the median over voxels.

Brain age depends more on the flexibility, because flexible curves fit noise in the age effects of weak component scores, which looks like age information. The second comparison trained on the lifespan sample and tested on 258 NKI adults, with 258 other NKI adults as controls. The error ratio is the standard deviation of the brain-age error divided by its standard error, which is 1 when the standard errors are calibrated:

| df of $`\mu`$ for all component scores | Gray matter error | White matter error | Gray matter error ratio | White matter error ratio |
|---|---|---|---|---|
| 1 | 5.65 | 6.73 | 1.05 | 1.15 |
| 2 | 5.07 | 5.78 | 1.15 | 1.14 |
| 5, the former default | 5.25 | 6.03 | 1.45 | 1.49 |
| 8 | 5.35 | 6.02 | 1.86 | 1.98 |
| Per component, Akaike criterion | 5.35 | 5.92 | 1.58 | 1.51 |
| Per component, Bayesian criterion | 5.19 | 5.98 | 1.18 | 1.17 |

Choosing the df per component did not beat one setting for all components. The third comparison used the automatic choice on the same data. It picked 3 df for $`\mu`$ and 1 or 2 for $`\sigma`$ for brain age, and 2 or 3 for $`\mu`$ and 2 for $`\sigma`$ for the z-maps:

| Brain age on NKI adults | Gray matter | White matter |
|---|---|---|
| Mean absolute error, former 5 and 3 df | 5.25 | 6.03 |
| Mean absolute error, chosen df | 5.11 | 5.98 |
| Error ratio, former 5 and 3 df | 1.45 | 1.49 |
| Error ratio, chosen df | 1.23 | 1.23 |

### Penalized splines

`--penalized` replaces the choice of one df per sample by penalized splines whose smoothness is estimated for every voxel and every component score. The natural spline bases of age get 10 df for $`\mu`$ and 6 for $`\sigma`$ (`--df-mu`, `--df-sigma` set other sizes), and the penalized log-likelihood is

```math
\ell_p = \ell - \frac{\lambda_\mu}{2} \boldsymbol\beta^\top \mathbf{S}_\mu \boldsymbol\beta - \frac{\lambda_\sigma}{2} \boldsymbol\theta^\top \mathbf{S}_\sigma \boldsymbol\theta,
```

where $`\boldsymbol\beta^\top \mathbf{S} \boldsymbol\beta`$ is the integrated squared second derivative of the curve. It is computed exactly: the curves are linear outside the boundary knots, and their second derivatives are piecewise linear between the knots. The linear term is not penalized, so $`\lambda \to \infty`$ gives a straight line.

- **Smoothing parameters.** They are estimated by the extended Fellner–Schall method (Wood and Fasiolo, 2017), a fixed-point update that increases the Laplace approximation of the restricted likelihood (REML), as `method = "REML"` in mgcv:

  ```math
  \lambda_j \leftarrow \frac{\operatorname{rank}(\mathbf{S}_j) - \lambda_j \operatorname{tr}(\mathbf{H}_p^{-1} \mathbf{S}_j)}{\boldsymbol\beta^\top \mathbf{S}_j \boldsymbol\beta},
  ```

  with the negative Hessian $`\mathbf{H}_p`$ of $`\ell_p`$: the expected information in the normal fits and the Newton Hessian in SHASH and GG. The normal fits update $`\lambda`$ after every RS iteration. SHASH and GG start from the values of the normal fit and update them only when the Newton steps have converged for the current values; updates after every Newton step oscillated between nearly linear and very flexible curves. A voxel has converged when the effective df of its curves change by less than 0.01. $`\lambda`$ stays within $`10^{\pm 6}`$ times the information per unit of penalty, where the curves are practically free or straight; a wider range made the joint Hessian of SHASH ill-conditioned.
- **Effective df.** Every fit stores $`\lambda`$ (`sp`) and the effective df of every penalized term (`edf`), the number of its coefficients minus $`\lambda_j \operatorname{tr}(\mathbf{H}_p^{-1} \mathbf{S}_j)`$.
- **Use.** No df search is needed, for the z-maps, the component scores of NormBrainAGE and BrainLOSS alike.

The comparison used 5-fold cross-validation of 1,000 random voxels or vertices, with the df of the default chosen again within every training fold, as on new data. Differences in held-out log-likelihood per subject, penalized minus chosen df, in thousandths of a nat:

| Data | Chosen df | Effective df of SHASH, $`\mu`$ and $`\sigma`$ | Normal, mean | Normal, median over voxels | SHASH, mean | SHASH, median over voxels |
|---|---|---|---|---|---|---|
| Lifespan gray matter 8 mm | 2 and 2 | 2.2, 1.8 | +0.14 | −0.01 | +0.19 | −0.04 |
| Lifespan white matter 8 mm | 2 and 1 | 2.7, 1.6 | −0.31 | −0.11 | +1.34 | +0.10 |
| NKI-Rockland gray matter 8 mm, 6–85 years | 3 and 1 or 2 | 4.0, 1.8 | +0.76 | +0.18 | +0.84 | +0.19 |
| NKI-Rockland thickness | 3 or 4 and 2 | 4.5, 1.9 | +0.22 | −0.12 | +0.18 | −0.22 |

- **Fit.** The differences are small. About half of the voxels gained, and the median over voxels was near 0. The penalized curves were better at the ends of the age range of white matter (+3.8 and +6.6 for the youngest and oldest 5% with SHASH) and in the children of NKI-Rockland (+1.3 to +1.9), and worse for the oldest 5% of the thickness data (−5.4).
- **Time.** Over the five folds, the penalized fits took 6 to 22 s, the choice of the df and the fits 23 to 54 s.
- **Brain age.** In 10-fold cross-validation of NormBrainAGE, penalized splines were less accurate than the df chosen for the error of the brain age, by 0.07 to 0.17 years: a mean absolute error of 5.17 instead of 5.09 years in gray matter and 5.91 instead of 5.74 in white matter of the lifespan sample, and 4.03 instead of 3.96 in gray matter of NKI-Rockland. REML fits every component score as well as possible, while the choice of the df optimizes the brain age itself.
- **Default.** It stays the choice of one df per sample, which was as good for the z-maps on average, better for brain age and simpler to report. `--penalized` is an option for z-maps of samples whose voxels need different flexibility.

### Boundary knots

The boundary knots of all splines lie at the 5th and 95th percentiles of the training values, and the inner knots at equally spaced quantiles between them. For 3 to 5 df, these are the default knots of restricted cubic splines (Harrell, 2015). The splines are linear below and above the boundary knots. Earlier versions placed the boundary knots at the youngest and the oldest age. Such knots depend on single subjects, and the curves are least certain near them. In the lifespan sample, whose oldest subject is 97 years old, the standard error of $`\mu`$ at the oldest age was 3.9, 5.9 and 6.3 times that at the median age with 2, 3 and 12 df. Boundary knots at the percentiles reduced it by 11%, 19% and 41%.

A smaller standard error alone does not show a better fit, because a stiffer curve can be biased. The held-out log-likelihood was therefore compared by age group with 5-fold cross-validation, as in the choice of the df: normal models of 1,000 random voxels of gray matter at 8 mm, 2 df for $`\sigma`$, and the median over voxels. Both knot placements had the best overall likelihood with the same df of $`\mu`$. The differences to boundary knots at the youngest and the oldest age, in thousandths of a nat per subject, were:

| Training sample | df of $`\mu`$ | All | Youngest 5% | Oldest 5% |
|---|---|---|---|---|
| Lifespan, 2,241 subjects aged 18–97 | 2 | 0.01 | 0.03 | 0.49 |
| Lifespan, white matter | 2 | 0.04 | 0.04 | 0.75 |
| UK Biobank, 14,735 subjects aged 46–82 | 2 | 0.00 | 0.05 | 0.06 |
| NKI-Rockland, 1,238 subjects aged 6–85 | 3 | −0.09 | −0.21 | 1.27 |
| NIH (879 subjects aged 5–22) and lifespan | 4 | 0.01 | 0.15 | 0.77 |

- **Oldest subjects.** The fit of the oldest 5% improved in every sample, least in UK Biobank, whose ages end at 82 years. With 8 df, the loss of the oldest 5% relative to the best setting fell from 4.15 to 2.26 thousandths of a nat in gray matter of the lifespan sample and from 4.90 to 2.18 in NKI-Rockland.
- **Children.** In NKI-Rockland, the youngest 5% are children aged 6 to 9 years, whose brains change fast. There, the linear segment cost a little: −0.21 for the youngest 5% and −0.16 for all 257 subjects younger than 18.
- **Size.** At equal df, 2 or 3 for $`\mu`$, the SHASH z-scores of the lifespan sample changed by a median of 0.002 to 0.003. In the oldest 5%, they changed by 0.004 to 0.005, and for the 6 subjects older than 90 years by 0.02 to 0.03.
- **Brain age.** NormBrainAGE, with the PCA of the raw values used at the time, chose the same df with both knot placements. In 10-fold cross-validation, the mean absolute error rose by 0.01 to 0.03 years: from 5.18 to 5.19 years in gray matter and from 5.91 to 5.93 in white matter of the lifespan sample, and from 4.25 to 4.28 in NKI-Rockland. In the youngest and the oldest 5%, the bias decreased in four of six comparisons, by up to 0.43 years, but the mean absolute error rose by 0.10 to 0.27 years. The coverage of the 95% intervals changed by at most one percentage point.

The z-maps and brain age use the same knots, so that both rest on the same age curves. The comparisons in [Evidence](#evidence), the analysis of the [mask](#voxels-and-vertices-that-are-modelled) and the test of the tails in the [diagnostics example](#example) were made with boundary knots at the youngest and the oldest age.

## Normal scores and new sites

The z-map of a test subject is its normal score $`\Phi^{-1}(F(y))`$ under the model at its chronological age, sex and covariates, at the reference site. For the normal family, this is the usual z-score.

For a new site, the normal scores of its control subjects are standardized for every voxel. If $`o`$ and $`r`$ are their mean and standard deviation, every test subject's normal score becomes $`(z - o)/r`$. For the normal family, this equals the location and scale adaptation of ComBat without empirical Bayes shrinkage (Johnson et al., 2007; Fortin et al., 2018).

## Sites and image quality

The comparisons in this section used models of the 2,241 lifespan subjects (CamCAN, IXI, OASIS-3 and SALD), data smoothed with 6 mm unless stated otherwise, and [scripts/iqm_comparison.py](../scripts/iqm_comparison.py).

### Site effects on the scale

`--site-sd` adds the training sites to the log scale, $`\delta_{c_i}`$ in the model above. ComBat models site differences in the variance in the same way, and Brain Charts with random effects of study. The option was compared with 5-fold cross-validation, as in the choice of the df: models of 1,000 random voxels at 8 mm, evaluated at the subjects' own sites, with the median over voxels. White matter was included here. Gains in held-out log-likelihood per subject over site effects on the location only, in thousandths of a nat:

| Sites | Gray matter, normal | Gray matter, SHASH | White matter, normal | White matter, SHASH |
|---|---|---|---|---|
| 4 samples, 494 to 651 subjects | 0.44 | 0.76 | 0.23 | 0.74 |
| 8 scanners, 67 to 651 subjects | −0.30 | 0.11 | −0.52 | 0.34 |

- **Size.** The sites differ in their standard deviation by only a few percent. The error of an estimated site effect costs about $`1/(2 n_c)`$ nats per subject of a site with $`n_c`$ training subjects, so the effects pay off only for large sites. With the scanners of IXI and OASIS-3 as sites, the small groups lost more than they gained.
- **New samples.** In the four samples below, the option changed the mean absolute error of brain age by at most 0.12 years and left the z-maps unchanged.
- **Default.** It is off. It may help with a few large sites that differ in noise, and with image quality measures in the log scale, see below.

### Random site effects

`--site-random` makes the site effects of the location, and with `--site-sd` of the log scale, random: every site gets its own effect, and a normal prior with a variance estimated for every voxel, by the same Fellner–Schall update as for [penalized splines](#penalized-splines), shrinks the effects of small sites toward the mean. This is the random intercept of Brain Charts and of hierarchical normative models. New data are predicted at the population mean, with the site effects at 0, instead of the size-weighted mean of the fixed effects; the adaptation to a new site with its controls is unchanged.

The comparison used gray matter at 8 mm, 1,000 random voxels and 5-fold cross-validation stratified by site, so every site was in the training and the test folds. The df were chosen once with fixed site effects and used for all variants. Two samples had sites of very different sizes: the 437 subjects of ABIDE (10–18 years) from 13 sites of 11 to 88 subjects, and 3,916 subjects (6–97 years) from 20 sites of 11 to 1,238 subjects (CamCAN, IXI split into its three scanners, OASIS-3, SALD, the 13 ABIDE sites and NKI-Rockland). Differences in held-out log-likelihood per subject relative to fixed site effects on the location, in thousandths of a nat:

| Sample and family | Random location | Sites of at most 30 subjects | Larger sites | Fixed effects also on the log scale (`--site-sd`) | Random effects on both |
|---|---|---|---|---|---|
| ABIDE, normal | +7.9 | +14.6 | +4.4 | −35.6 | +3.1 |
| ABIDE, SHASH | +29.8 | +68.2 | +10.1 | −72.2 | +24.5 |
| 20 sites, normal | +0.7 | +12.8 | +0.2 | −0.4 | +3.9 |
| 20 sites, SHASH | +65.3 | +1,603 | +4.9 | −131.1 | +69.0 |

- **Small sites.** Random effects helped mostly the small sites, in 93% to 96% of the voxels. With fixed effects, the SHASH fits of the smallest sites failed badly for held-out subjects of these sites, by 1.6 nats per subject in the 20-site sample.
- **Scale.** Fixed site effects on the log scale hurt small sites a lot. As random effects, they never did worse than fixed location effects alone; they added to random location effects in the 20-site sample (+3.9 instead of +0.7 with the normal family) but not in ABIDE (+3.1 instead of +7.9).
- **New sites.** Leaving every site out in turn and predicting it without adaptation, random effects did better for 70% to 77% of the sites. On average they gained +4.9 (normal) and +6.5 (SHASH) in ABIDE and lost −2.9 and −5.3 in the 20-site sample, where the population mean is further from some large sites than the size-weighted mean. New sites should be adapted with their controls in any case.
- **Time.** The variances need more iterations: the fits took 2.4 to 5.6 times as long as with fixed effects, and up to 14 times with random effects on both parameters. With many sites, the arrowhead layout keeps this manageable, see [Many sites](#many-sites-arrowhead-layout).
- **Default.** Fixed effects stay the default for samples of a few large sites, where random effects change little. Use `--site-random` for training samples with small sites.

### Image quality measures

CAT12 rates every scan by quality measures (IQMs): noise (NCR), inhomogeneity (ICR), resolution (res_RMS), edge sharpness (res_ECR) and tissue contrast (contrastr), combined into the ratings IQR and SIQR. They can be used in two ways:

- **In the normative models.** As covariates (`--train-cov`, `--cov-mean`, `--cov-sd`), their effects are learned in the training sample, and every test subject is scored at its own quality.
- **In the test sample.** Their effects are estimated in the test sample itself and removed from its data, with `--test-nuisance` or by harmonizing only the test data with ComCat.

`--test-nuisance` estimates the effects in the controls $`i \in C`$ of the test sample (`--adjust`). For every voxel, their values are regressed on an intercept, a natural spline $`f_j`$ of age, sex and the nuisance variables $`\mathbf{q}`$:

```math
y_{ij} = \alpha_j + f_j(a_i) + \beta_j s_i + \mathbf{h}(\mathbf{q}_i)^\top \boldsymbol\gamma_j + e_{ij}, \qquad i \in C .
```

- **Terms.** $`\mathbf{h}`$ is linear, or a natural spline with `--nuisance-df`. Age has up to 3 df, one per 20 controls, and is left out if the controls' ages span less than 5 years. Sex is left out if the controls have only one.
- **Brain age.** It uses the data $`y_{ij} - (\mathbf{h}(\mathbf{q}_i) - \bar{\mathbf{h}})^\top \hat{\boldsymbol\gamma}_j`$ of all test subjects, where $`\bar{\mathbf{h}}`$ is the mean over the controls.
- **z-maps.** They are corrected on their own scale: the controls' normal scores are regressed on the same terms, and the nuisance part is subtracted from all normal scores before the adaptation with the controls. For the normal family, both corrections are nearly the same. For SHASH, a shift of the data is not a constant shift of the skewed normal score.
- **Disease.** Effects of a disease are kept, because the controls have none. Estimated in all subjects instead, the IQM effects would absorb the part of a disease effect that goes along with worse image quality.

Within the training sample, IQMs improved the fit. Held-out log-likelihood as above, with the four samples as sites:

| IQMs in the normative models | Gray matter, normal | Gray matter, SHASH | White matter, normal | White matter, SHASH |
|---|---|---|---|---|
| IQR, mean and log scale | −0.22 | −0.10 | −0.39 | −0.29 |
| ICR, NCR, res_ECR, contrastr, mean | 3.47 | 3.18 | 1.87 | 1.11 |
| the same, mean and log scale | 3.64 | 4.16 | 1.43 | 1.36 |
| the same, with `--site-sd` | 5.05 | 5.76 | 2.40 | 3.36 |
| all seven, mean | 3.12 | 2.72 | 1.14 | 0.25 |
| all seven, mean and log scale | 0.29 | 0.05 | −2.49 | −2.41 |

The IQR hardly varies in the training sample, and SIQR duplicates res_ECR (correlation 0.99). The composite ratings therefore add nothing, and all seven IQMs in the log scale overfit.

Four independent samples tested the IQMs out of sample. In these samples the biology is fixed and the image quality varies:

- **ON-Harmony:** 10 subjects, 80 scans on 6 scanners.
- **Tohoku:** one subject, 121 acquisitions with different sequences.
- **Buchert:** one subject, 531 scans on 116 scanners.
- **MR-ART:** 148 subjects scanned without and with two levels of head motion.

The models of the lifespan sample were applied without adaptation. Brain age used four models (6 and 12 mm smoothing at 4 and 8 mm) with equal weights, and the z-maps SHASH models at 8 mm. The test samples contain no sex, which was therefore left out. Reproducibility is the within-subject standard deviation over a subject's scans, of BrainAGE in years and of z, the median over voxels. The motion effect is the paired difference between strong and no motion in MR-ART.

The test-sample corrections were estimated in two ways:

- **In-sample.** All scans of a test sample served as controls, and the same scans were scored, as when a whole sample is harmonized.
- **Held-out half.** The effects were estimated on one half of the subjects (the scans, for Tohoku and Buchert) and removed from the other half, which shows how they carry over to new subjects.

`--test-nuisance` used the four ratings ICR, NCR, res_ECR and contrastr. ComCat harmonized every test sample as one site, with all seven IQMs as B-splines (5 to 10 df per IQM, as ComCat chooses them) and age preserved where the subjects differ. These are the IQM-only conditions of the ComCat evaluation; scanners as sites were not used. Brain age:

| IQMs | BrainAGE SD, Tohoku | BrainAGE SD, Buchert | BrainAGE SD, ON-Harmony | BrainAGE SD, MR-ART | Motion effect |
|---|---|---|---|---|---|
| none | 7.21 | 4.47 | 1.52 | 4.24 | 4.33 |
| models: ICR, NCR, res_ECR, contrastr, mean and log scale | 6.32 | 4.09 | 1.63 | 3.74 | 2.97 |
| the same, with `--site-sd` | 5.36 | 3.81 | 1.49 | 3.72 | 2.29 |
| `--test-nuisance`, in-sample | 5.43 | 3.70 | 2.00 | 3.05 | 0.25 |
| `--test-nuisance`, held-out half | 5.74 | 3.78 | 2.18 | 3.09 | 0.50 |
| ComCat, in-sample | 1.46 | 1.54 | 2.17 | 2.94 | −0.21 |
| ComCat with `residual_delta`, in-sample | 2.91 | 2.35 | 3.18 | 3.35 | −0.16 |
| ComCat, held-out half | 2.94 | 3.27 | 14.38 | 4.14 | 0.77 |

z-maps, with the share of $`|z| \gt 2.58`$ in the MR-ART scans without motion, which would be 1% for calibrated scores:

| IQMs | z SD, Tohoku | z SD, Buchert | z SD, ON-Harmony | z SD, MR-ART | Mean \|Δz\| of motion | \|z\| > 2.58 without motion |
|---|---|---|---|---|---|---|
| none | 0.388 | 0.301 | 0.191 | 0.380 | 0.203 | 1.07% |
| models: ICR, NCR, res_ECR, contrastr, mean and log scale, `--site-sd` | 0.457 | 0.382 | 0.228 | 0.384 | 0.204 | 1.14% |
| `--test-nuisance`, in-sample | 0.312 | 0.285 | 0.312 | 0.365 | 0.068 | 1.11% |
| `--test-nuisance`, held-out half | 0.343 | 0.289 | 0.380 | 0.377 | 0.069 | 1.30% |
| ComCat, in-sample | 0.098 | 0.140 | 0.295 | 0.422 | 0.085 | 0.71% |
| ComCat with `residual_delta`, in-sample | 0.195 | 0.208 | 0.455 | 0.478 | 0.105 | 1.31% |
| ComCat, held-out half | 0.253 | 0.539 | 6.27 | 0.769 | 0.427 | 2.25% |

- **IQMs in the normative models.** They improved brain age in three of the four samples and halved the motion effect. They made the z-maps of ON-Harmony, Tohoku and Buchert less reproducible, with more extreme z, and did not improve those of MR-ART. With all seven IQMs in the mean and log scale, the z-scores of the strong-motion scans of MR-ART, whose IQMs lie beyond the training range, were on average 9 lower than without motion.
- **Why.** Image quality in the training sample varies mainly between subjects and scanners. In MR-ART, the IQM effects on the mean learned in the training sample predicted the spatial pattern of the motion effect on z only partly, with a correlation of 0.38 over voxels, and not its mean, +0.01 instead of −0.17.
- **`--test-nuisance`.** It removed most of the motion effect from brain age and z-maps, also for held-out subjects, and made both more reproducible in Tohoku and Buchert. In ON-Harmony, five subjects per half confounded anatomy with quality, and the correction made both worse. The in-sample results were only a little better than the held-out ones.
- **z scale.** Removing the IQM effects from the voxel values instead of the normal scores left more of the motion effect in the SHASH z-maps: 0.115 instead of 0.069 for held-out subjects.
- **ComCat.** Evaluated on the harmonized scans themselves, it gave the most reproducible brain age and z-maps of the two single subjects and the most reproducible brain age in MR-ART. The z-maps of MR-ART and ON-Harmony became less reproducible than without correction. Two things contribute besides the IQMs. The scaling by the site variance shrinks all deviations around the fitted values: the share of extreme z in the scans without motion fell to 0.7%. And 35 to 70 spline terms adapt to the sample. With site variances from the residuals (`residual_delta`), the gains shrank. For held-out subjects, the z-maps of Buchert and MR-ART became less reproducible than without correction, and ON-Harmony failed. For the brain age of the two single subjects, ComCat remained better than `--test-nuisance` also for held-out scans.

For z-maps, IQMs are best used on the test data with `--test-nuisance`. This needs enough controls with a spread of image quality: it worked with 74 subjects per half in MR-ART, but not with 5 in ON-Harmony. The controls should cover the image quality of the patients, whose values are otherwise extrapolated linearly. Few non-redundant ratings are safer than all seven. IQMs in the normative models should be a few ratings without the composites, and should stay out of the log scale if the test data may leave their training range.

## Diagnostics

`--diagnostics` evaluates the fit on the training data. The normal scores are computed at the subjects' own sites, without adaptation, in $`G = 10`$ age groups of equal size. In each group $`g`$ with $`m`$ subjects, four statistics are approximately standard normal for a correct model (Royston and Wright, 2000):

- $`Z_{1g} = \sqrt{m}\, \bar z`$ for the mean;
- $`Z_{2g} = [v^{1/3} - (1 - 2/(9(m-1)))] / \sqrt{2/(9(m-1))}`$ for the variance $`v`$, the Wilson–Hilferty transform;
- $`Z_{3g}`$, D'Agostino's test of the skewness;
- $`Z_{4g}`$, the Anscombe–Glynn test of the kurtosis.

$`Q_k = \sum_g Z_{kg}^2`$ is compared with a $`\chi^2`$ distribution with $`G - c_k`$ degrees of freedom. $`c_k`$ is the number of coefficients, including the intercept, of the age curve of the parameter that controls moment $`k`$:

| Moment | Parameter | $`c_k`$ |
|---|---|---|
| mean | $`\mu`$ | 1 + df of $`\mu`$ |
| variance | $`\sigma`$ | 1 + df of $`\sigma`$ |
| skewness | $`\nu`$ | 1 for constant shape, 1 + `--shape-df` where the shape depends on age, 0 for the normal family |
| kurtosis | $`\tau`$ | as for $`\nu`$ for SHASH, 0 for the normal family and GG |

The degrees of freedom are set for every voxel, because BIC chooses age-dependent shape voxel by voxel.

Worm plots (van Buuren and Fredriks, 2001) show, for six age groups, the difference between the sorted normal scores and the expected normal quantiles. The line is the median over voxels and the band covers 5% to 95% of the voxels. The dashed lines are the pointwise 95% band of a single well-calibrated voxel. The shape of a worm points to the kind of misfit:

- **Level.** A worm above or below zero means that the mean of the normal scores is not 0.
- **Slope.** A worm with a slope means that their variance is not 1.
- **U shape.** A U-shaped worm means right-skewed normal scores, and an inverted U left-skewed ones.
- **S shape.** An S shape rising to the right means heavier tails than normal, and an S falling to the right lighter tails.

### Example

Gray and white matter at 8 mm from 2,241 subjects of CamCAN, IXI, OASIS-3 and SALD gave the following shares of voxels with $`p \lt 0.05`$, which would be 5% for a perfect model. The models used the default mask and the df chosen by cross-validation, 2 for $`\mu`$ and 2 for $`\sigma`$ in both tissues:

| Tissue | Family | Mean | Variance | Skewness | Kurtosis | Median \|skewness\| |
|---|---|---|---|---|---|---|
| GM | normal | 16% | 25% | 87% | 84% | 0.51 |
| GM | shash | 23% | 13% | 38% | 34% | 0.02 |
| GM | shash, `--shape-df 3` | 21% | 12% | 32% | 33% | 0.02 |
| WM | normal | 17% | 29% | 88% | 86% | 0.65 |
| WM | shash | 24% | 16% | 31% | 32% | 0.02 |
| WM | shash, `--shape-df 3` | 22% | 14% | 28% | 30% | 0.02 |

SHASH removed most of the skewness and kurtosis misfit of the normal model. Age-dependent shape improved the fit only slightly. The mean is flagged more often with SHASH than with the normal model. Two effects probably contribute: least squares keeps the residual mean of the normal model close to zero in every age range by construction, while the location of SHASH also absorbs part of any change of shape with age. With 3 instead of 2 df for $`\mu`$, which cross-validation rated almost equally, the mean was flagged for 11% (normal) and 19% (SHASH) of the gray matter voxels. With 224 subjects per age group, the tests detect small misfit, so the worm plots and the median skewness are better guides to its size.

An independent sample tested the tails. The adults of NKI-Rockland were split into two halves of 258 subjects. One half adapted the models to the site, and the other half was scored. In 500 gray matter voxels with positive values, the shares of normal scores beyond two thresholds were:

| Family | Beyond ±2.58 | Beyond ±3.29 | Median \|skewness\| |
|---|---|---|---|
| expected | 1% | 0.1% | 0 |
| normal | 1.6% | 0.51% | 0.56 |
| shash | 1.1% | 0.15% | 0.13 |
| gg | 1.1% | 0.20% | 0.15 |

SHASH and GG came close to the expected tails, while the normal model gave five times too many extreme scores. GG also left more skewness in the training data, with a median absolute skewness of 0.07 against 0.02 for SHASH.

## Choice of the defaults

- **Normal for PCA scores and regional measures.** Principal component scores and averages over many voxels are close to Gaussian. The normal model is fast, and its $`\mu`$ and $`\sigma`$ are the mean and standard deviation. NormBrainAGE always uses it.
- **SHASH for voxel and vertex maps.** Single voxels are often skewed or heavy-tailed, especially near tissue borders. SHASH contains the normal model, starts from it and is shrunk toward it, so voxels without evidence of non-normality stay close to the normal fit. The bound on $`\tau`$ protects the extrapolated tails.
- **GG for Brain Charts compatibility.** GG suits positive measures such as global volumes and matches the distribution of Brain Charts. Brain Charts also used fractional polynomials of age and random effects of study (Bethlehem et al., 2022), which NeuroGAMLSS replaces by natural splines and fixed site effects.
- **Constant shape unless the diagnostics call for more.** Age-dependent shape adds parameters that are hard to estimate in sparse age ranges. Use `--shape-df` when Q statistics for skewness or kurtosis remain high, or when worm plots change shape across age groups.

## Principal components for NormBrainAGE

NormBrainAGE models the scores of the leading principal components of every model and region, 100 by default, instead of single voxels or vertices. Before the principal component analysis, every voxel or vertex is centered and divided by its standard deviation in the training sample, so that all of them enter with unit variance. Test data are standardized with the training means and standard deviations. `--no-standardize` uses the raw values, as earlier versions did.

- **Why.** The likelihood of brain age already weights the component scores by their uncertainty, through $`\sigma_j(a)`$ and the residual correlation $`\mathbf{R}`$. The voxels themselves, however, enter the analysis with their raw variance. Without standardization, the components follow the voxels with the largest variance, for example at tissue borders, whether or not this variance relates to age. Standardization changes which information the components keep, not how the likelihood weights it.
- **Scale.** With standardization, brain age does not depend on the scale of single voxels or vertices.
- **Mask.** Standardization gives near-empty voxels the same weight as all others, so it relies on the [mask](#voxels-and-vertices-that-are-modelled).

Brain age was compared with 10-fold cross-validation, with folds stratified by site and age, and on an independent sample: models trained on the lifespan sample were tested on 258 adults of NKI-Rockland, with 258 other adults as controls for the offset. The df were chosen for every setting by cross-validation. Mean absolute errors in years:

| Data | Test | Raw values | Standardized |
|---|---|---|---|
| GM 8 mm | lifespan, cross-validation | 5.19 | 5.07 |
| WM 8 mm | lifespan, cross-validation | 5.93 | 5.76 |
| GM 8 mm | NKI-Rockland, cross-validation | 4.28 | 3.96 |
| WM 8 mm | NKI-Rockland, cross-validation | 4.95 | 4.53 |
| GM 8 mm | NKI adults, independent | 5.17 | 5.00 |
| WM 8 mm | NKI adults, independent | 5.98 | 5.49 |
| GM 4 mm | lifespan, cross-validation | 5.10 | 5.03 |
| WM 4 mm | lifespan, cross-validation | 5.77 | 5.64 |
| GM 4 mm | NKI-Rockland, cross-validation | 4.16 | 3.92 |
| WM 4 mm | NKI-Rockland, cross-validation | 4.88 | 4.46 |
| GM 4 mm | NKI adults, independent | 5.10 | 4.82 |
| WM 4 mm | NKI adults, independent | 5.87 | 5.22 |

- **Volumes.** Standardization lowered the error in all 12 comparisons, by 0.07 to 0.65 years, most in white matter and in the data of NKI-Rockland. The coverage of the 95% intervals changed by at most one percentage point in cross-validation. On the independent sample, the error ratio, the standard deviation of the error divided by its standard error, changed from 1.17 to 1.23 in gray matter and from 1.23 to 1.18 in white matter at 8 mm. The correlation of BrainAGE with age changed by −0.10 to +0.06.
- **Surfaces.** In 10-fold cross-validation of NKI-Rockland, the results were mixed: standardization helped for sulc (8.67 to 7.81 years), fractal dimension (9.70 to 9.15) and area (11.04 to 10.92), hardly changed depth (10.71 to 10.68) and made thickness (5.49 to 5.68) and gyrification (9.08 to 9.23) worse.
- **Number of components.** In gray matter at 8 mm, 100 components were best or nearly best in cross-validation with both kinds of values. On the independent sample, 200 components were more accurate than 100, with raw values (4.86 instead of 5.17 years) and with standardization (4.79 instead of 5.00). The default stays at 100.

| Components | Lifespan, raw | Lifespan, standardized | NKI-Rockland, raw | NKI-Rockland, standardized | NKI adults, raw | NKI adults, standardized |
|---|---|---|---|---|---|---|
| 50 | 5.38 | 5.33 | 4.52 | 4.29 | 5.39 | 5.31 |
| 100 | 5.19 | 5.07 | 4.28 | 3.96 | 5.17 | 5.00 |
| 150 | 5.22 | 5.02 | 4.39 | 4.16 | 4.97 | 5.01 |
| 200 | 5.25 | 5.10 | 4.42 | 4.25 | 4.86 | 4.79 |

Other weights of the voxels were less useful in gray and white matter at 8 mm. Dividing by the residual standard deviation after removing age, sex and site gave errors similar to standardization. Weighting voxels by the strength of their age effect made brain age less accurate, for example 5.39 instead of 5.19 years in gray matter of the lifespan sample. In simulated data, in which the features with the largest variance carry the age signal, the raw values were better, so the advantage of standardization is a property of the image data. The other comparisons of brain age on this page, in [Evidence](#evidence) and [Boundary knots](#boundary-knots), used the raw values.

## Warp for NormBrainAGE

NormBrainAGE models principal component scores with normal distributions. `--warp` can first transform every voxel $`x`$, standardized by its training mean and standard deviation, with a sinh-arcsinh warp (Jones and Pewsey, 2009),

```math
w = \sinh(\delta\, \mathrm{asinh}(x) - \varepsilon).
```

As in warped Bayesian linear regression (Fraza et al., 2021), $`\varepsilon`$ and $`\delta`$ are estimated jointly with a normal location-scale model of $`w`$, by maximizing the likelihood of $`x`$ including the Jacobian of the warp. Unlike SHASH, the warp does not depend on age. The same transformation therefore applies to all subjects before PCA, which SHASH cannot provide. The warped values are standardized like raw values before the PCA. In 10-fold cross-validation of gray matter at 8 mm, the warp made brain age less accurate than standardization alone: 5.17 instead of 5.07 years in the lifespan sample and 4.22 instead of 3.96 years in NKI-Rockland. It is therefore off by default. The z-maps use the warp only in one case: with `--pca 0 --family normal`, the voxel-wise normal model of brain age, including the warp, is shared with the z-maps.

## BrainLOSS

BrainLOSS, for Loss Of Substance beyond Senescence, is the tissue loss that the brain age does not explain (`--brainloss`). For every part $`p`$ of NormBrainAGE, the whole brain and, with `--parcellation`, each lobe, the mean $`\bar y_p`$ of its voxels or vertices, in the units of the data, follows a normal normative model as in [Model](#model),

```math
\bar y_p \sim \mathcal{N}(\mu_p(a, s, c, \mathbf{x}),\ \sigma_p^2(a, s, \mathbf{x})).
```

Its df are chosen by cross-validation of the held-out likelihood, as for the z-maps. BrainLOSS is minus the normal score at the subject's brain age $`\hat a`$ of the whole brain, at the reference site,

```math
\mathrm{BrainLOSS}_p = -\frac{\bar y_p - \mu_p(\hat a)}{\sigma_p(\hat a)} .
```

`BrainLOSS_age` is the same normal score at the chronological age $`a`$. It is the total tissue loss for the subject's age and is not defined for subjects without a valid age:

```math
\mathrm{BrainLOSS}^{\mathrm{age}}_p = -\frac{\bar y_p - \mu_p(a)}{\sigma_p(a)} .
```

With controls (`--adjust`), both are standardized to mean 0 and standard deviation 1 in the controls. Positive values mean less tissue than expected at the brain age or at the chronological age.

- **Why at the brain age.** The two scores differ by the part of the deficit that the brain age explains. Before the standardization, and exactly for a linear $`\mu_p`$ with constant $`\sigma_p`$,

  ```math
  \mathrm{BrainLOSS}^{\mathrm{age}}_p = \mathrm{BrainLOSS}_p + b_p\,(\hat a - a), \qquad b_p = -\frac{\mu_p'}{\sigma_p} ,
  ```

  where $`b_p`$ is the normative decline of the mean per year in standard deviations. At chronological age, a tissue deficit therefore mixes an older-looking brain with loss of other kinds, and BrainLOSS_age overlaps with BrainAGE. At the brain age, the part that the aging pattern explains is removed, so BrainLOSS is nearly uncorrelated with BrainAGE. In a model that also contains BrainAGE, the two scores carry nearly the same information. On their own, BrainLOSS shows whether there is less tissue than expected for the brain age, and BrainLOSS_age whether there is less tissue than in people of the same age.
- **Why signed.** The non-aging deviation of NormBrainAGE is a Mahalanobis distance over all components. It counts atypicality in every direction, including more tissue than expected, so a loss that is specific to a disease is diluted. BrainLOSS measures one direction, less tissue.
- **Weights of the voxels.** The mean weights all voxels equally in the units of the data. In a voxel-wise version of the measure, the mean of the residuals separated pMCI from sMCI in gray matter (see below) at least as well as the mean of the z-scores (AUC 0.63 against 0.62) and better than a mean weighted by the inverse variance (0.60).

The evidence comes from ADNI, with CAT12 r1840 data of 835 subjects and models trained on the lifespan sample processed with the same version. 130 patients with mild cognitive impairment who converted to Alzheimer's disease within 36 months (pMCI) were compared with 128 who stayed stable (sMCI), and 128 patients with Alzheimer's disease with 112 stable controls, who also served as the controls of the test site. The groups follow the diagnoses at baseline and after 12, 24 and 36 months. Areas under the ROC curve at 8 mm, with 95% bootstrap intervals and the default correction:

| Measure | pMCI vs sMCI, GM | pMCI vs sMCI, WM | AD vs controls, GM | AD vs controls, WM |
|---|---|---|---|---|
| BrainAGE | 0.72 [0.66, 0.78] | 0.66 [0.59, 0.72] | 0.83 [0.78, 0.88] | 0.74 [0.68, 0.80] |
| Non-aging deviation | 0.50 [0.42, 0.57] | 0.49 [0.42, 0.57] | 0.58 [0.51, 0.66] | 0.61 [0.54, 0.68] |
| BrainLOSS, whole brain | 0.60 [0.53, 0.67] | 0.48 [0.41, 0.55] | 0.70 [0.63, 0.76] | 0.55 [0.48, 0.62] |
| BrainLOSS, temporal lobes | 0.66 [0.59, 0.72] | 0.52 [0.45, 0.59] | 0.81 [0.75, 0.86] | 0.64 [0.57, 0.71] |

- **Gray matter.** BrainLOSS separated the groups, most in the temporal lobes (the mean of the left and right lobe). In the patients with MCI, it was almost uncorrelated with BrainAGE: the Spearman correlation was −0.01 for the whole brain and 0.14 for the temporal lobes.
- **White matter.** BrainLOSS carried no information about conversion.
- **Added to BrainAGE.** In cross-validated logistic regression, BrainAGE of gray matter reached 0.716. With BrainLOSS of the whole brain, it reached 0.729, and with that of the temporal lobes 0.742. The bootstrap intervals of the gains, −0.018 to +0.044 and −0.012 to +0.064, include zero, so a larger sample has to confirm them.
- **Flexibility.** Cross-validation chose 3 df for $`\mu_p`$ in gray matter. With 2 df, which it rated almost equally, BrainLOSS of the whole brain reached 0.63.
- **Non-aging deviation.** It hardly separated pMCI from sMCI, also with the models adapted to the site (`--correction adapt`, 0.53), with raw or noise-weighted values before the PCA, or with fewer components.
- **BrainLOSS_age** was added after this comparison and has not been evaluated in these data.

## Relation to the gamlss R package

NeuroGAMLSS takes from gamlss and gamlss.dist the SHASHo2 and GG families, the RS algorithm, the autostep and the Q statistics and worm plots. It differs in four ways:

- **Vectorized fitting.** All voxels are fitted at once by batched linear algebra rather than one model at a time.
- **Smoothness.** By default, age enters through natural splines with degrees of freedom chosen per sample and boundary knots at the 5th and 95th percentiles. `--penalized` uses penalized splines with REML smoothing per voxel instead, similar to `pb()` in gamlss.
- **Newton steps.** The shape models use Newton steps on all coefficients with exact SHASH derivatives, rather than the RS or CG algorithms of gamlss.
- **Fewer features.** Random intercepts of the sites are available (`--site-random`). Other random effects, other families such as BCT or BCPE, and model selection by generalized AIC beyond the BIC choice of age-dependent shape are not implemented.

PCNtoolkit offers warped Bayesian linear regression (Fraza et al., 2021) and hierarchical Bayesian regression with a SHASH likelihood (de Boer et al., 2024). The latter found SHASH equal to or slightly better than warped regression for most measures.

## References

- Bethlehem, R.A.I., Seidlitz, J., White, S.R., et al., 2022. Brain charts for the human lifespan. Nature 604, 525–533.
- de Boer, A.A.A., Bayer, J.M.M., Kia, S.M., et al., 2024. Non-Gaussian normative modelling with hierarchical Bayesian regression. Imaging Neuroscience 2, 1–36.
- Fortin, J.-P., Cullen, N., Sheline, Y.I., et al., 2018. Harmonization of cortical thickness measurements across scanners and sites. NeuroImage 167, 104–120.
- Fraza, C.J., Dinga, R., Beckmann, C.F., Marquand, A.F., 2021. Warped Bayesian linear regression for normative modelling of big data. NeuroImage 245, 118715.
- Gardner, M., Shinohara, R.T., Bethlehem, R.A.I., et al., 2024. ComBatLS: a location- and scale-preserving method for multi-site image harmonization. bioRxiv.
- Harrell, F.E., 2015. Regression Modeling Strategies, 2nd ed. Springer, Cham.
- Hastie, T., Tibshirani, R., Friedman, J., 2009. The Elements of Statistical Learning, 2nd ed. Springer, New York.
- Johnson, W.E., Li, C., Rabinovic, A., 2007. Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics 8, 118–127.
- Jones, M.C., Pewsey, A., 2009. Sinh-arcsinh distributions. Biometrika 96, 761–780.
- Nocedal, J., Wright, S.J., 2006. Numerical Optimization, 2nd ed. Springer, New York.
- Rigby, R.A., Stasinopoulos, D.M., 2005. Generalized additive models for location, scale and shape. Journal of the Royal Statistical Society, Series C 54, 507–554.
- Rigby, R.A., Stasinopoulos, D.M., Heller, G.Z., De Bastiani, F., 2019. Distributions for Modeling Location, Scale, and Shape: Using GAMLSS in R. Chapman and Hall/CRC, Boca Raton.
- Royston, P., Wright, E.M., 2000. Goodness-of-fit statistics for age-specific reference intervals. Statistics in Medicine 19, 2943–2962.
- van Buuren, S., Fredriks, M., 2001. Worm plot: a simple diagnostic device for modelling growth reference curves. Statistics in Medicine 20, 1259–1277.
- Wood, S.N., Fasiolo, M., 2017. A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. Biometrics 73, 1071–1081.
