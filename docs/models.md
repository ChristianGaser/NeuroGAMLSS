# Statistical models

This page describes the models, the fitting algorithms and the diagnostics of NeuroGAMLSS as implemented in [neurogamlss.py](../neurogamlss.py). The [README](../README.md) explains how to run them, and [NDM_methods.md](../NDM_methods.md) describes NormBrainAGE.

## Model

Every voxel or vertex gets its own GAMLSS (Rigby and Stasinopoulos, 2005). For subject $i$ with age $a_i$, sex $s_i$, site $c_i$ and covariates $\mathbf{x}_i$,

$$
y_i \sim \mathcal{D}(\mu_i, \sigma_i, \nu_i, \tau_i),
$$

$$
\begin{aligned}
g(\mu_i) &= \beta_0 + \mathbf{b}_\mu(a_i)^\top \boldsymbol\beta + \beta_s s_i + \gamma_{c_i} + \mathbf{h}(\mathbf{x}_i)^\top \boldsymbol\beta_x, \\
\log \sigma_i &= \theta_0 + \mathbf{b}_\sigma(a_i)^\top \boldsymbol\theta + \theta_s s_i + \mathbf{h}(\mathbf{x}_i)^\top \boldsymbol\theta_x, \\
\nu_i &= \kappa_0 \;[+\, \mathbf{b}_\nu(a_i)^\top \boldsymbol\kappa], \\
\log \tau_i &= \lambda_0 \;[+\, \mathbf{b}_\nu(a_i)^\top \boldsymbol\lambda].
\end{aligned}
$$

- **Link of the location.** $g$ is the identity for the normal and SHASH families and the logarithm for GG.
- **Age.** $\mathbf{b}_\mu$, $\mathbf{b}_\sigma$ and $\mathbf{b}_\nu$ are natural cubic spline bases without intercept (Hastie et al., 2009). The degrees of freedom of $\mathbf{b}_\mu$ and $\mathbf{b}_\sigma$ are chosen for every training sample, see [Flexibility of the age curves](#flexibility-of-the-age-curves), and those of $\mathbf{b}_\nu$ are given by `--shape-df`. Their knots lie at equally spaced quantiles of the training ages. The splines are linear beyond the youngest and the oldest training age, so the models extrapolate linearly.
- **Sex** enters location and log scale as a main effect. It is left out when the training sample has only one sex or the test data contain no sex.
- **Site** enters the location as a fixed effect, with $\gamma_1 = 0$. Predictions for new data use the size-weighted mean of the site effects, as the standardized mean of ComBat.
- **Covariates** $\mathbf{h}(\mathbf{x})$ are natural splines with `--cov-df` degrees of freedom, linear by default, centred at their training median. A covariate is dropped from a predictor when it is constant or collinear with the other columns of that predictor.
- **Shape.** The shape parameters $\nu$ and $\tau$ are constant by default. The terms in square brackets are added with `--shape-df` where BIC prefers them, see [Age-dependent shape](#age-dependent-shape).

## Families

### Normal

$y \sim \mathcal{N}(\mu, \sigma^2)$, and the z-score is $z = (y - \mu)/\sigma$.

### SHASH

The sinh-arcsinh distribution of Jones and Pewsey (2009) in the SHASHo2 parameterization of gamlss.dist (Rigby et al., 2019). With

$$
z = \frac{y - \mu}{\sigma \tau}, \qquad t = \tau \operatorname{asinh}(z) - \nu, \qquad r = \sinh(t),
$$

$r$ follows a standard normal distribution, and the log-density is

$$
\log f(y) = \log\cosh t - \log\sigma - \tfrac{1}{2}\log(1 + z^2) - \tfrac{1}{2} r^2 - \tfrac{1}{2}\log 2\pi .
$$

- **Special case.** $\nu = 0$ and $\tau = 1$ give the normal distribution with mean $\mu$ and standard deviation $\sigma$.
- **Skewness.** $\nu > 0$ skews the distribution to the right and $\nu < 0$ to the left.
- **Tails.** $\tau < 1$ gives heavier tails and $\tau > 1$ lighter tails than the normal distribution.
- **Scale.** Near the centre, $y - \mu \approx \sigma r$ for every $\tau$ when $\nu = 0$. $\sigma$ therefore keeps the meaning of a scale while $\tau$ changes the tails. $\mu$ and $\sigma$ are still not the mean and standard deviation unless $\nu = 0$ and $\tau = 1$.
- **Normal score.** The z-score is $r$ itself, the standard normal quantile of the fitted cumulative distribution function.

### Generalized gamma

The GG family of gamlss.dist, which Brain Charts used (Bethlehem et al., 2022), for $y > 0$. With $\theta = 1/(\sigma\nu)^2$ and $w = (y/\mu)^\nu$, $\theta w$ follows a gamma distribution with shape $\theta$ and scale 1, and

$$
f(y) = \frac{|\nu|\, \theta^\theta\, w^\theta\, e^{-\theta w}}{\Gamma(\theta)\, y}.
$$

As $\nu \to 0$, GG becomes the log-normal distribution with $\log y \sim \mathcal{N}(\log\mu, \sigma^2)$. The density is evaluated in a form that is smooth through this limit. With $L = \log y - \log\mu$,

$$
\log f(y) = -\log\sigma - \tfrac{1}{2}\log 2\pi - \log y + S(\theta) - \frac{L^2}{\sigma^2} E(\nu L),
$$

where $S(\theta) = \theta\log\theta - \theta - \log\Gamma(\theta) - \tfrac{1}{2}\log\theta + \tfrac{1}{2}\log 2\pi$ is the remainder of Stirling's formula, which tends to 0, and $E(x) = (e^x - 1 - x)/x^2$, which tends to $\tfrac{1}{2}$. Both are computed with series expansions where the direct formulas lose precision.

The normal score is $\Phi^{-1}$ of the regularized incomplete gamma function $P(\theta, \theta w)$ for $\nu > 0$, and of $1 - P(\theta, \theta w)$ for $\nu < 0$. The tail with the smaller probability is used to avoid cancellation. For $\theta > 10^4$ the Wilson–Hilferty approximation is used, and at $\nu = 0$ the log-normal score $L/\sigma$.

## Estimation

### Normal family: RS algorithm

The normal model is fitted by the RS algorithm of GAMLSS, ported from ComBatLS (Gardner et al., 2024). Starting from ordinary least squares and a constant standard deviation, it alternates two updates:

1. a weighted least-squares update of the location coefficients with weights $\sigma^{-2}$;
2. a Fisher scoring step for the coefficients of $\log\sigma$, the least-squares regression of $(z^2 - 1)/2$ on the design of $\log\sigma$. The step is halved, up to 20 times, while the deviance increases.

Iterations stop when the location changes by less than $10^{-6}\sigma$ and the coefficients of $\log\sigma$ by less than $10^{-6}$, or after 2,000 iterations. All voxels are updated together with batched linear algebra.

### SHASH and GG: penalized Newton steps

The models with shape parameters maximize, for every voxel, the penalized log-likelihood

$$
\ell_p = \sum_i \log f(y_i) - \frac{1}{2 s^2} \lVert \boldsymbol\kappa_{\mathrm{all}} \rVert^2 - \frac{1}{2 s^2} \lVert \boldsymbol\lambda_{\mathrm{all}} \rVert^2 - \frac{\rho}{2} \sum_i \left[ (\log\tau_i - \log\tau_{\max})_+^2 + (\log\tau_{\min} - \log\tau_i)_+^2 \right].
$$

- **Priors.** $\boldsymbol\kappa_{\mathrm{all}}$ and $\boldsymbol\lambda_{\mathrm{all}}$ are all coefficients of $\nu$ and $\log\tau$. Their normal priors with standard deviation $s$ (`--shape-prior`, default 1) shrink the shape toward the normal distribution, or toward the log-normal distribution for GG.
- **Bound of τ.** A quadratic penalty with $\rho = 100$ keeps $\tau$ between $\tau_{\min} = 0.2$ and $\tau_{\max}$ (`--tau-max`, default 2). SHASH with $\tau > 1$ has lighter tails than the normal distribution, and its normal scores grow like $z^\tau$ far from the centre. Without the bound, values outside the training range, which are common in patients, can get extreme z-scores.
- **Start.** SHASH starts from the fitted normal model with $\nu = 0$ and $\tau = 1$, which is the same distribution. GG starts from the normal model of $\log y$, its log-normal limit. The data of each voxel are standardized for SHASH and divided by their geometric mean for GG, and the coefficients are transformed back after the fit.

The coefficients are updated by Newton steps for all voxels at once:

- **Derivatives.** The first and second derivatives of $\log f$ with respect to the linear predictors are exact for SHASH. For GG, they are central differences with step $10^{-4}$. The gradient and the Hessian of every voxel are assembled from these derivatives with one matrix product per pair of predictors.
- **Squared scores.** Where the negative Hessian plus the prior is not positive definite, the outer product of the scores replaces it (the BHHH approximation). This always gives an ascent direction. gamlss.dist uses the same squared-score approximation for the expected second derivatives of SHASH.
- **Step halving.** A step is halved, up to 30 times, until the penalized log-likelihood does not decrease, as in the autostep of gamlss.
- **Convergence.** A voxel stops when half its Newton decrement falls below $10^{-9}$ per subject. Voxels that reach 500 iterations, or whose step cannot be improved by halving, keep their best fit and are flagged as not converged.
- **Mixed algorithm.** The optimizer can start with iterations that ignore the cross-derivatives between predictors, as the RS phase of the mixed algorithm of gamlss (`rs_iter` of `fit_distribution`). This was slower without better fits for SHASH and is off by default.

In 8 mm gray and white matter of 2,241 subjects, 0.7% and 0.9% of the voxels did not converge. These voxels contain almost no tissue, with a median density of 0.001, and only a few subjects have nonzero values. Their raw skewness is about 10, and most of them have $\tau$ between 0.2 and 0.4.

### Age-dependent shape

With `--shape-df k`, every voxel is first fitted with constant shape and then with $\nu$ and $\log\tau$ as natural splines of age with $k$ degrees of freedom, starting from the constant fit. A voxel keeps the age-dependent shape if the Bayesian information criterion prefers it,

$$
2(\ell_2 - \ell_1) > m k \log n,
$$

where $\ell_1$ and $\ell_2$ are the unpenalized log-likelihoods of the two fits, $m$ is the number of shape parameters (2 for SHASH, 1 for GG) and $n$ is the number of subjects. In gray and white matter at 8 mm with $k = 3$, BIC chose age-dependent shape for 8% of the voxels.

## Flexibility of the age curves

By default, the degrees of freedom (df) of the age splines of $\mu$ and $\sigma$ are chosen for every training sample and model by 5-fold cross-validation. The folds are stratified by site and age: the subjects of every site are sorted by age and dealt to the folds in turn. The same df apply to all voxels and to all component scores.

- **Voxel-wise models.** Normal models of 1,000 random voxels are fitted to the training folds, and every setting is scored by the median over voxels of its held-out log-likelihood per subject, relative to the mean of the settings compared. The df of $\mu$ are searched first, from 2 to 12 with 3 for $\sigma$, and then those of $\sigma$, from 1 to 5. The normal model keeps the search fast, and for SHASH the same df were best in tests.
- **NormBrainAGE.** Every setting is scored by the mean absolute error of the held-out global brain age after removing its median. The df of $\mu$ are searched from 1 to 8 with 3 for $\sigma$, then those of $\sigma$ from 1 to 4. The principal components of every fold are computed once and reused for all settings, and the warp is left out during the search. With `--pca 0`, the voxel-wise model is shared with the z-maps, and its df are chosen as for them.

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

| df of $\mu$ for all component scores | Gray matter error | White matter error | Gray matter error ratio | White matter error ratio |
|---|---|---|---|---|
| 1 | 5.65 | 6.73 | 1.05 | 1.15 |
| 2 | 5.07 | 5.78 | 1.15 | 1.14 |
| 5, the former default | 5.25 | 6.03 | 1.45 | 1.49 |
| 8 | 5.35 | 6.02 | 1.86 | 1.98 |
| Per component, Akaike criterion | 5.35 | 5.92 | 1.58 | 1.51 |
| Per component, Bayesian criterion | 5.19 | 5.98 | 1.18 | 1.17 |

Choosing the df per component did not beat one setting for all components. The third comparison used the automatic choice on the same data. It picked 3 df for $\mu$ and 1 or 2 for $\sigma$ for brain age, and 2 or 3 for $\mu$ and 2 for $\sigma$ for the z-maps:

| Brain age on NKI adults | Gray matter | White matter |
|---|---|---|
| Mean absolute error, former 5 and 3 df | 5.25 | 6.03 |
| Mean absolute error, chosen df | 5.11 | 5.98 |
| Error ratio, former 5 and 3 df | 1.45 | 1.49 |
| Error ratio, chosen df | 1.23 | 1.23 |

## Normal scores and new sites

The z-map of a test subject is its normal score $\Phi^{-1}(F(y))$ under the model at its chronological age, sex and covariates, at the reference site. For the normal family, this is the usual z-score.

For a new site, the normal scores of its control subjects are standardized for every voxel. If $o$ and $r$ are their mean and standard deviation, every test subject's normal score becomes $(z - o)/r$. For the normal family, this equals the location and scale adaptation of ComBat without empirical Bayes shrinkage (Johnson et al., 2007; Fortin et al., 2018).

## Diagnostics

`--diagnostics` evaluates the fit on the training data. The normal scores are computed at the subjects' own sites, without adaptation, in $G = 10$ age groups of equal size. In each group $g$ with $m$ subjects, four statistics are approximately standard normal for a correct model (Royston and Wright, 2000):

- $Z_{1g} = \sqrt{m}\, \bar z$ for the mean;
- $Z_{2g} = \left[v^{1/3} - \left(1 - \frac{2}{9(m-1)}\right)\right] \big/ \sqrt{\frac{2}{9(m-1)}}$ for the variance $v$, the Wilson–Hilferty transform;
- $Z_{3g}$, D'Agostino's test of the skewness;
- $Z_{4g}$, the Anscombe–Glynn test of the kurtosis.

$Q_k = \sum_g Z_{kg}^2$ is compared with a $\chi^2$ distribution with $G - c_k$ degrees of freedom. $c_k$ is the number of coefficients, including the intercept, of the age curve of the parameter that controls moment $k$:

| Moment | Parameter | $c_k$ |
|---|---|---|
| mean | $\mu$ | 1 + df of $\mu$ |
| variance | $\sigma$ | 1 + df of $\sigma$ |
| skewness | $\nu$ | 1 for constant shape, 1 + `--shape-df` where the shape depends on age, 0 for the normal family |
| kurtosis | $\tau$ | as for $\nu$ for SHASH, 0 for the normal family and GG |

The degrees of freedom are set for every voxel, because BIC chooses age-dependent shape voxel by voxel.

Worm plots (van Buuren and Fredriks, 2001) show, for six age groups, the difference between the sorted normal scores and the expected normal quantiles. The line is the median over voxels and the band covers 5% to 95% of the voxels. The dashed lines are the pointwise 95% band of a single well-calibrated voxel. The shape of a worm points to the kind of misfit:

- **Level.** A worm above or below zero means that the mean of the normal scores is not 0.
- **Slope.** A worm with a slope means that their variance is not 1.
- **U shape.** A U-shaped worm means right-skewed normal scores, and an inverted U left-skewed ones.
- **S shape.** An S shape rising to the right means heavier tails than normal, and an S falling to the right lighter tails.

### Example

Gray and white matter at 8 mm from 2,241 subjects of CamCAN, IXI, OASIS-3 and SALD gave the following shares of voxels with $p < 0.05$, which would be 5% for a perfect model:

| Tissue | Family | Mean | Variance | Skewness | Kurtosis | Median \|skewness\| |
|---|---|---|---|---|---|---|
| GM | normal | 12% | 27% | 87% | 85% | 0.54 |
| GM | shash | 24% | 14% | 40% | 37% | 0.02 |
| GM | shash, `--shape-df 3` | 21% | 12% | 34% | 36% | 0.02 |
| WM | normal | 11% | 35% | 89% | 88% | 0.77 |
| WM | shash | 28% | 19% | 36% | 37% | 0.02 |
| WM | shash, `--shape-df 3` | 26% | 16% | 33% | 36% | 0.02 |

SHASH removed most of the skewness and kurtosis misfit of the normal model. Age-dependent shape improved the fit only slightly. The mean is flagged more often with SHASH than with the normal model. Two effects probably contribute: least squares keeps the residual mean of the normal model close to zero in every age range by construction, while the location of SHASH also absorbs part of any change of shape with age. With 224 subjects per age group, the tests detect small misfit, so the worm plots and the median skewness are better guides to its size.

An independent sample tested the tails. The adults of NKI-Rockland were split into two halves of 258 subjects. One half adapted the models to the site, and the other half was scored. In 500 gray matter voxels with positive values, the shares of normal scores beyond two thresholds were:

| Family | Beyond ±2.58 | Beyond ±3.29 | Median \|skewness\| |
|---|---|---|---|
| expected | 1% | 0.1% | 0 |
| normal | 1.6% | 0.51% | 0.56 |
| shash | 1.1% | 0.15% | 0.13 |
| gg | 1.1% | 0.20% | 0.15 |

SHASH and GG came close to the expected tails, while the normal model gave five times too many extreme scores. GG also left more skewness in the training data, with a median absolute skewness of 0.07 against 0.02 for SHASH.

## Choice of the defaults

- **Normal for PCA scores and regional measures.** Principal component scores and averages over many voxels are close to Gaussian. The normal model is fast, and its $\mu$ and $\sigma$ are the mean and standard deviation. NormBrainAGE always uses it.
- **SHASH for voxel and vertex maps.** Single voxels are often skewed or heavy-tailed, especially near tissue borders. SHASH contains the normal model, starts from it and is shrunk toward it, so voxels without evidence of non-normality stay close to the normal fit. The bound on $\tau$ protects the extrapolated tails.
- **GG for Brain Charts compatibility.** GG suits positive measures such as global volumes and matches the distribution of Brain Charts. Brain Charts also used fractional polynomials of age and random effects of study (Bethlehem et al., 2022), which NeuroGAMLSS replaces by natural splines and fixed site effects.
- **Constant shape unless the diagnostics call for more.** Age-dependent shape adds parameters that are hard to estimate in sparse age ranges. Use `--shape-df` when Q statistics for skewness or kurtosis remain high, or when worm plots change shape across age groups.

## Warp for NormBrainAGE

NormBrainAGE models principal component scores with normal distributions. `--warp` can first transform every voxel $x$, standardized by its training mean and standard deviation, with a sinh-arcsinh warp (Jones and Pewsey, 2009),

$$
w = \sinh(\delta \operatorname{asinh}(x) - \varepsilon).
$$

As in warped Bayesian linear regression (Fraza et al., 2021), $\varepsilon$ and $\delta$ are estimated jointly with a normal location-scale model of $w$, by maximizing the likelihood of $x$ including the Jacobian of the warp. Unlike SHASH, the warp does not depend on age. The same transformation therefore applies to all subjects before PCA, which SHASH cannot provide. In our tests, the warp did not change the accuracy of brain age, so it is off by default. The z-maps use the warp only in one case: with `--pca 0 --family normal`, the voxel-wise normal model of brain age, including the warp, is shared with the z-maps.

## Relation to the gamlss R package

NeuroGAMLSS takes from gamlss and gamlss.dist the SHASHo2 and GG families, the RS algorithm, step halving, the squared-score approximation and the Q statistics and worm plots. It differs in four ways:

- **Vectorized fitting.** All voxels are fitted at once by batched linear algebra rather than one model at a time.
- **Fixed smoothness.** Age enters through natural splines with fixed degrees of freedom instead of penalized splines with automatic smoothing, such as `pb()` in gamlss.
- **Newton steps.** The shape models use Newton steps on all coefficients with exact SHASH derivatives, rather than the RS or CG algorithms of gamlss.
- **Fewer features.** Random effects, other families such as BCT or BCPE, and model selection by generalized AIC beyond the BIC choice of age-dependent shape are not implemented.

PCNtoolkit offers warped Bayesian linear regression (Fraza et al., 2021) and hierarchical Bayesian regression with a SHASH likelihood (de Boer et al., 2024). The latter found SHASH equal to or slightly better than warped regression for most measures.

## References

- Bethlehem, R.A.I., Seidlitz, J., White, S.R., et al., 2022. Brain charts for the human lifespan. Nature 604, 525–533.
- de Boer, A.A.A., Bayer, J.M.M., Kia, S.M., et al., 2024. Non-Gaussian normative modelling with hierarchical Bayesian regression. Imaging Neuroscience 2, 1–36.
- Fortin, J.-P., Cullen, N., Sheline, Y.I., et al., 2018. Harmonization of cortical thickness measurements across scanners and sites. NeuroImage 167, 104–120.
- Fraza, C.J., Dinga, R., Beckmann, C.F., Marquand, A.F., 2021. Warped Bayesian linear regression for normative modelling of big data. NeuroImage 245, 118715.
- Gardner, M., Shinohara, R.T., Bethlehem, R.A.I., et al., 2024. ComBatLS: a location- and scale-preserving method for multi-site image harmonization. bioRxiv.
- Hastie, T., Tibshirani, R., Friedman, J., 2009. The Elements of Statistical Learning, 2nd ed. Springer, New York.
- Johnson, W.E., Li, C., Rabinovic, A., 2007. Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics 8, 118–127.
- Jones, M.C., Pewsey, A., 2009. Sinh-arcsinh distributions. Biometrika 96, 761–780.
- Rigby, R.A., Stasinopoulos, D.M., 2005. Generalized additive models for location, scale and shape. Journal of the Royal Statistical Society, Series C 54, 507–554.
- Rigby, R.A., Stasinopoulos, D.M., Heller, G.Z., De Bastiani, F., 2019. Distributions for Modeling Location, Scale, and Shape: Using GAMLSS in R. Chapman and Hall/CRC, Boca Raton.
- Royston, P., Wright, E.M., 2000. Goodness-of-fit statistics for age-specific reference intervals. Statistics in Medicine 19, 2943–2962.
- van Buuren, S., Fredriks, M., 2001. Worm plot: a simple diagnostic device for modelling growth reference curves. Statistics in Medicine 20, 1259–1277.
