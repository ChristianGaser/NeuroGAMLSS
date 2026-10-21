# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NeuroGAMLSS is a single Python module, `neurogamlss.py`, with two parts. It fits vectorized GAMLSS normative models to every voxel or vertex of brain MRI data and writes z-maps. Its brain-age part, NormBrainAGE, estimates brain age as the maximum-likelihood age under normative models of principal component scores. The inputs are mat-files written by `BA_data2mat.m` of the BrainAGE toolbox, for example from CAT12. The repository contains no real MRI data. A second tool, `mat2image.py`, writes the maps as NIfTI or GIFTI images.

- `README.md` is the user guide, with options and outputs.
- `docs/models.md` describes the statistics and holds the evidence for the defaults.
- `NDM_methods.md` is a draft methods section for NormBrainAGE.
- `docs/NeuroGAMLSS Method Scheme.html` explains the z-maps, NormBrainAGE and BrainLOSS step by step, with figures; `docs/NeuroGAMLSS Method Scheme.md` is its Markdown version.
- `scripts/iqm_comparison.py` compares image quality measures as covariates and site effects on the log SD on the real CAT12 samples, which live outside the repository.

## Commands

```bash
pip install -e ".[all,test]"                  # h5py, nibabel, matplotlib and pytest
python3 -m pytest                             # all tests, about 15 s
python3 -m pytest tests/test_neurogamlss.py::test_spline_boundary_knots   # one test
python3 -m pytest -k brainage                 # tests whose name matches
python3 neurogamlss.py --help                 # command line; installed as `neurogamlss`
python3 mat2image.py --help                   # images of the maps; installed as `neurogamlss-mat2image`
```

No linter or formatter is configured. The code keeps lines to about 100 characters and must stay compatible with Python 3.9.

The tests use simulated data only. The simulators are `simulate_shash`, `simulate_gg`, `simulate_trajectories`, `simulate_brain` and `masked_data` in `tests/test_neurogamlss.py`. Command-line tests write mat-files with `scipy.io.savemat` to `tmp_path` and call `NG.main([...])`. `tests/test_mat2image.py` converts such outputs with `M2I.main([...])`.

## Architecture

The module is organized in sections, in this order.

1. **Data.** `Data` holds `Y` (subjects × features, float32), age, sex and site.
   - **Sites and models.** `load_data` joins files given with `+` as sites. Files separated by spaces on the command line are separate *models* of the same subjects, such as tissues, resolutions or surface measures.
   - **Names.** The file name up to the resolution names a model (`_model_key`). `mesh` in the name marks surface data.
   - **Ages and mask.** An age of 0 or NaN means unknown. `feature_mask` keeps only voxels or vertices whose training mean reaches a threshold.
2. **Normative model.** `NormativeModel` is one GAMLSS per feature, fitted for all features at once.
   - **Designs.** μ uses intercept, natural spline of age, sex, site dummies and covariates. log σ uses the same terms except the site dummies, which `site_sd` (`--site-sd`) adds. Covariates are the last columns of both designs (`cov_effects`).
   - **Splines.** `NaturalSpline` places its boundary knots at the quantiles `SPLINE_BOUNDARY` and is linear outside them.
   - **Normal family.** `fit_location_scale` fits it with the RS algorithm, ported from ComBatLS. Subjects with the same rows of both designs form cells (`_cells`); with at most `CELL_SHARE` cells per subject, the updates use the cell means and the sums of squares within cells, which gives the same fit faster. Continuous covariates or exact ages (e.g. in days) give nearly one cell per subject, and then the subjects are used directly.
   - **SHASH and GG.** They start from the normal fit and are refined by `fit_distribution`: penalized Newton steps with exact SHASH or numerical GG derivatives, shape priors and a bound on τ. Where the Hessian is indefinite, its eigenvalues are replaced by their absolute values (modified Newton, not BHHH). For constant shape, a step across the τ bound is the Newton step of the model with the penalty (`bound_step`). Steps are shortened by quadratic interpolation, and trial steps are evaluated with their derivatives, so an accepted step is not evaluated again. Intercept-only shape blocks pass their predictor as a row (1, p), which `shash_terms` and `gg_terms` broadcast.
3. **NormBrainAGE.** `NDMBrainAge` works on parts: the whole brain and, with `--parcellation`, each lobe of the atlas in `atlases/`.
   - **Pipeline per part.** Standardized voxels → PCA (`_pca`, via the Gram matrix) → normal `NormativeModel` of the scores → `ResidualCorrelation` R → log-likelihood on an age grid (`_loglik_grid`) → maximum with Laplace SE (`_argmax_refine`).
   - **`pca=0`.** It fits every voxel with a low-rank R. With the normal family, `VoxelModel.from_ndm` shares this model with the z-maps.
   - **`Warp`.** An optional sinh-arcsinh transform applied before the PCA.
   - **BrainLOSS.** With `brainloss=True` (`--brainloss`), `_fit_loss` fits a normal `NormativeModel` of the mean of every part. `_loss` evaluates it at the global brain age (BrainLOSS) and at the chronological age (BrainLOSS_age), and `apply_models` standardizes both with the controls.
4. **`VoxelModel`.** The z-maps: normal scores of voxel-wise models, SHASH by default.
5. **Choice of the df.** Spline df of μ and σ are chosen once per training sample by cross-validation and stored with the models. `select_df_normative` uses the held-out likelihood of normal models on 1,000 random voxels. `select_df_brainage` uses the MAE of the held-out brain age. Both search `DF_GRID_*` with `_search_df`.
6. **Diagnostics.** Q statistics and worm plots by age group (`run_diagnostics`).
7. **Persistence.** `ModelSet` holds `info`, `brainage` and `voxel` for one input. Every model class has `get_state()` and `from_state()`. `save_models` writes `.npz`, or `.mat` with struct `NDMmodel`, plus a JSON description. `load_models` checks `MODEL_FORMAT` and `MODEL_VERSION`.
8. **Drivers.** `fit_models`, `apply_models` (test data, `--adjust` controls, `--correction`, and `--test-nuisance` through `remove_nuisance`, which estimates nuisance effects such as IQMs in the controls and removes them from the test data), `cross_validate` (`--kfold`), `train_test` and `main`. `gpr_baseline` is a replica of the GPR BrainAGE of the BrainAGE toolbox, run for comparison.

**`mat2image.py`** imports `neurogamlss` and writes the maps of `NDMzmap`, `NDMdiag` and input mat-files as images.

- **Volume data.** The grid is that of `cat_io_data2mat.m`: `round([181 217 181]/r)` voxels of r mm, first voxel at (−90, −126, −72) mm (`volume_grid`). The columns of `Y` are the voxels of the brain mask in MATLAB (column-major) order. Older versions of `cat_io_data2mat.m` stored an empty `ind` for volume data; then the indices come from `atlases/ind_<r>mm.mat` or `--ind`. `check_compatible` compares `ind` only where the data store it.
- **Surface data.** `ind` and `dim` (number of mesh vertices) are stored in the z-maps and diagnostics. The GIFTI files hold all vertices, NaN elsewhere, and `--mesh` sets the metadata `SurfaceID` for SPM.

## Conventions

- **Vectorized fitting.** Features are fitted together with batched linear algebra, in chunks that run in threads (`_map`, `--jobs`). The chunks must not depend on the number of threads; tests assert identical results with 1 and 4 threads.
- **Defaults.** Defaults are module constants at the top of the file, such as `VOXEL_DEFAULTS`, `MASK_DEFAULT`, `TAU_MIN`, `SPLINE_BOUNDARY` and `DF_GRID_*`.
- **Saved models.** They must load and predict exactly as when they were fitted.
  - Everything decided at fit time is stored in the state: knots, PCA axes and centers, mask, df.
  - A new state key needs a default in `from_state` that reproduces the older behavior, for example `s.get('standardize', False)`.
  - Change `MODEL_VERSION` only for incompatible formats.
- **Brain-age options.** They flow from `main` through the `kw` dict into `NDMBrainAge(**kw)`, `select_df_brainage` and the JSON settings (`model_description`).
- **Evidence in the docs.** Defaults are justified by comparisons on real data in `docs/models.md`, as held-out log-likelihood in thousandths of a nat or MAE in years. When a default changes, rerun the affected numbers in `README.md`, `docs/models.md` and `NDM_methods.md`, or state which setting a table used.
- **Math in Markdown.** Use GitHub math: inline `` $`\mu`$ ``, display blocks in ` ```math ` fences, and `\lt`/`\gt` instead of `<`/`>` inside math.
- **`NDM_methods.md`.** It is manuscript text. Values in [brackets] are placeholders, and the HTML comment at the end holds author notes that are not part of the text.
- **`figures/NDMScheme.py`.** It draws `figures/NDMScheme.png` and does not import `neurogamlss`.
- **`figures/method_scheme.py`.** It writes the figures of `docs/NeuroGAMLSS Method Scheme.html` as standalone SVG files to `docs/figures/method_scheme_*.svg`, with embedded subsets of the fonts. The HTML page is their source: edit a figure there, then run the script again.
