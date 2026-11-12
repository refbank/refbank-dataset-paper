# refbank-dataset-paper

Manuscript and analysis code for the Refbank dataset paper (`manuscript/refbank-dataset-paper.Rmd`).

## Layout

- `manuscript/` — the paper (`refbank-dataset-paper.Rmd`), bibliography (`RefBank.bib`), CSL style, and rendering output.
- `code/` — scripts to pull data from Redivis, fit the Bayesian models (on a SLURM cluster), and post-process the fits into what the manuscript reads.
- `cached_model_files/` — cached data and model outputs the manuscript reads directly via `read_rds()`. `data_for_mods/` and `mods/{summary,predicted,formulae}/` are committed; `mods/*.rds` (the raw fitted models) are gitignored because they're large.
- `code/tests/testthat/` — testthat tests for the helper functions in `code/mod_data_counts.R` and `code/re_fit_check.R` (a diagnostic that flags conditions whose fitted trajectories stray from their raw data; run over all fits with `code/run_re_fit_checks.R`). Not needed to reproduce the manuscript.
- `script/` — test runners: `script/test_fast` (unit tests only) and `script/test` (everything in `code/tests/testthat/`; the integration tests are skipped unless the gitignored fits in `cached_model_files/mods/*.rds` are present).

## Requirements

- R (>= 4.4) with: tidyverse, brms, cmdstanr, rstan, rstanarm, ggthemes, kableExtra, tidybayes, cowplot, viridis, papaja, loo.
- [`refbankr`](https://github.com/refbank/refbankr) (`remotes::install_github("refbank/refbankr")`) and the [`redivis`](https://github.com/redivis/redivis-r) R client it depends on.
- A free [Redivis](https://redivis.com) account — the first `refbankr`/`redivis` call in a fresh environment opens a browser to authenticate; after that it's cached.
- CmdStan — installed automatically by `code/install_deps.R`.
- A LaTeX distribution, to knit `papaja::apa6_pdf` (`manuscript/apa7.csl` and `manuscript/RefBank.bib` are already checked in).
- Access to a SLURM cluster for model fitting (scripts assume Stanford's Sherlock — partitions `owners,mcfrank` — but should be portable to any SLURM cluster with `module load R/4.4.2` available).

## Reproducing the manuscript

The fitted-model post-processing outputs (`cached_model_files/mods/{summary,predicted,formulae}/*.rds`) and the schema figure (`manuscript/figs/schema_v3_solarized.svg`) are committed, so **step 5 alone reproduces the PDF from what's already in the repo.** Step 5 still needs `refbankr` and a Redivis login: the manuscript pulls the dataset list and the corpus totals in Table 2 live from Redivis (Refbank `v13.4`). Steps 1–4 are only needed to refit everything from scratch (e.g. after a new Refbank data release or a model change).

### 1. Pull data from Redivis

```
Rscript code/get_data_for_mods.R
```

Writes `cached_model_files/data_for_mods/{condition_preds,per_matcher_for_model,per_describer_for_model,sims_for_model}.rds`. Pulls Refbank version `v13.4`, pinned at the top of the script — this must match the `version` set in the manuscript's data-loading chunk if you bump it.

### 2. Fit the models (SLURM)

One-time cluster setup (installs R packages + CmdStan, ~30–60 min):

```
sbatch code/install_deps.slurm
```

Submit every model-fitting job:

```
bash code/submit_mods
```

This submits all `code/run_*.slurm` jobs except `run_reduction_loo.slurm`. Once the four `run_reduction_mod_*` jobs have finished, run LOO model comparison separately — it depends on those four models already existing:

```
sbatch code/run_reduction_loo.slurm
```

Fitted models land in `cached_model_files/mods/*.rds`.

### 3. Post-process into what the manuscript reads

```
Rscript code/save_preds.R
```

Reads every fitted model and writes the summary statistics, model formulae, and predicted-value grids the Rmd actually loads, into `cached_model_files/mods/{summary,predicted,formulae}/`. It also writes `cached_model_files/mods/summary/model_data_counts.rds` (the number of datasets and trials in each model), using `code/mod_data_counts.R`.

```
Rscript code/acc_round_slope_at_12_options.R
```

Writes `cached_model_files/mods/summary/acc_round_slope_at_12_options.rds`: the accuracy model's round slope at 12 image options (the model's `log_round_num` coefficient is the slope at 0 options, because `option_set_size` is uncentred).


### 4. Regenerate the schema figure (only if the schema changed)

```
Rscript code/make_schema_diagram.R
```

Produces `manuscript/figs/schema_v3_solarized.svg` (Figure 1). Requires Graphviz via the DiagrammeR/DiagrammeRsvg R packages.

### 5. Knit the manuscript

```r
rmarkdown::render("manuscript/refbank-dataset-paper.Rmd")
```

(or open it in RStudio and click Knit). Produces `manuscript/refbank-dataset-paper.pdf`.
