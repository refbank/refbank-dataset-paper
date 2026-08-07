library(dplyr)
library(tibble)
library(brms)
library(stringr)

# Compares each group's raw observed mean to its in-sample fitted mean, to
# catch cases where a hierarchical model's random effects pull a group's
# prediction away from its own data (e.g. sparse groups with outlier
# observations getting an overconfident/extreme random slope).
group_fit_discrepancy <- function(data, response, fitted, groups, outlier_threshold = 3) {
  by_group <- data |>
    group_by(across(all_of(groups))) |>
    summarize(
      n = n(),
      raw_mean = mean(.data[[response]]),
      fitted_mean = mean(.data[[fitted]]),
      .groups = "drop"
    ) |>
    mutate(
      discrepancy = fitted_mean - raw_mean,
      abs_discrepancy = abs(discrepancy)
    )

  center <- median(by_group$abs_discrepancy)
  spread <- mad(by_group$abs_discrepancy)

  by_group |>
    mutate(
      is_outlier = if (spread == 0) FALSE else (abs_discrepancy - center) / spread > outlier_threshold
    ) |>
    arrange(desc(abs_discrepancy))
}

# Auto-detects the response variable from a brmsfit's formula (first LHS
# variable) unless response_var is given, and validates it's a real column in
# mod$data. Shared by check_re_fit() and check_re_fit_from_predictions() --
# split out so this decision logic is unit testable without a real fitted
# posterior (see test-re_fit_check.R's fake_brmsfit()).
resolve_response_var <- function(mod, response_var) {
  stopifnot("mod must be a brmsfit object" = inherits(mod, "brmsfit"))
  if (is.null(response_var)) {
    response_var <- all.vars(mod$formula$formula[[2]])[1]
  }
  stopifnot(
    "could not find response variable in mod$data" = response_var %in% names(mod$data)
  )
  response_var
}

# Normalizes group_vars into a named list of column-name vectors, one entry
# per discrepancy table to produce. An explicit scalar vector becomes a
# single entry keyed by its colon-joined name; an explicit list is used
# as-is. Left NULL, auto-detects one entry per random-effects factor from
# ranef(mod) (colon-separated for nested/interaction terms like
# dataset_id:condition_id) -- this branch needs a real fitted model and is
# covered by the integration tests, not unit tests.
resolve_group_vars <- function(mod, group_vars) {
  if (is.null(group_vars)) {
    factor_names <- names(ranef(mod))
    group_vars <- factor_names |> purrr::map(\(nm) str_split(nm, ":")[[1]])
    names(group_vars) <- factor_names
  } else if (!is.list(group_vars)) {
    group_vars <- list(group_vars)
    names(group_vars) <- paste(group_vars[[1]], collapse = ":")
  }
  group_vars
}

# Runs group_fit_discrepancy() against a fitted brms model: computes in-sample
# fitted values with full random effects (re_formula = NULL) and compares them
# to mod$data, one discrepancy table per grouping factor in the model's random
# effects structure (auto-detected from ranef(mod) unless group_vars given).
check_re_fit <- function(mod, group_vars = NULL, response_var = NULL, outlier_threshold = 3, ndraws = NULL) {
  response_var <- resolve_response_var(mod, response_var)
  group_vars <- resolve_group_vars(mod, group_vars)

  fitted_vals <- fitted(mod, newdata = mod$data, re_formula = NULL, summary = TRUE, ndraws = ndraws)[, "Estimate"]
  aug <- mod$data |> mutate(.fitted = fitted_vals)

  group_vars |>
    purrr::map(\(cols) {
      stopifnot(
        "grouping column missing from mod$data" = all(cols %in% names(aug))
      )
      group_fit_discrepancy(aug, response = response_var, fitted = ".fitted", groups = cols, outlier_threshold = outlier_threshold)
    })
}

# Same purpose as check_re_fit(), but instead of calling fitted() with
# re_formula = NULL (which builds a draws x rows matrix that OOM-kills on
# the larger cached models), joins a pre-computed prediction grid -- as saved
# by code/save_preds.R under cached_model_files/mods/predicted/ -- onto the
# raw data. The grid must have a `mean` column plus every other column it
# varies over (dataset_id, condition_id, round_num, and possibly moderators
# like option_set_size); the join uses all of those so rows aren't matched
# to the wrong grid cell.
group_fit_discrepancy_from_predictions <- function(data, response, predicted, outlier_threshold = 3) {
  stopifnot("predicted grid must have a mean column" = "mean" %in% names(predicted))

  # the grid may also include a marginal/overall row (dataset_id NA, used for
  # the blue "generic" line in manuscript plots) -- irrelevant here since raw
  # data rows always belong to a real dataset_id/condition_id.
  predicted_by_group <- predicted |> filter(!is.na(dataset_id), !is.na(condition_id))

  # brm() only keeps formula variables in mod$data, so a grid column like
  # round_num (derived for plotting in save_preds.R) may not exist there even
  # though log_round_num does -- join on whatever the two share instead of
  # requiring every grid column to be present in data. Some models'
  # condition-level rows also don't vary a predictor that the raw data does
  # (e.g. save_preds.R's acc_mod_log_rep grid marginalizes over
  # option_set_size per condition, leaving it structurally NA there even
  # though it's a real column) -- such a column would fail to match any row
  # if included in the join, so drop it instead of joining on it.
  candidate_cols <- intersect(setdiff(names(predicted_by_group), c("mean", "low", "high")), names(data))
  join_cols <- candidate_cols[sapply(candidate_cols, \(col) !all(is.na(predicted_by_group[[col]])))]
  stopifnot(
    "predicted grid and data share no round-number column to join on" =
      any(c("round_num", "log_round_num") %in% join_cols)
  )

  aug <- data |>
    left_join(
      predicted_by_group |> select(all_of(join_cols), .fitted = mean),
      by = join_cols
    )

  stopifnot(
    "predicted grid does not cover every row of data -- was it built from a different model fit?" =
      all(!is.na(aug$.fitted))
  )

  group_fit_discrepancy(aug, response = response, fitted = ".fitted", groups = c("dataset_id", "condition_id"), outlier_threshold = outlier_threshold)
}

# brmsfit-object wrapper around group_fit_discrepancy_from_predictions(),
# mirroring check_re_fit()'s auto-detection of response_var.
check_re_fit_from_predictions <- function(mod, predicted, response_var = NULL, outlier_threshold = 3) {
  response_var <- resolve_response_var(mod, response_var)
  group_fit_discrepancy_from_predictions(mod$data, response_var, predicted, outlier_threshold)
}

# Runs check_fn (default check_re_fit) against every model file found
# directly under mods_dir (does not recurse into subdirectories like
# predicted/ or summary/), keyed by filename without extension. read_fn and
# check_fn are overridable so the discovery/dispatch logic can be unit
# tested without loading real brms models.
run_re_fit_checks_all <- function(mods_dir, ..., check_fn = check_re_fit, read_fn = readRDS) {
  model_files <- list.files(mods_dir, pattern = "\\.rds$", full.names = TRUE)
  stopifnot("no .rds model files found in mods_dir" = length(model_files) > 0)

  names(model_files) <- tools::file_path_sans_ext(basename(model_files))

  model_files |> purrr::map(\(path) check_fn(read_fn(path), ...))
}
