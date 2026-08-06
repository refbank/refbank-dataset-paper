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

# Runs group_fit_discrepancy() against a fitted brms model: computes in-sample
# fitted values with full random effects (re_formula = NULL) and compares them
# to mod$data, one discrepancy table per grouping factor in the model's random
# effects structure (auto-detected from ranef(mod) unless group_vars given).
check_re_fit <- function(mod, group_vars = NULL, response_var = NULL, outlier_threshold = 3, ndraws = NULL) {
  stopifnot("mod must be a brmsfit object" = inherits(mod, "brmsfit"))

  if (is.null(response_var)) {
    response_var <- all.vars(mod$formula$formula[[2]])[1]
  }
  stopifnot(
    "could not find response variable in mod$data" = response_var %in% names(mod$data)
  )

  if (is.null(group_vars)) {
    factor_names <- names(ranef(mod))
    group_vars <- factor_names |> purrr::map(\(nm) str_split(nm, ":")[[1]])
    names(group_vars) <- factor_names
  } else if (!is.list(group_vars)) {
    group_vars <- list(group_vars)
    names(group_vars) <- paste(group_vars[[1]], collapse = ":")
  }

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
