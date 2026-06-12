library(tidyverse)
library(brms)
library(rstan)
library(tidybayes)
library(here)
mod_loc <- "cached_model_files/mods"


do_preds <- function(mod) {
  is_multinomial <- isTRUE(mod$family$family == "multinomial")

  draws_overall <- mod$data |>
    distinct(across(any_of(c("rep_num", "log_rep_num", ".category", "option_size", "total")))) |>
    add_epred_draws(mod, re_formula = NA, ndraws = 1000)

  if (is_multinomial) {
    draws_overall <- draws_overall |> mutate(.epred = .epred
    / total)
  }
  preds_overall <- draws_overall |>
    group_by(across(any_of(c("rep_num", "log_rep_num", ".category")))) |>
    summarize(
      mean = mean(.epred),
      low = quantile(.epred, .025),
      high = quantile(.epred, .975)
    )
  draws_condition <- mod$data |>
    distinct(across(any_of(c("rep_num", "log_rep_num", "condition_id", "dataset_id", ".category", "option_size", "total")))) |>
    add_epred_draws(mod, ndraws = 1000)

  if (is_multinomial) {
    draws_condition <- draws_condition |> mutate(
      .epred =
        .epred / total
    )
  }

  preds_condition <- draws_condition |>
    group_by(across(any_of(c("rep_num", "log_rep_num", "condition_id", "dataset_id", ".category")))) |>
    summarize(
      mean = mean(.epred),
      low = quantile(.epred, .025),
      high = quantile(.epred, .975)
    )

  preds <- preds_condition |> bind_rows(preds_overall)

  if ("log_rep_num" %in% names(preds) && !"rep_num" %in% names(preds)) {
    preds <- preds |> mutate(rep_num = exp(log_rep_num))
  }
  return(preds)
}

save_summary <- function(model) {
  intervals <- gather_draws(model, `b_.*`, regex = T) %>% mean_qi()

  stats <- gather_draws(model, `b_.*`, regex = T) %>%
    mutate(above_0 = ifelse(.value > 0, 1, 0)) %>%
    group_by(.variable) %>%
    summarize(pct_above_0 = mean(above_0)) %>%
    left_join(intervals, by = ".variable") %>%
    mutate(
      lower = .lower,
      upper = .upper,
      Term = str_sub(.variable, 3, -1),
      Estimate = .value
    ) %>%
    select(Term, Estimate, lower, upper)

  stats
}

do_model <- function(path) {
  model <- read_rds(here(mod_loc, path))
  save_summary(model) |> write_rds(here(mod_loc, "summary", path))
  model$formula |> write_rds(here(mod_loc, "formulae", path))
  if (!str_detect(path, "pred")) {
    do_preds(model) |> write_rds(here(mod_loc, "predicted", path))
  }
  gc()
}


mods <- list.files(path = here(mod_loc), pattern = ".*rds") |>
  discard(\(x) str_detect(x, "loo")) |>
  discard(\(x) str_detect(x, "red_mod_lin")) |>
  walk(~ do_model(.))
