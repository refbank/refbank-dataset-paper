library(brms)
library(here)
library(dplyr)
library(readr)
library(cmdstanr)
options(
  brms.backend = "cmdstanr",
  mc.cores = 4, # parallel::detectCores(),
  brms.cores = 4, # parallel::detectCores(),
  brms.iter = 4000,
  brms.chains = 4,
  control = list(adapt_delta = 0.99)
)
# take the fitted reduction models
# and create data for a slope predicting secondary model
mod_loc <- "cached_model_files/mods"

condition_info <- read_rds(here("cached_model_files/data_for_mods/condition_preds.rds")) |>
  mutate(image_type = factor(image_type, levels = c("tangram", "fribble", "photograph", "line drawing")), )

log_log_mod <- read_rds(here(mod_loc, "red_mod_log_log.rds"))
log_lin_mod <- read_rds(here(mod_loc, "red_mod_log_lin.rds"))

log_log_preds <- coef(log_log_mod)$`dataset_id:condition_id` |>
  as_tibble(rownames = "dataset_cond") |>
  select(dataset_cond,
    slope = Estimate.log_rep_num,
    se = Est.Error.log_rep_num
  ) |>
  left_join(condition_info)

log_lin_preds <- coef(log_lin_mod)$`dataset_id:condition_id` |>
  as_tibble(rownames = "dataset_cond") |>
  select(dataset_cond, slope = Estimate.rep_num, se = Est.Error.rep_num) |>
  left_join(condition_info)


p_beta_linear <- c(
  prior_string("normal(0, .2)", class = "b"),
  prior_string("normal(0, .2)", class = "Intercept"),
  prior_string("normal(0, .2)", class = "sigma", lb = 0)
)

log_lin_pred_mod <- brm(
  slope | mi(se) ~ n_players +
    option_size +
    image_type +
    partner_constancy +
    role_constancy +
    population +
    modality +
    feedback +
    backchannel,
  file = here("cached_model_files/mods/log_lin_pred.rds"),
  prior = c(p_beta_linear),
  data = log_lin_preds
)

log_log_pred_mod <- brm(
  slope | mi(se) ~ n_players +
    option_size +
    image_type +
    partner_constancy +
    role_constancy +
    population +
    modality +
    feedback +
    backchannel,
  file = here("cached_model_files/mods/log_log_pred.rds"),
  prior = c(p_beta_linear),
  data = log_log_preds
)
