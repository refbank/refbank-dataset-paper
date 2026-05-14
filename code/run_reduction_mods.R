library(brms)
library(here)
library(dplyr)
library(readr)
library(cmdstanr)
options(
  brms.backend = "cmdstanr",
  mc.cores = 4,
  brms.chains = 4,
  brms.iter = 2000,
  brms.threads = threading(4)
)

p_beta <- prior_string("normal(0,.5)", class = "b")
p_sd <- prior_string("normal(0,.5)", class = "sd")

p_intercept_logscale <- prior_string("normal(2,.5)", class = "Intercept")
p_intercept_linear <- prior_string("normal(10,10)", class = "Intercept")
p_beta_linear <- prior_string("normal(0,5)", class = "b")
p_sd_linear <- prior_string("normal(0,5)", class = "sd")

log_dv_priors <- c(p_intercept_logscale, p_beta, p_sd)
linear_dv_priors <- c(p_intercept_linear, p_beta_linear, p_sd_linear)

per_describer_for_model <- read_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds")) |>
  mutate(condition_id = as.factor(condition_id))

red_mod_log_log <- brm(log_words ~ log_rep_num + (log_rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_log.rds"),
  prior = log_dv_priors,
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1) |> slice_sample(n = 10000)
)

red_mod_log_lin <- brm(log_words ~ rep_num + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_lin.rds"),
  prior = log_dv_priors,
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

red_mod_lin_log <- brm(words ~ log_rep_num + (log_rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_lin_log.rds"),
  prior = linear_dv_priors,
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

red_mod_lin_lin <- brm(words ~ rep_num + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_lin_lin.rds"),
  prior = linear_dv_priors,
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)
