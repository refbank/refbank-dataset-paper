library(brms)
library(loo)
library(here)
library(dplyr)
library(readr)
library(cmdstanr)
local({
  versions <- sort(list.files(path.expand("~/.cmdstan"), pattern = "^cmdstan-", full.names = TRUE))
  if (length(versions) == 0) stop("No CmdStan installation found in ~/.cmdstan")
  set_cmdstan_path(versions[length(versions)])
})
options(
  brms.backend = "cmdstanr",
  mc.cores = 4,
  brms.chains = 4,
  brms.iter = 2000,
  brms.threads = threading(6)
)

p_beta <- prior_string("normal(0,.5)", class = "b")
p_sd <- prior_string("normal(0,.5)", class = "sd")
p_intercept_logscale <- prior_string("normal(2,.5)", class = "Intercept")

log_dv_priors <- c(p_intercept_logscale, p_beta, p_sd)

per_describer_for_model <- read_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds")) |>
  mutate(condition_id = as.factor(condition_id))

red_mod_log_log <- brm(log_words ~ log_round_num + (log_round_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_log.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  control = list(max_treedepth = 15),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)
