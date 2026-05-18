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
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1) |> slice_sample(n = 10000)
)

red_mod_log_lin <- brm(log_words ~ rep_num + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_lin.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

red_mod_lin_log <- brm(words ~ log_rep_num + (log_rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_lin_log.rds"),
  file_refit = "on_change",
  prior = linear_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

red_mod_lin_lin <- brm(words ~ rep_num + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_lin_lin.rds"),
  file_refit = "on_change",
  prior = linear_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

# LOO computation -------------------------------------------------------------------

# Standard LOO for same-DV comparisons (log_log vs log_lin; lin_log vs lin_lin).
# Uses moment matching to handle high Pareto-k values common in hierarchical models.
compute_and_cache_loo <- function(mod, path) {
  if (file.exists(path)) return(readRDS(path))
  result <- loo(mod, moment_match = TRUE)
  saveRDS(result, path)
  result
}

# Jacobian-corrected LOO for cross-DV comparison (all four models on original word-count scale).
# For log-DV models: log p(y) = log p(log y) - log(y), so subtract log_words from each
# observation's pointwise log-likelihood.
compute_and_cache_corrected_loo <- function(mod, path) {
  if (file.exists(path)) return(readRDS(path))
  ll <- log_lik(mod)
  nchains <- brms::nchains(mod)
  r_eff <- loo::relative_eff(exp(ll), chain_id = rep(seq_len(nchains), each = nrow(ll) / nchains))
  ll_corrected <- sweep(ll, 2, mod$data$log_words, "-")
  result <- loo::loo(ll_corrected, r_eff = r_eff)
  saveRDS(result, path)
  result
}

loo_log_log <- compute_and_cache_loo(red_mod_log_log, here("cached_model_files/mods/loo_log_log.rds"))
loo_log_lin <- compute_and_cache_loo(red_mod_log_lin, here("cached_model_files/mods/loo_log_lin.rds"))
loo_lin_log <- compute_and_cache_loo(red_mod_lin_log, here("cached_model_files/mods/loo_lin_log.rds"))
loo_lin_lin <- compute_and_cache_loo(red_mod_lin_lin, here("cached_model_files/mods/loo_lin_lin.rds"))

# Corrected LOO only needed for log-DV models; linear-DV models are already on original scale
loo_log_log_orig_scale <- compute_and_cache_corrected_loo(red_mod_log_log, here("cached_model_files/mods/loo_log_log_orig_scale.rds"))
loo_log_lin_orig_scale <- compute_and_cache_corrected_loo(red_mod_log_lin, here("cached_model_files/mods/loo_log_lin_orig_scale.rds"))
