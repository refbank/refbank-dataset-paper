library(brms)
library(ordbetareg)
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
# Priors now on logit scale (ordbeta uses logit link)
# logit(0.72) ≈ 0.94, so intercept around 1
p_beta_sim <- prior_string("normal(0, 0.5)", class = "b")
p_sd_sim <- prior_string("normal(0, 0.5)", class = "sd")
p_intercept_sim <- prior_string("normal(1, 1.5)", class = "Intercept")

sim_priors <- c(p_intercept_sim, p_beta_sim, p_sd_sim)

sims_for_model <- read_rds(here("cached_model_files/data_for_mods/sims_for_model.rds")) |>
  filter(stage_num == 1) |>
  mutate(log_round_num = log(round_num))

to_next_words_data <- sims_for_model |>
  filter(sim_type == "to_next", !is.na(log_words_earlier), !is.na(log_words_later))

# pool earlier+later so they share one mean/SD -- keeps a real earlier-vs-later
# difference visible instead of erasing it via independent centering
pooled_log_words <- c(to_next_words_data$log_words_earlier, to_next_words_data$log_words_later)
pooled_mean <- mean(pooled_log_words)
pooled_sd <- sd(pooled_log_words)

to_next_words_data <- to_next_words_data |>
  mutate(
    log_words_earlier_z = (log_words_earlier - pooled_mean) / pooled_sd,
    log_words_later_z = (log_words_later - pooled_mean) / pooled_sd
  )

stopifnot(
  "log_words_earlier_z/log_words_later_z have non-finite values" =
    all(is.finite(to_next_words_data$log_words_earlier_z), is.finite(to_next_words_data$log_words_later_z))
)

to_next_words_mod <- ordbetareg(
  sim ~ log_round_num * log_words_earlier_z * log_words_later_z +
    (log_round_num || dataset_id / condition_id),
  manual_prior = sim_priors,
  file = here("cached_model_files/mods/to_next_words_mod.rds"),
  file_refit = "on_change",
  data = to_next_words_data
)
