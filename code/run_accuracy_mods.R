library(brms)
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

p_beta_acc <- prior_string("normal(0, .5)", class = "b")
p_sd_acc <- prior_string("normal(0, .5)", class = "sd")
p_intercept_acc <- prior_string("normal(1,1)", class = "Intercept")

acc_priors <- c(p_intercept_acc, p_beta_acc, p_sd_acc)

acc_for_model <- read_rds(here("cached_model_files/data_for_mods/per_matcher_for_model.rds"))

acc_mod <- brm(correct ~ log_rep_num * option_size + (log_rep_num || dataset_id / condition_id),
  family = bernoulli(),
  prior = acc_priors,
  file = here("cached_model_files/mods/acc_mod_log_rep.rds"),
  data = acc_for_model |> filter(stage_num == 1)
)
