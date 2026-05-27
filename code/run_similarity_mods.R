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
# Priors now on logit scale (ordbeta uses logit link)
# logit(0.72) ≈ 0.94, so intercept around 1
p_beta_sim <- prior_string("normal(0, 0.5)", class = "b")
p_sd_sim <- prior_string("normal(0, 0.5)", class = "sd")
p_intercept_sim <- prior_string("normal(1, 1.5)", class = "Intercept")

sim_priors <- c(p_intercept_sim, p_beta_sim, p_sd_sim)

to_next_mod <- brm(sim ~ rep_num + (rep_num || dataset_id / condition_id),
  family = ordbeta(), # <-- this line
  file = here("cached_model_files/mods/to_next_mod.rds"),
  prior = sim_priors,
  data = sims_for_model |> filter(sim_type == "to_next")
)

diverge_mod <- brm(sim ~ rep_num + (rep_num || dataset_id / condition_id),
  family = ordbeta(),
  file = here("cached_model_files/mods/diverge_mod.rds"),
  prior = sim_priors,
  data = sims_for_model |> filter(sim_type == "diverge")
)
