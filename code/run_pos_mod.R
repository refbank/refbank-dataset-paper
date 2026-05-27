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

p_beta_pos <- prior_string("normal(0,.5)", class = "b", dpar = c("muDET", "muFUNCTION", "muMODIFIER", "muPRON", "muVERB"))
p_sd_pos <- prior_string("normal(0,.5)", class = "sd", dpar = c("muDET", "muFUNCTION", "muMODIFIER", "muPRON", "muVERB"))
p_intercept_pos <- prior_string("normal(0, 1.5)", class = "Intercept", dpar = c("muDET", "muFUNCTION", "muMODIFIER", "muPRON", "muVERB"))

logistic_pos_priors <- c(p_beta_pos, p_sd_pos, p_intercept_pos)

per_describer_for_model <- read_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds")) |>
  mutate(
    condition_id = as.factor(condition_id),
    total = NOUN + VERB + MODIFIER + FUNCTION + DET + PRON
  )

pos_mod <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ rep_num +
    (rep_num || dataset_id / condition_id),
  family = multinomial(),
  file = here("cached_model_files/mods/pos_mod.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)

pos_mod_log <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ log_rep_num +
    (log_rep_num || dataset_id / condition_id),
  family = multinomial(),
  file = here("cached_model_files/mods/pos_log_mod.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)

pos_mod_inv <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ inv_rep_num +
    (inv_rep_num || dataset_id / condition_id),
  family = multinomial(),
  file = here("cached_model_files/mods/pos_log_inv.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1) |> mutate(inv_rep_num = -1 / rep_num)
)
