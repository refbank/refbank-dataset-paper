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
p_intercept_pos <- prior_string("normal(0, 1.5)", class = "Intercept")

logistic_pos_priors <- c(p_beta, p_sd, p_intercept_pos)

per_describer_for_model <- read_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds")) |>
  mutate(condition_id = as.factor(condition_id))

pos_mod <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) ~ rep_num +
    (rep_num || dataset_id / condition_id),
  family = multinomial(),
  file = here("cached_model_files/mods/pos_mod.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)
