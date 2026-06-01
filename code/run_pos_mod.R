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
  brms.threads = threading(4)
)

p_beta_pos <- prior_string("normal(0,1.5)", class = "b", dpar = c("muDET", "muFUNCTION", "muMODIFIER", "muNOUN", "muVERB"))
p_sd_pos <- prior_string("normal(0,1.5)", class = "sd", dpar = c("muDET", "muFUNCTION", "muMODIFIER", "muNOUN", "muVERB"))
p_intercept_pos <- prior_string("normal(0, 1.5)", class = "Intercept", dpar = c("muDET", "muFUNCTION", "muMODIFIER", "muNOUN", "muVERB"))

logistic_pos_priors <- c(p_beta_pos, p_sd_pos, p_intercept_pos)

per_describer_for_model <- read_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds")) |>
  mutate(
    condition_id = as.factor(condition_id),
    total = NOUN + VERB + MODIFIER + FUNCTION + DET + PRON
  )

pos_mod <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ rep_num +
    (rep_num || dataset_id / condition_id),
  family = multinomial(refcat = "PRON"),
  file = here("cached_model_files/mods/pos_mod.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)

pos_mod_log <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ log_rep_num +
    (log_rep_num || dataset_id / condition_id),
  family = multinomial(refcat = "PRON"),
  file = here("cached_model_files/mods/pos_log_mod.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)

pos_mod_inv <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ inv_rep_num +
    (inv_rep_num || dataset_id / condition_id),
  family = multinomial(refcat = "PRON"),
  file = here("cached_model_files/mods/pos_log_inv.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1) |> mutate(inv_rep_num = -1 / rep_num)
)

# pos_mod_take_2 <- brm(
#   cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ rep_num +
#     (rep_num || condition_id),
#   family = multinomial(refcat="PRON"),
#   file = here("cached_model_files/mods/pos_mod_2.rds"),
#   prior = logistic_pos_priors,
#   data = per_describer_for_model |> filter(stage_num == 1)
# )
#
# pos_mod_log_take_2 <- brm(
#   cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ log_rep_num +
#     (log_rep_num || condition_id),
#   family = multinomial(refcat="PRON"),
#   file = here("cached_model_files/mods/pos_log_mod_2.rds"),
#   prior = logistic_pos_priors,
#   data = per_describer_for_model |> filter(stage_num == 1)
# )
#
# pos_mod_inv_take_2 <- brm(
#   cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ inv_rep_num +
#     (inv_rep_num || condition_id),
#   family = multinomial(refcat="PRON"),
#   file = here("cached_model_files/mods/pos_log_inv_2.rds"),
#   prior = logistic_pos_priors,
#   data = per_describer_for_model |> filter(stage_num == 1) |> mutate(inv_rep_num = -1 / rep_num)
# )

pos_mod_factor <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) | trials(total) ~ as.factor(rep_num) +
    (as.factor(rep_num) || condition_id),
  family = multinomial(refcat = "PRON"),
  file = here("cached_model_files/mods/pos_mod_factor.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)

p_beta_ordbeta <- prior_string("normal(0, 1.5)", class = "b")
p_sd_ordbeta <- prior_string("normal(0, 1.5)", class = "sd")
p_intercept_ordbeta <- prior_string("normal(0, 1.5)", class = "Intercept")
ordbeta_priors <- c(p_intercept_ordbeta, p_beta_ordbeta, p_sd_ordbeta)

pos_data <- per_describer_for_model |>
  filter(stage_num == 1, !is.na(total), total != 0) |>
  mutate(
    prop_NOUN     = NOUN / total,
    prop_VERB     = VERB / total,
    prop_MODIFIER = MODIFIER / total,
    prop_FUNCTION = FUNCTION / total,
    prop_DET      = DET / total,
    prop_PRON     = PRON / total
  )

ordbeta_NOUN <- ordbetareg(
  prop_NOUN ~ log_rep_num + (log_rep_num || condition_id),
  manual_prior = ordbeta_priors,
  file = here("cached_model_files/mods/pos_ordbeta_NOUN.rds"),
  data = pos_data
)

ordbeta_VERB <- ordbetareg(
  prop_VERB ~ log_rep_num + (log_rep_num || condition_id),
  manual_prior = ordbeta_priors,
  file = here("cached_model_files/mods/pos_ordbeta_VERB.rds"),
  data = pos_data
)

ordbeta_MODIFIER <- ordbetareg(
  prop_MODIFIER ~ log_rep_num + (log_rep_num || condition_id),
  manual_prior = ordbeta_priors,
  file = here("cached_model_files/mods/pos_ordbeta_MODIFIER.rds"),
  data = pos_data
)

ordbeta_FUNCTION <- ordbetareg(
  prop_FUNCTION ~ log_rep_num + (log_rep_num || condition_id),
  manual_prior = ordbeta_priors,
  file = here("cached_model_files/mods/pos_ordbeta_FUNCTION.rds"),
  data = pos_data
)

ordbeta_DET <- ordbetareg(
  prop_DET ~ log_rep_num + (log_rep_num || condition_id),
  manual_prior = ordbeta_priors,
  file = here("cached_model_files/mods/pos_ordbeta_DET.rds"),
  data = pos_data
)

ordbeta_PRON <- ordbetareg(
  prop_PRON ~ log_rep_num + (log_rep_num || condition_id),
  manual_prior = ordbeta_priors,
  file = here("cached_model_files/mods/pos_ordbeta_PRON.rds"),
  data = pos_data
)
