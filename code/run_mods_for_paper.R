library(brms)
library(here)
library(dplyr)
library(readr)
library(cmdstanr)
options(
  brms.backend = "cmdstanr",
  mc.cores = 1, # parallel::detectCores(),
  brms.cores = 1, # parallel::detectCores(),
  brms.iter = 200,
  brms.chains = 1,
  brms.threads = threading(1)
  # brms.control = list(adapt_delta = .95)
)

#### Priors

p_beta <- prior_string("normal(0,.5)", class = "b")
p_sd <- prior_string("normal(0,.5)", class = "sd")
# p_corr <- prior_string("lkj(1)", class = "cor")

p_intercept_pos <- prior_string("normal(0, 1.5)", class = "Intercept")
p_intercept_sim <- prior_string("normal(0.5,.2)", class = "Intercept")
p_beta_sim <- prior_string("normal(0,.1)", class = "b")
p_sd_sim <- prior_string("normal(0,.1)", class = "sd")


p_intercept_logscale <- prior_string("normal(2,.5)", class = "Intercept")
p_intercept_linear <- prior_string("normal(10,10)", class = "Intercept")
p_beta_linear <- prior_string("normal(0,5)", class = "b")
p_sd_linear <- prior_string("normal(0,5)", class = "sd")

logistic_priors <- c(p_beta, p_sd)
logistic_pos_priors <- c(p_beta, p_sd, p_intercept_pos)
log_dv_priors <- c(p_intercept_logscale, p_beta, p_sd)

linear_dv_priors <- c(p_intercept_linear, p_beta_linear, p_sd_linear)

sim_priors <- c(p_intercept_sim, p_beta_sim, p_sd_sim)


#### Describer model analyses

per_describer_for_model <- read_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds")) |>
  mutate(condition_id = as.factor(condition_id))


### Functional form considerations

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

### Moderators

# use whatever functional form was best above -- doing with log-lin for writing

# TODO add back age group
# red_mod_participants <- brm(log_words ~ rep_num + n_players + (rep_num | dataset_id / condition_id),
#   file = here("cached_model_files/mods/red_mod_participants.rds"),
#   prior = log_dv_priors,
#   data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
# )


# red_mod_modality <- brm(log_words~ rep_num+modality+feedback+backchannel+role_constancy + (rep_num | dataset_id / condition_id ),
#                    file=here("cached_model_files/mods/red_mod_modality.rds"),
#                    prior=log_dv_priors,
#                    data=per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num==1))


# red_mod_stim <- brm(log_words~ rep_num+option_size*image_type + (rep_num | dataset_id / condition_id ),
#                    file=here("cached_model_files/mods/red_mod_stim.rds"),
#                    prior=log_dv_priors,
#                    data=per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num==1))

#
# red_mod_stage <- brm(log_words~ rep_num*stage_num + (rep_num | dataset_id / condition_id ),
#                    file=here("cached_model_files/mods/red_mod_stage.rds"),
#                    prior=log_dv_priors,
#                    data=per_describer_for_model |> filter(!is.na(log_words)) )


### Pos/etc

#
# # TODO add back modality
pos_mod <- brm(
  cbind(NOUN, VERB, MODIFIER, FUNCTION, DET, PRON) ~ rep_num +
    (rep_num || dataset_id / condition_id),
  family = multinomial(),
  file = here("cached_model_files/mods/pos_mod.rds"),
  prior = logistic_pos_priors,
  data = per_describer_for_model |> filter(stage_num == 1)
)


### SBERT

sims_for_model <- read_rds(here("cached_model_files/data_for_mods/sims_for_model.rds")) |>
  mutate(condition_id = as.factor(condition_id)) |>
  filter(stage_num == 1)

to_next_mod <- brm(sim ~ rep_num + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/to_next_mod.rds"),
  prior = sim_priors,
  data = sims_for_model |> filter(sim_type == "to_next")
)

diverge_mod <- brm(sim ~ rep_num + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/diverge_mod.rds"),
  prior = sim_priors,
  data = sims_for_model |> filter(sim_type == "diverge")
)


# ### Matcher model analyses
#
# per_matcher_for_model <- read_rds(here("cached_model_files/data_for_mods/per_matcher_for_model.rds")) |>
#   mutate(condition_id = as.factor(condition_id)) |>
#   filter(stage_num == 1)
#
# accuracy_mod <- brm(correct ~ rep_num + (rep_num || dataset_id / condition_id),
#   family = bernoulli(link = "logit"),
#   file = here("cached_model_files/mods/accuracy_mod.rds"),
#   prior = logistic_priors,
#   data = per_matcher_for_model |> filter(!is.na(correct))
# )
#
# rt_mod <- brm(log_rt ~ rep_num + (rep_num || dataset_id / condition_id),
#   file = here("cached_model_files/mods/rt_mod.rds"),
#   prior = log_dv_priors,
#   data = per_matcher_for_model |> filter(!is.na(log_rt))
# )
#
# rt_mod_log_rep <- brm(log_rt ~ log_rep_num + (log_rep_num || dataset_id / condition_id),
#   file = here("cached_model_files/mods/rt_mod_log_rep.rds"),
#   prior = log_dv_priors,
#   data = per_matcher_for_model |> filter(!is.na(log_rt))
# )
