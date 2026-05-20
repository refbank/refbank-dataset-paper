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
  mutate(condition_id = as.factor(condition_id)) |>
  left_join(read_rds(here("cached_model_files/data_for_mods/condition_preds.rds")))




red_mod_log_log_participants <- brm(
  log_words ~ log_rep_num *
    # population*
    n_players + (log_rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_log_participants.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

red_mod_log_lin_participants <- brm(
  log_words ~ rep_num *
    # population*
    n_players + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_lin_participants.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

# red_mod_log_log_images <- brm(log_words ~ log_rep_num*(option_size+image_type) + (log_rep_num || dataset_id / condition_id),
#   file = here("cached_model_files/mods/red_mod_log_log_images.rds"),
#   file_refit = "on_change",
#   prior = log_dv_priors,
#   save_pars = save_pars(all = TRUE),
#   data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
# )
#
# red_mod_log_lin_images <- brm(log_words ~ rep_num*(option_size+image_type) + (rep_num || dataset_id / condition_id),
#   file = here("cached_model_files/mods/red_mod_log_lin_images.rds"),
#   file_refit = "on_change",
#   prior = log_dv_priors,
#   save_pars = save_pars(all = TRUE),
#   data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
# )

red_mod_log_log_channel <- brm(
  log_words ~ log_rep_num * (role_constancy +
    # modality+
    feedback + backchannel) + (log_rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_log_channel.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)

red_mod_log_lin_channel <- brm(
  log_words ~ rep_num * (role_constancy +
    # modality+
    feedback + backchannel) + (rep_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_lin_channel.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)
