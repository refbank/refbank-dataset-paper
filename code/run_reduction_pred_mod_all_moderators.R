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
  left_join(
    read_rds(here("cached_model_files/data_for_mods/condition_preds.rds")) |>
      select(dataset_id, condition_id, population, partner_constancy),
    by = c("dataset_id", "condition_id")
  ) |>
  mutate(
    log_n_players = log(n_players),
    log_option_set_size = log(option_set_size),
    image_type = factor(image_type, levels = c("tangram", "fribble", "photograph", "line drawing")),
    backchannel_bin = factor(if_else(backchannel == "full", "full", "restricted"), levels = c("full", "restricted")),
    feedback_bin = factor(if_else(feedback == "full", "full", "restricted"), levels = c("full", "restricted")),
    modality_bin = factor(if_else(modality == "written", "written", "oral"), levels = c("written", "oral")),
    population_bin = factor(if_else(population == "adult", "adult", "child"), levels = c("adult", "child"))
  )

stopifnot(
  "moderator recoding introduced NAs in the modeling sample" =
    per_describer_for_model |>
      filter(stage_num == 1, !is.na(log_words)) |>
      summarise(across(
        c(backchannel_bin, feedback_bin, modality_bin, population_bin, image_type),
        \(x) all(!is.na(x))
      )) |>
      unlist() |>
      all()
)

# we care about each moderator's interaction with round_num (does it change the
# rate of reduction), not its intercept (does it change overall verbosity) --
# so every moderator enters only via its interaction with log_round_num.
# backchannel_bin and modality_bin additionally interact with each other, still
# only through their joint interaction with log_round_num (no bare
# backchannel_bin:modality_bin term): log_round_num:(backchannel_bin*modality_bin)
# expands to log_round_num:backchannel_bin + log_round_num:modality_bin +
# log_round_num:backchannel_bin:modality_bin.


# it doesn't make sense to do partner constancy when we filter to stage 1 --
# where all partner in-constancy is by definition after a stage change

per_describer_for_model |>
  filter(stage_num == 1, !is.na(log_words)) |>
  count(image_type, backchannel_bin, feedback_bin, modality_bin, population_bin) |>
  print(n = Inf)

red_mod_log_log_all_moderators <- brm(
  log_words ~ log_round_num +
    log_round_num:(log_n_players + log_option_set_size +
      role_constancy + population_bin +
      feedback_bin + backchannel_bin + modality_bin) +
    (log_round_num || dataset_id / condition_id),
  file = here("cached_model_files/mods/red_mod_log_log_all_moderators.rds"),
  file_refit = "on_change",
  prior = log_dv_priors,
  save_pars = save_pars(all = TRUE),
  data = per_describer_for_model |> filter(!is.na(log_words)) |> filter(stage_num == 1)
)
