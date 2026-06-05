library(refbankr)
library(tidyr)
library(dplyr)
library(stringr)
library(readr)
favorite_datasets <- c("boyce2024_interaction", "hawkins2020_characterizing")

version <- "v12.1"

### get data from redivis
conditions <- get_conditions(datasets = favorite_datasets, version = version)
trials <- get_trials(datasets = favorite_datasets, version = version)
images <- get_images(datasets = favorite_datasets, version = version)
messages <- get_messages(datasets = favorite_datasets, version = version)
choices <- get_choices(datasets = favorite_datasets, version = version)
pos <- get_annotated_messages(datasets = favorite_datasets, version = version)
sbert <- get_cosine_similarities(datasets = favorite_datasets, sim_type = c("to_next", "diverge"), version = version)

combined_conditions <- trials |>
  left_join(images, by = c("target" = "image_id", "dataset_id")) |>
  rowwise() |>
  mutate(
    players = str_split(matchers, ";"),
    n_players = (players |> length()) + 1
  ) |>
  select(dataset_id, condition_id, option_size, image_type, n_players) |>
  group_by(dataset_id, condition_id, option_size, image_type, n_players) |>
  tally() |>
  group_by(dataset_id, condition_id) |>
  arrange(desc(n)) |>
  slice(1) |>
  select(-n) |>
  left_join(conditions) |>
  mutate(dataset_cond = str_c(dataset_id, "_", condition_id)) |>
  write_rds("cached_model_files/data_for_mods/condition_preds.rds")

# apply exclusions

valid_trials_misc <- trials |> filter(!exclusion_reason %in% c("describer didn't talk", "speaker typed nonsense"))

valid_trials_messages <- messages |>
  filter(role == "describer") |>
  filter(!is.na(text)) |>
  select(dataset_id, trial_id) |>
  unique()

valid_trials <- valid_trials_messages |>
  inner_join(valid_trials_misc)


# determine data-set-specific cutoffs
block_length <- valid_trials |>
  group_by(game_id, rep_num, dataset_id, condition_id) |>
  tally() |>
  group_by(rep_num, dataset_id, condition_id) |>
  summarize(max = max(n)) |>
  group_by(dataset_id, condition_id) |>
  summarize(block_length = median(max)) |>
  mutate(minimum = 2 * block_length) |>
  select(dataset_id, condition_id, minimum)


# calculate per-game lengths
valid_games <- valid_trials |>
  group_by(game_id, dataset_id, condition_id) |>
  tally() |>
  left_join(block_length) |>
  filter(n >= minimum) |>
  ungroup() |>
  select(game_id, dataset_id, condition_id)

valid_trials_games <- valid_trials |> inner_join(valid_games)


accuracy <- valid_trials_games |>
  left_join(choices) |>
  filter(!is.na(choice_id) & choice_id != "timed_out") |>
  mutate(correct = ifelse(choice_id == target, 1, 0)) |>
  select(correct, trial_id, dataset_id, rep_num, stage_num, condition_id, option_size) |>
  left_join(condition_info) |>
  mutate(log_rep_num = log(rep_num)) |>
  saveRDS("cached_model_files/data_for_mods/per_matcher_for_model.rds")

words <- messages |>
  inner_join(valid_trials_games) |>
  filter(role == "describer") |>
  filter(is.na(message_irrelevant) | !message_irrelevant) |>
  group_by(trial_id, dataset_id, player_id) |>
  summarise(words = sum(lengths(str_split(text, " ")), na.rm = TRUE)) |>
  mutate(log_words = log(words))

wanted_pos <- c(
  "NOUN", "VERB", "ADJ", "ADV",
  "PRON", "ADP", "DET", "AUX",
  "CCONJ", "SCONJ"
)

pos_summary <- pos |>
  filter(role == "describer") |>
  filter(is.na(message_irrelevant) | !message_irrelevant) |>
  select(trial_id, any_of(wanted_pos)) |>
  group_by(trial_id) |>
  summarize(across(all_of(wanted_pos), sum)) |>
  mutate(
    MODIFIER = ADV + ADJ,
    FUNCTION = ADP + AUX + CCONJ + SCONJ,
  ) |>
  select(-c(ADV, ADJ, ADP, AUX, CCONJ, SCONJ))


condition_info <- conditions |> select(condition_id, dataset_id, age_group = population, modality, feedback, backchannel, role_constancy)

stim_type <- images |> select(target = image_id, image_type, dataset_id)

per_describer_for_model <- valid_trials_games |>
  mutate(n_players = (matchers |> str_split(";") |>
    lengths()) + 1) |>
  left_join(stim_type) |>
  select(trial_id, dataset_id, rep_num, stage_num, condition_id, image_type, option_size, n_players) |>
  left_join(condition_info) |>
  left_join(words) |>
  left_join(pos_summary) |>
  mutate(log_rep_num = log(rep_num)) |>
  saveRDS("cached_model_files/data_for_mods/per_describer_for_model.rds")


to_next <- sbert |>
  select(sim_type, condition_id, stage_num, dataset_id, sim, rep_num, earlier, later) |>
  filter(sim_type == "to_next")

diverge <- sbert |>
  filter(sim_type == "diverge") |>
  mutate(game_id = game_id_1) |>
  bind_rows(sbert |> filter(sim_type == "diverge") |> mutate(game_id = game_id_2)) |>
  group_by(sim_type, game_id, target, condition_id, dataset_id, rep_num, stage_num) |>
  summarize(sim = mean(sim, na.rm = T))

sim_for_model <- to_next |>
  bind_rows(diverge) |>
  mutate(rep_num = case_when(
    sim_type %in% c("diverge") ~ rep_num,
    sim_type == "to_next" ~ earlier,
  )) |>
  select(sim_type, sim, rep_num, dataset_id, condition_id, stage_num) |>
  mutate(log_rep_num = log(rep_num)) |>
  filter(sim_type %in% c("to_next", "diverge")) |>
  saveRDS("cached_model_files/data_for_mods/sims_for_model.rds")
