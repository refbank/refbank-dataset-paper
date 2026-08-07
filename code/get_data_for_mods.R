library(refbankr)
library(tidyr)
library(dplyr)
library(stringr)
library(readr)
library(here)
# favorite_datasets <- c("boyce2024_interaction", "hawkins2020_characterizing")

version <- "v13.2"

### get data from redivis
conditions <- get_conditions(version = version)
trials <- get_trials(version = version)
images <- get_images(version = version)
messages <- get_messages(version = version)
selections <- get_selections(version = version)
pos <- get_annotated_messages(version = version)
sbert <- get_cosine_similarities(sim_type = c("to_next", "diverge"), version = version)

combined_conditions <- trials |>
  left_join(images, by = c("target_image" = "image_id", "dataset_id")) |>
  rowwise() |>
  mutate(
    players = str_split(matchers, ";"),
    n_players = (players |> length()) + 1
  ) |>
  select(dataset_id, condition_id, option_set_size, image_type, n_players) |>
  group_by(dataset_id, condition_id, option_set_size, image_type, n_players) |>
  tally() |>
  group_by(dataset_id, condition_id) |>
  arrange(desc(n)) |>
  slice(1) |>
  select(-n) |>
  left_join(conditions) |>
  mutate(dataset_cond = str_c(dataset_id, "_", condition_id)) |>
  write_rds("cached_model_files/data_for_mods/condition_preds.rds")

# apply exclusions

valid_trials_misc <- trials |>
  filter(!exclusion_reason %in% c("describer didn't talk", "speaker typed nonsense")) |>
  left_join(conditions) |>
  filter(!condition_label %in% c("expt1_new", "expt2_new", "expt3_new"))

valid_trials_messages <- messages |>
  filter(role == "describer") |>
  filter(!is.na(text)) |>
  select(dataset_id, trial_id) |>
  unique()

valid_trials <- valid_trials_messages |>
  inner_join(valid_trials_misc)


# determine data-set-specific cutoffs
block_length <- valid_trials |>
  group_by(game_id, round_num, dataset_id, condition_id) |>
  tally() |>
  group_by(round_num, dataset_id, condition_id) |>
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

valid_trials_games <- valid_trials |>
  inner_join(valid_games) |>
  group_by(target_image, game_id, dataset_id) |>
  mutate(count = n()) |>
  filter(count > 2) |>
  ungroup()

condition_info <- conditions |> select(condition_id, dataset_id, age_group = population, modality, feedback, backchannel, role_constancy)

accuracy <- valid_trials_games |>
  filter(confederates != "yes") |> # decision is that talk to bot is fine for the talking side, not for the accuracy side
  left_join(selections) |>
  filter(!is.na(selected_image) & selected_image != "timed_out") |>
  mutate(correct = ifelse(selected_image == target_image, 1, 0)) |>
  select(correct, trial_id, dataset_id, round_num, stage_num, condition_id, option_set_size) |>
  left_join(condition_info) |>
  mutate(log_round_num = log(round_num)) |>
  saveRDS(here("cached_model_files/data_for_mods/per_matcher_for_model.rds"))

words <- messages |>
  inner_join(valid_trials_games) |>
  filter(role == "describer") |>
  filter(is.na(message_irrelevant) | !message_irrelevant) |>
  group_by(trial_id, dataset_id, player_id) |>
  summarise(words = sum(lengths(str_split(text, " ")), na.rm = TRUE)) |>
  mutate(log_words = log(words))

stopifnot("some describer trials have 0 words -> log(0) = -Inf" = all(words$words > 0))

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


stim_type <- images |> select(target_image = image_id, image_type, dataset_id)

per_describer_for_model <- valid_trials_games |>
  mutate(n_players = (matchers |> str_split(";") |>
    lengths()) + 1) |>
  left_join(stim_type) |>
  select(trial_id, dataset_id, round_num, stage_num, condition_id, image_type, option_set_size, n_players) |>
  left_join(condition_info) |>
  left_join(words) |>
  left_join(pos_summary) |>
  mutate(log_round_num = log(round_num)) |>
  saveRDS("cached_model_files/data_for_mods/per_describer_for_model.rds")


valid_trials_games_for_sbert <- valid_trials_games |>
  group_by(game_id, round_num, room_num, dataset_id, target_image) |>
  arrange(trial_num) |>
  slice(1) # there exist cases where the same target occurs more than once per round, we dedupe here

to_next <- sbert |>
  filter(sim_type == "to_next") |>
  inner_join(valid_trials_games_for_sbert |> select(dataset_id, game_id, room_num, target_image,
    earlier = round_num, trial_id_earlier = trial_id
  )) |>
  inner_join(valid_trials_games_for_sbert |> select(dataset_id, game_id, room_num, target_image,
    later = round_num, trial_id_later = trial_id
  )) |>
  left_join(words |> select(dataset_id, trial_id_earlier = trial_id, log_words_earlier = log_words)) |>
  left_join(words |> select(dataset_id, trial_id_later = trial_id, log_words_later = log_words)) |>
  select(
    sim_type, condition_id, game_id, stage_num, dataset_id, sim, earlier, later,
    log_words_earlier, log_words_later
  )


diverge_filter <- sbert |>
  filter(sim_type == "diverge") |>
  inner_join(valid_trials_games_for_sbert |> select(dataset_id, room_num_1 = room_num, target_image, round_num, game_id_1 = game_id)) |>
  inner_join(valid_trials_games_for_sbert |> select(dataset_id, room_num_2 = room_num, target_image, round_num, game_id_2 = game_id))

diverge <- diverge_filter |>
  mutate(game_id = game_id_1) |>
  bind_rows(diverge_filter |> mutate(game_id = game_id_2)) |>
  group_by(sim_type, game_id, target_image, condition_id, dataset_id, round_num, stage_num) |>
  summarize(sim = mean(sim, na.rm = T), n = n()) |>
  filter(n > 3) |> # since we collapse, make sure we are collapsing over conditions with at least 5 to compare within/among
  ungroup()

diverge_trial_lookup <- valid_trials_games |>
  select(dataset_id, game_id, target_image, round_num, trial_id) |>
  distinct()

diverge <- diverge |>
  left_join(diverge_trial_lookup) |>
  left_join(words |> select(dataset_id, trial_id, log_words))


sim_for_model <- to_next |>
  bind_rows(diverge) |>
  mutate(round_num = case_when(
    sim_type %in% c("diverge") ~ round_num,
    sim_type == "to_next" ~ earlier,
  )) |>
  select(
    sim_type, sim, round_num, dataset_id, condition_id, stage_num,
    log_words, log_words_earlier, log_words_later
  ) |>
  mutate(log_round_num = log(round_num)) |>
  filter(sim_type %in% c("to_next", "diverge")) |>
  saveRDS("cached_model_files/data_for_mods/sims_for_model.rds")
