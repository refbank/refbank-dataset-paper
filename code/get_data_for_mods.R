library(refbankr)

favorite_datasets <- c("boyce2024_interaction", "hawkins2020_characterizing_cued")


### get data from redivis
conditions <- get_conditions(datasets = favorite_datasets)

trials <- get_trials(datasets = favorite_datasets)

choices <- get_choices(datasets = favorite_datasets)

messages <- get_messages(datasets = favorite_datasets)

images <- get_images(datasets = favorite_datasets)

pos <- get_annotated_messages(datasets = favorite_datasets)

sbert <- get_cosine_similarities(datasets = favorite_datasets, sim_type = c("to_next", "diverge"))

# apply exclusions

valid_trials_misc <- trials |> filter(!exclusion_reason %in% c("describer didn't talk", "speaker typed nonsense"))

valid_trials_choices <- choices |>
  filter(!is.na(choice_id)) |>
  filter(choice_id != "timed_out") |>
  select(dataset_id, trial_id) |>
  unique()

valid_trials_messages <- messages |>
  filter(role == "describer") |>
  filter(!is.na(text)) |>
  select(dataset_id, trial_id) |>
  unique()

valid_trials <- valid_trials_choices |>
  bind_rows(valid_trials_messages) |>
  unique() |>
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

### set up for models and write to disk

# for accuracy and rt model -- one row per matcher response
per_matcher_for_model <- choices |>
  inner_join(valid_trials_games) |>
  select(trial_id, condition_id, dataset_id, target, choice_id, time_stamp, rep_num, stage_num) |>
  filter(!is.na(choice_id)) |>
  mutate(correct = ifelse(choice_id == target, 1, 0)) |>
  mutate(log_rt = log(time_stamp)) |>
  mutate(log_rep_num = log(rep_num)) |>
  select(correct, log_rt, rep_num, log_rep_num, dataset_id, condition_id, stage_num)
write_rds(here("cached_model_files/data_for_mods/per_matcher_for_model.rds"))


# for all other models -- one row per describer/trial
words <- messages |>
  inner_join(valid_trials_games) |>
  filter(role == "describer") |>
  filter(is.na(message_irrelevant) | !message_irrelevant) |>
  group_by(trial_id, dataset_id, player_id) |>
  summarise(words = sum(lengths(str_split(text, " ")), na.rm = TRUE)) |>
  mutate(log_words = log(words))

pos_summary <- pos |>
  filter(role == "describer") |>
  filter(is.na(message_irrelevant) | !message_irrelevant) |>
  group_by(trial_id) |>
  summarize(across(c(n_hedges, bare, definite, indefinite, quantifier, proper_noun, demonstrative, possessive, total_np), sum)) |>
  mutate(across(-c(n_hedges, trial_id, total_np), \(c) c / total_np)) |>
  select(-total_np)


# Keep only real tokens (drop punctuation/symbols)
wanted_pos <- c(
  "NOUN", "VERB", "ADJ", "ADV",
  "PRON", "ADP", "DET", "AUX",
  "CCONJ", "SCONJ"
)

pos_summary <- pos |>
  inner_join(messages |> filter(role == "describer", is.na(message_irrelevant) | !message_irrelevant) |> select(trial_id, player_id)) |>
  select(trial_id, player_id, upos) |>
  filter(upos %in% wanted_pos) |>
  group_by(trial_id, player_id) |>
  mutate(total_token = n()) |>
  group_by(trial_id, player_id, upos, total_token) |>
  summarize(count = n()) |>
  mutate(pct = count / total_token) |>
  select(-count) |>
  pivot_wider(names_from = upos, values_from = pct, values_fill = 0) |>
  mutate(
    MODIFIER = ADV + ADJ,
    FUNCTION = ADP + AUX + CCONJ + SCONJ
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
  left_join(hedge_summary) |>
  left_join(pos_summary) |>
  mutate(log_rep_num = log(rep_num)) |>
  write_rds(here("cached_model_files/data_for_mods/per_describer_for_model.rds"))


sims_for_model <- sbert |>
  select(sim_type, condition_id, stage_num, dataset_id, sim, rep_num, earlier, later) |>
  mutate(rep_num = case_when(
    sim_type %in% c("diverge", "diff", "idiosyncrasy") ~ rep_num,
    sim_type == "to_next" ~ earlier,
    sim_type == "to_last" ~ earlier,
    sim_type == "to_first" ~ later
  )) |>
  select(sim_type, sim, rep_num, dataset_id, condition_id, stage_num) |>
  mutate(log_rep_num = log(rep_num)) |>
  filter(sim_type %in% c("to_next", "diverge")) |>
  write_rds(here("cached_model_files/data_for_mods/sims_for_model.rds"))
