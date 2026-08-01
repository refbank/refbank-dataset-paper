library(refbankr)
library(tidyr)
library(dplyr)
library(stringr)
library(readr)
# favorite_datasets <- c("boyce2024_interaction", "hawkins2020_characterizing")

version <- "v12.2"

### get data from redivis
datasets <- get_datasets(version = version)

condition_summary <- refbankr::get_dataset_summary(version = version)

dataset_features <- condition_summary |>
  left_join(datasets) |>
  group_by(dataset_id, short_cite) |>
  summarize(
    group_size_min = min(group_size),
    group_size_max = max(group_size),
    across(c(
      "language", "group_size", "option_set_size",
      "partner_constancy", "role_constancy", "population", "modality", "feedback", "backchannel"
    ), \(x) paste(unique(x), collapse = ", ")),
  ) |>
  mutate(group_size = case_when(
    group_size_min == group_size_max ~ str_c(group_size_min),
    T ~ str_c(group_size_min, " - ", group_size_max)
  )) |>
  select(-group_size_min, -group_size_max) |>
  mutate(
    option_set_size = case_when(
      dataset_id == "yoon2019_audience" ~ "16 (4)",
      dataset_id == "hawkins2026_fmri" ~ "18 (36)",
      T ~ option_set_size
    ),
    group_size = case_when(dataset_id == "hawkins2019_continual" ~ "2", T ~ group_size)
  ) |>
  ungroup() |>
  select(-dataset_id)

game_length <- condition_summary |>
  left_join(datasets) |>
  group_by(dataset_id, short_cite, condition_label) |>
  summarize(
    trials_per_game = sum(trials_per_game),
    reps_per_game = sum(reps_per_game)
  ) |>
  group_by(dataset_id, short_cite) |>
  summarize(
    min_trials = min(trials_per_game),
    max_trials = max(trials_per_game),
    min_reps = min(reps_per_game),
    max_reps = max(reps_per_game)
  ) |>
  mutate(
    trials_per_game = case_when(
      min_trials == max_trials ~ str_c(min_trials),
      T ~ str_c(min_trials, " - ", max_trials)
    ),
    reps_per_game = case_when(
      min_reps == max_reps ~ str_c(min_reps),
      T ~ str_c(min_reps, " - ", max_reps)
    )
  ) |>
  select(-min_trials, -max_trials, -min_reps, -max_reps) |>
  select(-dataset_id)


dataset_features_all <- dataset_features |>
  left_join(game_length) |>
  View(dataset_features_all)


data_quantity <- condition_summary |>
  left_join(datasets) |>
  group_by(dataset_id, short_cite) |>
  summarize(
    across(starts_with("total"), sum)
  ) |>
  select(-total_num_selections) |>
  ungroup() |>
  select(
    Paper = short_cite, Games = total_num_games, Players = total_num_players,
    Trials = total_num_trials, Words = total_num_words
  )

totals <- data_quantity |> summarize(
  across(c("Games", "Players", "Trials", "Words"), sum) |> mutate(Paper = "Total")
)

data_quantity_total <- data_quantity |> bind_rows(totals)

View(data_quantity_total)
