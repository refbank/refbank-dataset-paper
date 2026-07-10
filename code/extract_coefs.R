library(brms)
library(here)
library(tidyverse)
library(cmdstanr)
library(xgboost)
library(shapviz)
options(
  brms.backend = "cmdstanr",
  mc.cores = 4, # parallel::detectCores(),
  brms.cores = 4, # parallel::detectCores(),
  brms.iter = 4000,
  brms.chains = 4,
  control = list(adapt_delta = 0.99)
)

plot_lab <- \(x) {x |> str_replace_all("_", " ") |> str_to_sentence()}

##### data preprocessing #####
# take the fitted reduction models
# and create data for a slope predicting secondary model
mod_loc <- "cached_model_files/mods"

condition_info <- read_rds(here("cached_model_files/data_for_mods/condition_preds.rds")) |>
  mutate(image_type = factor(image_type, levels = c("tangram", "fribble", "photograph", "line drawing")), )

log_log_mod <- read_rds(here(mod_loc, "red_mod_log_log.rds"))
log_lin_mod <- read_rds(here(mod_loc, "red_mod_log_lin.rds"))

log_log_preds <- coef(log_log_mod)$`dataset_id:condition_id` |>
  as_tibble(rownames = "dataset_cond") |>
  select(dataset_cond,
    slope = Estimate.log_rep_num,
    se = Est.Error.log_rep_num
  ) |>
  left_join(condition_info)

log_lin_preds <- coef(log_lin_mod)$`dataset_id:condition_id` |>
  as_tibble(rownames = "dataset_cond") |>
  select(dataset_cond, slope = Estimate.rep_num, se = Est.Error.rep_num) |>
  left_join(condition_info)

moderators <- c(
  "n_players", "option_size", "image_type",
  "partner_constancy", "role_constancy", "population",
  "modality", "feedback", "backchannel"
)

log_log_xgb_preds <- log_log_preds |>
  select(all_of(moderators)) |>
  mutate(across(where(is.character), as.factor))

log_lin_xgb_preds <- log_lin_preds |>
  select(all_of(moderators)) |>
  mutate(across(where(is.character), as.factor))

log_log_xgb_data <- xgb.DMatrix(
  data = log_log_xgb_preds,
  label = log_log_preds$slope
)

log_lin_xgb_data <- xgb.DMatrix(
  data = log_lin_xgb_preds,
  label = log_lin_preds$slope
)

##### modelling #####
hyper_grid <- expand.grid(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  learning_rate = 0.01,
  max_depth = 3:6,
  min_child_weight = 3,
  subsample = 0.75,
  colsample_bytree = 0.5,
  min_split_loss = c(0, 1, 10, 100, 1000),
  reg_lambda = c(0, 1e-2, 0.1, 1, 10, 100),
  reg_alpha = c(0, 1e-2, 0.1, 1, 10, 100, 1000)
) |>
  mutate(objective = as.character(objective),
         eval_metric = as.character(eval_metric))

# log_log
log_log_xgb <- hyper_grid |>
  mutate(rmse = map_dbl(seq_len(nrow(hyper_grid)), \(i) {
    set.seed(123)
    message("Fitting model ", i, " of ", nrow(hyper_grid))
    params <- hyper_grid[i,] |>
      as.list()

    xgb_cv <- xgb.cv(params = params,
                     data = log_log_xgb_data,
                     nrounds = 1000,
                     nfold = 5,
                     verbose = FALSE,
                     early_stopping_rounds = 50)
    min(xgb_cv$evaluation_log$test_rmse_mean)
  }))

log_log_xgb_best <- xgb.train(data = log_log_xgb_data,
                              nrounds = 1000,
                              params = as.list(log_log_xgb |> slice_min(rmse) |> select(-rmse)))
saveRDS(log_log_xgb_best, here(mod_loc, "log_log_xgb_best.rds"))

log_log_shap <- shapviz(log_log_xgb_best,
                        X_pred = data.matrix(log_log_xgb_preds),
                        interaction = TRUE)
saveRDS(log_log_shap, here(mod_loc, "log_log_shap.rds"))

log_log_shap_plot <- sv_importance(log_log_shap) +
  scale_y_discrete(labels = plot_lab)
log_log_bee_plot <- sv_importance(log_log_shap, kind = "bee") +
  scale_y_discrete(labels = plot_lab)

# log_lin
log_lin_xgb <- hyper_grid |>
  mutate(rmse = map_dbl(seq_len(nrow(hyper_grid)), \(i) {
    set.seed(123)
    message("Fitting model ", i, " of ", nrow(hyper_grid))
    params <- hyper_grid[i,] |>
      as.list()

    xgb_cv <- xgb.cv(params = params,
                     data = log_lin_xgb_data,
                     nrounds = 1000,
                     nfold = 5,
                     verbose = FALSE,
                     early_stopping_rounds = 50)
    min(xgb_cv$evaluation_log$test_rmse_mean)
  }))

log_lin_xgb_best <- xgb.train(data = log_lin_xgb_data,
                              nrounds = 1000,
                              params = as.list(log_lin_xgb |> slice_min(rmse) |> select(-rmse)))
saveRDS(log_lin_xgb_best, here(mod_loc, "log_lin_xgb_best.rds"))

log_lin_shap <- shapviz(log_lin_xgb_best,
                        X_pred = data.matrix(log_lin_xgb_preds),
                        interaction = TRUE)
saveRDS(log_lin_shap, here(mod_loc, "log_lin_shap.rds"))

log_lin_shap_plot <- sv_importance(log_lin_shap) +
  scale_y_discrete(labels = plot_lab)
log_lin_bee_plot <- sv_importance(log_lin_shap, kind = "bee") +
  scale_y_discrete(labels = plot_lab)
