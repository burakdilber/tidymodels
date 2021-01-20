library(tidymodels)

gt %>%
  summary()

##ggplot
ggplot(gt, aes(x=TEY))+
  geom_histogram(bins = 50)

##rsample
set.seed(123)

gt_split <- initial_split(gt, prop = 0.75, strata = TEY)
gt_split

gt_train <- training(gt_split)
gt_test  <-  testing(gt_split)
gt_cv <- vfold_cv(gt_train, v = 10, strata = TEY)


###------------------------recipes---------------##

##recipes
gt_rec <- recipe(TEY ~ ., 
                 data = gt_train)


####-----------------------Linear Regression------------------------------

lm_spec <- 
  linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

lm_wf <-
  workflow() %>% 
  add_recipe(gt_rec) %>% 
  add_model(lm_spec)

lm_results <-
  lm_wf %>% 
  fit_resamples(resamples = gt_cv,
                metrics = metric_set(rmse,mae,rsq))

lm_results %>% 
  collect_metrics(summarize = TRUE)

lm_tuner <- 
  linear_reg( penalty = tune(),
              mixture = tune()) %>% 
  set_engine("lm") %>% 
  set_mode("regression")

lm_twf <-
  lm_wf %>% 
  update_model(lm_tuner)

lm_results <-
  lm_twf %>% 
  tune_grid(resamples = gt_cv)

lm_results %>% 
  show_best(metric = "rmse")

lm_best <-
  lm_results %>% 
  select_best(metric = "rmse")
lm_best

lm_wfl_final <- 
  lm_twf %>%
  finalize_workflow(lm_best)

lm_test_results <-
  lm_wfl_final %>% 
  last_fit(split = gt_split)

lm_test_results %>% 
  collect_metrics()

lm_test_results %>% 
  collect_predictions()


##parsnip
lm_model <- 
  linear_reg() %>% 
  set_engine("glmnet") %>%
  translate()
lm_model

lm_form_fit <- 
  lm_model %>% 
  fit(TEY ~ ., data = gt_train)
lm_form_fit

lm_form_fit %>% pluck("fit")

model_res <- 
  lm_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

param_est <- coef(model_res)

param_est

##broom
tidy(lm_form_fit)
glance(lm_form_fit)


##test data
gt_test_small <- gt_test %>% slice(1:5)
predict(lm_form_fit, new_data = gt_test_small)

gt_test_small %>% 
  select(TEY) %>% 
  bind_cols(predict(lm_form_fit, gt_test_small)) %>% 
  bind_cols(predict(lm_form_fit, gt_test_small, type = "pred_int")) 

##workflow
lm_wflow <- 
  workflow() %>% 
  add_model(lm_model)

lm_wflow

lm_wflow <- 
  lm_wflow %>% 
  add_formula(TEY ~ .)

lm_wflow

lm_fit <- fit(lm_wflow, gt_train)
lm_fit

predict(lm_fit, gt_test %>% slice(1:3))


lm_fit %>% 
  pull_workflow_fit() %>% 
  tidy() %>% 
  slice(1:5)

##performance (yardstick)
gt_test_res <- predict(lm_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)


####-----------------------Boosted Trees------------------------------

boost_spec <- 
  boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

boost_wf <-
  workflow() %>% 
  add_recipe(gt_rec) %>% 
  add_model(boost_spec)

boost_results <-
  boost_wf %>% 
  fit_resamples(resamples = gt_cv,
                metrics = metric_set(rmse,mae,rsq))

boost_results %>% 
  collect_metrics(summarize = TRUE)

boost_tuner <- 
  boost_tree( tree_depth = tune(),
              mtry = tune(),
              min_n = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

boost_twf <-
  boost_wf %>% 
  update_model(boost_tuner)

boost_results <-
  boost_twf %>% 
  tune_grid(resamples = gt_cv)

boost_results %>% 
  show_best(metric = "rmse")

boost_best <-
  boost_results %>% 
  select_best(metric = "rmse")
boost_best

boost_wfl_final <- 
  boost_twf %>%
  finalize_workflow(boost_best)

boost_test_results <-
  boost_wfl_final %>% 
  last_fit(split = gt_split)

boost_test_results %>% 
  collect_metrics()

boost_test_results %>% 
  collect_predictions()

boost_model <- 
  boost_tree(mtry = 1, min_n = 34, tree_depth = 8) %>% 
  set_engine("xgboost") %>%
  set_mode("regression") %>%
  translate()
boost_model

boost_form_fit <- 
  boost_model %>% 
  fit(TEY ~ ., data = gt_train)
boost_form_fit

boost_form_fit %>% pluck("fit")

model_res <- 
  boost_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
boost_wflow <- 
  workflow() %>% 
  add_model(boost_model)

boost_wflow

boost_wflow <- 
  boost_wflow %>% 
  add_formula(TEY ~ .)

boost_wflow

boost_fit <- fit(boost_wflow, gt_train)
boost_fit


##performance (yardstick)
gt_test_res <- predict(boost_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)

####---------------------Decision Tree--------------------------------


decision_spec <- 
  decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

decision_wf <-
  workflow() %>% 
  add_recipe(gt_rec) %>% 
  add_model(decision_spec)

decision_results <-
  decision_wf %>% 
  fit_resamples(resamples = gt_cv,
                metrics = metric_set(rmse,mae,rsq))

decision_results %>% 
  collect_metrics(summarize = TRUE)

decision_tuner <- 
  decision_tree( tree_depth = tune(),
                 min_n = tune(),
                 cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

decision_twf <-
  decision_wf %>% 
  update_model(decision_tuner)

decision_results <-
  decision_twf %>% 
  tune_grid(resamples = gt_cv)

decision_results %>% 
  show_best(metric = "rmse")

decision_best <-
  decision_results %>% 
  select_best(metric = "rmse")
decision_best

decision_wfl_final <- 
  decision_twf %>%
  finalize_workflow(decision_best)

decision_test_results <-
  decision_wfl_final %>% 
  last_fit(split = gt_split)

decision_test_results %>% 
  collect_metrics()

decision_test_results %>% 
  collect_predictions()



##parsnip
decision_model <- 
  decision_tree(cost_complexity = 0.00000000145,
                tree_depth = 15,
                min_n = 17) %>%
  set_engine("rpart") %>%
  set_mode("regression") %>%
  translate()
decision_model

decision_form_fit <- 
  decision_model %>% 
  fit(TEY ~ ., data = gt_train)
decision_form_fit

decision_form_fit %>% pluck("fit")

model_res <- 
  decision_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
decision_wflow <- 
  workflow() %>% 
  add_model(decision_model)

decision_wflow

decision_wflow <- 
  decision_wflow %>% 
  add_formula(TEY ~ .)

decision_wflow

decision_fit <- fit(decision_wflow, gt_train)
decision_fit


##performance (yardstick)
gt_test_res <- predict(decision_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)


####---------------------K - Nearest Neighbor--------------------------------

knn_spec <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

knn_wf <-
  workflow() %>% 
  add_recipe(gt_rec) %>% 
  add_model(knn_spec)

knn_results <-
  knn_wf %>% 
  fit_resamples(resamples = gt_cv,
                metrics = metric_set(rmse,rsq))

knn_results %>% 
  collect_metrics(summarize = TRUE)

knn_tuner <- 
  nearest_neighbor( neighbors = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

knn_twf <-
  knn_wf %>% 
  update_model(knn_tuner)

knn_results <-
  knn_twf %>% 
  tune_grid(resamples = gt_cv)

knn_results %>% 
  show_best(metric = "rmse")

knn_best <-
  knn_results %>% 
  select_best(metric = "rmse")
knn_best

knn_wfl_final <- 
  knn_twf %>%
  finalize_workflow(knn_best)

knn_test_results <-
  knn_wfl_final %>% 
  last_fit(split = gt_split)

knn_test_results %>% 
  collect_metrics()

knn_test_results %>% 
  collect_predictions()



##parsnip
knn_model <- 
  nearest_neighbor(neighbors = 8) %>% 
  set_engine("kknn") %>%
  set_mode("regression") %>%
  translate()
knn_model

knn_form_fit <- 
  knn_model %>% 
  fit(TEY ~ ., data = gt_train)
knn_form_fit

knn_form_fit %>% pluck("fit")

model_res <- 
  knn_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
knn_wflow <- 
  workflow() %>% 
  add_model(knn_model)

knn_wflow

knn_wflow <- 
  knn_wflow %>% 
  add_formula(TEY ~ .)

knn_wflow

knn_fit <- fit(knn_wflow, gt_train)
knn_fit


##performance (yardstick)
gt_test_res <- predict(knn_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)


####---------------------Random Forest--------------------------------


rf_spec <- 
  rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <-
  workflow() %>% 
  add_recipe(gt_rec) %>% 
  add_model(rf_spec)

rf_results <-
  rf_wf %>% 
  fit_resamples(resamples = gt_cv,
                metrics = metric_set(rmse,rsq))

rf_results %>% 
  collect_metrics(summarize = TRUE)

rf_tuner <- 
  rand_forest( min_n = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_twf <-
  rf_wf %>% 
  update_model(rf_tuner)

rf_results <-
  rf_twf %>% 
  tune_grid(resamples = gt_cv)

rf_results %>% 
  show_best(metric = "rmse")

rf_best <-
  rf_results %>% 
  select_best(metric = "rmse")
rf_best


rf_wfl_final <- 
  rf_twf %>%
  finalize_workflow(rf_best)

rf_test_results <-
  rf_wfl_final %>% 
  last_fit(split = gt_split)

rf_test_results %>% 
  collect_metrics()

rf_test_results %>% 
  collect_predictions()



##parsnip
rf_model <- 
  rand_forest(min_n = 4) %>% 
  set_engine("ranger") %>%
  set_mode("regression") %>%
  translate()
rf_model

rf_form_fit <- 
  rf_model %>% 
  fit(TEY ~ ., data = gt_train)
rf_form_fit

rf_form_fit %>% pluck("fit")

model_res <- 
  rf_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
rf_wflow <- 
  workflow() %>% 
  add_model(rf_model)

rf_wflow

rf_wflow <- 
  rf_wflow %>% 
  add_formula(TEY ~ .)

rf_wflow

rf_fit <- fit(rf_wflow, gt_train)
rf_fit


##performance (yardstick)
gt_test_res <- predict(rf_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)



####---------------------Support Vector Machine--------------------------------

svm_spec <- 
  svm_poly() %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

svm_wf <-
  workflow() %>% 
  add_recipe(gt_rec) %>% 
  add_model(svm_spec)

svm_results <-
  svm_wf %>% 
  fit_resamples(resamples = gt_cv,
                metrics = metric_set(rmse,mae,rsq))

svm_results %>% 
  collect_metrics(summarize = TRUE)

svm_tuner <- 
  svm_poly(cost = tune(),
           degree = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

svm_twf <-
  svm_wf %>% 
  update_model(svm_tuner)

svm_results <-
  svm_twf %>% 
  tune_grid(resamples = gt_cv)

svm_results %>% 
  show_best(metric = "rmse")

svm_best <-
  svm_results %>% 
  select_best(metric = "rmse")
svm_best

svm_wfl_final <- 
  svm_twf %>%
  finalize_workflow(svm_best)

svm_test_results <-
  svm_wfl_final %>% 
  last_fit(split = gt_split)

svm_test_results %>% 
  collect_metrics()

svm_test_results %>% 
  collect_predictions()



##parsnip
svm_model <- 
  svm_poly() %>% 
  set_engine("kernlab") %>%
  set_mode("regression") %>%
  translate()
svm_model

svm_form_fit <- 
  svm_model %>% 
  fit(TEY ~ ., data = gt_train)
svm_form_fit

svm_form_fit %>% pluck("fit")

model_res <- 
  svm_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
svm_wflow <- 
  workflow() %>% 
  add_model(svm_model)

svm_wflow

svm_wflow <- 
  svm_wflow %>% 
  add_formula(TEY ~ .)

svm_wflow

svm_fit <- fit(svm_wflow, gt_train)
svm_fit


##performance (yardstick)
gt_test_res <- predict(svm_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)


####---------------------Neural Networks--------------------------------

##parsnip
mlp_model <- 
  mlp(epochs = 100, hidden_units = 5) %>% 
  set_engine("nnet") %>%
  set_mode("regression") %>%
  translate()
mlp_model

mlp_form_fit <- 
  mlp_model %>% 
  fit(TEY ~ ., data = gt_train)
mlp_form_fit

mlp_form_fit %>% pluck("fit")

model_res <- 
  mlp_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
mlp_wflow <- 
  workflow() %>% 
  add_model(mlp_model)

mlp_wflow

mlp_wflow <- 
  mlp_wflow %>% 
  add_formula(TEY ~ .)

mlp_wflow

mlp_fit <- fit(mlp_wflow, gt_train)
mlp_fit


##performance (yardstick)
gt_test_res <- predict(mlp_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)



