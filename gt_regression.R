library(tidymodels)

gt %>%
  summary()

##ggplot
ggplot(gt, aes(x=TEY))+
  geom_histogram(bins = 50)

##rsample
set.seed(123)

gt_split <- initial_split(gt, prop = 0.80)
gt_split

gt_train <- training(gt_split)
gt_test  <-  testing(gt_split)

dim(gt_train)
dim(gt_test)

###------------------------recipes---------------##
##recipes

gt_recipe <- 
  recipe(TEY ~ .,data = gt_train)
summary(gt_recipe)


gt_recipe_steps <- gt_recipe %>% 
  step_meanimpute(all_numeric()) %>%
  step_nzv(all_predictors()) 
gt_recipe_steps

prepped_recipe <- prep(gt_recipe_steps, training = gt_train)
prepped_recipe

gt_train_preprocessed <- bake(prepped_recipe, gt_train) 
gt_train_preprocessed

gt_test_preprocessed <- bake(prepped_recipe, gt_test)
gt_test_preprocessed


####-----------------------Linear Regression------------------------------

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

##parsnip
boost_model <- 
  boost_tree() %>% 
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

##parsnip
decision_model <- 
  decision_tree() %>% ##??????????
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

##parsnip
knn_model <- 
  nearest_neighbor() %>% 
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

##parsnip
rand_model <- 
  rand_forest() %>% 
  set_engine("ranger") %>%
  set_mode("regression") %>%
  translate()
rand_model

rand_form_fit <- 
  rand_model %>% 
  fit(TEY ~ ., data = gt_train)
rand_form_fit

rand_form_fit %>% pluck("fit")

model_res <- 
  rand_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
rand_wflow <- 
  workflow() %>% 
  add_model(rand_model)

rand_wflow

rand_wflow <- 
  rand_wflow %>% 
  add_formula(TEY ~ .)

rand_wflow

rand_fit <- fit(rand_wflow, gt_train)
rand_fit


##performance (yardstick)
gt_test_res <- predict(rand_fit, new_data = gt_test %>% select(-TEY))
gt_test_res

gt_test_res <- bind_cols(gt_test_res, gt_test %>% select(TEY))
gt_test_res

rmse(gt_test_res, truth = TEY, estimate = .pred)

gt_metrics <- metric_set(rmse, rsq, mae)
gt_metrics(gt_test_res, truth = TEY, estimate = .pred)



####---------------------Support Vector Machine--------------------------------

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
  mlp() %>% 
  set_engine("keras") %>%
  set_mode("regression") %>%
  translate()
mlp_model

mlp_form_fit <- 
  mlp_model %>% 
  fit(TEY ~ ., data = gt_train)
mlp_form_fit

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



