library(tidymodels)

HTRU_2 <- read.csv("C:/Users/samsung1/Desktop/DR/seminer/classification/HTRU_2.csv")

class_htru<-factor(HTRU_2$class)
HTRU_2[,9]<-class_htru
str(HTRU_2)

View(HTRU_2)

HTRU_2 %>%
  summary()

##rsample
set.seed(123)

HTRU_2_split <- initial_split(HTRU_2, prop = 0.8)
HTRU_2_split

HTRU_2_train <- training(HTRU_2_split)
HTRU_2_test  <-  testing(HTRU_2_split)
HTRU_2_cv <- vfold_cv(HTRU_2_train, v = 5, strata = class)


###------------------------recipes---------------##

##recipes
HTRU_2_rec <- recipe(class ~ ., 
                     data = HTRU_2_train)

####-----------------------Logistic Regression------------------------------


log_spec <- 
  logistic_reg() %>% 
  set_engine("glm")

log_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(log_spec)

log_tuner <- 
  logistic_reg() %>% 
  set_engine("glm")

log_twf <-
  log_wf %>% 
  update_model(log_tuner)

log_results <-
  log_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

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
log_model <- 
  logistic_reg() %>% 
  set_engine("glm") %>%
  translate()
log_model

log_form_fit <- 
  log_model %>% 
  fit(class ~ ., data = HTRU_2_train)
log_form_fit

log_form_fit %>% pluck("fit")

model_res <- 
  log_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

param_est <- coef(model_res)

param_est

##broom
tidy(log_form_fit)
glance(log_form_fit)


##test data
HTRU_2_test_small <- HTRU_2_test %>% slice(1:5)
predict(log_form_fit, new_data = HTRU_2_test_small)

##workflow
log_wflow <- 
  workflow() %>% 
  add_model(log_model)

log_wflow

log_wflow <- 
  log_wflow %>% 
  add_formula(as.factor(class) ~ .)

log_wflow

log_fit <- fit(log_wflow, HTRU_2_train)
log_fit

predict(log_fit, HTRU_2_test %>% slice(1:3))


log_fit %>% 
  pull_workflow_fit() %>% 
  tidy() %>% 
  slice(1:5)



##performance (yardstick)
HTRU_2_test_res <- predict(log_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3242/(3242+70) #precision (0)
3242/(3242+16) #recall (0)
2*((3242/(3242+70)*(3242/(3242+16)))/(3242/(3242+70)+(3242/(3242+16)))) #F1 score (0)


251/(251+16) #precision(1)
251/(251+70) #recall(1)
2*((251/(251+16)*(251/(251+70)))/(251/(251+16)+(251/(251+70)))) #F1 score (1)


####-----------------------Boosted Trees------------------------------

boost_spec <- 
  boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

boost_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(boost_spec)

boost_results <-
  boost_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

boost_results %>% 
  collect_metrics(summarize = TRUE)

boost_tuner <- 
  boost_tree( tree_depth = tune(),
              mtry = tune(),
              min_n = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

boost_twf <-
  boost_wf %>% 
  update_model(boost_tuner)

boost_results <-
  boost_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

boost_results %>% 
  show_best(metric = "accuracy")

boost_best <-
  boost_results %>% 
  select_best(metric = "accuracy")
boost_best

boost_wfl_final <- 
  boost_twf %>%
  finalize_workflow(boost_best)

boost_test_results <-
  boost_wfl_final %>% 
  last_fit(split = HTRU_2_split)

boost_test_results %>% 
  collect_metrics()

boost_test_results %>% 
  collect_predictions()


##parsnip
boost_model <- 
  boost_tree(mtry = 8, min_n = 10, tree_depth = 4) %>% 
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  translate()
boost_model

boost_form_fit <- 
  boost_model %>% 
  fit(class ~ ., data = HTRU_2_train)
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
  add_formula(class ~ .)

boost_wflow

boost_fit <- fit(boost_wflow, HTRU_2_train)
boost_fit

##performance (yardstick)
HTRU_2_test_res <- predict(boost_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3238/(3238+59) #precision (0)
3238/(3238+20) #recall (0)
2*((3238/(3238+59)*(3238/(3238+20)))/(3238/(3238+59)+(3238/(3238+20)))) #F1 score (0)


257/(257+17) #precision(1)
257/(257+64) #recall(1)
2*((257/(257+17)*(257/(257+64)))/(257/(257+17)+(257/(257+64)))) #F1 score (1)

####-----------------------Decision Tree------------------------------

decision_spec <- 
  decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

decision_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(decision_spec)

decision_results <-
  decision_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

decision_results %>% 
  collect_metrics(summarize = TRUE)

decision_tuner <- 
  decision_tree( tree_depth = tune(),
                 min_n = tune(),
                 cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

decision_twf <-
  decision_wf %>% 
  update_model(decision_tuner)

decision_results <-
  decision_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

decision_results %>% 
  show_best(metric = "accuracy")

decision_best <-
  decision_results %>% 
  select_best(metric = "accuracy")
decision_best

decision_wfl_final <- 
  decision_twf %>%
  finalize_workflow(decision_best)

decision_test_results <-
  decision_wfl_final %>% 
  last_fit(split = HTRU_2_split)

decision_test_results %>% 
  collect_metrics()

decision_test_results %>% 
  collect_predictions()


##parsnip
decision_model <- 
  decision_tree(cost_complexity = 0.00000000265,
                tree_depth = 7,
                min_n = 33) %>% 
  set_engine("rpart") %>%
  set_mode("classification") %>%
  translate()
decision_model

decision_form_fit <- 
  decision_model %>% 
  fit(class ~ ., data = HTRU_2_train)
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
  add_formula(class ~ .)

decision_wflow

decision_fit <- fit(decision_wflow, HTRU_2_train)
decision_fit

##performance (yardstick)
HTRU_2_test_res <- predict(decision_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3237/(3237+69) #precision (0)
3237/(3237+21) #recall (0)
2*((3237/(3237+69)*3237/(3237+21))/(3237/(3237+69)+3237/(3237+21))) #F1 score (0)


262/(262+15) #precision(1)
262/(262+59) #recall(1)
2*((262/(262+15)*(262/(262+59)))/(262/(262+15)+(262/(262+59)))) #F1 score (1)

####-----------------------K - nearest Neighbor------------------------------

knn_spec <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

knn_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(knn_spec)

knn_results <-
  knn_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

knn_results %>% 
  collect_metrics(summarize = TRUE)

knn_tuner <- 
  nearest_neighbor( neighbors = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

knn_twf <-
  knn_wf %>% 
  update_model(knn_tuner)

knn_results <-
  knn_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

knn_results %>% 
  show_best(metric = "accuracy")

knn_best <-
  knn_results %>% 
  select_best(metric = "accuracy")
knn_best

knn_wfl_final <- 
  knn_twf %>%
  finalize_workflow(knn_best)

knn_test_results <-
  knn_wfl_final %>% 
  last_fit(split = HTRU_2_split)

knn_test_results %>% 
  collect_metrics()

knn_test_results %>% 
  collect_predictions()


##parsnip
knn_model <- 
  nearest_neighbor(neighbors = 11) %>% 
  set_engine("kknn") %>%
  set_mode("classification") %>%
  translate()
knn_model

knn_form_fit <- 
  knn_model %>% 
  fit(class ~ ., data = HTRU_2_train)
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
  add_formula(class ~ .)

knn_wflow

knn_fit <- fit(knn_wflow, HTRU_2_train)
knn_fit

##performance (yardstick)
HTRU_2_test_res <- predict(knn_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3234/(3234+63) #precision (0)
3234/(3234+24) #recall (0)
2*((3234/(3234+63)*3234/(3234+24))/(3234/(3234+63)+3234/(3234+24))) #F1 score (0)


254/(254+18) #precision(1)
254/(254+67) #recall(1)
2*((254/(254+18)*(254/(254+67)))/(254/(254+18)+(254/(254+67)))) #F1 score (1)

####-----------------------Random Forest------------------------------

rf_spec <- 
  rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(rf_spec)

rf_results <-
  rf_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

rf_results %>% 
  collect_metrics(summarize = TRUE)

rf_tuner <- 
  rand_forest( min_n = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_twf <-
  rf_wf %>% 
  update_model(rf_tuner)

rf_results <-
  rf_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

rf_results %>% 
  show_best(metric = "accuracy")

rf_best <-
  rf_results %>% 
  select_best(metric = "accuracy")
rf_best


rf_wfl_final <- 
  rf_twf %>%
  finalize_workflow(rf_best)

rf_test_results <-
  rf_wfl_final %>% 
  last_fit(split = HTRU_2_split)

rf_test_results %>% 
  collect_metrics()

rf_test_results %>% 
  collect_predictions()


##parsnip
rf_model <- 
  rand_forest(min_n = 10) %>% 
  set_engine("ranger") %>%
  set_mode("classification") %>%
  translate()
rf_model

rf_form_fit <- 
  rf_model %>% 
  fit(class ~ ., data = HTRU_2_train)
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
  add_formula(class ~ .)

rf_wflow

rf_fit <- fit(rf_wflow, HTRU_2_train)
rf_fit

##performance (yardstick)
HTRU_2_test_res <- predict(rf_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3240/(3240+61) #precision (0)
3240/(3240+18) #recall (0)
2*((3240/(3240+61)*3240/(3240+18))/(3240/(3240+61)+3240/(3240+18))) #F1 score (0)


261/(261+18) #precision(1)
261/(261+60) #recall(1)
2*((261/(261+18)*(261/(261+60)))/(261/(261+18)+(261/(261+60)))) #F1 score (1)

####-----------------------Supoort Vector Machines------------------------------

svm_spec <- 
  svm_poly() %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

svm_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(svm_spec)

svm_results <-
  svm_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

svm_results %>% 
  collect_metrics(summarize = TRUE)

svm_tuner <- 
  svm_poly(cost = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

svm_twf <-
  svm_wf %>% 
  update_model(svm_tuner)

svm_results <-
  svm_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

svm_results %>% 
  show_best(metric = "accuracy")

svm_best <-
  svm_results %>% 
  select_best(metric = "accuracy")
svm_best

svm_wfl_final <- 
  svm_twf %>%
  finalize_workflow(svm_best)

svm_test_results <-
  svm_wfl_final %>% 
  last_fit(split = HTRU_2_split)

svm_test_results %>% 
  collect_metrics()

svm_test_results %>% 
  collect_predictions()


##parsnip
svm_model <- 
  svm_poly(cost = 4.00) %>% 
  set_engine("kernlab") %>%
  set_mode("classification") %>%
  translate()
svm_model

svm_form_fit <- 
  svm_model %>% 
  fit(class ~ ., data = HTRU_2_train)
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
  add_formula(class ~ .)

svm_wflow

svm_fit <- fit(svm_wflow, HTRU_2_train)
svm_fit

##performance (yardstick)
HTRU_2_test_res <- predict(svm_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3242/(3242+70) #precision (0)
3242/(3242+16) #recall (0)
2*((3242/(3242+70)*3242/(3242+16))/(3242/(3242+70)+3242/(3242+16))) #F1 score (0)


251/(251+16) #precision(1)
251/(251+70) #recall(1)
2*((251/(251+16)*(251/(251+70)))/(251/(251+16)+(251/(251+70)))) #F1 score (1)


###------------------------------

##naive Bayes
nb_spec <- 
  naive_Bayes() %>% 
  set_engine("klaR")

nb_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(nb_spec)

nb_results <-
  nb_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

nb_results %>% 
  collect_metrics(summarize = TRUE)

nb_tuner <- 
  naive_Bayes(smoothness = tune(),
              Laplace = tune()) %>% 
  set_engine("klaR")

nb_twf <-
  nb_wf %>% 
  update_model(nb_tuner)

nb_results <-
  nb_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

nb_results %>% 
  show_best(metric = "accuracy")

nb_best <-
  nb_results %>% 
  select_best(metric = "accuracy")
nb_best

nb_wfl_final <- 
  nb_twf %>%
  finalize_workflow(nb_best)

nb_test_results <-
  nb_wfl_final %>% 
  last_fit(split = HTRU_2_split)

nb_test_results %>% 
  collect_metrics()

nb_test_results %>% 
  collect_predictions()



nb_model <- 
  naive_Bayes(smoothness = 0.505,
              Laplace = 2.66) %>% 
  set_engine("klaR") %>%
  translate()
nb_model

nb_form_fit <- 
  nb_model %>% 
  fit(class ~ ., data = HTRU_2_train)
nb_form_fit

nb_form_fit %>% pluck("fit")

model_res <- 
  nb_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
nb_wflow <- 
  workflow() %>% 
  add_model(nb_model)

nb_wflow

nb_wflow <- 
  nb_wflow %>% 
  add_formula(class ~ .)

nb_wflow

nb_fit <- fit(nb_wflow, HTRU_2_train)
nb_fit

##performance (yardstick)
HTRU_2_test_res <- predict(nb_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3967/(3967+67) #precision (0)
3967/(3967+101) #recall (0)
2*((3967/(3967+67)*3967/(3967+101))/(3967/(3967+67)+3967/(3967+101))) #F1 score (0)


341/(341+92) #precision(1)
341/(341+65) #recall(1)
2*((341/(341+92)*(341/(341+65)))/(341/(341+92)+(341/(341+65)))) #F1 score (1)


##Flexible Discriminant Models------------------

disc_flexible_spec <- 
  discrim_flexible() %>% 
  set_engine("earth")

disc_flexible_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(disc_flexible_spec)

disc_flexible_results <-
  disc_flexible_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

disc_flexible_results %>% 
  collect_metrics(summarize = TRUE)

disc_flexible_tuner <- 
  discrim_flexible(num_terms = tune(),
                   prod_degree = tune()) %>% 
  set_engine("earth")

disc_flexible_twf <-
  disc_flexible_wf %>% 
  update_model(disc_flexible_tuner)

disc_flexible_results <-
  disc_flexible_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

disc_flexible_results %>% 
  show_best(metric = "accuracy")

disc_flexible_best <-
  disc_flexible_results %>% 
  select_best(metric = "accuracy")
disc_flexible_best

disc_flexible_wfl_final <- 
  disc_flexible_twf %>%
  finalize_workflow(disc_flexible_best)

disc_flexible_test_results <-
  disc_flexible_wfl_final %>% 
  last_fit(split = HTRU_2_split)

disc_flexible_test_results %>% 
  collect_metrics()

disc_flexible_test_results %>% 
  collect_predictions()


disc_flexible_model <- 
  discrim_flexible(num_terms = 7,
                   prod_degree = 1) %>% 
  set_engine("earth") %>%
  translate()
disc_flexible_model

disc_flexible_form_fit <- 
  disc_flexible_model %>% 
  fit(class ~ ., data = HTRU_2_train)
disc_flexible_form_fit

disc_flexible_form_fit %>% pluck("fit")

model_res <- 
  disc_flexible_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
disc_flexible_wflow <- 
  workflow() %>% 
  add_model(disc_flexible_model)

disc_flexible_wflow

disc_flexible_wflow <- 
  disc_flexible_wflow %>% 
  add_formula(class ~ .)

disc_flexible_wflow

disc_flexible_fit <- fit(disc_flexible_wflow, HTRU_2_train)
disc_flexible_fit

##performance (yardstick)
HTRU_2_test_res <- predict(disc_flexible_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


4041/(4041+82) #precision (0)
4041/(4041+27) #recall (0)
2*((4041/(4041+82)*4041/(4041+27))/(4041/(4041+82)+4041/(4041+27))) #F1 score (0)


325/(325+28) #precision(1)
325/(325+81) #recall(1)
2*((325/(325+28)*(325/(325+81)))/(325/(325+28)+(325/(325+81)))) #F1 score (1)



##Linear Discriminant Models-----------------------

disc_linear_spec <- 
  discrim_linear() %>% 
  set_engine("MASS")

disc_linear_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(disc_linear_spec)

disc_linear_results <-
  disc_linear_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

disc_linear_results %>% 
  collect_metrics(summarize = TRUE)

disc_linear_tuner <- 
  discrim_linear(penalty = tune()) %>% 
  set_engine("MASS")

disc_linear_twf <-
  disc_linear_wf %>% 
  update_model(disc_linear_tuner)

disc_linear_results <-
  disc_linear_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

disc_linear_results %>% 
  show_best(metric = "accuracy")

disc_linear_best <-
  disc_linear_results %>% 
  select_best(metric = "accuracy")
disc_linear_best

disc_linear_wfl_final <- 
  disc_linear_twf %>%
  finalize_workflow(disc_linear_best)

disc_linear_test_results <-
  disc_linear_wfl_final %>% 
  last_fit(split = HTRU_2_split)

disc_linear_test_results %>% 
  collect_metrics()

disc_linear_test_results %>% 
  collect_predictions()


disc_linear_model <- 
  discrim_linear() %>% 
  set_engine("MASS") %>%
  translate()
disc_linear_model

disc_linear_form_fit <- 
  disc_linear_model %>% 
  fit(class ~ ., data = HTRU_2_train)
disc_linear_form_fit

disc_linear_form_fit %>% pluck("fit")

model_res <- 
  disc_linear_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
disc_linear_wflow <- 
  workflow() %>% 
  add_model(disc_linear_model)

disc_linear_wflow

disc_linear_wflow <- 
  disc_linear_wflow %>% 
  add_formula(class ~ .)

disc_linear_wflow

disc_linear_fit <- fit(disc_linear_wflow, HTRU_2_train)
disc_linear_fit

##performance (yardstick)
HTRU_2_test_res <- predict(disc_linear_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


4054/(4054+101) #precision (0)
4054/(4054+14) #recall (0)
2*((4054/(4054+101)*4054/(4054+14))/(4054/(4054+101)+4054/(4054+14))) #F1 score (0)


305/(305+14) #precision(1)
305/(305+101) #recall(1)
2*((305/(305+14)*(305/(305+101)))/(305/(305+14)+(305/(305+101)))) #F1 score (1)

##Regularized Discriminant Models------------------------------
disc_regularized_spec <- 
  discrim_regularized() %>% 
  set_engine("klaR")

disc_regularized_wf <-
  workflow() %>% 
  add_recipe(HTRU_2_rec) %>% 
  add_model(disc_regularized_spec)

disc_regularized_results <-
  disc_regularized_wf %>% 
  fit_resamples(resamples = HTRU_2_cv,
                metrics = metric_set(accuracy))

disc_regularized_results %>% 
  collect_metrics(summarize = TRUE)

disc_regularized_tuner <- 
  discrim_regularized(frac_common_cov = tune(),
                      frac_identity = tune()) %>% 
  set_engine("klaR")

disc_regularized_twf <-
  disc_regularized_wf %>% 
  update_model(disc_regularized_tuner)

disc_regularized_results <-
  disc_regularized_twf %>% 
  tune_grid(resamples = HTRU_2_cv)

disc_regularized_results %>% 
  show_best(metric = "accuracy")

disc_regularized_best <-
  disc_regularized_results %>% 
  select_best(metric = "accuracy")
disc_regularized_best

disc_regularized_wfl_final <- 
  disc_regularized_twf %>%
  finalize_workflow(disc_regularized_best)

disc_regularized_test_results <-
  disc_regularized_wfl_final %>% 
  last_fit(split = HTRU_2_split)

disc_regularized_test_results %>% 
  collect_metrics()

disc_regularized_test_results %>% 
  collect_predictions()



disc_regularized_model <- 
  discrim_regularized(frac_common_cov = 0.767,
                      frac_identity = 0.0857) %>% 
  set_engine("klaR") %>%
  translate()
disc_regularized_model

disc_regularized_form_fit <- 
  disc_regularized_model %>% 
  fit(class ~ ., data = HTRU_2_train)
disc_regularized_form_fit

disc_regularized_form_fit %>% pluck("fit")

model_res <- 
  disc_regularized_form_fit %>% 
  pluck("fit") %>% 
  summary()
model_res

##workflow
disc_regularized_wflow <- 
  workflow() %>% 
  add_model(disc_regularized_model)

disc_regularized_wflow

disc_regularized_wflow <- 
  disc_regularized_wflow %>% 
  add_formula(class ~ .)

disc_regularized_wflow

disc_regularized_fit <- fit(disc_regularized_wflow, HTRU_2_train)
disc_regularized_fit

##performance (yardstick)
HTRU_2_test_res <- predict(disc_regularized_fit, new_data = HTRU_2_test %>% select(-class))
HTRU_2_test_res

HTRU_2_test_res <- bind_cols(HTRU_2_test_res, HTRU_2_test %>% select(class))
HTRU_2_test_res

accuracy(HTRU_2_test_res, class, .pred_class)

conf_mat(HTRU_2_test_res, class, .pred_class)


3989/(3989+91) #precision (0)
3989/(3989+79) #recall (0)
2*((3989/(3989+91)*3989/(3989+79))/(3989/(3989+91)+3989/(3989+79))) #F1 score (0)


290/(290+38) #precision(1)
290/(290+116) #recall(1)
2*((290/(290+38)*(290/(290+116)))/(290/(290+38)+(290/(290+116)))) #F1 score (1)
