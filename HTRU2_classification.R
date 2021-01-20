library(tidymodels)

HTRU_2 <- read.csv("C:/Users/samsung1/Desktop/DR/seminer/classification/HTRU_2.csv")

class_htru<-factor(HTRU_2$class)
HTRU_2[,9]<-class_htru
str(HTRU_2)

View(HTRU_2)

HTRU_2 %>%
  summary()

pairs(HTRU_2[,c(1,2,3,4,7,8)], col=HTRU_2$class)

ggplot(HTRU_2, aes(class)) +
  geom_bar(fill="steelblue") +
  labs(title="Bar Chart of Each Class", x="Class", y = "Count") +
  theme_minimal()

ggplot(data=HTRU_2, aes(x=HTRU_2$class)) +
  geom_bar(stat="identity", fill="steelblue") +
  geom_text(aes(label=17898), vjust=1.6, color="white", size=3.5)+
  labs(title="Popularity of Tidymodels", x="Years", y = "Number of Downloads")+
  theme_minimal()

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

##parsnip
boost_model <- 
  boost_tree() %>% 
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


262/(262+20) #precision(1)
262/(262+59) #recall(1)
2*((262/(262+20)*(262/(262+59)))/(262/(262+20)+(262/(262+59)))) #F1 score (1)

####-----------------------Decision Tree------------------------------

##parsnip
decision_model <- 
  decision_tree() %>% 
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


252/(252+21) #precision(1)
252/(252+69) #recall(1)
2*((252/(252+21)*(252/(252+69)))/(252/(252+21)+(252/(252+69)))) #F1 score (1)

####-----------------------K - nearest Neighbor------------------------------

##parsnip
knn_model <- 
  nearest_neighbor() %>% 
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


258/(258+24) #precision(1)
258/(258+63) #recall(1)
2*((258/(258+24)*(258/(258+63)))/(258/(258+24)+(258/(258+63)))) #F1 score (1)

####-----------------------Random Forest------------------------------

##parsnip
rf_model <- 
  rand_forest() %>% 
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


260/(260+18) #precision(1)
260/(260+61) #recall(1)
2*((260/(260+18)*(260/(260+61)))/(260/(260+18)+(260/(260+61)))) #F1 score (1)

####-----------------------Supoort Vector Machines------------------------------

##parsnip
svm_model <- 
  svm_poly() %>% 
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

##----------------------------------------------------

library(discrim)

##Naive Bayes
nb_model <- 
  naive_Bayes() %>% 
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


339/(339+101) #precision(1)
339/(339+67) #recall(1)
2*((339/(339+101)*(339/(339+67)))/(339/(339+101)+(339/(339+67)))) #F1 score (1)


##Flexible Discriminant Models------------------
disc_flexible_model <- 
  discrim_flexible() %>% 
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


324/(324+27) #precision(1)
324/(324+82) #recall(1)
2*((324/(324+27)*(324/(324+82)))/(324/(324+27)+(324/(324+82)))) #F1 score (1)


##Linear Discriminant Models-----------------------

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

disc_regularized_model <- 
  discrim_regularized() %>% 
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


315/(315+79) #precision(1)
315/(315+91) #recall(1)
2*((315/(315+79)*(315/(315+91)))/(315/(315+79)+(315/(315+91)))) #F1 score (1)
