library(keras)
library(lime)
library(readr)
library(dplyr)
library(tidyr)
library(reticulate)
library(tensorflow)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

library(ggplot2)
library(forcats)
#Functions
NcalcMeasures<-function(TP,FN,FP,TN){
  
  NcalcAccuracy<-function(TP,FP,TN,FN){return(100.0*((TP+TN)/(TP+FP+FN+TN)))}
  NcalcPgood<-function(TP,FP,TN,FN){return(100.0*(TP/(TP+FP)))}
  NcalcPbad<-function(TP,FP,TN,FN){return(100.0*(TN/(FN+TN)))}
  NcalcFPR<-function(TP,FP,TN,FN){return(100.0*(FP/(FP+TN)))}
  NcalcTPR<-function(TP,FP,TN,FN){return(100.0*(TP/(TP+FN)))}
  NcalcMCC<-function(TP,FP,TN,FN){return( ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))}
  
  retList<-list(  "TP"=TP,
                  "FN"=FN,
                  "TN"=TN,
                  "FP"=FP,
                  "accuracy"=NcalcAccuracy(TP,FP,TN,FN),
                  "pgood"=NcalcPgood(TP,FP,TN,FN),
                  "pbad"=NcalcPbad(TP,FP,TN,FN),
                  "FPR"=NcalcFPR(TP,FP,TN,FN),
                  "TPR"=NcalcTPR(TP,FP,TN,FN),
                  "MCC"=NcalcMCC(TP,FP,TN,FN)
  )
  return(retList)}
tab_int2double <- function(intTab){
  doubleTab <- intTab
  for(i in nrow(intTab)){
    for (j in ncol(intTab)) {
      doubleTab[i,j] <- as.double(intTab[i,j])
    }
  }
  return(doubleTab)
}





#DATSET

churn_data_raw <- read_csv("D:WA_Fn-UseC_-Telco-Customer-Churn.csv")
glimpse(churn_data_raw)


churn_data_tbl <- churn_data_raw %>%
  drop_na() %>%
  dplyr::select(Churn, everything())

# Split test/training sets
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split

# Retrieve train and test sets
train_tbl_with_ids <- training(train_test_split)
test_tbl_with_ids  <- training(train_test_split)

train_tbl <- select(train_tbl_with_ids, -customerID)
test_tbl <- select(test_tbl_with_ids, -customerID)

churn_data_tbl %>% 
  ggplot(aes(x = tenure)) + 
  geom_histogram()

churn_data_tbl %>% 
  ggplot(aes(x = tenure)) + 
  geom_histogram(bins = 6)


churn_data_tbl %>% 
  ggplot(aes(x = TotalCharges)) + 
  geom_histogram(bins = 100)

churn_data_tbl %>% 
  ggplot(aes(x = log(TotalCharges))) + 
  geom_histogram(bins = 100)

rec_obj <- recipe(Churn ~ ., data = train_tbl) %>% 
  step_discretize(tenure, options = list(cuts = 6)) %>% 
  step_log(TotalCharges) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data = train_tbl)

rec_obj

x_train_tbl <- bake(rec_obj, train_tbl) %>%  dplyr::select(-Churn)
x_test_tbl  <- bake(rec_obj, test_tbl) %>%  dplyr::select(-Churn)

glimpse(x_train_tbl)

y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)

model_keras <- keras_model_sequential() %>% 
  layer_dense(
    units = 64, 
    kernel_initializer = "uniform",
    activation = "relu",
    input_shape = ncol(x_train_tbl)
  ) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(
    units = 64, 
    kernel_initializer = "uniform",
    activation = "relu"
  ) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(
    units = 1,
    kernel_initializer = "uniform",
    activation = "sigmoid")
model_keras %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam", 
  metrics = "accuracy"
)

history <- model_keras %>% keras::fit(
  as.matrix(x_train_tbl), 
  y_train_vec, 
  batch_size = 50,
  epochs = 35, 
  validation_split = 0.30
)
save_model_hdf5(model_keras, 'D:/R/Churn Prediction/customer_churn.hdf5', overwrite = TRUE, include_optimizer = TRUE)
yhat_keras_class_vec <- model_keras %>% predict_classes(as.matrix(x_test_tbl)) %>% as.vector()
yhat_keras_prob_vec  <- model_keras %>% predict_proba(as.matrix(x_test_tbl)) %>% as.vector()

estimates_keras_tbl <- tibble(
  truth = as.factor(y_test_vec) %>% forcats::fct_recode(yes = "1", no = "0"),
  estimate = as.factor(yhat_keras_class_vec) %>% forcats::fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)
estimates_keras_tbl

estimates_keras_tbl %>%
  conf_mat(truth, estimate) %>%
  pluck(1) %>%
  as_tibble() %>%
  
  # Visualize with ggplot
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)

confmat4 <- estimates_keras_tbl %>% conf_mat(truth, estimate)

cm1 <- confmat4$table
confmat4$table
measures1 <- NcalcMeasures(as.double(cm1[2,2]),cm1[1,2],
                          cm1[2,1],cm1[1,1])
measures1

x <- confusionMatrix(truth, estimate, positive = "Yes")
TN <- confmat4$table[1]/1760
FP <- confmat4$table[2]/1760
FN <- confmat4$table[3]/1760
TP <- confmat4$table[4]/1760
cost_simple = FN*300 + TP*60 + FP*60 + TN*0
cost_simple

confmat4 <- estimates_keras_tbl %>% conf_mat(truth, estimate)
#Churn Rate
TN <- confmat4$table[1]/12
FP <- confmat4$table[2]/12
FN <- confmat4$table[3]/12
TP <- confmat4$table[4]/12
churn_rate = TP / FP + TP
churn_rate 

#Hit Rate 
TN <- confmat4$table[1]/13
FP <- confmat4$table[2]/13
FN <- confmat4$table[3]/13
TP <- confmat4$table[4]/13
hit_rate = TP / FN + TP
hit_rate 



thresh <- seq(0.1,1.0, length = 10)
cost = rep(0,length(thresh))

dat <- data.frame(
  model = c(rep("optimized",10),"simple"),
  cost_thresh = c(cost,cost_simple),
  thresh_plot = c(thresh,0.5)
)
savings_per_customer = cost_simple - min(cost)
total_savings = 500000*savings_per_customer

total_savings

options(yardstick.event_first = TRUE)
options(yardstick.event_first = FALSE)

estimates_keras_tbl %>% roc_auc(truth, class_prob)

estimates_keras_tbl %>% conf_mat(truth, estimate)

estimates_keras_tbl %>% metrics(truth, estimate)

estimates_keras_tbl %>% precision(truth, estimate)

estimates_keras_tbl %>% recall(truth, estimate)


estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)


class(model_keras)

model_type.keras.models.Sequential <- function(x, ...) {
  "classification"}


predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
}

predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()

explainer <- lime::lime (
  x              = x_train_tbl, 
  model          = model_keras, 
  bin_continuous = TRUE)

explanation <- lime::explain(
  x_test_tbl[1:10,], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 4,
  kernel_width = 0.5
)

plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")



rna <- keras_model_sequential()

rna %>% 
  layer_dense(units = 4, input_shape = 6, kernel_initializer = 'normal', activation = 'relu') %>%
  layer_dense(units = 4, kernel_initializer = 'normal', activation = 'relu') %>%
  layer_dense(units = 1, kernel_initializer = 'normal', activation = 'linear')

summary(rna)

rna %>%
  compile(loss = 'mse', optimizer = optimizer_adam())
save(list = ls(), file = 'D:/R/Churn Prediction/customer_churn.RData')


