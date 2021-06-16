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

churn_data_raw <- readxl::read_excel("D:/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
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

confmat4 <- estimates_keras_tbl %>% conf_mat(truth, estimate)
cm1 <- confmat4$table

measures1 <- NcalcMeasures(as.double(cm1[2,2]),cm1[1,2],
                          cm1[2,1],cm1[1,1])
measures1

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
#Navie Bayes,Regression Tree;
library(dplyr)
library(ggplot2)
library(corrplot)
library(GGally)
knitr::opts_chunk$set(echo = TRUE)
getwd()
telco <- readxl::read_excel("D:/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.xlsx")


str(telco)
summary(telco)

telco$SeniorCitizen <- gsub('0', 'No', telco$SeniorCitizen)
telco$SeniorCitizen <- gsub('1', 'Yes', telco$SeniorCitizen)
telco$SeniorCitizen <- as.factor(telco$SeniorCitizen)

telco$Churn<-as.factor(telco$Churn)

class(telco$SeniorCitizen)
class(telco$Churn)

levels(telco$PaymentMethod) <- c("Bank transfer", "Credit card", "Electronic check", "Mailed check")

subset(telco, is.na(telco$TotalCharges))
telco <- na.omit(telco)
sum(is.na(telco))

boxplot(telco$tenure, main = "tenure")
boxplot(telco$MonthlyCharges, main = "MonthlyCharges")
boxplot(telco$TotalCharges, main = "TotalCharges")

quant <- telco[,c(6,19,20)]

ggcorr(quant,label = T, digits = 3, low = "#ff0f0f", mid = "#ffffff", high = "#0206ed", midpoint = 0)

for (i in c(2:length(colnames(telco)))){
  if (is.numeric(telco[,i]) == T ){
    print(ggplot(telco, aes(x = telco[,i], fill = Churn)) + 
            geom_histogram(bins = 10) + labs(x = colnames(telco)[i]))
  }
}

for (i in c(2:length(colnames(telco)))){
  if (is.factor(telco[,i]) == T ){
    print(
      ggplot(telco, aes(telco[,i], fill = Churn)) + 
        geom_bar(position = "fill") + 
        labs(x = colnames(telco)[i]) + 
        theme(axis.text.x = element_text(size = 8)))
  }
}

ggplot(telco, aes(x = Churn, y = (..count../sum(..count..)), fill = Churn)) + 
  geom_bar() + 
  scale_y_continuous(labels=scales::percent) +
  ylab("Frequency")

library(unbalanced)
levels(telco$Churn) <- c(0,1)

ind_var <- telco[,1:20]
cl <- telco$Churn


osdat <- ubOver(X=ind_var, Y=cl) #best result
os_cust <- cbind(osdat$X, osdat$Y)
colnames(os_cust)[21] <- "Churn"
barplot(table(os_cust[,21]), main = "Balanced Data through Oversampling", ylab = "Count", xlab = colnames(os_cust)[21])

set.seed(40)
train <- sample(nrow(os_cust), 0.7*nrow(os_cust), replace = FALSE)
TrainSet <- os_cust[train, -1]
TestSet <- os_cust[-train, -1]

summary(TrainSet)
summary(TestSet)

library(e1071)
library(ROCR)
library(Rcpp)

nbmod <- naiveBayes(Churn ~., data = TrainSet)
nbmod
nbTrain <- predict(nbmod, TrainSet, type = "class")
table(nbTrain, TrainSet$Churn)
nbTest <- predict(nbmod, TestSet, type = "class")
nbacc <- mean(nbTest == TestSet$Churn)
nbacc
table(nbTest, TestSet$Churn)
test_label <- TestSet[, "Churn"]
table(nbTrain)

plot(nbTest, TestSet$Churn, main="Predicted Vs. Actual - Naive Bayes", ylab="Predicted",xlab="Actual")

plot(nbTrain, TrainSet$Churn, main="Predicted Vs. Actual - Naive Bayes", ylab="Predicted",xlab="Actual")

#Confusion Matrix
library(caret)
confusionMatrix(as.factor(nbTest), as.factor(TestSet$Churn))

confusionMatrix(as.factor(nbTrain), as.factor(TrainSet$Churn))



# UnBalanced Naive Bayes:

library(e1071)
library(ROCR)
library(Rcpp)
set.seed(5)
inTrain <- createDataPartition(telco$Churn, p=0.75, list=FALSE)
Train<- telco[inTrain,]
Test <- telco[-inTrain,]
nbmod <- naiveBayes(Churn ~., data = Train)
nbmod
nbTrain <- predict(nbmod, Train, type = "class")
table(nbTrain, Train$Churn)
nbTest <- predict(nbmod, Test, type = "class")
nbacc <- mean(nbTest == Test$Churn)
nbacc
table(nbTest, Test$Churn)
test_label <- Test[, "Churn"]
table(nbTrain)

plot(nbTest, Test$Churn, main="Predicted Vs. Actual - Naive Bayes", ylab="Predicted",xlab="Actual")

plot(nbTrain, Train$Churn, main="Predicted Vs. Actual - Naive Bayes", ylab="Predicted",xlab="Actual")

#Confusion Matrix
library(caret)
confusionMatrix(as.factor(nbTest), as.factor(Test$Churn))

confusionMatrix(as.factor(nbTrain), as.factor(Train$Churn))



# Random Forest
library(randomForest)
library(caret)
set.seed(5)
inTrain <- createDataPartition(telco$Churn, p=0.75, list=FALSE)
train<- telco[inTrain,]
test <- telco[-inTrain,]
rfModel <- randomForest(Churn ~., data = train)
print(rfModel)
#Functions:
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
#Prediction 
pred_rf1 <- predict(rfModel, test)
confMat1<- caret::confusionMatrix(pred_rf1, test$Churn)

plot(rfModel)
train <- as.data.frame(train)

rfModel_new <- randomForest(Churn ~., data = train, ntree = 200, mtry = 2, importance = TRUE, proximity = TRUE)
print(rfModel_new)


pred_rf2 <- predict(rfModel_new, test)
confMat2<- caret::confusionMatrix(pred_rf2, test$Churn)

matrix <- tab_int2double(confMat2$table)

matrix
measures1 <- NcalcMeasures(matrix[2,2],matrix[1,2],
                          matrix[2,1],matrix[1,1])

measures1


#DEcession Tree
library(party)
library(partykit)
telco[, 'Churn'] <- lapply(telco[, 'Churn'], factor)
telco[, 'Contract'] <- lapply(telco[, 'Contract'], factor)
telco[, 'tenure'] <- lapply(telco[, 'tenure'], factor)
telco[, 'PaperlessBilling'] <- lapply(telco[, 'PaperlessBilling'], factor)

intrain<- createDataPartition(telco$Churn,p=0.7,list=FALSE)
set.seed(2017)
training<- telco[intrain,]
testing<- telco[-intrain,]
tree <- ctree(Churn~Contract+tenure+PaperlessBilling, training)

plot(tree, type='simple')

pred_tree <- predict(tree, testing)
confMat3 <- confusionMatrix(pred_tree,testing$Churn)
matrix3 <- tab_int2double(confMat3$table) 
measures3 <- NcalcMeasures(matrix3[2,2],matrix3[1,2],
                          matrix3[2,1],matrix3[1,1])


measures3

p1 <- predict(tree, training)
tab1 <- table(Predicted = p1, Actual = training$Churn)
tab2 <- table(Predicted = pred_tree, Actual = testing$Churn)

print(paste('Decision Tree Accuracy',sum(diag(tab2))/sum(tab2)))
save(list = ls(), file = 'D:/R/Churn Prediction/customer_churn.RData')
