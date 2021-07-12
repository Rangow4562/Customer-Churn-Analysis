library(caret)
library(ggplot2)
library(data.table)
library(car)
library(corrplot)
library(rattle)
library(randomForest)
library(C50)
library(rpart)
library(ROCR)
library(e1071)
library(gmodels)
library(doParallel)
library(data.table)
library(install.load)
library(tidyr)
library(lubridate)
library(Hmisc)
library(caret)
library(dummies)
library(caTools)
library(MASS)
library(gridExtra)
library(klaR)
library(arules)
library(class)
library(scales)
library(purrr)
library(kernlab)
library(png)

registerDoParallel(cores =4 )
cust_data <- fread("D:/WA_Fn-UseC_-Telco-Customer-Churn.csv", header = TRUE, sep = ",")
cust_data <- cust_data[, -1]

cust_data[is.na(cust_data)] <- 0

cust_data$Churn <- replace(cust_data$Churn, cust_data$Churn == "No", 0)
cust_data$Churn <- replace(cust_data$Churn, cust_data$Churn == "Yes", 1)
cust_data$Churn <- as.numeric(cust_data$Churn)

cust_data$gender <- car::recode(cust_data$gender, "'Male'=1; 'Female'=0")
cust_data$Partner <- car::recode(cust_data$Partner, "'Yes'=1; 'No'=0")
cust_data$Dependents <- car::recode(cust_data$Dependents, "'Yes'=1; 'No'=0")
cust_data$PhoneService <- car::recode(cust_data$PhoneService, "'Yes'=1; 'No'=0")
cust_data$MultipleLines <- car::recode(cust_data$MultipleLines, "'Yes'=1; 'No'=0;'No phone service'=3")
cust_data$InternetService <- car::recode(cust_data$InternetService, "'No'=0; 'DSL'=1;'Fiber optic'=2")
cust_data$OnlineSecurity <- car::recode(cust_data$OnlineSecurity, "'No'=0; 'Yes'=1;'No internet service'=2")
cust_data$OnlineBackup <- car::recode(cust_data$OnlineBackup, "'No'=0; 'Yes'=1;'No internet service'=2")
cust_data$DeviceProtection <- car::recode(cust_data$DeviceProtection, "'No'=0; 'Yes'=1;'No internet service'=2")
cust_data$TechSupport <- car::recode(cust_data$TechSupport, "'No'=0; 'Yes'=1;'No internet service'=2")
cust_data$StreamingTV <- car::recode(cust_data$StreamingTV, "'No'=0; 'Yes'=1;'No internet service'=2")
cust_data$StreamingMovies <- car::recode(cust_data$StreamingMovies, "'No'=0; 'Yes'=1;'No internet service'=2")
cust_data$Contract <- car::recode(cust_data$Contract, "'Month-to-month'=0; 'One year'=1;'Two year'=2")
cust_data$PaperlessBilling <- car::recode(cust_data$PaperlessBilling, "'Yes'=1; 'No'=0")
cust_data$PaymentMethod <- car::recode(cust_data$PaymentMethod, "'Electronic check'=1; 'Mailed check'=2;'Bank transfer (automatic)'=3; 'Credit card (automatic)'=4")

#convert column to factor
cust_data[, 'Churn'] <- lapply(cust_data[, 'Churn'], factor)

summary(cust_data)
str(cust_data)

corrmatrix <- round(cor(cust_data[, - 'Churn']), digits = 2)
corrmatrix

png('correlation_matrix.png')
qplot(x = Var1, y = Var2, data = reshape2::melt(cor(cust_data[, - 'Churn'], use = "p")), fill = value, geom = "tile") + scale_fill_gradient2(limits = c(-1, 1)) + labs(title = "Correlation Matrix")
dev.off()

## Convert to categorical/factor variables
cust_data$gender <- factor(cust_data$gender)
cust_data$SeniorCitizen <- factor(cust_data$SeniorCitizen )
cust_data$Partner <- factor(cust_data$Partner)
cust_data$Dependents <- factor(cust_data$Dependents)
cust_data$PhoneService <- factor(cust_data$PhoneService)
cust_data$MultipleLines <- factor(cust_data$MultipleLines)
cust_data$InternetService <- factor(cust_data$InternetService)
cust_data$OnlineSecurity <- factor(cust_data$OnlineSecurity)
cust_data$OnlineBackup <- factor(cust_data$OnlineBackup)
cust_data$DeviceProtection <- factor(cust_data$DeviceProtection)
cust_data$TechSupport <- factor(cust_data$TechSupport)
cust_data$StreamingTV <- factor(cust_data$StreamingTV)
cust_data$StreamingMovies <- factor(cust_data$StreamingMovies)
cust_data$Contract <- factor(cust_data$Contract)
cust_data$PaperlessBilling <- factor(cust_data$PaperlessBilling)
cust_data$PaymentMethod <- factor(cust_data$PaymentMethod)

library(caret)
set.seed(1234)
intrain <- createDataPartition(y = cust_data$Churn, p = 0.8, list = FALSE, times = 1)
training <- cust_data[intrain,]
testing <- cust_data[ - intrain,]

# Support Vector Machine (SVM) Model

# Implement the SVM algorithm using the optimal cost.
tune.svm = tune(svm,as.factor(Churn)~.,data=training,
                kernel="linear",ranges = list(cost=c(0.0001,0.001,0.01,0.1,1,10,100,1000)))
tune.svm$best.model

svm_predict <- predict(tune.svm$best.model,testing)


#calculate the confusion matrix for the best model
table(predicted=svm_predict,truth=testing$Churn)
confusionMatrix(svm_predict,testing$Churn,positive='1')


svm.model_1 <- svm(as.factor(Churn)~.,data=training,
                   kernel="linear",cost=0.01,scale = FALSE,decision.values=T,probability=TRUE)

svm1_predict <- predict(svm.model_1,testing,decision.values=T,probability=T)
confusionMatrix(svm1_predict,testing$Churn,positive='1')

#Tune the model using train function and radial kernel and check the results
tune_ksvm <- train(as.factor(Churn)~.,data=training,method="svmRadial",
                   tuneGrid=expand.grid(sigma=c(0.001,0.01,0.1,1,0.0001),C=c(0.01,0.1,1,20,21,22,23,24,25,26,27,28,29,1,10,30,40,50)),
                   metric="Accuracy",trControl=trainControl(method='repeatedcv',
                                                            number=5,repeats=10))
tune_ksvm$finalModel
ksvm_predict <- predict(tune_ksvm,testing)
confusionMatrix(ksvm_predict,testing$Churn,positive ='1')

ksvm_linear <- ksvm (as.factor(Churn)~.,data=training,kernel="vanilladot",prob.model=T,
                     C=0.01)
ksvm_predict <- as.data.frame(predict(ksvm_linear,testing,type="probabilities"))
ksvm_predict_1 <- predict(ksvm_linear,testing,type='response')
confusionMatrix(ksvm_predict_1,testing$Churn,positive = '1')


rocplot <- function(pred, truth, ...){
  predob =  prediction(pred, truth)
  perf = performance(predob, 'tpr', 'fpr')
  plot(perf, colorize=T,...)
  abline(a=0,b=1,lwd=2,lty=2)
  print(performance(predob,'auc'))
}

#plot the ROC curve and see the auc
rocplot(ksvm_predict[,2],testing$Churn)

x <- pred_rf %>% conf_mat(truth, prediction)
TN <- x$table[1]/6
FP <- x$table[2]/6
FN <- x$table[3]/6
TP <- x$table[4]/6
churn_rate = TP / FP + TP
churn_rate 
