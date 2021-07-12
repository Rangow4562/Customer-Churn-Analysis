library(randomForest)
library(caret)
telco <- readxl::read_excel("D:/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
set.seed(5)
inTrain <- createDataPartition(telco$Churn, p=0.75, list=FALSE)
train<- telco[inTrain,]
test <- telco[-inTrain,]
churn.rf = randomForest(Churn~., data = train, importance = T)

churn.rf
churn.predict.prob <- predict(churn.rf, test, type="prob")

churn.predict <- predict(churn.rf, test)
confusionMatrix(churn.predict, test$Churn, positive = "Yes")
library(ROCR)
# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# plotting ROC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
importance(churn.rf)