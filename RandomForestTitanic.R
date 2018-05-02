
set.seed(1234)
train <- read.csv('C:/Users/ckim25/Documents/Titanic/train.csv', stringsAsFactors=FALSE)
test <- read.csv('C:/Users/ckim25/Documents/Titanic/test.csv', stringsAsFactors=FALSE)

library(caret)
library(randomForest)

extractFeatures <- function(data) {
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  fea$Age[is.na(fea$Age)] <- -1
  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm = TRUE)
  fea$Embarked[fea$Embarked==""] = "S"
  fea$Sex <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  return(fea)
}

# randomForest function
rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)

# Submission format
submission <- data.frame(PassengerId = test$PassengerId)
submission$Survived <- predict(rf, extractFeatures(test))

write.csv(submission, file = "random_forest_submission.csv", row.names = FALSE)








