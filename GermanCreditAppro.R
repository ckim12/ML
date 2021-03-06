# Read German raw credit score data

german <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
names(german) <- c("checking", "duration", "creditHistory","purpose", "credit", "savings", "employment", "installmentRate",
                   "personal", "debtors", "presentResidence", "property", "age","otherPlans", "housing", "existingBankCredits", "job",
                   "dependents", "telephone", "foreign", "risk")

# Since it have a lot of categorical features, we will use dummyVars() fn to create dummy binary for these. For risk variable we convert
# as factor with 0 for good credit and 1 for bad credit.

library(caret)
dummies <- dummyVars(risk ~ ., data = german)                   
head(dummies, n=3)
str(dummies)

german_n <- data.frame(predict(dummies, newdata = german), risk=factor((german$risk-1)))
dim(german_n)
[1] 1000   62

# Split data to train and test sets.
set.seed(977)
german_sampling_vector <- createDataPartition(german_n$risk, p=0.8, list=FALSE)
german_train <- german_n[german_sampling_vector,]
german_test <- german_n[-german_sampling_vector,]

dim(german_train)
[1] 800  62

# Create vector class.weights parameters to specify the cost of misclassifying and observation to each class
# and to incorporate our asymmetric error cost info into our model. Then we will use the tune() function to train
# various SVM model with a radial kernel.
class_weights <- c(1, 5)
names(class_weights) <- c("0", "1")
class_weights
0 1 
1 5 

# In order to run support vector machine learning, have to install "e1071"
install.packages("e1071")
library(e1071)

set.seed(2423)
german_radial_tune <- tune(svm,risk ~ ., data = german_train,
                           kernel = "radial", ranges = list(cost = c(0.01, 0.1, 1, 10, 100),
                           gamma = c(0.01, 0.05, 0.1, 0.5, 1)), class.weights = class_weights)


german_radial_tune$best.parameters
   cost gamma
9   10  0.05

german_radial_tune$best.performance
[1] 0.2675
# which mean the best model cost=10 and gamma=0.05 and have 74 percent trainig accuracy.

# Now let testing the portion of test data set 
german_model <- german_radial_tune$best.model
test_predict <- predict(german_model, german_test[,1:61])
mean(test_predict == german_test[,62])
[1] 0.735

table(predicted = test_predict, actual = german_test[,62])
          actual
predicted   0   1
         0 134  47
         1   6  13
# There are 73.5 percent for test accuracy which is very close to training 
