# Read heart data from UCI 
heart <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
names(heart) <- c("AGE", "SEX", "CHESTPAIN", "RESTBP", "CHOL", "SUGAR", "ECG", "MAXHR", "ANGINA", "DEP", "EXERCISE", "FLOUR", "THAL", "OUTPUT")

# Check first five rows 
head(heart)

AGE SEX CHESTPAIN RESTBP CHOL SUGAR ECG MAXHR ANGINA DEP EXERCISE FLOUR THAL OUTPUT
1  70   1         4    130  322     0   2   109      0 2.4        2     3    3      2
2  67   0         3    115  564     0   2   160      0 1.6        2     0    7      1
3  57   1         2    124  261     0   0   141      0 0.3        1     0    7      2
4  64   1         4    128  263     0   0   105      1 0.2        2     1    7      1
5  74   0         2    120  269     0   2   121      1 0.2        1     1    3      1
6  65   1         4    120  177     0   0   140      0 0.4        1     0    7      1

# Convert from numeric to factor
heart$CHESTPAIN = factor(heart$CHESTPAIN)
heart$ECG = factor(heart$ECG)
heart$THAL = factor(heart$THAL)
heart$EXERCISE = factor(heart$EXERCISE)

# Check how the convertion go
str(heart)

'data.frame':	270 obs. of  14 variables:
  $ AGE      : num  70 67 57 64 74 65 56 59 60 63 ...
$ SEX      : num  1 0 1 1 0 1 1 1 1 0 ...
$ CHESTPAIN: Factor w/ 4 levels "1","2","3","4": 4 3 2 4 2 4 3 4 4 4 ...
$ RESTBP   : num  130 115 124 128 120 120 130 110 140 150 ...
$ CHOL     : num  322 564 261 263 269 177 256 239 293 407 ...
$ SUGAR    : num  0 0 0 0 0 0 1 0 0 0 ...
$ ECG      : Factor w/ 3 levels "0","1","2": 3 3 1 1 3 1 3 3 3 3 ...
$ MAXHR    : num  109 160 141 105 121 140 142 142 170 154 ...
$ ANGINA   : num  0 0 0 1 1 0 1 1 0 0 ...
$ DEP      : num  2.4 1.6 0.3 0.2 0.2 0.4 0.6 1.2 1.2 4 ...
$ EXERCISE : Factor w/ 3 levels "1","2","3": 2 2 1 2 1 1 2 2 2 2 ...
$ FLOUR    : num  3 0 0 1 1 0 1 1 2 3 ...
$ THAL     : Factor w/ 3 levels "3","6","7": 1 3 3 3 1 3 2 3 3 3 ...
$ OUTPUT   : int  2 1 2 1 1 1 2 2 2 2 ...

# Turn an output feature from 1 and 2 to 0 and 1 by subtract from output -1
heart$OUTPUT = heart$OUTPUT - 1
head(heart)

AGE SEX CHESTPAIN RESTBP CHOL SUGAR ECG MAXHR ANGINA DEP EXERCISE FLOUR THAL OUTPUT
1  70   1         4    130  322     0   2   109      0 2.4        2     3    3      1
2  67   0         3    115  564     0   2   160      0 1.6        2     0    7      0
3  57   1         2    124  261     0   0   141      0 0.3        1     0    7      1
4  64   1         4    128  263     0   0   105      1 0.2        2     1    7      0
5  74   0         2    120  269     0   2   121      1 0.2        1     1    3      0
6  65   1         4    120  177     0   0   140      0 0.4        1     0    7      0

# Set a seed so the split between train and test are not going change every time you run
library(caret)
set.seed(987954)

heart_sampling_vector <- createDataPartition(heart$OUTPUT, p = 0.85, list = FALSE)

heart_train <- heart[heart_sampling_vector,]
heart_train_labels <- heart$OUTPUT[heart_sampling_vector]

heart_test <- heart[-heart_sampling_vector,]
heart_test_labels <- heart$OUTPUT[-heart_sampling_vector]

dim(heart_train)
[1] 230  14

dim(heart_test)
[1] 40 14

####################################################
# Run logistic regression model with train dataset #
####################################################
heart_model <- glm(OUTPUT ~ ., data = heart_train, family = binomial("logit"))
summary(heart_model)

Call:
  glm(formula = OUTPUT ~ ., family = binomial("logit"), data = heart_train)

Deviance Residuals: 
  Min       1Q   Median       3Q      Max  
-2.7137  -0.4421  -0.1382   0.3588   2.8118  

Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept) -7.946051   3.477686  -2.285 0.022321 *  
AGE         -0.020538   0.029580  -0.694 0.487482    
SEX          1.641327   0.656291   2.501 0.012387 *  
CHESTPAIN2   1.308530   1.000913   1.307 0.191098    
CHESTPAIN3   0.560233   0.865114   0.648 0.517255    
CHESTPAIN4   2.356442   0.820521   2.872 0.004080 ** 
RESTBP       0.026588   0.013357   1.991 0.046529 *  
CHOL         0.008105   0.004790   1.692 0.090593 .  
SUGAR       -1.263606   0.732414  -1.725 0.084480 .  
ECG1         1.352751   3.287293   0.412 0.680699    
ECG2         0.563430   0.461872   1.220 0.222509    
MAXHR       -0.013585   0.012873  -1.055 0.291283    
ANGINA       0.999906   0.525996   1.901 0.057305 .  
DEP          0.196349   0.282891   0.694 0.487632    
EXERCISE2    0.743530   0.560700   1.326 0.184815    
EXERCISE3    0.946718   1.165567   0.812 0.416655    
FLOUR        1.310240   0.308348   4.249 2.15e-05 ***
THAL6        0.304117   0.995464   0.306 0.759983    
THAL7        1.717886   0.510986   3.362 0.000774 ***
  ---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: 315.90  on 229  degrees of freedom
Residual deviance: 140.36  on 211  degrees of freedom
AIC: 178.36

Number of Fisher Scoring iterations: 6

# From the model of Logistic Regression show there are three features such as "FLUOR", "CHESTPAIN4",
# and THAL7 are the strongest feature predictors for heart diseases. The other features have p-value
# greater than 0.05 are not good indicators of heart disease.

# Using the predict() function to compute an output. 
train_predict <- predict(heart_model, newdata = heart_train, type = "response")
train_class_predict <- as.numeric(train_predict > 0.5)
mean(train_class_predict == heart_train$OUTPUT)
[1] 0.8869565

test_predict <- predict(heart_model, newdata = heart_test, type = "response")
test_class_predict <- as.numeric(test_predict > 0.5)
mean(test_class_predict == heart_test$OUTPUT)

# To assess the effect of changing the threshold on performance metrics.
install.packages("ROCR")
library(ROCR)

train_predict <- predict(heart_model, newdata = heart_train, type = "response")
pred <- prediction(train_predict, heart_train$OUTPUT)
perf <- performance(pred, measure = "prec", x.measure = "rec")

#########################################
# Predicting heart disease with bagging #
#########################################

# Draw samples with replacement and use these to train models
M <- 11
seeds <- 70000 : (70000 + M -1)
n <- nrow(heart_train)
sample_vectors <- sapply(seeds, function(x) { set.seed(x); return(sample(n, n, replace = T))})

train_1glm <- function(sample_indices) { 
              data <- heart_train[sample_indices,];
              model <- glm(OUTPUT ~ ., data = data, family = binomial("logit"));
              return(model)
              }

models <- apply(sample_vectors, 2, train_1glm)

# Add a new column ID to data frame that stores the original row names from the heart_train df.
get_1bag <- function(sample_indices) {
  unique_sample <- unique(sample_indices);
  df <- heart_train[unique_sample, ];
  df$ID <- unique_sample;
  return(df)
}

bags <- apply(sample_vectors, 2, get_1bag)

glm_predictions <- function(model, data, model_index) {
  colname <- paste("PREDICTIONS", model_index);
  data[colname] <- as.numeric(
    predict(model, data, type = "response") > 0.5);
  return(data[,c("ID", colname), drop = FALSE])
}

training_predictions <- mapply(glm_predictions, models, bags, 1 : M, SIMPLIFY = F)

train_pred_df <- Reduce(function(x, y) merge(x, y, by = "ID", all = T), training_predictions)

head(train_pred_df[,1:5])
ID PREDICTIONS 1 PREDICTIONS 2 PREDICTIONS 3 PREDICTIONS 4
1  1             1            NA             1            NA
2  2             0            NA            NA             0
3  3            NA             0             0            NA
4  4            NA             1             1             1
5  5             0             0             0            NA
6  6             0             1             0             0

train_pred_vote <- apply(train_pred_df[,-1], 1, function(x) as.numeric(mean(x, na.rm = TRUE) > 0.5))

mean(train_pred_vote == heart_train$OUTPUT[as.numeric(train_pred_df$ID)])
[1] 0.9173913

# Function out-of-bag
get_1oo_bag <- function(sample_indices) {
  unique_sample <- setdiff(1 : n, unique(sample_indices));
  df <- heart_train[unique_sample,];
  df$ID <- unique_sample;
  if (length(unique(heart_train[sample_indices,]$ECG)) < 3)
    df[df$ECG == 1,"ECG"] = NA;
  return(df)
}

oo_bags <- apply(sample_vectors, 2, get_1oo_bag)

# Using glm_predictions() function to predict out-of-bag samples.
oob_predictions <- mapply(glm_predictions, models, oo_bags, 1 : M, SIMPLIFY = F)
oob_pred_df <- Reduce(function(x, y) merge(x, y, by = "ID", all = T), oob_predictions)
oob_pred_vote <- apply(oob_pred_df[,-1], 1,function(x) as.numeric(mean(x, na.rm = TRUE) > 0.5))

mean(oob_pred_vote == heart_train$OUTPUT[as.numeric(oob_pred_df$ID)], na.rm = TRUE)
[1] 0.8515284
