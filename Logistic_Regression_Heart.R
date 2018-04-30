
# Read call heart data from UCI 
heart <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
names(heart) <- c("AGE", "SEX", "CHESTPAIN", "RESTBP", "CHOL", "SUGAR", "ECG", "MAXHR", "ANGINA", "DEP", "EXERCISE", "FLOUR", "THAL", "OUTPUT")

# Check first five rows 
head(heart)

# Convert from numeric to factor
heart$CHESTPAIN = factor(heart$CHESTPAIN)
heart$ECG = factor(heart$ECG)
heart$THAL = factor(heart$THAL)
heart$EXERCISE = factor(heart$EXERCISE)

# Check how the convertion go
str(heart)

# Turn an output feature from 1 and 2 to 0 and 1 by subtract from output -1
heart$OUTPUT = heart$OUTPUT - 1
head(heart)

# Set a seed so the split between train and test are not going change every time you run
library(caret)
set.seed(987954)

heart_sampling_vector <- createDataPartition(heart$OUTPUT, p = 0.85, list = FALSE)
heart_train <- heart[heart_sampling_vector,]
heart_train_labels <- heart$OUTPUT[heart_sampling_vector]

heart_test <- heart[-heart_sampling_vector,]
heart_test_labels <- heart$OUTPUT[-heart_sampling_vector]

# Run logistic regression model with train dataset
heart_model <- glm(OUTPUT ~ ., data = heart_train, family = binomial("logit"))
summary(heart_model)
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

