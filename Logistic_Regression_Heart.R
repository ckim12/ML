
# Read heart data from UCI 
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
set.seed(987954)
heart_sampling_vector <- createDataPartition(heart$OUTPUT, p = 0.85, list = FALSE)
heart_train <- heart[heart_sampling_vector,]
heart_train_labels <- heart$OUTPUT[heart_sampling_vector]
heart_test <- heart[-heart_sampling_vector,]
heart_test_labels <- heart$OUTPUT[-heart_sampling_vector]

# Run logistic regression model with train dataset
heart_model <- glm(OUTPUT ~ ., data = heart_train, family = binomial("logit"))
summary(heart_model)


