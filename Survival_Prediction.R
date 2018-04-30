# data laod ---------------------------------------------------------------


# download data set
actg320_colnames <- c('id','time','censor','time_d','censor_d','treatment','treatment_group',
                      'strat2','sex','raceth','ivdrug','hemophil','karnof','cd4','priorzdv','age')
actg320 <- read.table('https://www.umass.edu/statdata/statdata/data/actg320.dat', col.names = actg320_colnames)
dim(actg320)
head(actg320)

# we're removing time_d and censor_2 as it has a rarer outcome balance
actg320 <- actg320[,c('time', 'censor', 'treatment','treatment_group',
                      'strat2','sex','raceth','ivdrug','hemophil','karnof','cd4','priorzdv','age')]

# install.packages('ranger')
library(ranger)
# install.packages('survival')
library(survival)

survival_formula <- formula(paste('Surv(', 'time', ',', 'censor', ') ~ ','treatment+treatment_group',
                                  '+strat2+sex+raceth+ivdrug+hemophil+karnof+cd4+priorzdv+age'))

survival_formula

survival_model <- ranger(survival_formula,
                         data = actg320,  
                         seed = 1234,
                         importance = 'permutation',
                         mtry = 2,
                         verbose = TRUE,
                         num.trees = 50,
                         write.forest=TRUE)

# print out coefficients
sort(survival_model$variable.importance)

plot(survival_model$unique.death.times, survival_model$survival[1,], type='l', col='orange', ylim=c(0.4,1))
lines(survival_model$unique.death.times, survival_model$survival[56,], col='blue')

actg320[1,]
actg320[56,]


plot(survival_model$unique.death.times, survival_model$survival[1,], type='l', col='orange', ylim=c(0.4,1))
for (x in c(2:100)) {
  lines(survival_model$unique.death.times, survival_model$survival[x,], col='red')
}


set.seed(1234)
random_splits <- runif(nrow(actg320))
train_df_official <- actg320[random_splits < .5,]
dim(train_df_official)
validate_df_official <- actg320[random_splits >= .5,]
dim(validate_df_official)

period_choice <- 82 # 103 
table(train_df_official$time)


# classification data set
train_df_classificaiton  <- train_df_official 
train_df_classificaiton$ReachedEvent <- ifelse((train_df_classificaiton$censor==1 & 
                                                  train_df_classificaiton$time<=period_choice), 1, 0)
summary(train_df_classificaiton$ReachedEvent)

validate_df_classification  <- validate_df_official 
validate_df_classification$ReachedEvent <- ifelse((validate_df_classification$censor==1 & 
                                                     validate_df_classification$time<=period_choice), 1, 0)
summary(validate_df_classification$ReachedEvent)



feature_names <- setdiff(names(train_df_classificaiton), c('ReachedEvent', 'time', 'censor'))

# isntall.packages('gbm')
library(gbm)
classification_formula <- formula(paste('ReachedEvent ~ ','treatment+treatment_group',
                                        '+strat2+sex+raceth+ivdrug+hemophil+karnof+cd4+priorzdv+age'))

set.seed(1234)
gbm_model = gbm(classification_formula, 
                data =  train_df_classificaiton,
                distribution='bernoulli',
                n.trees=500,         
                interaction.depth=3,
                shrinkage=0.01,
                bag.fraction=0.5,
                keep.data=FALSE,
                cv.folds=5)

nTrees <- gbm.perf(gbm_model)
validate_predictions <- predict(gbm_model, newdata=validate_df_classification[,feature_names], type="response", n.trees=nTrees)

# install.packages('pROC')
library(pROC)
roc(response=validate_df_classification$ReachedEvent, predictor=validate_predictions)

survival_model <- ranger(survival_formula,
                         data = train_df_official,
                         seed=1234,
                         verbose = TRUE,
                         num.trees = 50,
                         mtry = 2,
                         write.forest=TRUE )

survival_model$unique.death.times


suvival_predictions <- predict( survival_model, validate_df_official[, c('treatment','treatment_group',
                                                                         'strat2','sex','raceth','ivdrug',
                                                                         'hemophil','karnof','cd4',
                                                                         'priorzdv','age')])

roc(response=validate_df_classification$ReachedEvent, predictor=1 - suvival_predictions$survival[,which(suvival_predictions$unique.death.times==period_choice)])



# blend both together -------------------------------------------------------
roc(predictor = (validate_predictions + (1 - suvival_predictions$survival[,which(suvival_predictions$unique.death.times==period_choice)]))/2, 
    response = validate_df_classification$ReachedEvent)



# split training into two datasets
set.seed(1234)
random_splits <- runif(nrow(train_df_official))
train_1 <- train_df_official[random_splits < .5,]
dim(train_1)
train_2 <- train_df_official[random_splits >= .5,]
dim(train_2)

# split testing set in two
set.seed(1234)
random_splits <- runif(nrow(validate_df_official))
test_1 <- validate_df_official[random_splits < .5,]
dim(test_1)
test_2 <- validate_df_official[random_splits >= .5,]
dim(test_2)



surv_1 <- ranger(survival_formula,
                 data =  train_1,
                 verbose = TRUE,
                 seed=1234,
                 num.trees = 50,
                 mtry = 2,
                 write.forest=TRUE )
surv_1$unique.death.times

preds <- predict( surv_1, rbind(train_2[,feature_names], test_2[,feature_names]))
preds_1 <- data.frame(preds$survival)

surv_2 <- ranger(survival_formula,
                 data = train_2,
                 verbose = TRUE,
                 seed=1234,
                 num.trees = 50,
                 mtry = 2,
                 write.forest=TRUE )
surv_2$unique.death.times

preds <- predict( surv_2, rbind(train_1[,feature_names], test_1[,feature_names]))
preds_2 <- data.frame(preds$survival)



# NOTE: can't use period_choice here as second data set doesn't have that period
surv_1$unique.death.times 
train_2_ensemble <- cbind(train_2, preds_1[1:nrow(train_2),which(surv_1$unique.death.times == period_choice)])
names(train_2_ensemble)[ncol(train_2_ensemble)] <- 'survival_probablities'
dim(train_2_ensemble)
names(train_2_ensemble)

test_2_ensemble <- cbind(test_2, preds_1[((nrow(train_2_ensemble)+1):nrow(preds_1)),which(surv_1$unique.death.times == period_choice)])
names(test_2_ensemble)[ncol(test_2_ensemble)] <- 'survival_probablities'


surv_2$unique.death.times
train_1_ensemble <- cbind(train_1, preds_2[1:nrow(train_1),which(surv_2$unique.death.times == period_choice)])
names(train_1_ensemble)[ncol(train_1_ensemble)] <- 'survival_probablities'

test_1_ensemble <- cbind(test_1, preds_2[((nrow(train_1_ensemble)+1):nrow(preds_2)),which(surv_2$unique.death.times == period_choice)])
names(test_1_ensemble)[ncol(test_1_ensemble)] <- 'survival_probablities' 

# finally bring them both back together
train_df_final <- rbind(train_1_ensemble, train_2_ensemble)
validate_df_final <- rbind(test_1_ensemble, test_2_ensemble)



# enjoy fruits of our labor
train_df_final$ReachedEvent <- ifelse((train_df_final$censor==1 & 
                                         train_df_final$time <= period_choice), 1, 0)
summary(train_df_final$ReachedEvent)

validate_df_final$ReachedEvent <- ifelse((validate_df_final$censor==1 & 
                                            validate_df_final$time<= period_choice), 1, 0)


feature_names <- setdiff(names(train_df_final), c('ReachedEvent', 'time', 'censor'))

classification_formula <- formula(paste('ReachedEvent ~ ','treatment+treatment_group',
                                        '+strat2+sex+raceth+ivdrug+hemophil+karnof+cd4+priorzdv+age+survival_probablities'))

set.seed(1234)
gbm_model = gbm(classification_formula, 
                data =  train_df_final,
                distribution='bernoulli',
                n.trees=500,         
                interaction.depth=1,
                shrinkage=0.01,
                bag.fraction=0.5,
                keep.data=FALSE,
                cv.folds=5)

nTrees <- gbm.perf(gbm_model)
validate_predictions <- predict(gbm_model, newdata=validate_df_final[,feature_names], type="response", n.trees=nTrees)
roc(response=validate_df_final$ReachedEvent, predictor=validate_predictions)