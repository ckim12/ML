install.packages("fscaret")
require(fscaret)

data("funcRegPred")
print(funcRegPred)

require(caret)
print(paste('Total models in caret:', length(getModelInfo())))

install.packages("fscaret", dependencies = c("Depens", "Suggests"))

titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')
titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',ifelse(grepl('Miss'
                  titanicDF$Name),'Miss','Nothing')))

titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',ifelse(grepl('Miss',titanicDF$Name),'Miss','Nothing')))
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)


titanicDF <- titanicDF[c('PClass', 'Age', 'Sex', 'Title', 'Survived')]

titanicDF$Title <- as.factor(titanicDF$Title)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank = F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))
print(names(titanicDF))

set.seed(1234)
splitIndex <- createDataPartition(titanicDF$Survived, p=.75, list = FALSE, times = 1)
trainDF <- titanicDF[splitIndex,]
testDF <- titanicDF[-splitIndex,]

fsModels <- c("glm", "gbm", "treebag", "ridge", "lasso")
myFS <- fscaret(trainDF, testDF, myTimeLimit = 40, preprocessData = TRUE,
                Used.funcRegPred = 'gbm', with.labels = TRUE,
                supress.output = FALSE, no.cores = 2)

myFS$VarImp$matrixVarImp.MSE

results <- myFS$VarImp$matrixVarImp.MSE
results$Input_no <- as.numeric(results$Input_no)
results <- results[c("SUM","SUM%","ImpGrad","Input_no")]
myFS$PPlabels$Input_no <- as.numeric(rownames(myFS$PPlabels))
results <- merge(x=results, y=myFS$PPlabels, by="Input_no", all.x = T)
results <- results[c('Labels', 'SUM')]
results <- subset(results,results$SUM !=0)
results <- results[order(-results$SUM),]
print(results)
