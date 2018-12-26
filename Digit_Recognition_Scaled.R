#Setting the working directory
setwd("F:/Data Science/SVM/Assignment")

#importing mnist data set
train<- read.csv("mnist_train.csv", stringsAsFactors = F,header = F)
test<- read.csv("mnist_test.csv", stringsAsFactors = F,header = F)

#changing dependent column label name to "Digit" for both train and test
names(train)[1]<-"Digit"
names(test)[1]<-"Digit"

#As per instructions, taking 15% of data from each dataset,
set.seed(100)
trainindices = sample(1:nrow(train), 0.15*nrow(train))
train = train[trainindices,] #selecting 15% of train data
#Test data can be of any size as in real case scenarios, the test data is a lot in more size than train set on which alogorithm is applied.

#importing additional packages
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)

#------Data Cleaning and Preparation------#

#checking for any NA values
sapply(train, function(x) sum(is.na(x))) #no NA in train dataset

sapply(test, function(x) sum(is.na(x))) #no NA in test dataset

#checking dimensions of both dataset
dim(train)
dim(test)
#both data sets have equal number of columns

#checking structure of both datasets for all columns
str(train, list.len=ncol(df))
str(test, list.len=ncol(df))

#changing dependent column to factor type
train$Digit<- factor(train$Digit)
test$Digit<- factor(test$Digit)

#Checking for maximum value of data in the data set
max(train[, 2:ncol(train)]) #maximum is 255

#Scaling data as various attributes of each digit ranges from 0 to 255
train[ , 2:ncol(train)] <- train[ , 2:ncol(train)]/255
test[ , 2:ncol(test)] <- test[ , 2:ncol(test)]/255

#-----Modelling and Evaluation-------#

#using Linear Kernel and C as 1
model_linear_1<- ksvm(Digit~., data=train, scale=FALSE, kernel="vanilladot", C=1)
evaluate_linear_1<- predict(model_linear_1, test) #evaluating the model on test dataset base
confusionMatrix(evaluate_linear_1, test$Digit)

#Accuracy=0.9203%
#Sensitivity ranges from 0.848 to 0.985
#Specificity ranges from 0.984 to 0.996

#using Linear Kernel and C as 10
model_linear_10<- ksvm(Digit~., data=train, scale=FALSE, kernel="vanilladot", C=10)
evaluate_linear_10<- predict(model_linear_10, test)
confusionMatrix(evaluate_linear_10, test$Digit)

#Accuracy=0.918% Accuracy dips a bit
#Sensitivity ranges from 0.8439 to 0.985
#Specificity ranges from 0.9835 to 0.9962
#the stats remain almost same, model might be overfitting


#-------------------Hyperparameter tuning and Cross Validation------------#

#By using traincontrol function, we Control the computational nuances of the train function
#C=5,using 5 fold and method as cross validation

trainControl <- trainControl(method="cv", number=5)

# making a grid of C values.
grid <- expand.grid(C=seq(1, 5, by=1))

fit.svm <- train(Digit~., data=train, method="svmLinear", metric="Accuracy", 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)
#Accuracy is best at .9146 at C=1

plot(fit.svm)

#using hyperparameter C, we dont get a clear picture.
#We need to check with non-linear kernel

#----------Using Radial Basis Function Kernel--------#
model_RBF<-ksvm(Digit~ ., data = train, scale = FALSE, kernel = "rbfdot")
evaluate_RBF<- predict(model_RBF, test)
confusionMatrix(evaluate_RBF, test$Digit)
#using RBF kernel, accuracy shoots upto .9554

# Making grid of "sigma" and C values. 
grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))

# Performing 5-fold cross validation
fit.svm_radial <- train(Digit~., data=train, method="svmRadial", metric="Accuracy", 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm_radial)
# Best tune at sigma = 0.03 & C=3 with accuracy of 0.9687776

plot(fit.svm_radial)
# Plotting model results

# Validating the model results on test data
evaluate_non_linear_rbf<- predict(fit.svm_radial, test)
confusionMatrix(evaluate_non_linear_rbf, test$Digit)

#Accuracy : 0.9682
#Highest Sensitivity : 0.9903 for Class : 1
#Highest Specificity : 0.9983 for Class : 1

test$Predict<-cbind(test, evaluate_non_linear_rbf) #predicting the test dataset as per our model.