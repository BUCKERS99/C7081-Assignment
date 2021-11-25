#### Header ####
## Statistical analysis for data science
## Assignment code for C7081
## Harry Buckley - 17239400
## last edited: 25/11/21


# CONTENTS ####
## 0.0 Setup ####
# working directory
harperwd <- "G:/MSc/Stat analysis" 
homewd <- "C:/Users/harry/Desktop/MSc/CORE/Statistical Analysis for Data Science/Statistical Analysis for Data Science Assignment"
githubwd <- "C:/Users/harry/Documents/MSc/Git Repository/C7081-Assignment"
setwd(githubwd)
getwd()
set.seed(99)

# Set Librarys
c(install.packages("readxl"),
  install.packages("ggplot2"),
  install.packages("visreg"),
  install.packages("tree"),
  install.packages("boot"),
  install.packages("MASS"),
  install.packages("leaps"),
  install.packages("rstanarm"),
  install.packages("caret"),
  install.packages("randomForest"),
  install.packages("car"),
  install.packages("lmtest"),
  install.packages("dplyr"),
  install.packages("kableExtra"))

c(library(readxl),
  library(ggplot2),
  library(visreg),
  library(tree),
  library(boot),
  library(MASS),
  library(leaps),
  library(rstan),
  library(caret),
  library(randomForest),
  library(rstanarm),
  library(car),
  library(lmtest),
  library(dplyr),
  library(kableExtra))


  require(readxl)
  require(ggplot2)
  require(visreg)
  require(tree)
  require(boot)
  require(MASS)
  require(leaps)
  require(rstan)
  require(caret)
  require(randomForest)
  require(rstanarm)
  require(car)
  require(lmtest)
  require(dplyr)

## 0.1 Loading data and convert to factors ####
lol_orig <- read_xlsx("high_diamond_ranked_10min_tidy.xlsx")
lol <- read_xlsx("high_diamond_ranked_10min_tidy.xlsx")

# turn the data into a factor that has the levels 1 or 0 as yes or no
lol$b_wins <- as.factor(lol$b_wins)
lol$b_fir_blo <- as.factor(lol$b_fir_blo)
lol$r_fir_blo <- as.factor(lol$r_fir_blo)

## 0.2 Collinearity and removal of variables ####

# looking at the correlation
cor_lol <- cor(lol_orig)
print(cor_lol)

# getting the column names that have a correlation of 1
colnames(lol_orig)[colSums(abs(cor_lol)==1)==1]

# these need removing to prevent collinearity in the model
removal_columns <- grep(pattern = paste0( 
  c("b_fir_blo", "b_d", "r_d", "game_id", "b_tot_mons", "r_tot_mons", 
    "b_gp_min", "r_gp_min" ), 
  collapse = "|"), colnames(lol))

lol_rem <- lol[, -removal_columns]

# checking that the variable have been removed
colnames(lol)
colnames(lol_rem)


## 0.3 Equal classes for the regression ####
# create a table to check the class length
y <- as.numeric(lol_orig$b_wins)
x <- lol_orig[, -2]

table(y) # Class lengths are roughly equal

## 0.4 test and train data ####

set.seed(99)

tr_te_split <- sample(1:9879, size = 7902)

train_lol <- lol_rem[tr_te_split, ]

test_lol <- lol_rem[ -tr_te_split, ]

## 1.0 binary logistical model ####

# Logistical regression of train data
log_reg_train <- glm(formula = b_wins ~.,
               family = binomial,
               data = train_lol)

log_summ <- summary(log_reg_train)
print(log_summ)


# plotting prediction that blue team wins for test and train data
predicted.data <- data.frame(prob.of.win = log_reg_train$fitted.values,
                             win=test_lol$b_wins)
predicted.data <- predicted.data[
  order(predicted.data$prob.of.win, decreasing = F), ]
predicted.data$rank <- 1:nrow(predicted.data)

ggplot(data = predicted.data, aes (x = rank, y = prob.of.win)) +
  geom_point(position = "jitter", aes(color = win), alpha = 2) +
  xlab("Index") +
  ylab("Prediction that blue team wins")

# show summary

log_summ <- summary(log_reg)
print(log_summ)

hist(log_reg_train$fitted.values)

# make predicitions using this model 
log_reg_preds <- predict.glm(log_reg_train, test_lol)

print(log_reg_preds)

# calculate the MSE and RMSE
MSE_log_reg <- sum(log_reg_train$residuals^2) / log_reg_train$df.residual
print(MSE_log_reg)

#RMSE for Log Reg
RMSE_log_reg <- sqrt(MSE_log_reg)
print(RMSE_log_reg)

log_reg_accu <- print(mean(round(log_reg_preds)==test_lol[,1]))

## 1.1 Taking the significant variables forward ####

# highlight the significant variables
sigs <- rownames(log_summ$coefficients)[which (log_summ$coefficients [, 4] <1e-1)]
print(sigs)

# use these for the next GLM
glm_sig <- log_reg <- glm(formula = b_wins ~  b_tot_gp + r_tot_gp + b_e_mon +
                            b_tot_xp + b_cs_min + r_a + r_e_mon + r_tot_xp, 
                          family = binomial,
                          data = train_lol)
glm_sig_summ <- summary(glm_sig)
print(glm_sig_summ)
# make predictions

glm_sig_preds <- predict(glm_sig, test_lol)
glm_sig_preds 

# Get MSE and RMSE

MSE_glm_sig <- sum(glm_sig$residuals^2) / glm_sig$df.residual
print(MSE_glm_sig)

#RMSE for glm sig
RMSE_glm_sig <- sqrt(MSE_glm_sig)
print(RMSE_glm_sig)

# histogram of the fitted values
hist(glm_sig$fitted.values)

# validating the models for binomial GLMs

# 1.get the devience residuals and look at distribution
devresid <- resid(glm_sig, type = "deviance")
hist(devresid)
# this is clearly binomial although residuals with an output of over 2 may not fit 
# the model

# 2. check the variance of the residuals
predicted.data <- data.frame(prob.of.win = log_reg$residuals,
                             win=train_lol$b_wins)
predicted.data <- predicted.data[
  order(predicted.data$prob.of.win, decreasing = F), ]
predicted.data$rank <- 1:nrow(predicted.data)

ggplot(data = predicted.data, aes (x = rank, y = prob.of.win)) +
  geom_point(aes(color = win), alpha = 1, shape = 4, stroke = 1) +
  xlab("Index") +
  ylab("Prediction that blue team wins") +
  geom_hline(mapping = NULL, data = NULL, yintercept = 2*MSE_glm_sig) +
  geom_hline(mapping = NULL, data = NULL, yintercept = -2*MSE_glm_sig) 
  
# Majority of data falls between the standard error lines

# 3. Examining the cooks distance
par(mfrow = c(2,2))
plot(glm_sig)
# there are very few outliers

# accuracy of the model
glm_accu <- print(mean(round(glm_sig_preds)==test_lol[,1]))

par(mfrow= c(1,1))


## 2.0 Subset selection ####

non_sig_subset <- regsubsets(b_wins ~. ,
                             data = train_lol)
summary(non_sig_subset)


# Create the loop for finding validaiton error
train = sample(c(T,F), nrow(lol_rem), rep = T)
test = (!train)

test.mat = model.matrix(b_wins~., data = lol_rem[test,])
sub.val.errors = rep(NA,8)
for (i in 1:8) {
  coefi = coef(non_sig_subset, id = i)
  pred = test.mat[,names(coefi)] %*% coefi
  sub.val.errors[i] = mean((as.numeric(lol_rem$b_wins[test]) - pred)^2)
}


# this shows that the best model contains 8 variables
which.min(sub.val.errors)
coef(regfit.best, which.min(sub.val.errors))


# plot the errors
par(mfrow = c(2,2))
plot(sub.val.errors, type = "b",
     main = "Validation error in best subset",
     xlab = "Number of variables in model",
     ylab = "Validation Error" )
# plot the models to see which is best
plot(non_sig_subset$rss, xlab = "Number of variables",
     ylab = "RSS" )

plot(non_sig_subset, scale = "adjr2")

# shows that the model with the least variables and smallest RSS trade off


## 3.0 Stepwise selection ####

# 3.1 forward stepwise selection
forw_step_train <- glm(formula = b_wins ~. ,
                      family = binomial,
                      data = train_lol)
summary(forw_step_train)


forw_step <- step(forw_step_train, direction = "forward")

summary(forw_step)
names(forw_step)
plot(forw_step$residuals)

# Validation error
set.seed(99)
train = sample(c(T,F), nrow(lol_rem), rep = T)
test = (!train)

regfit.best <- regsubsets(b_wins ~., data = lol_rem[train,],
                          nvmax = 19)
test.mat = model.matrix(b_wins~., data = lol_rem[test,])
val.errors = rep(NA,19)
for (i in 1:19) {
  coefi = coef(regfit.best, id = i)
  pred = test.mat[,names(coefi)] %*% coefi
  val.errors[i] = mean((as.numeric(lol_rem$b_wins[test]) - pred)^2)
}
val.errors
which.min(val.errors)



# this shows that the best model contains 13 variables
coef(regfit.best,which.min(val.errors))

# plot the errors
plot(val.errors, type = "b",
     main = "Validation error in forward stepwise",
     xlab = "Number of variables in model",
     ylab = "Validation Error" )

# predictions of forward stepwise

forw_pred <- predict(forw_step, test_lol, type = "response", se.fit =T)


# accuracy of forward stepwise model
forw_accu <- print(mean(round(forw_pred$fit)==test_lol[,1]))


# 3.2 Backwards stepwise selection

backw_step_train <- glm(formula = b_wins ~. ,
                       family = binomial,
                       data = train_lol)
summary(backw_step_train)


backw_step <- step(backw_step_train, direction = "backward")

summary(backw_step)
names(backw_step)
plot(backw_step$residuals)

# Validation error
set.seed(99)
train = sample(c(T,F), nrow(lol_rem), rep = T)
test = (!train)

bkw.val.errors = rep(NA,25)
for (i in 1:25) {
  coefi = coef(backw_step, id = i)
  pred = test.mat[,names(coefi)] %*% coefi
  bkw.val.errors[i] = mean((as.numeric(lol_rem$b_wins[test]) - pred)^2)
}
bkw.val.errors
which.min(bkw.val.errors)

# plot the errors
plot(bkw.val.errors, type = "b",
     main = "Validation error in backwards stepwise",
     xlab = "Number of variables in model",
     ylab = "Validation Error" )
# weird results here

# predictions of backwards stepwise

backw_pred <- predict(backw_step, test_lol, type = "response", se.fit =T)

#Accuracy of backwards stepwise model
backw_accu <- print(mean(round(backw_pred$fit)==test_lol[,1]))

## 4.0 K-Nearest Neighbour ####
ctrl  <- trainControl(method  = "cv",number  = 10) # 10-fold cross-validation

fit.cv <- train(b_wins ~ ., data = train_lol, method = "knn", # k nearest neighbors
                trControl = ctrl,  # Add the control
                preProcess = c("center","scale"), 
                tuneGrid =data.frame(k=seq(5,100,by=10))) 


pred <- predict(fit.cv,test_lol) # predict the output classes
pred.prob <- predict(fit.cv,test_lol, type = "prob") # predict the output probability

print(fit.cv) # Plot the results. See at the end the chosen value of "k"
print(pred.prob) # print the probability results
plot(pred.prob) # Plot the probability results
plot(fit.cv) # Plot the Cross-validation output


# Accuracy of KNN
print(fit.cv$results)
print(fit.cv$bestTune)
KNN_accu <- fit.cv$results[rownames(fit.cv$bestTune),2]

## 5.0 Decision Tree ####

# using a classification tree to analyse the data
tree.lol <- tree(b_wins ~., data = train_lol)
summary(tree.lol)
# misclassification rate is 32%


# blue and red total gold are used as variables
plot(tree.lol)
text(tree.lol,
     pretty = 0)

# the most important indicator is the blue total gold 
print(tree.lol)
cv_tree <- cv.tree(tree.lol)
plot(cv_tree$size, cv_tree$dev, type = "b")

# see if we can improve the tree
lol.tree.prune <- prune.tree(tree.lol, best = 4)
?prune.tree
plot(lol.tree.prune)
text(lol.tree.prune,
     pretty = 0)
summary(lol.tree.prune)
# less nodes but same error rate and higher mean deviance

# applying the predict function
prune.tree.predict <- predict(lol.tree.prune, test_lol)

# accuracy of model
tree_accu <- print(mean(round(prune.tree.predict)==test_lol[,1]))

# accuracy of 32% means that this model is not accurate
## 6.0 Random Forest ####

# as m= sqrt(p) for classification problems we will use m = 4.47 
set.seed(99)
bag.lol <- randomForest(b_wins ~., data = train_lol, mtry = 4,
                        imprtance = T)
print(bag.lol)

yhat.bag <- predict(bag.lol, test_lol, type = "prob")
plot(yhat.bag)

bag.lol.MSE <- mean((yhat.bag-as.numeric(test))^2)
names(yhat.bag)
importance(bag.lol)

# Accuracy of random forest
rndm_accu <- print(mean(round(yhat.bag)==test_lol[,1]))




## 7.0 Results ####
Accuracy_of_predictions <-c(log_reg_accu,
                            glm_accu,
                            forw_accu,
                            backw_accu,
                            KNN_accu,
                            tree_accu,
                            rndm_accu) 

Accuracy_names <- c("logistic regression accuracy",
                    "significant variables model accuracy",
                    "forward stepwise selection accuracy",
                    "backwards stepwise selection accuracy",
                    "K-nearest neighbour accuracy",
                    "decision tree accuracy",
                    "random forrest accuracy")
table_of_accuracy <- cbind(Accuracy_names, Accuracy_of_predictions)
table_of_accuracy

# make a data frame
table.accu <- data.frame(
  name=Accuracy_names,
  value = Accuracy_of_predictions)

# arrange the data frame into a decending order of accuracy
table.accu <- arrange(table.accu, desc(value))

table.accu$name <- factor(table.accu$name, levels = c("K-nearest neighbour accuracy",
                          "forward stepwise selection accuracy",
                          "backwards stepwise selection accuracy",
                          "decision tree accuracy",
                          "logistic regression accuracy",
                          "significant variables model accuracy",
                          "random forrest accuracy"))
                          
# Plot the model accuracy
ggplot(table.accu, aes(x = name, y = value*100, fill=name)) +
  geom_bar(stat = "identity", width = 0.5 ) +
  scale_fill_hue(c = 50) +
  ggtitle("Accuracy of model predictions")+
  theme(plot.title = element_text(hjust = 0.5, vjust = 1, face = "bold"), 
        legend.position = "none")+
  xlab("") +
  ylab("Percentage accuracy")+
  coord_flip()


