# Performance Assessment: practice
# --------------------------------

# Use libraries.
library(rpart)

# Load data into session.
train <- read.csv("assignment4/CollegeTrain.csv")
train$X = NULL
test <- read.csv("assignment4/CollegeTest.csv")
test$X = NULL

# Formula used to compute the relation.
formula <- Outcome ~ .

# Q1
# --

# Set rpart options.
option <- rpart.control(minsplit=1, cp=0)

# Construct the decision tree.
nsamp <- round(0.05 * nrow(train))
train5 = train[sample(nrow(train), nsamp), ]
fit <- rpart(formula, data=train5, control=option)

# Compute test accuracy.
randtest <- test[sample(nrow(test), 100), ]
p <- predict(fit, randtest)
tab <- table(randtest$Outcome, predict(fit, randtest, type='class'))
acc <- sum(diag(tab))/sum(tab)

# Compute confidence interval.
zn <- qnorm(1 - 0.05/2)
n <- 100
sep <- zn * sqrt((acc * (1 - acc))/n)
sprintf("acc, CI lower, CI upper: %.3g, %.3g, %.3g", acc, acc - sep, acc + sep)

# Q2
# --
for (i in 1:100) {
  test100 = test[sample(nrow(test), 100), ]
  # Compute test accuracy.
  p <- predict(fit, test100)
  tab <- table(test100$Outcome, predict(fit, test100, type='class'))
  acc <- c(acc, sum(diag(tab))/sum(tab))
}
sprintf("mean, q2.5, q97.5, %.3g, %.3g, %.3g", mean(acc), quantile(acc, 0.025), quantile(acc, 0.975))

# Q3
# --
test_acc <- c()
q25 <- c()
q975 <- c()
CILO <- c()
CIHI <- c()
avacc <- c()

nsamp <- round(0.05 * nrow(train))
for (j in 1:20) {
  # Construct the decision tree.
  train5 = train[sample(nrow(train), nsamp), ]
  fit <- rpart(formula, data=train5, control=option)
  
  # Compute test accuracy.
  randtest <- test[sample(nrow(test), 100), ]
  p <- predict(fit, randtest)
  tab <- table(randtest$Outcome, predict(fit, randtest, type='class'))
  acc <- (sum(diag(tab))/sum(tab))
  test_acc <- c(test_acc, acc)
  
  # Compute confidence interval.
  zn <- qnorm(1 - 0.05/2)
  n <- 100
  sep <- zn * sqrt((acc * (1 - acc))/n)
  CILO <- c(CILO, acc - sep)
  CIHI <- c(CIHI, acc + sep)
  
  acc <- c()
  for (i in 1:100) {
    test100 = test[sample(nrow(test), 100), ]
    # Compute test accuracy.
    p <- predict(fit, test100)
    tab <- table(test100$Outcome, predict(fit, test100, type='class'))
    acc <- c(acc, sum(diag(tab))/sum(tab))
  }
  avacc <- c(avacc, mean(acc))
  q25 <- c(q25, quantile(acc, 0.025))
  q975 <- c(q975, quantile(acc, 0.975))
}

student_frame <- data.frame(CILO, CIHI, test_acc, q25, q975, avacc)
names(student_frame) <- c("CI.Low", "CI.High", "Test.Acc", "2.5%", "97.5%", "Average.Acc")
save(student_frame, file="assignment4/A4_3_3.RData")

# Q4
# --
# - The average accuracy (over 100 test sets) of a model usually falls inside the 95 % confidence interval
#   computed on a single test set.
# - The columns 2.5 % and 97.5 % provide a tighter confidence interval on the accuracy of the model
#   than the columns CI.Low and CI.high.
# - The column Test.Acc. puts into light the large variability caused by
#   both the sampling of the train and test set.
# - The column Average.Acc. puts into light the large variability caused by the sampling of the train set.