# Decision Tree Learning: bagging
# -------------------------------

# CART
# ----

# Use libraries.
library(rpart)
library(ipred)
library(doParallel)
library(foreach) 
library(pracma)

# Load data into session.
bhtrain <- read.csv("assignment2/BostonHouseTrain.csv")
test <- read.csv("assignment2/BostonHouseTest.csv")

# Formula used to compute the relation.
formula <- class ~ . - X

# Set rpart options.
option <- rpart.control(minsplit=-1, cp=-1, maxdepth=2)

# Construct the decision tree.
fit <- rpart(formula, data=bhtrain, control=option)

# Question 1
# ----------

table_test <- table(test$class, predict(fit, test, type="class"))
acc_test <- sum(diag(table_test))/sum(table_test)
print(acc_test)

# Question 2
# ----------
n <- 10
acc <- 0
ntrees <- 100
for (i in 1:n) {
  print(i)
  
  # Create a parallel socket cluster
  cl <- makeCluster(8) # use 8 workers
  registerDoParallel(cl) # register the parallel backend
  
  # Fit trees in parallel and compute predictions on the test set
  predictions <- foreach(
    icount(ntrees), 
    .packages = "rpart", 
    .combine = cbind
  ) %dopar% {
    # bootstrap copy of training data
    index <- sample(nrow(bhtrain), replace = TRUE)
    bhtrain_boot <- bhtrain[index, ]  
    
    # fit tree to bootstrap copy
    bagged_tree <- rpart(
      formula, 
      control = option,
      data = bhtrain_boot
    ) 
    
    predict(bagged_tree, newdata = test, type="class")
  }
  
  stopCluster(cl)
  
  pred <- c()
  for (j in 1:nrow(predictions)) {
    pred <- c(pred, Mode(predictions[j, ]))
  }
  
  table_test <- table(test$class, pred)
  acc <- acc + sum(diag(table_test))/sum(table_test)
}

print(acc / n)

# Question 3
# ----------
acc_list <- c()
ntreeslist <- c(10, 20, 50)
for (ntrees in ntreeslist) {
  acc_tmp <- 0
  for (i in 1:n) {
    # Create a parallel socket cluster
    cl <- makeCluster(8) # use 8 workers
    registerDoParallel(cl) # register the parallel backend
    
    # Fit trees in parallel and compute predictions on the test set
    predictions <- foreach(
      icount(ntrees), 
      .packages = "rpart", 
      .combine = cbind
    ) %dopar% {
      # bootstrap copy of training data
      index <- sample(nrow(bhtrain), replace = TRUE)
      bhtrain_boot <- bhtrain[index, ]  
      
      # fit tree to bootstrap copy
      bagged_tree <- rpart(
        formula, 
        control = option,
        data = bhtrain_boot
      ) 
      
      predict(bagged_tree, newdata = test, type="class")
    }
    
    stopCluster(cl)
    
    pred <- c()
    for (j in 1:nrow(predictions)) {
      pred <- c(pred, Mode(predictions[j, ]))
    }
    
    table_test <- table(test$class, pred)
    acc_tmp <- acc_tmp + sum(diag(table_test))/sum(table_test)
  }
  acc_list <- c(acc_list, acc_tmp / n)
}
plot(c(ntreeslist, 100), c(acc_list, acc / n))

# - At some point, the accuracy reaches a plateau, so there is no real benefit in accuracy when adding more trees.
# - Compared to our weak learner (a single tree), using bagging drastically improves the accuracy,
#   even with only a few (e.g. 10) trees.
# - Using bagging, with an appropriate number of trees, will yield better results
#   than using a single tree (like you used in A2.4).