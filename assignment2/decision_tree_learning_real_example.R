# Decision Tree Learning: a real-life example
# -------------------------------------------

# CART
# ----

# Use libraries.
library(rpart)

# Load data into session.
train <- read.csv("assignment2/BostonHouseTrain.csv")
test <- read.csv("assignment2/BostonHouseTest.csv")

# Formula used to compute the relation.
formula <- class ~ . - X

# Set rpart options.
option <- rpart.control(minsplit=1, cp=0)

# Construct the decision tree.
fit <- rpart(formula, data=train, control=option)

# Output the fit.
print(fit)
plot(fit)
text(fit)
printcp(fit)

# Question 1
# ----------

# Compute number of nodes.
nb_nodes <- max(fit$where)

# Compute confusion matrices
table_train <- table(train$class, predict(fit, train, type='class'))
table_test <- table(test$class, predict(fit, test, type='class'))

acc_train <- sum(diag(table_train))/sum(table_train)
acc_test <- sum(diag(table_test))/sum(table_test)

sprintf(fmt = "%g, %g, %g", nb_nodes, acc_train, acc_test)

# Question 2
# ----------

# Yes, if the data is consistent.

# Question 3
# ----------
n <- 100
acc_train <- 0
acc_test <- 0
nb_nodes <- 0
for (i in 1:n) {
  train25 <- train[sample(nrow(train), nrow(train)/4), ]
  fit <- rpart(formula, data=train25, control=option)
  nb_nodes <- nb_nodes + max(fit$where)
  table_train <- table(train25$class, predict(fit, train25, type='class'))
  table_test <- table(test$class, predict(fit, test, type='class'))
  acc_train <- acc_train + sum(diag(table_train))/sum(table_train)
  acc_test <- acc_test + sum(diag(table_test))/sum(table_test)
}

sprintf(fmt = "%g, %g, %g", nb_nodes/n, acc_train/n, acc_test/n)