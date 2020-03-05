# Decision Tree Learning: learning curve
# --------------------------------------

# CART
# ----

# Use libraries.
library(rpart)

# Load data into session.
bhtrain <- read.csv("assignment2/BostonHouseTrain.csv")
test <- read.csv("assignment2/BostonHouseTest.csv")

# Formula used to compute the relation.
formula <- class ~ . - X

# Set rpart options.
option <- rpart.control(minsplit=1, cp=0)

# Question 1
# ----------
n <- 25
round <- c(rep(1:n, 5))
tsz <- c()
nb_nodes <- c()
acc <- c()
for (sz in c(5, 10, 20, 50, 99)) {
  for (i in 1:n) {
    train <- bhtrain[sample(nrow(bhtrain), sz*nrow(bhtrain)/100), ]
    tsz <- c(tsz, sz*nrow(bhtrain)/100)
    fit <- rpart(formula, data=train, control=option)
    nb_nodes <- c(nb_nodes, max(fit$where))
    table_test <- table(test$class, predict(fit, test, type='class'))
    acc <- c(acc, sum(diag(table_test))/sum(table_test))
  }
}

student_frame <- data.frame(round, tsz, nb_nodes, acc)
names(student_frame) <- c("Round", "Train size", "Num nodes", "Test acc")
save(student_frame, file="assignment2/A2_3_1.RData")

# Question 2
# ----------
boxplot(nb_nodes[1:n], nb_nodes[n+1:2*n], nb_nodes[2*n+1:3*n], nb_nodes[3*n+1:4*n], nb_nodes[4*n+1:5*n])
boxplot(acc[1:n], acc[n+1:2*n], acc[2*n+1:3*n], acc[3*n+1:4*n], acc[4*n+1:5*n])


# - The variance of the test acc1uracy decreases with the training set size
#   because the overlap between random data split increases.
# - The number of nodes of the trees increases with the training set size
#   because more splits are needed to classify all training examples.
# - The test accuracy increases first rapidly with the training set size, then reaches a plateau.

# Question 3
# ----------
cp <- 0.01
pruned <- rpart.control(minsplit=1, cp=cp)

n <- 25
round <- c(rep(1:n, 5))
tsz <- c()
nb_nodes1 <- c()
nb_nodes2 <- c()
acc1 <- c()
acc2 <- c()
for (sz in c(5, 10, 20, 50, 99)) {
  for (i in 1:n) {
    train <- bhtrain[sample(nrow(bhtrain), sz*nrow(bhtrain)/100), ]
    tsz <- c(tsz, sz*nrow(bhtrain)/100)
    ffit <- rpart(formula, data=train, control=option)
    pfit <- rpart(formula, data=train, control=pruned)
    nb_nodes1 <- c(nb_nodes1, max(ffit$where))
    nb_nodes2 <- c(nb_nodes2, max(pfit$where))
    ftable_test <- table(test$class, predict(ffit, test, type='class'))
    ptable_test <- table(test$class, predict(pfit, test, type='class'))
    acc1 <- c(acc1, sum(diag(ftable_test))/sum(ftable_test))
    acc2 <- c(acc2, sum(diag(ptable_test))/sum(ptable_test))
  }
}

student_frame <- data.frame(round, tsz, nb_nodes2, acc2)
names(student_frame) <- c("Round", "Train size", "Num nodes", "Test acc")
save(file="assignment2/A2_3_3.RData", student_frame, cp)

# Question 4
# ----------

boxplot(nb_nodes1[1:n], nb_nodes1[n+1:2*n], nb_nodes1[2*n+1:3*n], nb_nodes1[3*n+1:4*n], nb_nodes1[4*n+1:5*n], nb_nodes2[1:n], nb_nodes2[n+1:2*n], nb_nodes2[2*n+1:3*n], nb_nodes2[3*n+1:4*n], nb_nodes2[4*n+1:5*n])
boxplot(acc1[1:n], acc1[n+1:2*n], acc1[2*n+1:3*n], acc1[3*n+1:4*n], acc1[4*n+1:5*n], acc2[1:n], acc2[n+1:2*n], acc2[2*n+1:3*n], acc2[3*n+1:4*n], acc2[4*n+1:5*n])

# - The number of nodes of pruned trees first increases with the number of observations,
#   then stops increasing.
# - Pruning the tree prevents overfitting.
# - An adequatly pruned tree will have a higher test accuracy.
# - Pruning the tree is mostly useful for large number of training examples.

# Question 5
# ----------

fit <- rpart(formula, data=bhtrain, control=pruned)
plot(fit)
text(fit)
print(fit)
printcp(fit)

# The most important three are lstat, rm, and crim.