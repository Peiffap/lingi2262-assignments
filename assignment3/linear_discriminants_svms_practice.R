# Linear discriminants and SVMs: practice
# ---------------------------------------

# Exploring the meta-parameters of an SVM
# ---------------------------------------
library(e1071)
library(tcltk)

# Load data into session.
train <- read.csv("assignment3/Letters/LettersTrain.csv")
test <- read.csv("assignment3/Letters/LettersTest.csv")
valid <- read.csv("assignment3/Letters/LettersValid.csv")

# Load indices.
load("assignment3/selected_indices.RData")

# Compute various classifiers.
# Parameters: kernel, cost, degree, gamma.
formula = labels ~ . - X

compute_acc <- function(classifier, set) {
  t <- table(set$labels, predict(classifier, set, type='labels'))
  sum(diag(t))/sum(t)
}

# Q1
# --

r <- c()
sizes <- c()
train_acc <- c()
valid_acc <- c()
ker <- c()
co <- c()
degr <- c()
gam <- c()

defaultco <- 1
defaultdegr <- 3
defaultgam <- 1/617

# Progress bar.
pb <- tkProgressBar(title = "SVM progress bar", label = "",
              min = 0, max = 480, initial = 0, width = 300)
ctr <- 0
for (i in 1:10) {
  for (size in c(95, 189, 473, 710, 851, 937)) {
    r <- c(r, rep(i, 8))
    sizes <- c(sizes, rep(size, 8))
    ker <- c(ker, rep("linear", 3), rep("polynomial", 2), rep("radial", 2), "sigmoid")
    co <- c(co, 0.1, 10, 1000, rep(defaultco, 5))
    degr <- c(degr, rep(defaultdegr, 3), 3, 9, rep(defaultdegr, 3))
    gam <- c(gam, rep(defaultgam, 6), 0.1, defaultgam)
    
    # Define training set.
    tset <- train[selected_indices[[i]][1:size], ]
    
    # Linear kernel tests.
    for (C in c(0.1, 10, 1000)) {
      classifier <- svm(formula, data=tset, kernel="linear", cost=C)
      train_acc <- c(train_acc, compute_acc(classifier, tset))
      valid_acc <- c(valid_acc, compute_acc(classifier, valid))
      ctr <- ctr + 1
      setTkProgressBar(pb, value=ctr, title = NULL, label = NULL)
    }
    
    # Polynomial kernel tests.
    for (deg in c(3, 9)) {
      classifier <- svm(formula, data=tset, kernel="polynomial", degree=deg)
      train_acc <- c(train_acc, compute_acc(classifier, tset))
      valid_acc <- c(valid_acc, compute_acc(classifier, valid))
      ctr <- ctr + 1
      setTkProgressBar(pb, value=ctr, title = NULL, label = NULL)
    }
    
    # Radial basis kernel tests.
    for (g in c(defaultgam, 0.1)) {
      classifier <- svm(formula, data=tset, kernel="radial", gamma=g)
      train_acc <- c(train_acc, compute_acc(classifier, tset))
      valid_acc <- c(valid_acc, compute_acc(classifier, valid))
      ctr <- ctr + 1
      setTkProgressBar(pb, value=ctr, title = NULL, label = NULL)
    }
    
    # Sigmoid kernel tests.
    classifier <- svm(formula, data=tset, kernel="sigmoid")
    train_acc <- c(train_acc, compute_acc(classifier, tset))
    valid_acc <- c(valid_acc, compute_acc(classifier, valid))
    ctr <- ctr + 1
    setTkProgressBar(pb, value=ctr, title = NULL, label = NULL)
  }
}

setTkProgressBar(pb, value=ctr, title = NULL, label = NULL)

# Save data frame.
student_frame <- data.frame(r, sizes, train_acc, valid_acc, ker, co, degr, gam)
names(student_frame) <- c("Round", "Train size", "Train acc", "Valid acc", "kernel", "cost", "degree", "gamma")
save(student_frame, file="assignment3/A3_2_1.RData")

close(pb)

# Q2
# --

linclass = svm(formula, data=train, kernel="linear", C = 1000)
acc = compute_acc(linclass, test)

# - A linear kernel seems like a good kernel choice.
# - Hard margin hyperplanes are learnt when the C parameter of the primal problem tends to infinity.
# - An RBF kernel with C = 1 and gamma = 1 is roughly equivalent to random guessing.
# - The validation set accuracy increases with the number of observations.
# - Choosing the meta-parameters that maximize the accuracy on the test set causes an optimistic bias.
#   That's why a validation set is used.
# - The accuracy on our specific test set is worse than the one on the validation set
#   for most of the meta-parameter choices.
#   It simply reflects that the data distribution is harder on the test set.

# Q3
# --

linclass = svm(formula, data=train, kernel="linear")
lintrain_acc = compute_acc(linclass, train)
lintest_acc = compute_acc(linclass, test)

rbfclass = svm(formula, data=train, kernel="radial", gamma = 1)
rbftrain_acc = compute_acc(rbfclass, train)
rbftest_acc = compute_acc(rbfclass, test)

sprintf(fmt = "%d, %g, %g, %d, %g, %g", linclass$tot.nSV, lintrain_acc, lintest_acc, rbfclass$tot.nSV, rbftrain_acc, rbftest_acc)

# Q4
# --

# For the second set of parameters (RBF kernel), each training point is essentially
# only similar to itself and dissimilar from all other points.
# Thus, the number of support vectors is maximal and generalization is impossible.