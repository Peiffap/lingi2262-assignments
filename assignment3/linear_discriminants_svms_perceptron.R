# Linear discriminants and SVMs: implementing a perceptron
# --------------------------------------------------------

# Perceptron with margin
# ----------------------

load("assignment3/RestrictedLetters.RData")

# Q1
# --

perceptron <- function(x, y, b, w_init, eta, epoch) {
  # Implements a binary single-sample perceptron with margin
  # Inputs:    x a feature matrix containing a sample on each row
  #            y a vector with the class of each sample (either +1 or -1)
  #            b a margin
  #            w_init a vector with the initial weight values (intercept in w_init[1])
  #            eta a fixed learning rate
  #            epoch the maximal number of iterations
  # Precond:   Data is assumed to be homogeneous
  # Output:    A weight vector
  x <- cbind(rep(1, nrow(x)), x)
  for (k in 1:nrow(x)) {
    x[k, ] <- x[k, ] * y[k]
  }
  convergence <- function() {
    for (k in 1:nrow(x)) {
      if (w %*% x[k, ] <= b) {
        return(FALSE)
      }
    }
    return(TRUE)
  }
  k <- 0
  i <- 0
  w <- w_init
  while (i < epoch && !convergence()) {
    k <- (k%%nrow(x)) + 1
    if (w %*% x[k,] <= b) {
      w <- w + eta * x[k, ]
    }
    i <- i + 1
  }
  w
}

# Q2
# --

b = 10
y = c()
for (yk in train$labels) {
  if (yk == "A") {
    y = c(y, 1)
  } else {
    y = c(y, -1)
  }
}
trainmat <- matrix(as.numeric(unlist(train)), nrow=nrow(train))[, 1:10]
weights <- perceptron(trainmat, y, b, 1:ncol(train), 0.1, 20000)

compute_acc <- function(wei, set) {
  mat <- matrix(as.numeric(unlist(set)), nrow=nrow(set))[, 1:10]
  mat <- cbind(rep(1, nrow(mat)), mat)
  ctr <- 0
  for (k in 1:nrow(set)) {
    pred <- sign(wei %*% mat[k, ])
    if (set[k, ]$labels == "A" && pred == 1 || set[k, ]$labels == "E" && pred == -1) {
      ctr <- ctr + 1
    }
  }
  ctr / nrow(set)
}

compute_acc(weights, train)
compute_acc(weights, valid)
compute_acc(weights, test)

# - For the training set, the algorithm doesn't seem to converge (the values of w keep oscillating)
#   even for a high epoch; the training data is therefore probably not linearly separable.
# - With the best choice of metaparameters (evaluated using the validation set),
#   the accuracy of the perceptron on the test set is close to the one obtained with a linear SVM.
# - If the problem is linearly separable, increasing the b / eta ratio
#   should increase the number of iterations before convergence.
# - With a small or zero margin, the final hyperplane is strongly dependant on the values of w_init.
#   This is less true for a larger margin.