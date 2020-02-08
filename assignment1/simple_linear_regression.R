# A simple linear regression example
# ----------------------------------

# Introduction
# ------------

# Linear model.
linearmodel1 <- function(w, x) {
  w * x
}

# Error function
sqloss <- function(w, x, y) {
  sum((linearmodel1(w, x) - y)^2)
}

# Generate and plot data.
x <- 1:10
y <- 3 * x
plot(x, y)

# Since y depends linearly on x, this should be easy to represent as a linear model; let's try it.
w0 <- 0
w1 <- 3
abline(w0, w1, col="red")

# We would expect the loss to be null with such a model.
sqloss(w1, x, y)

# Generate some other values for the parameter.
w1.range <- seq(-2, 5, by=0.1)

# Compute the loss values for all parameter values in the specified range.
loss.values <- sapply(w1.range, sqloss, x, y)

# Plot the result.
plot(w1.range, loss.values, col="green", type="l")

# To find the optimal parameter value, optimize along w1.
optimize(f=sqloss, w1.range, x, y)

# Exercises
# ---------

# Add a (seeded) random fluctuation (uniformly distributed) and find the optimal parameter value
# and the value of the squared loss with the optimal value.
y <- 3 * x
set.seed(7)
y <- y + runif(10, min=-1, max=1)

optimize(sqloss, w1.range, x, y)