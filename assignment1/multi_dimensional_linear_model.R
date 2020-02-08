# A multi-dimensional linear model
# --------------------------------

# Introduction
# ------------

# Load data into session.
bodyfat <- read.csv("bodyfat.csv", row.names=1)

# Looking at this data frame can be done as follows.
bodyfat

# One can look at the dimensions of the data and their names as follows.
dim(bodyfat)
dimnames(bodyfat)

# Look at the data in pairs, and eventually restrict the visualization to specific variables.
pairs(bodyfat)
pairs(bodyfat[, c("BFI", "Weight", "Height")])

# An alternative plot relies on a lattice plot.
library (lattice)
xyplot(BFI ~ Weight | Height, data=bodyfat)

# Let's try to predict the BFI as a linear function of the Height and Weight of a person.
linearmodel <- lm(BFI ~ Height + Weight, data=bodyfat)
w <- linearmodel$coefficients

# The computation of predicted values of such a model can be encoded as follows.
predict <- function(x, w) {
  intercept <- w[1]
  intercept + as.matrix(x) %*% w[2:length(w)]
}

# This can be used as follows.
x <- bodyfat[, c("Height", "Weight")]
predicted <- predict(x, w)

# We can compare the predicted values to the real BFI values.
plot(bodyfat[, "BFI"], predicted, xlab="Actual BFI", ylab="Predicted BFI")
abline(0, 1, col="blue")

# Exercises
# ---------

# Write a simple function to compute the associated squared loss.
squaredloss <- function (y, predicted) {
  sum((predicted - y)^2)
}

y <- bodyfat[, "BFI"]
squaredloss(y, predicted)

# Propose a model estimating the linear dependence between
# the body fat index (the BFI variable) and the 3 following covariates:
# Weight, Abdomen, Biceps denoting respectively
# the weight, abdomen circumference and the biceps circumference of the person,
# and give its coefficients.
model <- lm(BFI ~ Weight + Abdomen + Biceps, data=bodyfat)
w <- model$coefficients
x <- bodyfat[, c("Weight", "Abdomen", "Biceps")]
predicted <- predict(x, w)
w
squaredloss(y, predicted)

# Model choice.
# The second model is better because
# it predicts values closer to the actual ones, and hence has a lower loss.

# Model coefficients analysis
# All other things being equal, BFI increases positively with the Biceps circumference,
# which means that a person with larger Biceps will have a higher BFI.
# All other things being equal, BFI decreases when the Weight of the person increases.

# Splitting the data in half.
bodyfat_split <- split(bodyfat, sample(rep(1:2, dim(bodyfat)[1]/2)))
training <- data.frame(bodyfat_split[1])
test <- data.frame(bodyfat_split[2])

# Repeat multiple times and evaluate average loss.
n <- 100
sum_training <- 0
sum_test <- 0
for (i in 1:n) {
  bodyfat_split <- split(bodyfat, sample(rep(1:2, dim(bodyfat)[1]/2)))
  training <- data.frame(bodyfat_split[1])
  test <- data.frame(bodyfat_split[2])
  
  model <- lm(X1.BFI ~ X1.Weight + X1.Abdomen + X1.Biceps, data=training)
  w <- model$coefficients
  x_training <- training[, c("X1.Weight", "X1.Abdomen", "X1.Biceps")]
  y_training <- training[, "X1.BFI"]
  x_test <- test[, c("X2.Weight", "X2.Abdomen", "X2.Biceps")]
  y_test <- test[, "X2.BFI"]
  predicted_training <- predict(x_training, w)
  predicted_test <- predict(x_test, w)
  sum_training <- sum_training + squaredloss(y_training, predicted_training)
  sum_test <- sum_test + squaredloss(y_test, predicted_test)
}
sum_training / n / (dim(bodyfat)[1]/2) # per_sample_train_loss
sum_test / n / (dim(bodyfat)[1]/2) # per_sample_test_loss

# Compare to loss when training = test
model <- lm(BFI ~ Weight + Abdomen + Biceps, data=bodyfat)
w <- model$coefficients
x <- bodyfat[, c("Weight", "Abdomen", "Biceps")]
y <- bodyfat[, "BFI"]
predicted <- predict(x, w)
squaredloss(y, predicted) / dim(bodyfat)[1]

# Optimistic bias
# The optimistic bias is confirmed because
# the per sample test loss is higher than the per sample train loss.