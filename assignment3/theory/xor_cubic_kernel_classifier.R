# XOR classifier with cubic kernel trick.

library(e1071)

x3 <- c(0, 0, 1, 1)
threex2y <- c(0, 0, 0, 1.73205081)
threexy2 <- c(0, 0, 0, 1.73205081)
y3 <- c(0, 1, 0, 1)
xor <- c(0, 1, 1, 0)

dat <- data.frame(x3, threex2y, threexy2, y3, xor)
cl <- svm(xor ~ ., data=dat, kernel='linear')
t <- table(dat$xor, round(predict(cl, dat, type='xor')))
sprintf("Prediction accuracy for XOR = %g", sum(diag(t))/sum(t))