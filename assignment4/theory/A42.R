# Q1
# --
phat <- 0.88
n <- 100
zn <- 1.96
sep <- zn * ((phat * (1 - phat))/n)**0.5
sprintf("Q1: confidence interval: [%.3g, %.3g]", phat - sep, phat + sep)

# Q2
# --
phat <- 0.84
n <- 100
zn <- 1.96
sep <- zn * ((phat * (1 - phat))/n)**0.5
sprintf("Q2: confidence interval: [%.3g, %.3g]", phat - sep, phat + sep)

# Q3
# --
# no, since the intervals are not disjoint.

# Q4
# --
n = 100
p1 = 0.88
p2 = 0.84
m1 <- p1*n ; m2 <- p2*n
perf <- data.frame(rbind(c(m1,m2),c(n-m1,n-m2)))
sprintf("Q4: p-value: %.3g", fisher.test(perf)$p.value)

# Q5
# --
# no, since the p-value is far too high.

# Q6
# --
# cf python script, rounding error sucks.

# Q7
# --
# /

# Q8
# --
for (k in c(2, 5, 10, 20, 50, 100)) {
  av = c()
  for (i in 1:100) {
    av = c(av, mean(runif(k, 0, 1)))
  }
  plot(density(av))
  x <- seq(0, 1, length=1000)
  y <- dnorm(x, mean=0.5, sd=1/(12*k)**0.5)
  lines(x, y, type="l", lwd=1)
  readline(prompt="Press enter to plot for next value of k")
}

# Q9
# --
for (k in c(2, 5, 10, 20, 50, 100)) {
  av = c()
  for (i in 1:100) {
    av = c(av, mean(rnorm(k, 0.5, 1/12**0.5)))
  }
  plot(density(av))
  x <- seq(0, 1, length=1000)
  y <- dnorm(x, mean=0.5, sd=1/(12*k)**0.5)
  lines(x, y, type="l", lwd=1)
  readline(prompt="Press enter to plot for next value of k")
}
  
# Q10
# ---
# no, not sufficient, as normal distributed X_i would give the same results. this is because of the CLT.
