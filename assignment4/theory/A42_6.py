import scipy.stats as stats
import matplotlib.pyplot as plt

n = 1000
p1 = 0.88
p2 = 0.84
vals = []
while n < 1100:
    m1, m2  = p1 * n, p2 * n
    _, pvalue = stats.fisher_exact([[m1, m2], [n - m1, n - m2]])
    vals.append(pvalue)
    print("n = %d, pval = %.8g" % (n, pvalue))
    if pvalue < 0.01:
        print("solution: n = %d gives pval < 0.01" % (n))
    n += 1

plt.plot(range(1000, 1100), vals, '.')
plt.plot(range(1000, 1100), [0.01 for x in range(1000, 1100)], '-')