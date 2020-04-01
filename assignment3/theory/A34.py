#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:26:27 2020

@author: admin
"""

# Script to solve A3.4
# --------------------

# Q1
# --

import numpy as np
import matplotlib.pyplot as plt

Wxh = np.array([[0.79, 1.34], [0.87, 1.08]])
bh = np.array([[0.1], [-1.12]])
Why = np.array([[0.68], [-2.01]])
by = np.array([-0.3])

x1 = np.array([[0], [0]])
x2 = np.array([[0], [1]])
x3 = np.array([[1], [0]])
x4 = np.array([[1], [1]])

y = [0, 1, 1, 0]

def sigmoid(a):
    return 1 / (1 + np.exp(-a[0]))

def rlu(a):
    return np.maximum([[0], [0]], a)

h = []
a = [[], []]
yhat = []
i = 0
for x in [x1, x2, x3, x4]:
    a[0].append(np.transpose(Wxh) @ x + bh)
    h.append(rlu(a[0][i]))
    a[1].append(np.transpose(Why) @ h[i] + by)
    yhat.append(sigmoid(a[1][i]))
    print("y%d = %g" % (i+1, yhat[i]))
    i += 1
print()

# Q2
# --

# By using g(y) = round(y), we find 3 examples are well-classified.
print("correctly classified = 3\n")

# Q3
# --

def cross_entropy(y, yhat):
    return -sum([y1 * np.log(y2) + (1 - y1) * np.log(1 - y2) for (y1, y2) in zip(y, yhat)])

print("cross-entropy = %g\n" % (cross_entropy(y, yhat)))

# Q4
# --

# output layer gradient = -y3/yhat3 - (1 - y3) / (1 - yhat3)
grad_yhat_J = -1 /  yhat[2]
print("output layer gradient = %g\n" % (grad_yhat_J))

# Q5
# --
exp = np.exp(-by - np.transpose(Why) @ h[2])
grad_w = grad_yhat_J * np.transpose(np.divide(h[2] * exp, np.power(1 + exp, 2)))

def sigmoid_derivative(a):
    sa = sigmoid(a)
    return sa * (1 - sa)

grad_b = grad_yhat_J * sigmoid_derivative(a[1][2])

print("grad_w_1 = %g\ngrad_w_2 = %g\ngrad_b = %g\n" % (grad_w[0, 0], grad_w[0, 1], grad_b))
q81 = grad_w
q82 = grad_b

# Q6
# --

g = grad_b
grad_h_J = np.transpose(Why) * g
print("grad_h_1 = %g\ngrad_h_2 = %g\n" % (grad_h_J[0, 0], grad_h_J[0, 1]))

# Q7
# --

def rlu_derivative(a):
    return (np.sign(a) + 1) / 2

g = grad_h_J
g = np.reshape([gg * fa for (gg, fa) in zip(g, rlu_derivative(a[0][2]))], (2, 1))
grad_b = g
grad_w = g @ np.transpose(x3)
print("grad_w_11 = %g\ngrad_w_12 = %g\ngrad_w_21 = %g\ngrad_w_22 = %g\ngrad_b_1 = %g\ngrad_b_2 = %g\n" %
      (grad_w[0, 0], grad_w[0, 1], grad_w[1, 0], grad_w[1, 1], grad_b[0], grad_b[1]))

# Q8
# --

eta = 0.2
Why = Why - eta * np.transpose(q81)
by = by - eta * q82
print("w_1 = %g\nw_2 = %g\nb = %g\n" % (Why[0][0], Why[1][0], by))

# Q9
# --

Wxh = Wxh - eta * np.transpose(grad_w)
bh = bh - eta * grad_b
print("w_11 = %g\nw_12 = %g\nw_21 = %g\nw_22 = %g\nb_1 = %g\nb_2 = %g\n" %
      (Wxh[0, 0], Wxh[0, 1], Wxh[1, 0], Wxh[1, 1], bh[0], bh[1]))

# Q10
# ---

h = []
a = [[], []]
yhat = []
i = 0
for x in [x1, x2, x3, x4]:
    a[0].append(np.transpose(Wxh) @ x + bh)
    h.append(rlu(a[0][i]))
    a[1].append(np.transpose(Why) @ h[i] + by)
    yhat.append(sigmoid(a[1][i]))
    i += 1
i = 0
for [a, b] in h:
    if y[i] == 1:
        plt.plot(a, b, marker="$1$", color="red")
    else:
        plt.plot(a, b, marker="$0$", color="green")
    i += 1
plt.show()

# Q11
# ---

print("yes, linearly separable\n")

# Q12
# ---

for i in range(len(yhat)):
    print("y%d = %g" % (i+1, yhat[i]))

# Q13
# ---

# By using g(y) = round(y), we find 4 examples are well-classified.
print("correctly classified = 4\n")

# Q14
# ---

print("cross-entropy = %g\n" % (cross_entropy(y, yhat)))

# Q15
# ---

print("- if we let the gradient descent algorithm continue, we could expect it to optimise the network to a point where the output values would be something like yhat1 = 0.01, yhat2 = 0.99, yhat3 = 0.99, yhat4 = 0.01")
print("- the gradient descent algorithm will continue iterating to decrease the loss, until the latter converges")