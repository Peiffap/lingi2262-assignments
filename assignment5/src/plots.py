#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import pickle
import matplotlib.pyplot as plt
import tikzplotlib # pip install tikzplotlib

import seaborn as sns

plt.style.use("ggplot")

res = pickle.load(open("../report/data/res.p", "rb"))

keys = res.keys()

lengths = [1000 * (k-1)/k for k in keys]

test = []
testerr = []
train = []
trainerr = []

for k in keys:
    test.append(res[k]['test'].mean() * 100)
    testerr.append(res[k]['test'].std() * 100)

    train.append(res[k]['train'].mean() * 100)
    trainerr.append(res[k]['train'].std() * 100)

clrs = sns.color_palette("husl", 2)

plt.figure()
plt.plot(lengths, test, c=clrs[0])
plt.fill_between(lengths, [i-j for i, j in zip(test, testerr)], [i+j for i, j in zip(test, testerr)], alpha=0.3, facecolor=clrs[0])

plt.plot(lengths, train, c=clrs[1])
plt.fill_between(lengths, [i-j for i, j in zip(train, trainerr)], [i+j for i, j in zip(train, trainerr)], alpha=0.3, facecolor=clrs[1])

plt.xlabel("Training set size (samples)")
plt.ylabel("Balanced classification rate (\\si{\percent})")
plt.legend(["Test set", "Training set"])

tikzplotlib.save("../report/plots/learning_rate.tikz", axis_width="0.65\\linewidth")