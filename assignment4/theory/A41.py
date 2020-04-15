#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import statistics


def samp(p, lim):
    n = 0
    ctr = 0
    while ctr < lim:
        r = random.randint(1, 100)
        if r > p * 100:
            ctr += 1
        n += 1
    return n


s = []
n = 100000
p = 0.8
for i in range(n):
    s.append(samp(p, 1))

print("examples until first error: %g" % (statistics.mean(s))) # true val: 5

print("standard dev: %g" % (statistics.stdev(s))) # true val: 4.472


s = []
for i in range(n):
    s.append(samp(p, 3))

print("examples until third error: %g" % (statistics.mean(s))) # true val: 15



