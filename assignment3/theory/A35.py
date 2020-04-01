#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:24:00 2020

@author: admin
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

x1 = [0, 0]
x2 = [0, 1]
x3 = [1, 0]
x4 = [1, 1]
y = [0, 1, 1, 0]

# Q1
# --

def phi(x):
    return [x[0]**2, 2**0.5*x[0]*x[1], x[1]**2, 2**0.5*x[0], 2**0.5*x[1], 1]

# All are dimensions.

phi1 = phi(x1)
phi2 = phi(x2)
phi3 = phi(x3)
phi4 = phi(x4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

i = 0
for phi in [phi1, phi2, phi3, phi4]:
    # Last 3 dimensions are useless.
    if y[i] == 1:
        ax.scatter(phi[0], phi[1], phi[2], marker="$1$", color="red")
    else:
        ax.scatter(phi[0], phi[1], phi[2], marker="$0$", color="green")
    i += 1
plt.show()

# Q2
# --

# Just the first three are sufficient.

# Q3
# --

def phi(x):
    return [x[0]**3, 3**0.5 * x[0]**2 * x[1], 3**0.5 * x[0] * x[1]**2, x[1]**3]

# Q4
# --

# Allow using lists for markers.
def mscatter(x,y,z, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,z,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

phi1 = phi(x1)
phi2 = phi(x2)
phi3 = phi(x3)
phi4 = phi(x4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

phivec = np.array([phi1, phi2, phi3, phi4])
markers = ["$0$", "$1$", "$1$", "$0$"]

# Third dimension is useless.
mscatter(phivec[:, 0], phivec[:, 1], phivec[:, 3], ax=ax, m=markers)
plt.show()