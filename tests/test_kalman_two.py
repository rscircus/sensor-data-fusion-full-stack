# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sdf.kalman_two import multivariate_gaussian

# %% - static situation

resolution = 60
X = np.linspace(-3, 3, resolution)
Y = np.linspace(-3, 4, resolution)
X, Y = np.meshgrid(X, Y)

mu = np.array([1.0, 1.0])
covar = np.array([[1.0, -0.5], [-0.5, 1.5]])  # skewed covar for nice effect

# Pack X and Y into a single 3-dimensional array
state = np.empty(X.shape + (2,))
state[:, :, 0] = X
state[:, :, 1] = Y

fig = plt.figure(figsize=(25 / 2.54, 20 / 2.54))
ax = fig.gca(projection="3d")
ax.plot_surface(
    X,
    Y,
    multivariate_gaussian(state, mu, covar),
    rstride=3,
    cstride=3,
    linewidth=1,
    antialiased=True,
    cmap=cm.viridis,
)
cset = ax.contourf(
    X,
    Y,
    multivariate_gaussian(state, mu, covar),
    zdir="z",
    offset=-0.2,
    cmap=cm.viridis,
)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.2, 0.2)
ax.set_zticks(np.linspace(-0.2, 0.2, 11))
ax.view_init(27, -45)

# %% - and now the animation
