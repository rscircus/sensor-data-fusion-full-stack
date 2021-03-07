# %% -- imports
# Basically implements p.60 to p.62
# y - the innovation vector could be used for gating here now.
#
# This took me basically the whole day to implement 2.5 pages.
# However, numpy sometimes is not that straight forward and understanding
# the innovation covariance matrix took a while.

import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sdf.kalman_two import multivariate_gaussian
from matplotlib.animation import FuncAnimation

debug = False

#
# Setup for visualization
#

resolution = 100
X = np.linspace(-3, 30, resolution * 2)
Y = np.linspace(-5, 5, resolution // 2)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
state = np.empty(X.shape + (2,))
state[:, :, 0] = X
state[:, :, 1] = Y

#
# And now the Kalman Filter Part
#

# %% -- animation setup

fig = plt.figure(figsize=(25 / 2.54, 20 / 2.54))
ax = fig.gca(projection="3d")

x = np.array([[0.0], [1.0]])  # state - location and velocity
# P = np.diag( [1.0, 0.05] )
# covariance matrix - location and velocity again / for simplicty only diag
P = np.array(
    [[2.0, -1.5], [-1.5, 2.5]]
)  # skewed covar for nice effect, corresponds to a negative covar between location and speed!

# evaluation model
F = np.array(
    [[1.0, 1.0], [0.0, 1.0]]
)  # state transition matrix shifts loc by vel*(dt=1.0)
D = np.array(
    [[2.0, 0.0], [0.0, 2.0]]
)  # evolution covariance, or state model noise covariance
# how much do we trust our sensors vs model D large: trust sensors more

# likelihood / filter model
H = np.array(
    [[1.0, 0.0], [0.0, 0.0]]
)  # measurement function - measures location, ignores velocity for now
R = np.array([[2.5, 0], [0.0, 2.5]])  # measurement error covariance matrix


x_measurements = [
    0.07,  # first is completely off #
    1.05,
    2.51,
    3.47,
    5.76,
    6.93,  # huge offset
    6.53,
    9.01,
    9.53,
    10.68,
    11.15,  # another offset
    12.76,
    13.45,
    14.15,
    15.05,
    14.87,
    17.90,
    18.75,  # extreme offset
    19.16,
    20.50,
    21.75,
    22.05,
    23.5,
    24.27,
    25.0,
    26.1,
    26.9,
    29.3,
    29.15,
    30,
]
y_measurements = 30 * [1.0]
x_measurements.reverse()
y_measurements.reverse()


def animate(frame):
    global x, P, F, D, R, H, ax

    cur_label = ""

    print(f"P_incoming: {P}")
    # extract meshgrid
    X = state[:, :, 0]
    Y = state[:, :, 1]

    z = np.array([[x_measurements.pop()], [y_measurements.pop()]])

    # Prediction by evolution model
    x = F @ x
    P = F @ P @ F.T + D

    ## Plot related - prediction
    # Numpy has a horrible transpose for 1d-vectors...
    mu = x.reshape(1, -1)

    # odd frames -> prediction
    if frame % 2 == 1:
        Z = multivariate_gaussian(state, mu, P.round(2))
        cur_label = "Prediction"

    # Update by likelihood
    y = z - H @ x  # the innovation vector!
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)  # Kalman Gain
    # how much need state and cov from prediction to be corrected?

    # print("Kalman input:")
    # print(f"P: \n{P}")
    # print(f"H: \n{H}")
    # print(f"S: \n{S}")

    if debug:
        print(f"Kalman Gain {K}")

    # correct state and covariance
    x = x + K @ y
    P = P - K @ S @ K.T

    # print(f"x: {x}")
    # print(f"P_corr: {P.round(2)}")
    # print(f"covar: {covar}")

    ## Plot related - prediction
    # Numpy has a horrible transpose for 1d-vectors...
    mu = x.reshape(1, -1)

    # even frames -> Filtering
    if frame % 2 == 0:
        Z = multivariate_gaussian(state, mu, P.round(2))
        cur_label = "Filtering"

    ax.cla()
    ax.plot_surface(
        X,
        Y,
        multivariate_gaussian(state, mu, P),
        rstride=3,
        cstride=3,
        linewidth=1,
        antialiased=True,
        cmap=cm.viridis,
    )
    cset = ax.contourf(
        X,
        Y,
        Z,
        zdir="z",
        offset=-0.2,
        cmap=cm.viridis,
    )

    # make things look good
    ax.set_xlim(0, 30)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-0.2, 0.2)
    ax.set_zticks(np.linspace(-0.2, 0.2, 11))
    ax.view_init(15, -47)
    plt.title(cur_label)


def init():
    """This is a bug in matplotlib?"""
    pass


frames = 30

animation = FuncAnimation(fig, animate, frames, interval=500, init_func=init)

filename = "animation.gif"

basename = os.path.splitext(filename)[0]
animation.save(basename + ".mp4", writer="ffmpeg")

os.system("ffmpeg -y -i {}.mp4 {}.gif".format(basename, basename))
os.remove(basename + ".mp4")

# %%

# %%
