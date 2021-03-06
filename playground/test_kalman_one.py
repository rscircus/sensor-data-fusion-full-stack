# %%

import os
import math
from sdf.kalman_one import gauss_add, gauss_multiply
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from matplotlib.animation import FuncAnimation
from scipy.stats import norm


def plot_gaussian_pdf(
    mean=0.0,
    variance=1.0,
    std=None,
    ax=None,
    xlim=None,
    ylim=None,
    label=None,
    color="green",
    marker="",
):
    """
    Plots a normal distribution PDF.
    """

    # sanity
    if ax is None:
        ax = plt.gca()

    if variance is not None and std is not None:
        raise ValueError("Specify only one of variance and std")

    if variance is None and std is None:
        raise ValueError("Specify variance or std")

    if variance is not None:
        std = math.sqrt(variance)

    # the actual Gaussian from scipy
    n = norm(mean, std)

    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.0)
    ax.plot(xs, n.pdf(xs), color=color, marker=marker, label=label)

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


state = (0, 10000)  # Gaussian N(mu=0, var=insanely big)
velocity = 1
velocity_error = 0.05
sensor_error = 2.5

measurements = [
    -2.07,  # first is completely off
    5.05,
    0.51,
    3.47,
    5.76,
    0.93,  #  huge offset
    6.53,
    9.01,
    7.53,
    11.68,
    9.15,  # another offset
    14.76,
    19.45,
    16.15,
    19.05,
    14.87,
    7.90,
    5.75,  # extreme offset
    7.16,
    20.50,
    21.75,
    22.05,
    23.5,
    24.27,
    25.0,
]
zs = []  # measurements (locations)
ps = [0]  # filter outputs (locations) / added a 0 s.t. the ax1.legend() works

N = 25


def animate(frame):
    global legend, state, zs, ps, N, measurements, ax1, ax2, fig

    # predict we add up cur location with velocity*1sec
    state = gauss_add(state[0], state[1], velocity, velocity_error)
    Z = measurements[frame]
    zs.append(Z)

    # update - Gauss multiplication of new state with likelihood :)
    state = gauss_multiply(state[0], state[1], Z, sensor_error)
    ps.append(state[0])

    ax1.plot(zs, "ro", label="measurement")
    ax1.set_xlim([0, N * 1.2])
    ax1.set_ylim([0, N * 1.2])

    if len(ps) > 1:
        ax1.plot(ps, "b", label="filter")

    ax2.cla()
    plot_gaussian_pdf(state[0], state[1], xlim=[0, N * 1.2], ax=ax2, label="filter")
    ax2.set_ylim(0, 1)
    fig.tight_layout()

    if frame == 0:
        ax1.legend()
        ax2.legend()


def init():
    """This is a bug in matplotlib?"""
    pass


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
animation = FuncAnimation(fig, animate, N, interval=750, init_func=init)

# %%

filename = "animation.gif"

basename = os.path.splitext(filename)[0]
animation.save(basename + ".mp4", writer="ffmpeg")

os.system("ffmpeg -y -i {}.mp4 {}.gif".format(basename, basename))
os.remove(basename + ".mp4")
# %%

# %%
