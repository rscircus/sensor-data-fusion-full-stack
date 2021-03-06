# %%

import os
from sdf.kalman_one import gauss_add, gauss_multiply, plot_gaussian_pdf
import matplotlib.pyplot as plt
import numpy.random as random
from matplotlib.animation import FuncAnimation


state = (0, 10000)  # Gaussian N(mu=0, var=insanely big)
groundtruth = 0
velocity = 1
velocity_error = 0.05
sensor_error = 1.5

measurements = [
    -2.07,  # first is completely off #
    5.05,
    0.51,
    3.47,
    5.76,
    0.93,  # huge offset
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
    26.1,
    26.9,
    29.3,
    29.15,
    30,
]
zs = [0]  # measurements (locations)
ps = [0]  # filter outputs (locations) / added a 0 s.t. the ax1.legend() works
ns = [0]  # noise/error offset

N = 30
dt = 1


def animate(frame):
    global dt, groundtruth, state, zs, ps, ns, N, measurements, ax1, ax2, fig

    # predict: we progress state with evolution model
    state = gauss_add(state[0], state[1], velocity * dt, velocity_error)

    # memorize for plotting later
    groundtruth = groundtruth + velocity * dt
    prediction_state = state

    Z = measurements[frame]
    zs.append(Z)

    # update: We correct the state using the measurement (as likelihood in Bayes manner)
    state = gauss_multiply(state[0], state[1], Z, sensor_error)
    ps.append(state[0])

    # plot measurement
    ax1.plot(zs, color="orange", marker="o", label="measurement")
    ax1.set_xlim([0, N * 1.2])
    ax1.set_ylim([-5, N * 1.2])

    # plot filter output (state*likelihood)
    if len(ps) > 1:
        ax1.plot(ps, "green", label="filter")

    # plot the current filter output (a Gaussian)
    ax2.cla()
    plot_gaussian_pdf(
        state[0], state[1], xlim=[0, N * 1.2], ax=ax2, color="green", label="filtering"
    )
    plot_gaussian_pdf(
        prediction_state[0],
        prediction_state[1],
        xlim=[0, N * 1.2],
        ax=ax2,
        color="blue",
        label="prediction",
    )

    # debug
    print(groundtruth)

    # plot groundtruth
    ax2.axvline(x=int(groundtruth), color="red", label="groundtruth")
    ax2.set_ylim(0, 1)

    noise = Z - groundtruth
    ns.append(noise)

    print(noise)
    ax3.plot(ns, color="red", marker="o", label="measurement - groundtruth")
    ax3.set_xlim([0, N * 1.2])
    ax3.set_ylim(-15, 15)

    # make things look nice
    if frame == 0:
        ax1.legend()
        ax1.grid()
        ax3.legend()
        ax3.grid()

    # ax2 gets cleared all the time, hence we redraw the legend and xlabel
    ax2.legend()
    ax2.grid()
    ax2.set(xlabel="location")
    fig.tight_layout()


def init():
    """This is a bug in matplotlib?"""
    pass


fig = plt.figure(figsize=(25 / 2.54, 20 / 2.54))

ax1 = fig.add_subplot(311)
ax1.set(ylabel="location", xlabel="time")

ax2 = fig.add_subplot(312)

ax3 = fig.add_subplot(313)
animation = FuncAnimation(fig, animate, N, interval=750, init_func=init)


# %%

filename = "animation.gif"

basename = os.path.splitext(filename)[0]
animation.save(basename + ".mp4", writer="ffmpeg")

os.system("ffmpeg -y -i {}.mp4 {}.gif".format(basename, basename))
os.remove(basename + ".mp4")

# %%
