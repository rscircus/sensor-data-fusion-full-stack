# %%

print("Generate noise models with a little bit of causality.")
# %%

import random
import numpy as np
import matplotlib.pyplot as plt

# 1D first:
# %% - Setup distribution to be sampled


def generate_1d_noisy_movement(x_start=0, x_vel=1.0, num_steps=100, pause_ratio=0.5):
    """Generates a linear noisy movement.

    Parameters:
    -----------

    x_start: int
        Starting position of movement.

    x_vel: float
        Velocity of the movement.

    num_steps: int
        The length of the movement or size of the returned position array.

    pause_ratio: float
        A value between 0 and 1 to reflect how often nothing happens.
    """

    DOWN = 0
    UP = 1

    # 1D first

    up_prob = pause_ratio

    start = x_start
    positions = [start]

    # generate a bunch of random numbers and spread them
    # 1.0 - up_prob - down_prob of the time we do not move
    sample = np.random.random(num_steps)
    up_samples = sample > 1.0 - up_prob

    for up in up_samples:
        positions.append(positions[-1] + up / pause_ratio)

    return positions


positions = generate_1d_noisy_movement()
plt.plot(positions)

# %%
# 1D random walk:


DOWN = 0
UP = 1

# 1D first

up_prob = 0.05
down_prob = up_prob

start = 0
positions = [start]
num_steps = 1000

# generate a bunch of random numbers and spread them
# 1.0 - up_prob - down_prob of the time we do not move
sample = np.random.random(num_steps)
down_samples = sample < down_prob
up_samples = sample > 1.0 - up_prob

for down, up in zip(down_samples, up_samples):
    positions.append(positions[-1] - down + up)


# 2D now:

# %%

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3

X = 0
Y = 1

# 1D first
up_prob = 0.05
down_prob = up_prob
left_prob = up_prob
right_prob = up_prob

start = [10, 10]
positions = [start]
num_steps = 10000

vertical_sample = np.random.random(num_steps)
up_samples = vertical_sample < up_prob
down_samples = vertical_sample > 1.0 - down_prob

horizontal_sample = np.random.random(num_steps)
left_samples = horizontal_sample < left_prob
right_samples = horizontal_sample > 1.0 - right_prob

nswe = zip(up_samples, down_samples, left_samples, right_samples)

for u, d, l, r in nswe:
    positions.append([positions[-1][X] + u - d, positions[-1][Y] + r - l])

x, y = zip(*positions)
plt.plot(x, y)
# %%

# And finally a noise sine to have something to work with
def gen_sine(carrier_freq, duration):
    sin = []
    for t in range(duration):
        sin.append(np.sin(2 * np.pi * t * carrier_freq))
    return sin


def gen_gauss_noise(max_amp, snr, duration):
    return np.random.normal(0, max_amp / snr, duration)


def gen_noisy_signal(signal, duration):
    noisy_signal = signal(0.01, duration)
    noisy_signal += gen_gauss_noise(1, 10, duration)
    return noisy_signal


uuuu = gen_noisy_signal(gen_sine, 1000)
plt.plot(uuuu)

# %%

# Now lets use our target

from sdf.kalman_1d import Target

tar = Target(1, 0.5)
path = []

for i in range(100):
    path.append(tar.location())
    tar.step()

plt.xlabel("time")
plt.ylabel("location")
plt.plot(path, label="true position")

tar = Target(1, 0.5)
noisy_path = []

for i in range(100):
    noisy_path.append(tar.location())
    tar.noisy_step()

plt.xlabel("time")
plt.ylabel("location")
plt.plot(noisy_path, label="measured position")
plt.legend()
# %%

# %%
