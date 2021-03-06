import random
import numpy as np
import matplotlib.pyplot as plt

# 1D first:


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

    up_prob = pause_ratio

    start = x_start
    positions = [start]

    sample = np.random.random(num_steps)
    up_samples = sample > 1.0 - up_prob

    for up in up_samples:
        positions.append(positions[-1] + up / pause_ratio)

    return positions