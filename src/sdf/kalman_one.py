# This module covers a 1-D version of SDF

# - [X] simple target moving forward (with opt. noise)
# - [X] 1D Kalman filter

import math
import random
import numpy as np
from scipy.stats import norm


class Target:
    def __init__(self, start, velocity):
        self.position = start
        self.velocity = velocity

        # const
        self.STEP = 0.1

        # local
        self.time = 0

    def step(self):
        self.position += self.velocity * self.STEP
        self.time += 1

    def noisy_step(self):
        self.step()
        self.position += self.velocity * (0.5 - random.random())

    def location(self):
        return self.position

    def __str__(self):
        return "time: " + str(self.time) + " | position: " + str(self.position)


# Preparing Gaussians
# TODO: Move into own class


def gaussian_1d(x, mu, sigma):
    """Returns gaussian value at x with given params."""
    norm = 1.0 / np.sqrt(2 * np.pi * sigma ** 2)
    exp = np.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))
    return norm * exp


def gauss_multiply(mu1, sigma1, mu2, sigma2):
    """Multiplies two gaussians according to product rule."""

    new_mu = (sigma2 ** 2 * mu1 + sigma1 ** 2 * mu2) / (sigma1 ** 2 + sigma2 ** 2)
    new_sigma = np.sqrt(1 / (1 / sigma1 ** 2 + 1 / sigma2 ** 2))
    return new_mu, new_sigma


def gauss_add(mu1, sigma1, mu2, sigma2):
    new_mu = mu1 + mu2
    new_sigma = sigma1 + sigma2
    return [new_mu, new_sigma]


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


# 1D Kalman Filter - see README.md
# The one below was the first shot at it, following the books notation.
# TODO: Move into own class

Z = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
Z.reverse()

dt = 1.0
v = 1.0
p_v = 0.01

q = 0.01
r = 0.01


def initialize():
    state = 5
    covariance = 0.5
    return state, covariance


def predict(state, covariance):
    state = (
        state + dt * v
    )  # State Transition Equation (Dynamic Model or Prediction Model)
    covariance = covariance + (dt ** 2 * p_v) + q  # Predicted Covariance equation
    return state, covariance


def measure():
    z = Z.pop()
    return z


def update(state, covariance, z):
    k = covariance / (covariance + r)  # Kalman Gain
    state = state + k * (z - state)  # State Update - k decides about adding the offset
    covariance = (
        1 - k
    ) * covariance  # Covariance Update - again k can "decrease" covariance
    return state, covariance


def run_filter():
    state, covariance = initialize()

    # 5 Kalman Filter steps
    for j in range(1, 5):
        state, covariance = predict(state, covariance)
        print(f"prediction: {state}, {covariance}")

        z = measure()
        state, covariance = update(state, covariance, z)
        print(f"correction: {state}, {covariance}")


def main():
    """
    Run Kalman in 1D
    """

    # simple run with Z
    run_filter()