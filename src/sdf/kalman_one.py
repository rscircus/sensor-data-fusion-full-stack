# %%
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


class SimpleKalman:
    def __init__(self, state, covariance, measurements, dt, v, p_v, q, r):
        self.state = state
        self.covariance = covariance
        self.Z = measurements
        self.dt = dt
        self.v = v
        self.p_v = p_v
        self.q = q
        self.r = r

    def predict(self):
        dt = self.dt
        v = self.v
        p_v = self.p_v
        q = self.q

        self.state = (
            self.state + dt * v
        )  # State Transition Equation (Dynamic Model or Prediction Model)
        self.covariance = (
            self.covariance + (dt ** 2 * p_v) + q
        )  # Predicted Covariance equation

    def measure(self):
        self.z = self.Z.pop()

    def update(self):
        r = self.r

        k = self.covariance / (self.covariance + r)  # Kalman Gain
        self.state = self.state + k * (
            self.z - self.state
        )  # State Update - k decides about adding the offset
        self.covariance = (
            1 - k
        ) * self.covariance  # Covariance Update - again k can "decrease" covariance

    def run_filter(self):

        # Kalman Filter steps
        for j in range(1, 5):
            self.predict()
            print(f"prediction: {self.state}, {self.covariance}")

            z = self.measure()
            self.update()
            print(f"correction: {self.state}, {self.covariance}")


def main():
    """
    Run Kalman in 1D
    """

    init_state = 5
    init_var = 0.5
    Z = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    Z.reverse()

    dt = 1.0
    v = 1.0
    p_v = 0.01

    q = 0.01
    r = 0.01

    sf = SimpleKalman(init_state, init_var, Z, dt, v, p_v, q, r)

    # simple run with Z
    print("Running filter")
    sf.run_filter()