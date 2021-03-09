import numpy as np


class Gaussian:
    def __init__(self, dim):
        self.mu = np.array([dim * [0.0]])
        self.shape = tuple(i for i in [dim * [dim]])
        self.covar = np.zeros(self.shape)

        # TODO: This will be a hell of a covariance matrix

    def add(other: Gaussian):
        """Adds another Gaussian to itself."""

        # TODO: Sanity checks
        pass

    def multiply(other: Gaussian):
        """Multiplies itself with another Gaussian."""
        pass


# TODO: evolution model as descendant of Gaussian

# TODO: sensor model as descendant of Gaussian