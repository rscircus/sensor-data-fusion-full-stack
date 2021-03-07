import numpy as np


def multivariate_gaussian(state, mu, covar):
    """Return the multivariate Gaussian distribution on array state.

    Arguments:
    ----------

    state: n-dim array
        Has to be an array constructed by packing the meshed arrays
        of variables x_1, x_2, x_3, ..., x_k into its last dimension.

    mu: float
        center of the Gaussian.

    covar: float
        This is explicitly upper case, as we are talking about the
        covariance matrix now.

    """

    dim = mu.shape[0]
    covar_det = np.linalg.det(covar)
    covar_inv = np.linalg.inv(covar)
    norm = np.sqrt((2 * np.pi) ** dim * covar_det)

    factor = np.einsum("...k,kl,...l->...", state - mu, covar_inv, state - mu)
    # Einstein summation is awesome!
    # This einsum call calculates (x-mu)T.covar-1.(x-mu) in a vectorized way across all input

    return np.exp(-0.5 * factor) / norm
