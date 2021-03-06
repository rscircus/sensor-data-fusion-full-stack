# After a little research I found a nice implementation
# to test the Kalman Filter against:
#
# https://pypi.org/project/filterpy/

# %%

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
from 

N = 100


f = KalmanFilter(dim_x=2, dim_z=1)

f.x = np.array([[0.0], [1.0]])  # dim = 2, state/location + velocity

# Newton:

f.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # state transition matrix

# Measurement

f.H = np.array([[1.0, 0.0]])  # measurement function

# Generate some measurement



# Uncertainties:

f.P *= 1000.0  # covariance matrix
f.R = 5  # state uncertainty
f.Q = Q_discrete_white_noise(2, 0.1, 0.1)  # process uncertainty

# TODO: How does the lib generate it?
# def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):

# %% - Test the filter

while True:
    f.predict()
    f.update()
    x = f.x