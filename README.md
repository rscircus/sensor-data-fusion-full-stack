# Sensor Data Fusion - Full Stack

This project implements the three plus one main pillars of Sensor Data Fusion, respectively Bayesian inference.

That is:

- initiation
- prediction
- filtering
- retrodiction

as described in [Wolfgang Koch's book Tracking and Sensor Data Fusion](https://www.springer.com/gp/book/9783642392702).

* * *

If time allows, these models will also cover

- Interacting Multiple Models (incorporate maneuver classes into the dynamics model)
- Multiple Hypothesis Tracking (allow multiple 'realities' to converge in a survival-of-the-fittest manner)

To realize this, we need a set of features:

1. Generation of sensor data from one or many sensors with some added noise.
2. Visualization of the true and the SDF-stack generated approximations.
3. An UI to learn through play

## A rough development plan:

_Reduced the problem to 1D for now._

- [x] Generate sources and add adjustable noise
    - Created `Target` base class
        - Comes with 1D location and velocity
        - Measurement noise is artificially added in `noisy_step`
- [x] Visualize them to get a feeling for what we are doing
    - Using jupyter in `playground` for now
    - Project runs with `poetry run sdf`
    - NEXT: Add Gaussian above current position and animate
- [ ] Model evolution model as Gauss-Markov transition density
- [ ] Implement one sensor which can potentially move (for Dopplereffect, which enables (G)MTI)
- [ ] Implement measurement equation as Gauss likelihood
- [ ] Visualize static situation (replicate Fig 2.1)
- [ ] Add multiple sensors and pull the signals together using eff. measurement error covariance
- [ ] Move sensors to allow TDoA and FDoA (FDoA needs two! => geometric fusion gain? p.42ff)
- [ ] Add explicit noise to generate false positives with 1-3 models
- [ ] Implement a Kalman Filter (if time allows an Extended Kalman Filter, because nature is not linear)
- [ ] Implement 'drop-outs' to motivate retrodiction
- [ ] Implement Retrodiction to compensate drop-outs
- [ ] Implement expectation gates to deal with false positives
- [ ] Add MHT tracking to see how the chain individual gating -> local combining -> pruning works
- [ ] Optional: IMM
- [ ] Optional: Sequential Likelihood Test
- [ ] Optional: Add map-data/tracks as artificial measurements

## Mathematical basis

### 1D Kalman Filter

The Kalman Filter is a two-step [predictor-corrector method](https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method).

#### Predict

- (1) predicted state estimate
- (2) uncertainty of the predicted state estimate

or in equations:

```math
x_t = x_{t-1} + v_t \cdot \delta t\\
\sigma_t = \sigma_{t-1} + \delta t^2 \cdot \sigma_t^v + q_t
```

#### Correct

- Using the Kalman gain
- Update/correct state estimate
- Update/correct uncertainty of the state estimate

or in equations:

```math
k_t = \frac{\sigma_t}{\sigma_t + \sigma_r}\\
\bar{x}_t = x_{t} + k_t (z_t - x_t)\\
\bar{\sigma}_t = (1 - k_t) \sigma_t
```

With the vars being in the order of appearance: predicted state estimate, old state, velocity, timestepsize, variance of pred state estimate, previous variance, variance of velocity estimate and process noise variance.

In 1D the Kalman gain is the uncertainty in state divided by the sum of state estimate uncertainty and measurement uncertainty. Imagining what would happen with small measurement uncertainty? What would happen with big state estimate uncertainty?

__In one sentence:__ The Kalman filter estimates the state of a system using a weighted average of the system's predicted state and the (noisy) observed state (= measurement).

## Log

### Signal sources:

### Noisy 1D with constant velocity

![](./assets/noise_1d_const_speed.png)

#### Noisy dancer in 2D:

![](./assets/noisy_dancer.png)

orig: `x(t) = sin(t)`

#### Very noisy static position in 2D:

![](./assets/random_walk_2d.png)

orig: `[10, 10]`

## Sources

### *Computational science and engineering* by Gilbert Strang

https://math.mit.edu/~gs/cse/

Especially helpful in explaining the Linear Algebra aspects of the Kalman Filter. Especially p. 207 ff.
This also opens the door to parallelization.

### Tracking and Sensor Data Fusion by Wolfgang Koch

https://www.springer.com/gp/book/9783642392702
