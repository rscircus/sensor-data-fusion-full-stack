# Sensor Data Fusion - Full Stack

This project implements the three main pillars of Sensor Data Fusion, respectively Bayesian inference.

That is:

- prediction
- filtering
- retrodiction

as described in [Wolfgang Koch's book Tracking and Sensor Data Fusion](https://www.springer.com/gp/book/9783642392702).

If time allows, these models will also cover

- Interacting Multiple Models (incorporate maneuver classes into the dynamics model)
- Multiple Hypothesis Tracking (allow multiple 'realities' to converge in a survival-of-the-fittest manner)

To realize this, we need a set of features:

1. Generation of sensor data from one or many sensors with some added noise.
2. Visualization of the true and the SDF-stack generated approximations.
3. An UI to learn through play (optional)
