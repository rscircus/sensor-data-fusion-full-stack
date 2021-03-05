# This module covers a 1-D version of SDF

# - [ ] simple target moving forward (with opt. noise)
# - [ ] 1D Kalman filter

import random


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


def main():
    """
    Run Kalman in 1D
    """

    t = Target(1, 0.5)
    print(t)

    for i in range(10):
        t.step()
        print(t)

    for i in range(10):
        t.noisy_step()
        print(t)