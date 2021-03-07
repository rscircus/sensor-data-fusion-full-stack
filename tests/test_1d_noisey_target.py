from sdf.kalman_one import Target

t = Target(1, 0.5)
print(t)

for i in range(10):
    t.step()
    print(t)

for i in range(10):
    t.noisy_step()
    print(t)