import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

MAX_LEN = 1000 #100
COUNT = 1000

# feature size
F_SIZE = 100 #10

# dent feature
f_dent = np.concatenate((np.linspace(0,1,int(F_SIZE / 2)),np.linspace(1,0,int(F_SIZE / 2))))

# sin curve feature
f_sin = np.sin(np.linspace(0,np.pi,F_SIZE))
print(f_sin)
# double dent like this: --'\,---
f_2dent = np.concatenate((np.linspace(0,1,int(F_SIZE / 4)),np.linspace(1,-1,int(F_SIZE / 2)),np.linspace(-1,0,int(F_SIZE / 4))))

print(f_2dent)


# labels
y = np.ones(COUNT)
y[:int(COUNT/2)] = 0

x = 0.5 + (np.random.rand(COUNT,MAX_LEN) - 0.5) * 0.1

# Labeling rule:
# if f_dent , f_sin are present together, then 0
# if f_2dent then label 1


for i in range(int(COUNT/2)):
    f_start = np.random.randint(MAX_LEN - F_SIZE)
    x[i,f_start:f_start + F_SIZE] = f_dent * 0.5 + 0.5
    f2_start = np.random.randint(MAX_LEN - F_SIZE)
    if f_start - F_SIZE < f2_start < f_start + F_SIZE:
        if f_start - 3 * F_SIZE < 0:
            f2_start = np.random.randint(f_start + 2 * F_SIZE,f_start + 3 * F_SIZE)
        elif f_start + 3 * F_SIZE > MAX_LEN:
            f2_start = np.random.randint(f_start - 3 * F_SIZE,f_start - 2 * F_SIZE)
    x[i, f2_start:f2_start + F_SIZE] = -f_sin * 0.5 + 0.5

for i in range(int(COUNT/2),int(COUNT)):
    f_start = np.random.randint(MAX_LEN - F_SIZE)
    x[i, f_start:f_start + F_SIZE] = f_2dent * 0.5 + 0.5


for xi in range(0,1000):
    fig = plt.figure()
    plt.plot(range(1000),x[xi],'.-')
    fig.savefig('visuals_tmp/sample_{}.jpg'.format(xi))
    plt.close()

torch.save(x, open('time_series_x_100x.pt', 'wb'))
torch.save(y, open('time_series_y_100x.pt', 'wb'))

print(x[0])





