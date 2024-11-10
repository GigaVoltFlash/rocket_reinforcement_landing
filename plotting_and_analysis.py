import matplotlib.pyplot as plt
import numpy as np

# different plots that need to be created
# spaghetti plots of state across the first 100 trials?

# spaghetti plots of the actions across the first 100 trials?

data1 = np.load('state_buffer_storage_1.npz')
action1 = np.load('action_buffer_storage_1.npz')
import pdb
# pdb.set_trace()

fig = plt.figure()
for i in range(1, 13):
    data = np.load(f'state_buffer_storage_{i}.npz')
    for element in data:
        plt.plot(data[element][:, 0], data[element][:, 1])

plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
