import numpy as np
import matplotlib.pyplot as plt
from utils_dqn import plot_rewards

total_reward = []
for i in range(0, 49):
    data = np.load(f'dqn_results_wind/state_buffer_storage_{i}.npz')
    action = np.load(f'dqn_results_wind/action_buffer_storage_{i}.npz')
    reward = np.load(f'dqn_results_wind/reward_storage_{i}.npz')
    total_reward.append(reward['arr_0'].squeeze())


full_reward_list = (np.array(total_reward)).reshape(-1)

# Calculate the running average
window_size = 100
running_avg = np.convolve(full_reward_list, np.ones(window_size)/window_size, mode='valid')

# Adjust running average to align with the original array
# Padding the beginning with NaN for elements that don't have a full window
padded_avg = np.concatenate((np.full(window_size - 1, np.nan), running_avg))


plt.figure(figsize=(10, 6))
plt.plot(full_reward_list, label='Raw reward data', alpha=0.7)
plt.plot(padded_avg, label=f'Running Average (Last {window_size} elements)', color='red', linewidth=2)
plt.xlabel('Episode No.')
plt.ylabel('Reward value')
plt.title('Training DQN policy with wind-based policy')
plt.legend()
plt.grid(True)
plt.savefig('./imgs/reward_over_episodes_wind.pdf')