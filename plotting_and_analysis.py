import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  # for color maps

# different plots that need to be created
# spaghetti plots of state across the first 100 trials?

# spaghetti plots of the actions across the first 100 trials?

############# RANDOM RUNS ###############
fig,ax = plt.subplots()
fig2,ax2 = plt.subplots()
cmap = cm.viridis  # You can use other colormaps, like cm.plasma, cm.inferno, etc.
norm = plt.Normalize(vmin=-50.0, vmax=50.0)  # Normalize z values for color mapping
data = np.load(f'data/state_buffer_storage.npz')
action = np.load(f'data/action_buffer_storage.npz')
reward = np.load(f'data/reward_storage.npz')
for i, element in enumerate(data):
    ax.plot(data[element][:, 0], data[element][:, 1], color=cmap(norm(reward['arr_0'][i])))
    ax2.plot(data[element][:, 1], np.rad2deg(data[element][:, 4]), color=cmap(norm(reward['arr_0'][i])))

# for i in range(1, 13):
#     data = np.load(f'data/state_buffer_storage_{i}.npz')
#     action = np.load(f'data/action_buffer_storage_{i}.npz')
#     reward = np.load(f'data/reward_storage_{i}.npz')
#     for i, element in enumerate(data):
#         ax.plot(data[element][:, 0], data[element][:, 1], color=cmap(norm(reward['arr_0'][i])))
#         ax2.plot(data[element][:, 1], np.rad2deg(data[element][:, 4]), color=cmap(norm(reward['arr_0'][i])))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed to add the color bar; no data passed here
plt.colorbar(sm, ax=ax, label='Reward value')
ax.grid()
ax.set_xlabel('Position in x (m)')
ax.set_ylabel('Position in y (m)')
fig.savefig('imgs/random_actions_discrete_action.pdf')
plt.colorbar(sm, ax=ax2, label='Reward value')
ax2.grid()
ax2.set_xlabel('Position in y (m)')
ax2.set_ylabel('Pitch angle (deg)')
ax2.invert_xaxis()
fig2.savefig('imgs/random_actions_theta_discrete_action.pdf')


######### RUNS WITH POLICY ###########
# fig,ax = plt.subplots()
# fig2,ax2 = plt.subplots()
# cmap = cm.viridis  # You can use other colormaps, like cm.plasma, cm.inferno, etc.
# norm = plt.Normalize(vmin=-300.0, vmax=-200.0)  # Normalize z values for color mapping
# # for i in range(15, 30):
# #     data = np.load(f'data/state_buffer_storage_{i}.npz')
# #     action = np.load(f'data/action_buffer_storage_{i}.npz')
# #     reward = np.load(f'data/reward_storage_{i}.npz')
# #     for i, element in enumerate(data):
# #         ax.plot(data[element][:, 0], data[element][:, 1], color=cmap(norm(reward['arr_0'][i])))
# #         ax2.plot(data[element][:, 1], np.rad2deg(data[element][:, 4]), color=cmap(norm(reward['arr_0'][i])))

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Only needed to add the color bar; no data passed here
# plt.colorbar(sm, ax=ax, label='Reward value')
# ax.grid()
# ax.set_xlabel('Position in x (m)')
# ax.set_ylabel('Position in y (m)')
# fig.savefig('imgs/combined_actions.pdf')
# plt.colorbar(sm, ax=ax2, label='Reward value')
# ax2.grid()
# ax2.set_xlabel('Position in y (m)')
# ax2.set_ylabel('Pitch angle (deg)')
# ax2.invert_xaxis()
# fig2.savefig('imgs/combined_actions_theta.pdf')


######### COMPARING RANDOM AND POLICY ACTIONS ###########
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# data1 = np.load(f'data/state_buffer_storage_1.npz')
# data1 = np.load(f'data/state_buffer_storage_25.npz')
# action1 = np.load(f'data/action_buffer_storage_1.npz')
# action2 = np.load(f'data/action_buffer_storage_25.npz')
# ax1.plot(action1['arr_0'][:200, 0], label='Example Random Run')
# ax1.plot(action2['arr_0'][:200, 0], label='Example Policy Run')
# ax1.axhline(0.0, color='red', linestyle='--', label='Min thrust')
# ax1.axhline(20.0, color='red', linestyle='--', label='Max thrust')
# ax2.plot(np.rad2deg(action1['arr_0'][:200, 1]), label='Example Random Run')
# ax2.plot(np.rad2deg(action2['arr_0'][:200, 1]), label='Example Policy Run')
# ax2.axhline(-30.0, color='red', linestyle='--', label='Min gimbal')
# ax2.axhline(30.0, color='red', linestyle='--', label='Max gimbal')

# ax1.grid()
# ax2.grid()
# ax1.set_ylim(-5.0, 25)
# ax2.set_ylim(-35.0, 35)
# ax1.set_ylabel('Thrust (N)')
# ax2.set_ylabel('Gimbal (deg)')
# ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
# ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()
# fig.savefig('imgs/action_comparison.pdf')


# random_reward = 0
# random_reward_len = 0
# for i in range(1, 13):
#     reward = np.load(f'data/reward_storage_{i}.npz')
#     random_reward += np.sum(reward['arr_0'])
#     random_reward_len += len(reward['arr_0'])

# avg_random_reward = random_reward/random_reward_len

# policy_reward = 0
# policy_reward_len = 0
# for i in range(14, 50):
#     reward = np.load(f'data/reward_storage_{i}.npz')
#     policy_reward += np.sum(reward['arr_0'])
#     policy_reward_len += len(reward['arr_0'])

# avg_policy_reward = policy_reward/policy_reward_len

# print('Random policy average reward = ', avg_random_reward)
# print('Main policy average reward = ', avg_policy_reward)
# print('number of random runs',  random_reward_len)
# print('number of policy runs',  policy_reward_len)

