import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  # for color maps
from rocket_discrete_action import StarshipEnvDiscrete

TEST = False



if TEST:
    data = np.load(f'test_data/state_buffer_storage.npz')
    action = np.load(f'test_data/action_buffer_storage.npz')
    reward = np.load(f'test_data/reward_storage.npz')
    landed = np.load(f'test_data/landed_storage.npz')

    print('Landed runs: ', landed['arr_0'].squeeze())
    run_no = np.argmax(reward['arr_0'].squeeze())
    # run_no = 20
    print('Max reward = ', np.max(reward['arr_0'].squeeze()))
    state_data = data[f'arr_{run_no}']
    action_data = action[f'arr_{run_no}']

    env = StarshipEnvDiscrete()

    for i in range(len(state_data)):
        env.state = state_data[i, :]
        env.last_u = action_data[i]
        f, vphi = env.action_table[action_data[i]]
        env.last_u = np.array([f, vphi])
        env.render()

else:
    File_no = 5
    data = np.load(f'data/state_buffer_storage_{File_no}.npz')
    action = np.load(f'data/action_buffer_storage_{File_no}.npz')
    reward = np.load(f'data/reward_storage_{File_no}.npz')
    landed = np.load(f'data/landed_storage_{File_no}.npz')

    print('Landed runs: ', len(landed['arr_0'].squeeze()))
    run_no = np.argmax(reward['arr_0'].squeeze())
    # run_no = 57
    print('Max reward = ', np.max(reward['arr_0'].squeeze()))
    state_data = data[f'arr_{run_no}']
    action_data = action[f'arr_{run_no}']

    env = StarshipEnvDiscrete()

    for i in range(len(state_data)):
        env.state = state_data[i, :]
        env.last_u = action_data[i]
        f, vphi = env.action_table[action_data[i]]
        env.last_u = np.array([f, vphi])
        env.render()
