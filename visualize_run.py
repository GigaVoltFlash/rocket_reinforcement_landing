import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  # for color maps
from rocket_discrete_action import StarshipEnvDiscrete

TEST = False
WIND_EVALUATION = True



if TEST:
    if WIND_EVALUATION:
        data = np.load(f'wind_evaluations_with_nowind_policy/state_buffer_storage_5.npz')
        action = np.load(f'wind_evaluations_with_nowind_policy/action_buffer_storage_5.npz')
        reward = np.load(f'wind_evaluations_with_nowind_policy/reward_storage_5.npz')
        landed = np.load(f'wind_evaluations_with_nowind_policy/landed_storage_5.npz')
    else:
        data = np.load(f'test_data/state_buffer_storage_nowind.npz')
        action = np.load(f'test_data/action_buffer_storage_nowind.npz')
        reward = np.load(f'test_data/reward_storage_nowind.npz')
        landed = np.load(f'test_data/landed_storage_nowind.npz')

    print('Landed runs: ', landed['arr_0'].squeeze())
    # run_no = np.argmax(reward['arr_0'].squeeze())
    run_no = 5
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
    best_policy = 0
    largest_lands = 0
    for i in range(49):
        landed = np.load(f'data/landed_storage_{i}.npz')
        landed_runs = landed['arr_0']
        if landed_runs is None:
            continue
        else:
            num_landed = len(landed_runs)
            if num_landed > largest_lands:
                largest_lands = num_landed
                best_policy = i
    print(best_policy)
    print(largest_lands)
    File_no = 44
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
