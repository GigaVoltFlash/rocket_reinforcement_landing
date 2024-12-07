from itertools import count
import numpy as np
import torch
from rocket_discrete_action import StarshipEnvDiscrete
from dqn_model import ModelArch
from utils_dqn import plot_rewards

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def evaluate_dqn(wind_path=None, weights_path='dqn_results/policy_net_weights_34.pt', savefile_suffix='nowind', save_folder='test_data'):

    env = StarshipEnvDiscrete(wind_profile_path=wind_path)

    # Get number of actions from gym action space
    n_actions = 9
    # Get the number of state observations
    state = env.reset()
    n_observations = len(state)

    model = ModelArch(n_actions, n_observations, env.action_space)
    # Load existing policy in to test
    model.policy_net.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.policy_net.eval()

    reward_buffer = []
    state_buffer_storage = []
    action_buffer_storage = []
    landed_episodes_in_epoch = []

    num_episodes = 100

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        
        for t in count():
            
            action = model.select_action(state)
            observation, reward, terminated, landed = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            total_reward += reward
            done = terminated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                if landed:
                    landed_episodes_in_epoch.append((i_episode))
                if (i_episode % 20 == 0):
                    print('Finished episode', i_episode, 'with reward ', np.array(total_reward.cpu()))
                state_buffer_storage.append(np.array(env.state_buffer))
                action_buffer_storage.append(np.array(env.action_buffer))
                reward_buffer.append(np.array(total_reward.cpu()))
                break

    np.savez(f'{save_folder}/state_buffer_storage_{savefile_suffix}.npz', *state_buffer_storage)
    np.savez(f'{save_folder}/action_buffer_storage_{savefile_suffix}.npz', *action_buffer_storage)
    np.savez(f'{save_folder}/reward_storage_{savefile_suffix}.npz', np.array(reward_buffer))
    np.savez(f'{save_folder}/landed_storage_{savefile_suffix}.npz', np.array(landed_episodes_in_epoch))
    print('Number of landed runs = ', np.array(landed_episodes_in_epoch).shape[0])
    plot_rewards(reward_buffer, show_result=True, test=True, save_folder=save_folder, suffix=f'_{savefile_suffix}')
    print('Complete')

    return np.array(landed_episodes_in_epoch).shape[0]

if __name__ == '__main__':
    evaluate_dqn()
    # evaluate_dqn(wind_path='./wind_data/wind_profile_11.csv', weights_path='dqn_results_wind/policy_net_weights_10.pt')
    # evaluate_dqn(wind_path='./wind_data/wind_profile_11.csv')