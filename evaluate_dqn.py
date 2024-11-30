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

RENDER = False
env = StarshipEnvDiscrete()
PATH_FOR_WEIGHTS = 'data/policy_net_weights_4.pt'


# Get number of actions from gym action space
n_actions = 9
# Get the number of state observations
state = env.reset()
n_observations = len(state)

model = ModelArch(n_actions, n_observations, env.action_space)
# Load existing policy in to test
model.policy_net.load_state_dict(torch.load(PATH_FOR_WEIGHTS, weights_only=True))
model.policy_net.eval()

steps_done = 0
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
        if (i_episode % 100 == 0) and RENDER:
            env.render()
        
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

np.savez(f'test_data/state_buffer_storage.npz', *state_buffer_storage)
np.savez(f'test_data/action_buffer_storage.npz', *action_buffer_storage)
np.savez(f'test_data/reward_storage.npz', np.array(reward_buffer))
np.savez(f'test_data/landed_storage.npz', np.array(landed_episodes_in_epoch))
print('Number of landed runs = ', np.array(landed_episodes_in_epoch).shape[0])
plot_rewards(reward_buffer, show_result=True, test=True)
print('Complete')
