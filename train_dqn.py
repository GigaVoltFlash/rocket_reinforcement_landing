from itertools import count
import pdb
import numpy as np
import torch
from rocket_discrete_action import StarshipEnvDiscrete
from dqn_model import ModelArch
from utils_dqn import plot_rewards

RENDER = False

env = StarshipEnvDiscrete()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Get number of actions from gym action space
n_actions = 9
# Get the number of state observations
state = env.reset()
n_observations = len(state)

model = ModelArch(n_actions, n_observations, env.action_space)

episodes_per_epoch = 200
epoch_no = 0
steps_done = 0
reward_buffer = []
state_buffer_storage = []
action_buffer_storage = []
landed_episodes_in_epoch = [-1]

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 10000
else:
    num_episodes = 500

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

        # Store the transition in memory
        model.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        model.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = model.target_net.state_dict()
        policy_net_state_dict = model.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*model.TAU + target_net_state_dict[key]*(1-model.TAU)
        model.target_net.load_state_dict(target_net_state_dict)

        if done:
            if landed:
                landed_episodes_in_epoch.append((i_episode % episodes_per_epoch))
            if (i_episode % 20 == 0):
                print('Finished episode', i_episode, 'with reward ', np.array(total_reward.cpu()))
            state_buffer_storage.append(np.array(env.state_buffer))
            action_buffer_storage.append(np.array(env.action_buffer))
            reward_buffer.append(np.array(total_reward.cpu()))
            if (i_episode % episodes_per_epoch == 0) and i_episode > 0.0:
                np.savez(f'data/state_buffer_storage_{str(epoch_no)}.npz', *state_buffer_storage)
                np.savez(f'data/action_buffer_storage_{str(epoch_no)}.npz', *action_buffer_storage)
                np.savez(f'data/reward_storage_{str(epoch_no)}.npz', np.array(reward_buffer[epoch_no*episodes_per_epoch:(epoch_no + 1)*episodes_per_epoch]))
                np.savez(f'data/landed_storage_{str(epoch_no)}.npz', np.array(landed_episodes_in_epoch))
                torch.save(model.policy_net.state_dict(), f'data/policy_net_weights_{str(epoch_no)}.pt')
                plot_rewards(reward_buffer)
                state_buffer_storage = []
                action_buffer_storage = []
                landed_episodes_in_epoch = [-1]            
                epoch_no += 1
                
            break

print('Complete')
plot_rewards(reward_buffer, show_result=True)