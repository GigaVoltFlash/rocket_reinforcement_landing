from itertools import count
import numpy as np
import torch
from rocket_discrete_action import StarshipEnvDiscrete
from dqn_model import ModelArch
from utils_dqn import plot_rewards

def evaluate_over_x_y(model, first_var, second_var):
    var1_values = torch.tensor(first_var, dtype=torch.float32)
    var2_values = torch.tensor(second_var, dtype=torch.float32)
    meshgrid_val1, meshgrid_val2 = torch.meshgrid(var1_values, var2_values)
    inputs = torch.stack((meshgrid_val1.flatten(), meshgrid_val2.flatten()), dim=-1)
    zeros = torch.zeros((inputs.shape[0], 5))  # 5 zero columns for the remaining values
    
    # Concatenate the zeros with the meshgrid
    final_inputs = torch.cat((inputs, zeros), dim=1)
    
    # Pass through the model
    action = model(final_inputs).max(1).indices.view(1, 1)
    
    return action

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

env = StarshipEnvDiscrete()
PATH_FOR_WEIGHTS = 'data/policy_net_weights_1.pt'

# Get number of actions from gym action space
n_actions = 9
# Get the number of state observations
state = env.reset()
n_observations = len(state)

model = ModelArch(n_actions, n_observations, env.action_space)
# Load existing policy in to test
model.policy_net.load_state_dict(torch.load(PATH_FOR_WEIGHTS, weights_only=True))
model.policy_net.eval()

x = np.linspace(-300, 300, 1000)
y = np.linspace(-30, 500, 1000)

actions = evaluate_over_x_y(model.policy_net, x, y)
