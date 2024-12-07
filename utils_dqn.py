import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_rewards(reward_buffer, show_result=False, test=False, save_folder='imgs', suffix=''):
    reward_np = np.array(reward_buffer)
    rewards = torch.tensor(reward_np, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.plot(rewards.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards) >= 100 and not test:
        means = rewards.unfold(0, 100, 1).mean(2).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if test:
        file_name = f'{save_folder}/test_runs_rewards{suffix}.pdf'
    else:
        file_name = f'{save_folder}/training_over_time.pdf'

    plt.savefig(file_name)
    print(f'Plot saved to {file_name}')
    plt.close()


