import matplotlib.pyplot as plt
import torch


def plot_rewards(reward_buffer, show_result=False, test=False):
    rewards = torch.tensor(reward_buffer, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.grid()
    plt.plot(rewards.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards) >= 100 and not test:
        means = rewards.unfold(0, 100, 1).mean(2).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if test:
        file_name = 'imgs/test_runs_rewards.pdf'
    else:
        file_name = 'imgs/training_over_time.pdf'

    plt.savefig(file_name)
    print(f'Plot saved to {file_name}')
    plt.close()