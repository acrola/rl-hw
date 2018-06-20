import pickle
import os.path
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/home/lior/PycharmProjects/rl-hw/project")

with open(os.path.join('.', "experiments", 'adv_model_0.0000', "statistics.pkl"), "rb") as f:
	unpickler = pickle.Unpickler(f)
	scores_1 = unpickler.load()
with open(os.path.join('.', "experiments", 'adv_model_1.0000', "statistics.pkl"), "rb") as f:
	unpickler = pickle.Unpickler(f)
	scores_2 = unpickler.load()

leng = min(len(scores_1['mean_episode_rewards']),len(scores_2['mean_episode_rewards']))
steps = np.arange(leng)
mean_episode_rewards_1 = scores_1['mean_episode_rewards'][:leng]
best_mean_episode_rewards_1 = scores_1['best_mean_episode_rewards'][:leng]
mean_episode_rewards_2 = scores_2['mean_episode_rewards'][:leng]
best_mean_episode_rewards_2 = scores_2['best_mean_episode_rewards'][:leng]
legend = []
plt.figure(2)
plt.plot(steps, mean_episode_rewards_1, 'b')
legend.append('mean episode rewards for given DNN')
plt.plot(steps, best_mean_episode_rewards_1, 'r')
legend.append('best mean episode rewards for given DNN')
plt.plot(steps, mean_episode_rewards_2, 'g')
legend.append('mean episode rewards for our DNN')
plt.plot(steps, best_mean_episode_rewards_2, 'c')
legend.append('best mean episode rewards for our DNN')

plt.legend(legend, loc=4)
# plt.xscale('log')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Time Steps vs. Rewards, different model for Pong (game 3)')
img_save = 'Project_Q2_adv_model_game_3'
plt.savefig(img_save)
