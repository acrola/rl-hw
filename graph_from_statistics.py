import pickle
import os.path
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/home/lior/PycharmProjects/rl-hw/project")

with open(os.path.join('.', "experiments", 'default', "statistics.pkl"), "rb") as f:
	# for key, val in pickle.load(f).items():
	# 	print("{} {}".format(key, val))
	unpickler = pickle.Unpickler(f)
	scores = unpickler.load()
	# steps = np.arange(len(pickle.load(f).items()['mean_episode_rewards']))
	# mean_episode_rewards = pickle.load(f).items()['mean_episode_rewards']
	# best_mean_episode_rewards = pickle.load(f).items()['best_mean_episode_rewards']
steps = np.arange(len(scores['mean_episode_rewards']))
mean_episode_rewards = scores['mean_episode_rewards']
best_mean_episode_rewards = scores['best_mean_episode_rewards']
legend = []
plt.figure(2)
plt.plot(steps, mean_episode_rewards, 'b')
legend.append('mean episode rewards')
plt.plot(steps, best_mean_episode_rewards, 'g')
legend.append('best mean episode rewards')

plt.legend(legend, loc=2)
plt.xscale('log')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Time Steps vs. Rewards')
img_save = 'Project_Q1'
plt.savefig(img_save)
