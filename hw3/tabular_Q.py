import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Set learning parameters
lr = .8
y = .95
num_episodes = 2000

def q_learning():
    # Implement Q-Table learning algorithm
    # Initialize table with all zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # create lists to contain total rewards and steps per episode
    # jList = []
    rList = []
    epsilon = 1
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0  # Total reward during current episode
        d = False
        j = 0

        # The Q-Table learning algorithm
        while j < 99 and not d:
            j += 1

            # 1. Choose an action by greedily (with noise) picking from Q table
            # 2. Get new state and reward from environment
            # 3. Update Q-Table with new knowledge
            # 4. Update total reward
            # 5. Update episode if we reached the Goal State

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[s])

            next_s, reward, d, _ = env.step(action)
            Q[s, action] += lr * (reward + y * np.max(Q[next_s]) - Q[s, action])
            s = next_s
            rAll += reward

        if epsilon < 0.1:
            epsilon = 0  # Quit exploration after a while
        else:
            epsilon = 1./((i/55) + 1.18)

        rList.append(rAll)

    return rList, Q


# Reports
rList, Q = q_learning()
score = sum(rList) / num_episodes
# Avoid non-convergence
while score < 0.01:
    rList, Q = q_learning()
    score = sum(rList) / num_episodes
print("Score over time: " + str(score))
print("Final Q-Table Values")
print(Q)
