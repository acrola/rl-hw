import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

# Load environment
env = gym.make('FrozenLake-v0')


def to_one_hot(state):
    one_hot = torch.zeros([1, env.observation_space.n])
    one_hot[0, state] = 1
    return Variable(one_hot)

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss


# S because it's shallow
class SQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SQN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.fc1(x)


q_net = SQN(input_size=env.observation_space.n, output_size=env.action_space.n)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(q_net.parameters(), lr=0.5)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
q_net.train()
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        Q = q_net(to_one_hot(s))
        _, a = torch.max(Q, 1)
        a = int(a.data.numpy())

        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        Q1 = q_net(to_one_hot(s1))

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        maxQ1 = torch.max(Q1).data[0]
        Qtarget = Q.data.clone()
        Qtarget[0, a] = r + y * maxQ1
        Qtarget = Variable(Qtarget, requires_grad=False)

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        optimizer.zero_grad()
        loss = criterion(Q, Qtarget)
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1. / ((i / 50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))