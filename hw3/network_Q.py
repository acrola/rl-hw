import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss


def to_one_hot(s):
    s_one_hot = np.reshape(np.identity(16)[s:s + 1], [16])
    return Variable(torch.FloatTensor(s_one_hot), requires_grad=False)


# S because it's shallow
class SQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SQN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        # nn.init.uniform(self.fc1.weight, 0, 0.01)

    def forward(self, x):
        return self.fc1(x)


q_net = SQN(input_size=env.observation_space.n, output_size=env.action_space.n)
criterion = nn.MSELoss(reduce=False)
optimizer = torch.optim.SGD(q_net.parameters(), lr=0.1)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
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
        _, a = torch.max(Q, 0)
        a = a.data

        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a[0])

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        Q1 = q_net(to_one_hot(s1))

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        maxQ1 = torch.max(Q1).data[0]
        Qtarget = Q.data.clone()
        Qtarget[a[0]] = r + y * maxQ1
        Qtarget = Variable(Qtarget, requires_grad=False)

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        optimizer.zero_grad()
        output = q_net(to_one_hot(s))
        loss = torch.sum(criterion(output, Qtarget))
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))