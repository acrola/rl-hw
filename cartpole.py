import gym
import torch
import abc
import matplotlib.pyplot as plt
import argparse
from sys import platform as sys_pf
if sys_pf == 'darwin':  # Required to fix Mac OSX matplotlib crash
    import matplotlib
    matplotlib.use("TkAgg")

RENDER_SIM = False                  # Render simulation on screen (slower training)
POLE_OBSERVATION_DIM = 4            # Number of dimensions for pole observations
EPISODES_COUNT = 1                  # Total episodes to train
STEPS_COUNT = 200                   # Max number of steps per episode (if no failure occurs)
RANDOM_SEARCH_SAMPLE_COUNT = 10000  # Number of samples for random search
EVAL_ITERATIONS = 1000              # Number of iterations to run random search for
EVAL_SCORE = 200                    # Target score for random search evaluation


class BaseAgent(object, metaclass=abc.ABCMeta):
    ''' Base class for pole agents '''
    def __init__(self):
        self.w = None

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError('Base class for agents - non instantiable')

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError('Base class for agents - non instantiable')

    @staticmethod
    def sample_weights():
        return torch.rand(POLE_OBSERVATION_DIM).mul_(2).sub_(1)  # weights initialized in range [-1, 1]

    def take_action(self, observation):
        if self.w is None:
            raise RuntimeError("Agent has to be trained first.")

        return 1 if torch.dot(self.w, torch.FloatTensor(observation)) >= 0 else 0


class SimpleAgent(BaseAgent):
    ''' Simple agent - samples a random weight when trained '''
    def train(self):
        self.w = BaseAgent.sample_weights()

    def __str__(self):
        return 'Simple Agent'


class RandomSearchAgent(BaseAgent):
    ''' Performs random-search by sampling multiple times in a greedy manner '''

    def random_search(self, samples_count):
        sim = PoleSimulation()

        max_score = 0
        best_w = None

        for _ in range(samples_count):
            self.w = BaseAgent.sample_weights()
            score = sim.run(self)
            if score >= max_score:
                max_score = score
                best_w = self.w

        self.w = best_w
        return max_score

    def train(self):
        ''' Run random search using RANDOM_SEARCH_SAMPLE_COUNT samples '''
        self.random_search(RANDOM_SEARCH_SAMPLE_COUNT)

    def eval(self):
        '''
            Runs random search until we hit the target score.
            Note: does not evaluate the trained model,
            but the training scheme itself (thus overrides training results)
        '''
        total_episodes = 1
        score = self.random_search(1)
        while score < EVAL_SCORE:
            total_episodes += 1
            score = self.random_search(1)

        return total_episodes

    def __str__(self):
        return 'Random Search Agent'


class PoleSimulation:

    @staticmethod
    def run(agent):
        '''
        :return: Runs agent on the pole simulation for a single episode and returns it's score (total reward accumulated)
        '''

        try:
            env = gym.make('CartPole-v0')
            score = 0
            for i_episode in range(EPISODES_COUNT):
                observation = env.reset()
                for t in range(STEPS_COUNT):
                    if RENDER_SIM:
                        env.render()
                    action = agent.take_action(observation)
                    observation, reward, done, info = env.step(action)
                    score += reward
                    if done:
                        break

            print(str(agent) + ' finished {0} episode(s) after {1} timesteps'.format(i_episode+1, t+1))
            return score

        finally:
            env.close()  # Required to fix ImportError: sys.meta_path bug


def run_simple_agent_simulation():
    sim = PoleSimulation()
    agent = SimpleAgent()
    agent.train()
    score = sim.run(agent)
    print(str(agent) + ' achieved a score of {}'.format(score))


def run_random_search_simulation():
    sim = PoleSimulation()
    agent = RandomSearchAgent()
    agent.train()
    score = sim.run(agent)
    print(str(agent) + ' achieved a score of {}'.format(score))


def evaluate_random_search_agent():

    episodes_accum = []

    # Run multiple search steps
    for idx in range(EVAL_ITERATIONS):
        agent = RandomSearchAgent()
        episodes = agent.eval()
        print(str(agent) + ' #{0} finished evaluation step after {1} episodes'.format(idx + 1, episodes))
        episodes_accum.append(episodes)

    # Calc average number of episodes required
    sum = 0
    for elem in episodes_accum:
        sum += elem
    avg = sum / len(episodes_accum)
    print("The average number of episodes needed to hit the target score is {0:.2f}".format(avg))

    # Plot results
    plt.hist(episodes_accum, bins=range(min(episodes_accum), max(episodes_accum)+2))
    plt.xlabel('Episode Count')
    plt.ylabel('Frequency')

    plt.savefig('RandomSearchEval.png')
    plt.show()


parser = argparse.ArgumentParser(description='Choose simulation scheme')
parser.add_argument('--run', help='Type of simulation to run')
args = parser.parse_args()
if args.run == 'simple':
    run_simple_agent_simulation()
elif args.run == 'randomsearch':
    run_random_search_simulation()
elif args.run == 'eval_randomsearch':
    evaluate_random_search_agent()
