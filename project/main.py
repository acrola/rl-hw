import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
from utils.experiments_mgr import start_experiments_generator

# Program parameters are set via the experiments_mgr
# (to support custom arguments and sampling from ranges)

def main(env, num_timesteps, experiment_config, experiment_name):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=experiment_config['lr'], alpha=experiment_config['alpha'], eps=experiment_config['eps']),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        experiment_name=experiment_name,
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=experiment_config['replay_size'],
        batch_size=experiment_config['batch'],
        gamma=experiment_config['gamma'],
        learning_starts=experiment_config['learning_start'],
        learning_freq=experiment_config['learning_freq'],
        frame_history_len=experiment_config['frame_hist'],
        target_update_freq=experiment_config['target_update_freq'],
        output_path=experiment_config['output']
    )


if __name__ == '__main__':

    experiments_generator = start_experiments_generator()

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    for exp_num, (experiment_config, experiment_name) in enumerate(experiments_generator):
        print('Beginning experiment: #' + str(exp_num+1) + ': ' + experiment_name)

        # Change the index to select a different game.
        assert 0 <= experiment_config['game'] < len(benchmark.tasks), 'Illegal Atari game id'
        task = benchmark.tasks[experiment_config['game']] # By default - use Pong (id 3)

        # Run training
        seed = experiment_config['seed']   # By default - use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)

        # Take minimum between custom configuration and task configuration
        max_timestamps = min(experiment_config['max_steps'], task.max_timesteps)

        main(env, max_timestamps, experiment_config, experiment_name)

        print('Ended experiment: #' + str(exp_num + 1) + ': ' + experiment_name)
