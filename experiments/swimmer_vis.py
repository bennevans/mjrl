
import pickle
import argparse

from mjrl.utils.gym_env import GymEnv
import mjrl.envs


parser = argparse.ArgumentParser(description='visualize experiment')
parser.add_argument('exp_dir')
parser.add_argument('-n', '--episodes', type=int, default=5)

args = parser.parse_args()
print(args)

exp_dir = args.exp_dir
# policy_dir = exp_dir + '/iterations/best_policy.pickle'
policy_dir = exp_dir + '/iterations/policy_25.pickle'

policy = pickle.load(open(policy_dir, 'rb'))

e = GymEnv('mjrl_swimmer-v0')
e.visualize_policy(policy, num_episodes=args.episodes, horizon=e.horizon, mode='evaluation')

