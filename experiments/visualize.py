
import pickle
import argparse

from mjrl.utils.gym_env import GymEnv
import mjrl.envs


parser = argparse.ArgumentParser(description='visualize experiment')
parser.add_argument('exp_dir')

args = parser.parse_args()
print(args)

exp_dir = args.exp_dir
policy_dir = exp_dir + '/iterations/best_policy.pickle'

policy = pickle.load(open(policy_dir, 'rb'))

e = GymEnv('mjrl_point_mass-v0')
e.visualize_policy(policy, num_episodes=5, horizon=e.horizon, mode='evaluation')

