
from mjrl.utils.gym_env import GymEnv

import mjrl.envs

import argparse
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize an experiment directory')
    parser.add_argument('exp', type=str, help='experiment directory')
    parser.add_argument('env', type=str, help='eniroment type')

    parser.add_argument('-E', '--episodes', type=int, default=1)
    parser.add_argument('-H', '--horizon', type=int)
    parser.add_argument('-P', '--policy', type=str)

    args = parser.parse_args()

    e = GymEnv(args.env)

    if args.policy is None:
        policy_dir = os.path.join(args.exp, 'iterations/best_policy.pickle')
    else:
        policy_dir = args.policy

    with open(policy_dir, 'rb') as f:
        policy = pickle.load(f)


    if args.horizon is None:
        horizon = e.horizon
    else:
        horizon = args.horizon
    e.visualize_policy(policy, num_episodes=args.episodes, horizon=horizon, mode='evaluation')

    print(args)
