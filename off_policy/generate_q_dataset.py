
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.trajectory_sampler import sample_paths
from mjrl.utils.replay_buffer import ReplayBuffer

import mjrl.envs

import pickle

mode = 'acrobot'

if mode == 'pm':
    policy_dir = 'point_mass_exp1/iterations/best_policy.pickle'
    e = GymEnv('mjrl_point_mass-v0')
elif mode == 'acrobot':
    policy_dir = 'acrobot_exp1/iterations/best_policy.pickle'
    e = GymEnv('mjrl_acrobot-v0')
else:
    raise Exception('bad mode: {}'.format(mode))

policy = pickle.load(open(policy_dir, 'rb'))


# dataset params
K = 100
T = e.horizon
seed = 11

rb = ReplayBuffer()

paths = sample_paths(K, policy, T=T, env=e, env_name='myenv_name', mode='evaluation')

rb.update(paths)

pickle.dump(rb, open('rb.pickle', 'wb'))
pickle.dump(paths, open('paths.pickle', 'wb'))
