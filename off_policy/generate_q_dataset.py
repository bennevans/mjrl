
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
from mjrl.utils.replay_buffer import ReplayBuffer

import mjrl.envs

import pickle

mode = 'pm'

if mode == 'pm':
    policy_dir = 'point_mass_exp1/iterations/best_policy.pickle'
    e = GymEnv('mjrl_point_mass-v0')
elif mode == 'acrobot':
    policy_dir = 'acrobot_exp1/iterations/best_policy.pickle'
    e = GymEnv('mjrl_acrobot-v0')
elif mode =='swimmer':
    policy_dir = 'swimmer_testing/iterations/best_policy.pickle'
    e = GymEnv('mjrl_swimmer-v0')
else:
    raise Exception('bad mode: {}'.format(mode))

policy = pickle.load(open(policy_dir, 'rb'))


# dataset params
K = 20
T = e.horizon
seed = 2

print("K: {} T: {} KT: {} seed: {} mode: {}".format(K, T, K*T, seed, mode))

rb = ReplayBuffer(max_dataset_size=10000)

paths = sample_paths(K, e, policy, horizon=T, eval_mode='evaluation')

rb.update(paths)

# pickle.dump(rb, open('rb_test.pickle', 'wb'))
# pickle.dump(paths, open('paths_test.pickle', 'wb'))

pickle.dump(rb, open('replay_buf.pickle', 'wb'))
pickle.dump(paths, open('paths.pickle', 'wb'))
