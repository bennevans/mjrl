
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.process_samples import discount_sum
from mjrl.q_baselines.mlp_baseline import MLPBaseline

import mjrl.envs

import pickle

policy_dir = 'point_mass_exp1/iterations/best_policy.pickle'

policy = pickle.load(open(policy_dir, 'rb'))
rb = pickle.load(open('rb.pickle', 'rb'))
paths = pickle.load(open('paths.pickle', 'rb'))

e = GymEnv('mjrl_point_mass-v0')

learn_rate = 1e-3
batch_size = 64
epochs = 10
hidden_sizes = [64, 64]

baseline = MLPBaseline(e, learn_rate=learn_rate, batch_size=batch_size, epochs=epochs, hidden_sizes=hidden_sizes)

# params
gamma = 0.95

baseline.fit_off_policy_many(rb, policy, gamma)

for path in paths:
    pred = baseline.predict(path)
    mc = discount_sum(path['rewards'], gamma)
    print('----------')
    print(pred)
    print(mc)
    print('----------')


