
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.process_samples import discount_sum
from mjrl.q_baselines.mlp_baseline import MLPBaseline

import matplotlib.pyplot as plt

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
rb = pickle.load(open('rb.pickle', 'rb'))
paths = pickle.load(open('paths.pickle', 'rb'))


learn_rate = 1e-4
batch_size = 32
epochs = 1
hidden_sizes = [256, 256, 256]
fit_iters = 250

baseline = MLPBaseline(e,
    learn_rate=learn_rate, batch_size=batch_size, epochs=epochs,
    hidden_sizes=hidden_sizes, fit_iters=fit_iters)

# params
gamma = 0.995

def get_first(path): 
    new_path = {} 
    for k in path: 
        try: 
            new_path[k] = path[k][0:1] 
        except: 
            pass 
    return new_path 

def get_last(path): 
    new_path = {} 
    for k in path: 
        try: 
            new_path[k] = path[k][-1:] 
        except: 
            pass 
    return new_path 

def evaluate(path):
    T = len(path['actions'])
    p0 = get_first(path)
    pl = get_last(path)
    pred = baseline.predict(p0)
    last = baseline.predict(pl)
    mc = discount_sum(path['rewards'], gamma)
    # print('----------')
    # print("Q(s,a), MC + terminal")
    # print(pred, mc[0] + + gamma**T * last)
    # print("Monte Carlo Rollout")
    # print(mc[0])
    # print("MC + gamma^T * Q(s_T, a_T)")
    # print(mc[0] + + gamma**T * last)
    # print("max q", (1-gamma**T) / (1-gamma))
    # print('----------')
    return pred, mc[0] + + gamma**T * last

print('fitting baseline')
baseline.fit_off_policy_many(rb, policy, gamma)

preds = []
mc_terms = []
for path in paths:
    pred, mc_term = evaluate(path)
    preds.append(pred[0])
    mc_terms.append(mc_term[0])
    print(pred, mc_term)

plt.xlabel('$Q(s_1,a_1)$')
plt.ylabel('$MC + \\gamma^T Q(s_T, a_T)$')
plt.scatter(preds, mc_terms)
plt.show()
plt.savefig('q_vs_mc.png')

if mode == 'pm':
    pickle.dump(baseline, open('baseline.pickle', 'wb'))
elif mode == 'acrobot':
    pickle.dump(baseline, open('baseline_acro.pickle', 'wb'))