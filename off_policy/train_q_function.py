
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.process_samples import discount_sum
from mjrl.q_baselines.mlp_baseline import MLPBaseline

import matplotlib.pyplot as plt

import mjrl.envs
import pickle

mode = 'acrobot'



def train_baseline(env, rb, policy, epochs, fit_iters, learn_rate, batch_size, hidden_sizes, gamma, return_error=False):

    baseline = MLPBaseline(env,
        learn_rate=learn_rate, batch_size=batch_size, epochs=epochs,
        hidden_sizes=hidden_sizes, fit_iters=fit_iters)

    final_bellman_error = baseline.fit_off_policy_many(rb, policy, gamma)
    
    if return_error:
        return baseline, final_bellman_error
        
    return baseline


if __name__ == '__main__':

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
    batch_size = 128
    epochs = 1
    hidden_sizes = [64, 64]
    fit_iters = 100
    gamma = 0.995

    baseline = train_baseline(e, rb, policy, epochs, fit_iters, learn_rate, batch_size, hidden_sizes, gamma)

    if mode == 'pm':
        pickle.dump(baseline, open('baseline.pickle', 'wb'))
    elif mode == 'acrobot':
        pickle.dump(baseline, open('baseline_acro.pickle', 'wb'))

    # do some light evaluation right after training
    from evaluate_q_function import *

    print('evaluating')
    pred_1, mc_1 = evaluate_n_step(1, gamma, paths, baseline)
    pred_5, mc_5 = evaluate_n_step(5, gamma, paths, baseline)
    pred_start_end, mc_start_end = evaluate_start_end(gamma, paths, baseline)

    print('1 step')
    print('mse', mse(pred_1, mc_1))
    print('line_fit', line_fit(pred_1, mc_1))

    plt.figure()
    plt.xlabel('$Q(s_1,a_1)$')
    plt.ylabel('$MC + \\gamma^T Q(s_2, a_2)$')
    plt.title('1 step')
    plt.scatter(pred_1, mc_1)

    print('5 step')
    print('mse', mse(pred_5, mc_5))
    print('line_fit', line_fit(pred_5, mc_5))

    plt.figure()
    plt.xlabel('$Q(s_1,a_1)$')
    plt.ylabel('$MC + \\gamma^T Q(s_5, a_5)$')
    plt.title('5 step')
    plt.scatter(pred_5, mc_5)


    print('T step')
    print('mse', mse(pred_start_end, mc_start_end))
    print('line_fit', line_fit(pred_start_end, mc_start_end))

    plt.figure()
    plt.xlabel('$Q(s_1,a_1)$')
    plt.ylabel('$MC + \\gamma^T Q(s_T, a_T)$')
    plt.title('T step')
    plt.scatter(pred_start_end, mc_start_end)

    plt.show()
