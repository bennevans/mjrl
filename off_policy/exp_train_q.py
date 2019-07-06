

from mjrl.utils.gym_env import GymEnv

from train_q_function import train_baseline
from evaluate_q_function import *

import mjrl.envs
import pickle
import os
import itertools
import json

def run_exp_many(env, n, rb, policy, params, gamma):
    epochs = params['epochs']
    fit_iters = params['fit_iters']
    batch_size = params['batch_size']
    hidden_sizes = params['hidden_sizes']
    lr = params['lr']

    baselines = []
    for _ in range(n):
        baseline = train_baseline(e, rb, policy, epochs, fit_iters, lr, batch_size, hidden_sizes, gamma)
        baselines.append(baseline)

    return baselines

def get_param_list(possible_params):
    epochs = possible_params['epochs']
    fit_iters = possible_params['fit_iters']
    batch_size = possible_params['batch_size']
    hidden_sizes = possible_params['hidden_sizes']
    lrs = possible_params['lrs']

    params = [epochs, fit_iters, batch_size, hidden_sizes, lrs]

    permutations = itertools.product(*params)

    param_list = []
    for item in permutations:
        param = {
            'epochs': item[0],
            'fit_iters': item[1],
            'batch_size': item[2],
            'hidden_sizes': item[3],
            'lr': item[4]
        }
        param_list.append(param)
    return param_list

def evaluate_baseline(baseline, paths, base_dir, baseline_name, gamma):

    baseline_dir = os.path.join(base_dir, baseline_name)
    os.mkdir(baseline_dir)

    baseline_file = os.path.join(baseline_dir, 'baseline.pickle')
    exp_info_file = os.path.join(baseline_dir, 'info.json')

    with open(baseline_file, 'wb') as f:
        pickle.dump(baseline, f)
    
    pred_1, mc_1 = evaluate_n_step(1, gamma, paths, baseline)
    pred_5, mc_5 = evaluate_n_step(5, gamma, paths, baseline)
    pred_start_end, mc_start_end = evaluate_start_end(gamma, paths, baseline)

    mse_1 = mse(pred_1, mc_1)
    mse_5 = mse(pred_5, mc_5)
    mse_start_end = mse(pred_start_end, mc_start_end)

    line_fit_1 = line_fit(pred_1, mc_1)
    line_fit_5 = line_fit(pred_5, mc_5)
    line_fit_start_end = line_fit(pred_start_end, mc_start_end)
    
    m_1, b_1, r2_1 = line_fit_1
    m_5, b_5, r2_5 = line_fit_5
    m_end, b_end, r2_end = line_fit_start_end

    exp_info = {
        '1': {
            'mse': mse_1,
            'm': m_1,
            'b': b_1,
            'r2': r2_1
        },
        '5': {
            'mse': mse_5,
            'm': m_5,
            'b': b_5,
            'r2': r2_5
        },
        'end': {
            'mse': mse_start_end,
            'm': m_end,
            'b': b_end,
            'r2': r2_end
        }
    }

    with open(exp_info_file, 'w') as f:
        json.dump(exp_info, f, sort_keys=True, indent=4)
    
    return exp_info

def combine_infos(exp_info_list):
    exp_info_dict = {
        k : { 
            'mse': [], 
            'm': [], 
            'b': [], 
            'r2': [], 
        } for k in ['1', '5', 'end'] 
    }

    for exp_info in exp_info_list:
        for k, v in exp_info.items():
            for k1 in exp_info_dict[k]:
                exp_info_dict[k][k1].append(v[k1])

    return exp_info_dict

def evaluate_and_save_baselines(baselines, params, paths, base_dir, i, gamma):
    exp_dir = os.path.join(base_dir, 'exp_{}/'.format(i))
    metadata = os.path.join(exp_dir, 'params.json')
    results = os.path.join(exp_dir, 'results.json')

    os.mkdir(exp_dir)

    with open(metadata, 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

    exp_info_list = []

    for i, baseline in enumerate(baselines):
        exp_info = evaluate_baseline(baseline, paths, exp_dir, 'baseline_{}'.format(i), gamma)
        exp_info_list.append(exp_info)
    
    exp_info_dict = combine_infos(exp_info_list)

    with open(results, 'w') as f:
        json.dump(exp_info_dict, f, sort_keys=True, indent=4)

    return exp_info_dict

def enumerate_run_save(env, n, rb, paths, policy, possible_params, gamma, base_dir):

    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print('experiment directory already exists! exiting')
        return

    metadata = os.path.join(base_dir, 'possible_params.json')
    results = os.path.join(base_dir, 'results.json')

    with open(metadata, 'w') as f:
        json.dump(possible_params, f, sort_keys=True, indent=4)

    param_list = get_param_list(possible_params)

    all_info = {}

    for i, params in enumerate(param_list):
        print(i, params)
        baselines = run_exp_many(env, n, rb, policy, params, gamma)

        exp_info_dict = evaluate_and_save_baselines(baselines, params, paths, base_dir, i, gamma)

        all_info[i] = {
            'params': params,
            'info': exp_info_dict
        }
    
    with open(results, 'w') as f:
        json.dump(all_info, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    # policy_dir = 'point_mass_exp1/iterations/best_policy.pickle'
    # e = GymEnv('mjrl_point_mass-v0')
    policy_dir = 'acrobot_exp1/iterations/best_policy.pickle'
    e = GymEnv('mjrl_acrobot-v0')
    policy = pickle.load(open(policy_dir, 'rb'))
    rb = pickle.load(open('rb.pickle', 'rb'))
    paths = pickle.load(open('paths.pickle', 'rb'))

    base_dir = './q_exps/vary_epochs/'

    gamma = 0.995
    n = 2

    possible_params = {
        'epochs': [1, 2, 5, 20, 100],
        'fit_iters': [50],
        'batch_size': [64],
        'hidden_sizes': [(64, 64)],
        'lrs': [1e-4]
    }

    enumerate_run_save(e, n, rb, paths, policy, possible_params, gamma, base_dir)
