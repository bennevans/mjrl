

from mjrl.utils.gym_env import GymEnv

from train_q_function import train_baseline
from evaluate_q_function import *

import mjrl.envs
import pickle
import os
import itertools
import json
import time

def run_exp_many(env, n, rb, policy, params, gamma, return_errors=False):
    epochs = params['epochs']
    fit_iters = params['fit_iters']
    batch_size = params['batch_size']
    hidden_sizes = params['hidden_sizes']
    lr = params['lr']
    use_time = params['use_time']

    baselines = []
    errors = []
    for _ in range(n):
        baseline, error = train_baseline(e, rb, policy, epochs, fit_iters,
            lr, batch_size, hidden_sizes, gamma, return_error=True, use_time=use_time)
        baselines.append(baseline)
        errors.append(error)

    if return_errors:
        return baselines, errors

    return baselines

def get_param_list(possible_params):
    epochs = possible_params['epochs']
    fit_iters = possible_params['fit_iters']
    batch_size = possible_params['batch_size']
    hidden_sizes = possible_params['hidden_sizes']
    lrs = possible_params['lrs']
    use_time = possible_params['use_time']

    params = [epochs, fit_iters, batch_size, hidden_sizes, lrs, use_time]

    permutations = itertools.product(*params)

    param_list = []
    for item in permutations:
        param = {
            'epochs': item[0],
            'fit_iters': item[1],
            'batch_size': item[2],
            'hidden_sizes': item[3],
            'lr': item[4],
            'use_time': item[5]
        }
        param_list.append(param)
    return param_list

def evaluate_baseline(baseline, paths, base_dir, baseline_name, gamma, write_files=True, suffix=''):

    if write_files:
        baseline_dir = os.path.join(base_dir, baseline_name)
        if not os.path.exists(baseline_dir):
            os.mkdir(baseline_dir)

        baseline_file = os.path.join(baseline_dir, 'baseline.pickle')
        exp_info_file = os.path.join(baseline_dir, 'info{}.json'.format(suffix))
        plot_1_file = os.path.join(baseline_dir, 'q_vs_mc_1{}.png'.format(suffix))
        plot_5_file = os.path.join(baseline_dir, 'q_vs_mc_5{}.png'.format(suffix))
        plot_end_file = os.path.join(baseline_dir, 'q_vs_mc_end{}.png'.format(suffix))

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

    if write_files:
        with open(exp_info_file, 'w') as f:
            json.dump(exp_info, f, sort_keys=True, indent=4)
        
        plt.clf()
        plt.title('1 step Q vs MC')
        plt.xlabel('$Q(s_1,a_1)$')
        plt.ylabel('$MC + \\gamma^T Q(s_2, a_2)$')
        plt.scatter(pred_1, mc_1)
        plt.savefig(plot_1_file)

        plt.clf()
        plt.title('5 step Q vs MC')
        plt.xlabel('$Q(s_1,a_1)$')
        plt.ylabel('$MC + \\gamma^T Q(s_6, a_6)$')
        plt.scatter(pred_5, mc_5)
        plt.savefig(plot_5_file)

        plt.clf()
        plt.title('end step Q vs MC')
        plt.xlabel('$Q(s_1,a_1)$')
        plt.ylabel('$MC + \\gamma^T Q(s_T, a_T)$')
        plt.scatter(pred_start_end, mc_start_end)
        plt.savefig(plot_end_file)
    
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

def evaluate_and_save_baselines(baselines, params, paths, paths_test, base_dir, i, gamma):
    exp_dir = os.path.join(base_dir, 'exp_{}/'.format(i))
    metadata = os.path.join(exp_dir, 'params.json')
    results = os.path.join(exp_dir, 'results.json')

    os.mkdir(exp_dir)

    with open(metadata, 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

    exp_info_list = []
    exp_info_list_test = []

    for i, baseline in enumerate(baselines):
        exp_info = evaluate_baseline(baseline, paths, exp_dir, 'baseline_{}'.format(i), gamma)
        exp_info_list.append(exp_info)

        exp_info_test = evaluate_baseline(baseline, paths_test, exp_dir, 'baseline_{}'.format(i), gamma, suffix='_test')
        exp_info_list_test.append(exp_info_test)
    
    exp_info_dict = combine_infos(exp_info_list)
    exp_info_dict_test = combine_infos(exp_info_list_test)

    with open(results, 'w') as f:
        json.dump(exp_info_dict, f, sort_keys=True, indent=4)

    return exp_info_dict, exp_info_dict_test

def enumerate_run_save(env, n, rb, paths, paths_test, policy, possible_params, gamma, base_dir):

    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print('experiment directory already exists! exiting')
        return

    results = os.path.join(base_dir, 'results.json')

    param_list = get_param_list(possible_params)

    all_info = {}

    for i, params in enumerate(param_list):
        print(i, params)
        start_time = time.time()
        baselines, errors = run_exp_many(env, n, rb, policy, params, gamma, return_errors=True)
        run_end_time = time.time()

        exp_info_dict, exp_test_dict = evaluate_and_save_baselines(baselines, params, paths, paths_test, base_dir, i, gamma)

        eval_end_time = time.time()

        all_info[i] = {
            'params': params,
            'info': exp_info_dict,
            'info_test': exp_test_dict,
            'bellman_errors': errors,
            'start_time': start_time,
            'run_end_time': run_end_time,
            'eval_end_time': eval_end_time,
            'total_runtime': eval_end_time - start_time,
            'run_runtime': run_end_time - start_time,
            'eval_runtime': eval_end_time - run_end_time
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

    paths_test = pickle.load(open('paths_test.pickle', 'rb'))

    base_dir = './q_exps/acro_fit_iters_0/'
    # base_dir = './q_exps/testing/'

    gamma = 0.995
    n = 5

    possible_params = {
        'epochs': [1],
        'fit_iters': [1000, 2000, 5000, 10000],
        'batch_size': [64],
        'hidden_sizes': [(64, 64)],
        'lrs': [1e-4],
        'use_time': [True]
    }
    
    start_time = time.time()
    
    metadata = {
        'start_time': start_time,
        'n': n,
        'gamma': gamma,
        'policy_dir': policy_dir,
        'base_dir': base_dir,
        'possible_params': possible_params
    }

    enumerate_run_save(e, n, rb, paths, paths_test, policy, possible_params, gamma, base_dir)

    metadata['end_time'] = time.time()
    metadata['total_time'] = metadata['end_time'] - metadata['start_time']

    metadata_dir = os.path.join(base_dir, 'metadata.json')

    with open(metadata_dir, 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4)
