

import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_log_max(log, element):
    return np.max(log[element])

def get_log_avg(log, element):
    return np.average(log[element])

def get_logs_avg(logs, element):
    total = 0.0
    for log in logs:
        total += get_log_avg(log, element)
    return total / len(logs)

def get_logs_max(logs, element):
    total = 0.0
    for log in logs:
        total += get_log_max(log, element)
    return total / len(logs)

def experiment_statistics(logs, stats):

    info = {}

    for element, avg in stats:
        if avg:
            info[element] = get_logs_avg(logs, element)
        else:
            info[element] = get_logs_max(logs, element)  

    return info

def run_statistics(logs, stats):
    info = {}
    for element, avg in stats:
        for log in logs:
            if avg:
                if element in info:
                    info[element].append(get_log_avg(log, element))
                else:
                    info[element] = [get_log_avg(log, element)]
            else:
                if element in info:
                    info[element].append(get_log_avg(log, element))
                else:
                    info[element] = [get_log_avg(log, element)]
    return info

def get_non_unique(experiments):
    all_kv = {}
    for exp in experiments:
        param = exp['params']
        for k, v in param.items():
            if k in all_kv:
                if type(v) is list:
                    all_kv[k].add(tuple(v))
                else:
                    all_kv[k].add(v)
            else:
                if type(v) is list:
                    all_kv[k] = {tuple(v)}
                else:
                    all_kv[k] = {v}
    non_unique = {}
    for k, v in all_kv.items():
        if len(v) > 1:
            non_unique[k] = v
    return non_unique

def scatter_stats(experiments, k1, k2):
    l1 = []
    l2 = []
    
    for exp in experiments:
        l1.append(exp['stats'][k1])
        l2.append(exp['stats'][k2])

    plt.xlabel(k1)
    plt.ylabel(k2)
    plt.scatter(l1, l2)
    plt.show()


def scatter_run_stats(experiments, k1, k2, color_key):

    info = {}
    
    for exp in experiments:
        key = exp['params'][color_key]
        info[key] = {}
        info[key]['x'] = exp['run_stats'][k1]
        info[key]['y'] = exp['run_stats'][k2]
        info[key]['color'] = [exp['params'][color_key]] * len(exp['run_stats'][k2])

    plt.xlabel(k1)
    plt.ylabel(k2)

    for k,v in info.items():
        plt.scatter(v['x'], v['y'], label='{}: {}'.format(color_key, v['color'][0]))
    plt.legend()
    plt.show()

def examine_run_stat(experiments, param, stat):
    
    xs = []
    ys = []
    for exp in experiments:
        n = len(exp['run_stats'][stat])
        xs += [exp['params'][param]] * n
        ys += exp['run_stats'][stat]

    lin_reg = LinearRegression()
    lin_reg.fit(np.expand_dims(np.array(xs), 1), ys)
    
    pltxs = np.linspace(np.min(xs), np.max(xs))
    pltys = lin_reg.predict(np.expand_dims(np.array(pltxs), 1))

    m = lin_reg.coef_[0]
    b = lin_reg.intercept_
    r_sq = lin_reg.score(np.expand_dims(np.array(xs), 1), ys)

    print(m, b, r_sq)

    plt.xlabel(param)
    plt.ylabel(stat)
    plt.scatter(xs, ys)
    plt.plot(pltxs, pltys)
    plt.show()

def plot_curves(experiment, param):
    for log in experiment['logs']:
        plt.title('{} vs iterations'.format(param))
        plt.xlabel('iterations')
        plt.ylabel(param)
        plt.plot(log[param])
    
    plt.show()


if __name__ == '__main__':
    base_dir = './pg_exp/time_swimmer_dataset_size_0'
    # base_dir = '/tmp/swimmer_dataset_fit_iter_0/'
    # base_dir = '/tmp/swimmer_random_epochs_fit_iters_lr_size_0/'

    STATS = [('MSE_end_after', True), ('eval_score', False), ('stoc_pol_max', False)]

    sub_dirs = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    experiments = []

    for sub_dir in sub_dirs:
        exp_dir = os.path.join(base_dir, sub_dir)

        sub_exp_dirs = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))]

        param_dir = os.path.join(exp_dir, 'params.json')

        with open(param_dir, 'r') as f:
            param = json.load(f)


        experiment_logs = []

        for sub_exp_dir in sub_exp_dirs:
            run_dir = os.path.join(exp_dir, sub_exp_dir)

            log_dir = os.path.join(run_dir, 'logs/log.pickle')

            with open(log_dir, 'rb') as f:
                log = pickle.load(f)
                experiment_logs.append(log)

        experiment_stats = experiment_statistics(experiment_logs, STATS)
        run_stats = run_statistics(experiment_logs, STATS)
        # info = process_logs(experiment_logs)

        exp = {
            'stats': experiment_stats,
            'run_stats': run_stats,
            'params': param,
            'logs': experiment_logs
        }

        experiments.append(exp)

    non_unique_params = get_non_unique(experiments)
        