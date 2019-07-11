
import json
import os
import csv

import matplotlib.pyplot as plt
import numpy as np



def visualize_attribute(experiment, attr):
    plt.figure(1)
    plt.title('1, 5, end attribute: {}'.format(attr))

    for i, length in enumerate(experiment['info']):
        plt.subplot(131 + i)
        plt.hist(experiment['info'][length][attr])
        plt.xlabel(attr)
    plt.show()

def summarize(experiment):
    results = {}
    for length in experiment['info']:
        results[length] = {
            'mse': np.mean(experiment['info'][length]['mse']) # just mse for now, can add others later
        }
    return results

def summarize_all(experiments):
    results = []
    for _, exp in experiments.items():
        results.append((summarize(exp), exp['params']))
    return results

def get_best_mse(summary, length):
    res = (summary[0])
    best = -1
    for i, (result, params) in enumerate(summary):
        if result[length]['mse'] < res[0][length]['mse']:
            res = (result, params)
            best = i

    return res, best

def get_csv(results):
    params_header = ['batch_size', 'epochs', 'fit_iters', 'hidden_sizes', 'lr', 'use_time']
    results_header = ['mse', 'm', 'b', 'r2']
    all_rows = [params_header + ['length'] + results_header]
    for res in results.values():
        params_row = [res['params'][p] for p in params_header]
        for length, data in res['info_test'].items():
            n = len(data['mse'])
            rows = [[]] * n
            for i in range(n):
                res_row = [data[p][i] for p in results_header]
                rows[i] = params_row + [length] + res_row
            all_rows += rows

    return all_rows


if __name__ == '__main__':
    base_dir = './q_exps/acro_testing'
    data_dir = os.path.join(base_dir, 'results.json')
    output_dir = os.path.join(base_dir, 'results.csv')

    results = json.load(open(data_dir, 'r'))
    
    # visualize_attribute(results['0'], 'mse')
    summary = summarize_all(results)
    best, idx = get_best_mse(summary, 'end')

    all_rows = get_csv(results)

    with open(output_dir, 'w', newline='') as csvfile:
        sw = csv.writer(csvfile, delimiter=';')
        sw.writerows(all_rows)
    
