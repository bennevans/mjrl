


import argparse
import os
import pickle
import csv

from process_pg_exp import get_experiments, get_non_unique

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key', type=str)
    parser.add_argument('exp', type=str, help='experiments')

    args = parser.parse_args()

    base_dir = args.exp

    STATS = [('MSE_end_after', True), ('eval_score', False), ('stoc_pol_max', False)]

    experiments = get_experiments(args.exp, STATS)

    non_unique_params = get_non_unique(experiments)


    param_name = list(non_unique_params.keys())[0]

    data = {}

    for exp in experiments:
        param_val = exp['params'][param_name]
        for i, run in enumerate(exp['logs']):
            iter_data = run[args.key]
            header = '{}: {} run: {}'.format(param_name, param_val, i)
            data[header] = iter_data
        
    out_csv_dir = os.path.join(base_dir, '{}.csv'.format(args.key))
    field_names = sorted(data.keys())
    with open(out_csv_dir, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for i in range(len(data[field_names[0]])):
            row = {}
            for name in field_names:
                row[name] = data[name][i]
            writer.writerow(row)