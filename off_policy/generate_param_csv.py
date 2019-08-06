


import argparse
import os
import pickle
import csv
import json
import pyperclip

from process_pg_exp import get_experiments, get_non_unique

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, help='experiments')

    args = parser.parse_args()

    base_dir = args.exp

    experiments = get_experiments(base_dir, [])
    non_unique = list(get_non_unique(experiments).keys())
    
    with open(os.path.join(base_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    data = {}

    for param_dict in metadata['param_list']:
        for param, value in param_dict.items():
            if param not in non_unique:
                data[param] = value
    
    kv = sorted(data.items())

    csv_str = ""
    for k, v in kv:
        csv_str += '{}; {}\n'.format(k, v)

    with open(os.path.join(base_dir, 'params.csv'), 'w') as f:
        f.write(csv_str)
    