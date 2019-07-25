import argparse
import pyperclip
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, help='experiment directory')
    parser.add_argument('col', type=str, help='log key')

    args = parser.parse_args()

    log_path = os.path.join(args.exp, 'logs/log.pickle')

    with open(log_path, 'rb') as f:
        log = pickle.load(f)
    
    values = log[args.col]

    csv_str = str(args.col + '\n')

    for v in values:
        csv_str += str(v) +'\n'
    
    pyperclip.copy(csv_str)

    print(csv_str)