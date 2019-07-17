
import numpy as np

import argparse
import os
import pickle

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def get_key(d, idx):
    return list(d.keys())[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='view the log of an experiment directory')
    parser.add_argument('exp', type=str, help='experiment directory')

    args = parser.parse_args()

    log_dir = os.path.join(args.exp, 'logs/log.pickle')


    with open(log_dir, 'rb') as f:
        log = pickle.load(f)

    del log['running_score']
    # for k, v in log.items():
    #     plt.title(k)
    #     plt.plot(v)
    #     plt.show()

    d = len(log.keys())
    n = len(log['b_end_before'])

    X = np.zeros((n,d))

    for i in range(n):
        for j, (k, v) in enumerate(log.items()):
            X[i, j] = v[i]

    X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0)+1e-4)

    var_ratios = []
    for i in range(10):
        pca = PCA(n_components=i+1)
        pca.fit(X_scaled)
        var_ratios.append(np.sum(pca.explained_variance_ratio_))

    plt.plot(var_ratios)
    plt.show()