
from mjrl.utils.process_samples import discount_sum
import matplotlib.pyplot as plt
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression


def get_ith(path, i):
    new_path = {} 
    for k in path: 
        try: 
            new_path[k] = path[k][i:i+1] 
        except: 
            pass 
    return new_path 

def get_first(path):
    return get_ith(path, 0)

def get_last(path): 
    last_idx = len(path['observations']) - 1
    return get_ith(path, last_idx)

def evaluate(path, gamma, baseline):
    T = len(path['actions'])
    p0 = get_first(path)
    pl = get_last(path)
    pred = baseline.predict(p0)
    last = baseline.predict(pl)
    mc = discount_sum(path['rewards'], gamma)
    return pred, mc[0] + + gamma**T * last

def evaluate_idx(path, start_idx, end_idx, gamma, baseline):
    if start_idx >= end_idx:
        raise IndexError('start_idx should be < than end_idx')
    
    p0 = get_ith(path, start_idx)
    pl = get_ith(path, end_idx)
    pred = baseline.predict(p0)
    last = baseline.predict(pl)
    mc = discount_sum(path['rewards'][start_idx:end_idx], gamma)

    return pred, mc[0] + + gamma**(end_idx - start_idx) * last

def evaluate_start_end(gamma, paths, baseline):
    preds = []
    mc_terms = []
    for path in paths:
        pred, mc_term = evaluate(path, gamma, baseline)
        preds.append(pred[0])
        mc_terms.append(mc_term[0])
    return preds, mc_terms

def evaluate_n_step(n, gamma, paths, baseline):
    preds = []
    mc_terms = []
    for path in paths:
        T = len(path['observations'])
        for t in range(T-n):
            pred, mc_term = evaluate_idx(path, t, t+n, gamma, baseline)
            preds.append(pred[0])
            mc_terms.append(mc_term[0])
    return preds, mc_terms

def mse(pred, mc):
    pred = np.array(pred)
    mc = np.array(mc)
    n = len(mc)
    return np.sum((pred - mc)**2) / n

def line_fit(pred, mc):
    X = np.array(pred).reshape((-1,1))
    y = np.array(mc)
    model = LinearRegression()
    model.fit(X, y)
    r_sq = model.score(X, y)
    b = model.intercept_.tolist()
    m = model.coef_[0].tolist()
    return m, b, r_sq

if __name__ == '__main__':
    baseline_file = 'baseline_acro.pickle'
    baseline = pickle.load(open(baseline_file, 'rb'))

    paths = pickle.load(open('paths.pickle', 'rb'))

    gamma = 0.995

    pred_1, mc_1 = evaluate_n_step(1, gamma, paths, baseline)
    pred_5, mc_5 = evaluate_n_step(5, gamma, paths, baseline)
    pred_start_end, mc_start_end = evaluate_start_end(gamma, paths, baseline)

    print('mse', mse(pred_1, mc_1))
    print('line_fit', line_fit(pred_1, mc_1))

    # plt.xlabel('$Q(s_1,a_1)$')
    # plt.ylabel('$MC + \\gamma^T Q(s_2, a_2)$')
    # plt.scatter(pred_1, mc_1)


    print('mse', mse(pred_5, mc_5))
    print('line_fit', line_fit(pred_5, mc_5))

    # plt.xlabel('$Q(s_1,a_1)$')
    # plt.ylabel('$MC + \\gamma^T Q(s_5, a_5)$')
    # plt.scatter(pred_5, mc_5)
    
    print('mse', mse(pred_start_end, mc_start_end))
    print('line_fit', line_fit(pred_start_end, mc_start_end))

    plt.xlabel('$Q(s_1,a_1)$')
    plt.ylabel('$MC + \\gamma^T Q(s_T, a_T)$')
    plt.scatter(pred_start_end, mc_start_end)

    plt.show()

    # plt.xlabel('$Q(s_1,a_1)$')
    # plt.ylabel('$MC + \\gamma^T Q(s_T, a_T)$')
    # plt.scatter(preds, mc_terms)
    # plt.show()
    # plt.savefig('q_vs_mc.png')