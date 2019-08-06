

import time
import os
import json
import itertools
import inspect
import copy
import socket
import mjrl.envs

import numpy as np

from mjrl.utils.gym_env import GymEnv
from mjrl.algos.npg_cg import NPG
from mjrl.algos.npg_cg_off_policy import NPGOffPolicy
from mjrl.policies.gaussian_mlp import MLP
from mjrl.utils.replay_buffer import ReplayBuffer
from mjrl.utils.train_agent import train_agent


import mjrl.q_baselines.mlp_baseline as qmlp
import mjrl.baselines.mlp_baseline as mlp


BASE_SEED = 0x5EED

def get_param_list(possible_params):
    keys = []
    values = []

    for key, value in possible_params.items():
        keys.append(key)
        values.append(value)

    permutations = itertools.product(*values)

    param_list = []

    for item in permutations:
        param = {}
        for i, key in enumerate(keys):
            param[key] = item[i]
        param_list.append(param)

    return param_list

def sanatize_possible(possible_params):
    ret = copy.deepcopy(possible_params)
    for i, fit_iter_fn in enumerate(ret['baseline_fit_iter_fn']):
        if ret['baseline_fit_iter_fn'][i] is None:
            ret['baseline_fit_iter_fn'][i] = 'None'
        else:
            ret['baseline_fit_iter_fn'][i] = inspect.getsource(fit_iter_fn)
    return ret

def sanatize(params):
    ret = copy.deepcopy(params)
    if ret['baseline_fit_iter_fn'] is not None:
        fn = ret['baseline_fit_iter_fn']
        ret['baseline_fit_iter_fn'] = inspect.getsource(fn)
    return ret


def run_exp(params, env_name, job_name, i):
    epochs = params['baseline_epochs']
    fit_iters = params['baseline_fit_iters']
    fit_iter_fn = params['baseline_fit_iter_fn']
    batch_size = params['baseline_batch_size']
    baseline_hidden_size = params['baseline_hidden_size']
    lr = params['baseline_lr']
    use_time = params['baseline_use_time']
    off_policy = params['baseline_off_policy']

    normalized_step_size = params['agent_normalized_step_size']
    max_dataset_size = params['agent_max_dataset_size']
    drop_mode = params['agent_drop_mode']

    num_update_actions = params['agent_num_update_actions']
    num_update_states = params['agent_num_update_states']

    policy_hidden_size = params['policy_hidden_size']

    niter = params['niter']
    gamma = params['gamma']
    gae_lambda = params['gae_lambda']
    num_cpu = params['num_cpu']
    num_traj = params['num_traj']
    save_freq = params['save_freq']
    evaluation_rollouts = params['evaluation_rollouts']

    e = GymEnv(env_name)

    policy = MLP(e.spec, hidden_sizes=policy_hidden_size, seed=BASE_SEED+i)

    if off_policy:

        baseline = qmlp.MLPBaseline(e.spec, learn_rate=lr, batch_size=batch_size,
            epochs=epochs, fit_iters=fit_iters, hidden_sizes=baseline_hidden_size,
            use_time=use_time)

        agent = NPGOffPolicy(e, policy, baseline, normalized_step_size=normalized_step_size,
            seed=BASE_SEED+i+1, save_logs=True, fit_iter_fn=fit_iter_fn, max_dataset_size=max_dataset_size,
            drop_mode=drop_mode, num_update_actions=num_update_actions, num_update_states=num_update_states)

    else:
        baseline = mlp.MLPBaseline(e.spec, learn_rate=lr, batch_size=batch_size,
            epochs=epochs)
        
        agent = NPG(e, policy, baseline, normalized_step_size=normalized_step_size,
            seed=BASE_SEED+i+2, save_logs=True)

    os.mkdir(job_name)
    param_file = os.path.join(job_name, 'params.json')

    with open(param_file, 'w') as f:
        json.dump(sanatize(params), f, sort_keys=True, indent=4)

    train_agent(job_name=job_name,
        agent=agent,
        seed=BASE_SEED+i+3,
        niter=niter,
        gamma=gamma,
        gae_lambda=gae_lambda,
        num_cpu=num_cpu,
        sample_mode='trajectories',
        num_traj=num_traj,
        save_freq=save_freq,
        evaluation_rollouts=evaluation_rollouts,
        include_i=off_policy)

    

def run_exp_many(params, env_name, n, exp_dir):

    os.mkdir(exp_dir)

    param_file = os.path.join(exp_dir, 'params.json')

    with open(param_file, 'w') as f:
        json.dump(sanatize(params), f, sort_keys=True, indent=4)

    for i in range(n):
        run_dir = os.path.join(exp_dir, 'run_{}'.format(i))
        print('run_dir', run_dir)
        run_exp(params, env_name, run_dir, i)

def enumerate_run_save(param_list, base_dir, env_name, n):

    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print('experiment directory already exists! exiting')
        return
    
    for i, params in enumerate(param_list):
        print(i, params)
        exp_dir = os.path.join(base_dir, 'exp_{}'.format(i))
        run_exp_many(params, env_name, n, exp_dir)


def generate_param_list_combinatorial(possible_params):

    param_list = get_param_list(possible_params)

    return param_list

def get_params(**kwargs):
    default = {
        'baseline_epochs': 1,
        'baseline_fit_iters': 50,
        'baseline_fit_iter_fn': None,
        'baseline_batch_size': 64,
        'baseline_hidden_size': (64, 64),
        'baseline_lr': 1e-4,
        'baseline_use_time': True,
        'baseline_off_policy': True,

        'agent_normalized_step_size': 0.1,
        'agent_drop_mode': ReplayBuffer.DROP_MODE_RANDOM,
        'agent_max_dataset_size': 100000,
        'agent_num_update_actions': 10,
        'agent_num_update_states': 256,
        'policy_hidden_size': (32, 32), 

        'niter': 150,
        'gamma': 0.995,
        'gae_lambda': 0.97,
        'num_cpu': 6,
        'num_traj': 5,
        'save_freq': 5,
        'evaluation_rollouts': 10
    }

    for k,v in kwargs.items():
        default[k] = v
    return default

def sanatize_param_list(param_list):
    return param_list

def generate_param_list_fixed():
    param_list = []

    param_list.append(get_params(agent_max_dataset_size=1000000, baseline_fit_iters=1))
    param_list.append(get_params(agent_max_dataset_size=100000, baseline_fit_iters=10))
    param_list.append(get_params(agent_max_dataset_size=50000, baseline_fit_iters=20))
    param_list.append(get_params(agent_max_dataset_size=10000, baseline_fit_iters=100))
    param_list.append(get_params(agent_max_dataset_size=1000, baseline_fit_iters=1000))

    return param_list

def log_sample(lower, upper):
    t = type(lower)
    return t(np.exp(np.random.uniform(np.log(lower), np.log(upper))))

def sample(lower, upper):
    t = type(lower)
    return t(np.random.uniform(lower, upper))

def generate_param_list_random(n):
    # if there is one argument, use it
    # else, define (low, high, logscale)
    param_limits = {
        'baseline_epochs': [1, 3, False],
        'baseline_fit_iters': [25, 250, True],
        'baseline_fit_iter_fn': [None],
        'baseline_batch_size': [64],
        'baseline_hidden_size': [(64, 64)],
        'baseline_lr': [1e-5, 1e-3, True],
        'baseline_use_time': [False],
        'baseline_off_policy': [True],

        'agent_normalized_step_size': [0.01, 0.5, True],
        'agent_drop_mode': [ReplayBuffer.DROP_MODE_RANDOM],
        'agent_max_dataset_size': [1000, 100000, True],
        'agent_num_update_actions': [2, 128, True],
        'agent_num_update_states': [16, 10000, True],
        'policy_hidden_size': [(32, 32)], 

        'niter': [100],
        'gamma': [0.995],
        'gae_lambda': [0.97],
        'num_cpu': [6],
        'num_traj': [16],
        'save_freq': [5],
        'evaluation_rollouts': [10]
    }

    param_list = []
    for i in range(n):
        param = {}
        for k,v in param_limits.items():
            if len(v) == 1:
                param[k] = v[0]
            elif len(v) == 3:
                if v[2]:
                    param[k] = log_sample(v[0], v[1])
                else:
                    param[k] = sample(v[0], v[1])
            else:
                raise Exception('bad param limits')
        param_list.append(param)
    return param_list


possible_params = {
    'baseline_epochs': [1],
    'baseline_fit_iters': [50],
    'baseline_fit_iter_fn': [None],
    'baseline_batch_size': [64],
    'baseline_hidden_size': [(64, 64)],
    'baseline_lr': [1e-3],
    'baseline_use_time': [False],
    'baseline_off_policy': [True],

    'agent_normalized_step_size': [0.01, 0.05, 0.1, 0.2, 0.5],
    'agent_drop_mode': [ReplayBuffer.DROP_MODE_RANDOM],
    'agent_max_dataset_size': [30000],
    'agent_num_update_actions': [6],
    'agent_num_update_states': [256],
    'policy_hidden_size': [(32, 32)], 

    'niter': [100],
    'gamma': [0.995],
    'gae_lambda': [0.97],
    'num_cpu': [6],
    'num_traj': [20],
    'save_freq': [5],
    'evaluation_rollouts': [10]
}

if __name__ == '__main__':

    env_name = 'mjrl_hopper-v0'
    base_dir = 'pg_exp/hopper_step_size_off_policy_0/'
    machine = 'ben-mcl'

    hostname = socket.gethostname()
    if machine != hostname:
        print('mismatching hostnames! expected: {} given: {}'.format(machine, hostname))
        exit()

    n = 3

    start_time = time.time()

    param_list = generate_param_list_combinatorial(possible_params)
    # param_list = generate_param_list_fixed()
    # param_list = generate_param_list_random(10)

    for param in param_list:
        print('param:\n', param)

    enumerate_run_save(param_list, base_dir, env_name, n)
    
    metadata = {
        'start_time': start_time,
        'end_time': time.time(),
        'n': n,
        'base_dir': base_dir,
        'param_list': sanatize_param_list(param_list),
        'env_name': env_name,
        'base_seed': BASE_SEED,
        'machine': machine
    }

    metadata['total_time'] = metadata['end_time'] - metadata['start_time']

    metadata_dir = os.path.join(base_dir, 'metadata.json')

    with open(metadata_dir, 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4)



