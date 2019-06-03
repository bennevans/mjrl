from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP

from mjrl.q_baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_off_policy import NPGOffPolicy
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer

import numpy as np

import multiprocessing
import subprocess
import json
import datetime
import os

np_seed = 501 + int(timer.time())
host = subprocess.check_output('hostname').decode('utf-8').strip()
np.random.seed(np_seed)

e = GymEnv('mjrl_swimmer-v0')

sizes = [2000 * 1, 2000 * 10, 2000 * 20, 2000 * 50, 2000*100, -1]
num_iters = [1, 10, 50, 100, 250]
num_cpus = [32]
num_trajs = [5, 25, 50, 100, 250]
npg_size = 20 * 2000
ni = 50
train_iter = 1000
num_traj = 40
num_cpu = 4
epochs = 1
baseline_lr = 1e-3
baseline_batch_size = 10000

exp_baseline_epochs = [1, 2, 5, 10, 25]
exp_baseline_lr = [1e-3, 1e-4, 1e-5]
exp_baseline_batch_size = [64, 512, 4096, 10000, 20000]
exp_max_dataset_size = [50000, 100000, 500000, 1000000, -1]
exp_kl_dist = [0.05]
exp_agent_epochs = [1]
exp_num_traj = [5]

def sample_and_add(info, param_name, options):
    param = np.random.choice(options).tolist()
    info[param_name] = param

def generate_params():
    baseline_info = {}
    agent_info = {}
    train_info = {}

    sample_and_add(baseline_info, 'epochs', exp_baseline_epochs)
    sample_and_add(baseline_info, 'lr', exp_baseline_lr)
    sample_and_add(baseline_info, 'batch_size', exp_baseline_batch_size)
    sample_and_add(agent_info, 'max_dataset_size', exp_max_dataset_size)
    sample_and_add(agent_info, 'kl_dist', exp_kl_dist)
    sample_and_add(agent_info, 'epochs', exp_agent_epochs)
    sample_and_add(train_info, 'num_traj', exp_num_traj)

    train_info['num_cpu'] = 12
    train_info['train_iter'] = 100
    train_info['save_freq'] = train_info['train_iter'] // 10
    train_info['np_seed'] = np_seed


    return baseline_info, agent_info, train_info


def experiment(e, i, exp_num, baseline_info, agent_info, train_info):
    baseline_epochs, baseline_lr, baseline_batch_size \
        = baseline_info['epochs'],baseline_info['lr'], baseline_info['batch_size']
    max_dataset_size, kl_dist, agent_epochs \
        = agent_info['max_dataset_size'], agent_info['kl_dist'], agent_info['epochs']
    num_traj, num_cpu, save_freq, SEED, train_iter \
        = train_info['num_traj'], train_info['num_cpu'], train_info['save_freq'], train_info['SEED'], train_info['train_iter']

    policy = MLP(e.spec)
    baseline = MLPBaseline(e.spec, epochs=baseline_epochs, learn_rate=baseline_lr, batch_size=baseline_batch_size)
    agent = NPGOffPolicy(e, policy, baseline, seed=SEED, save_logs=True,
        fit_off_policy=True, fit_on_policy=False, max_dataset_size=max_dataset_size,
        kl_dist=kl_dist, epochs=agent_epochs, use_batches=False)

    suffix = datetime.datetime.strftime(datetime.datetime.now(), '%b_%d_%H_%M_%S')
    dirname = 'exp_{}_run_{}_swimmer_{}_{}'.format(exp_num, i, host, suffix)

    info = {
        'baseline': baseline_info,
        'agent': agent_info,
        'train': train_info
    }

    info_json = json.dumps(info, sort_keys=True, indent=4)
    print(info_json)

    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)

    with open('{}/info.json'.format(dirname), 'w') as f:
        f.write(info_json)

    train_agent(job_name=dirname,
                agent=agent,
                seed=SEED,
                niter=train_iter,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=num_cpu,
                sample_mode='trajectories',
                num_traj=num_traj,      # samples = 40*25 = 1000
                save_freq=save_freq,
                evaluation_rollouts=10)
    
    return dirname
    

if __name__ == '__main__':
    times = []
    exp_dirs = []
    exp_num = 1

    num_exps = 2

    with open('exp_{}_info_{}.txt'.format(exp_num, host), 'a') as f:
        f.write('exp,exp_dir,times\n')
        
        for i in range(num_exps):
            baseline_info, agent_info, train_info = generate_params()
            train_info['SEED'] = i + int(timer.time())
            ts = timer.time()
            d = experiment(e, i, exp_num, baseline_info, agent_info, train_info)
            run_time = timer.time() - ts
            print(d, run_time)
            exp_dirs.append(d)
            times.append(run_time)
            f.write('{},{},{}\n'.format(i, d, run_time))
    
    for e,t in zip(exp_dirs, times):
        print(e, t)


