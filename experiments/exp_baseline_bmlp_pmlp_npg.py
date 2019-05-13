from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP

from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer

import numpy as np

import multiprocessing
import subprocess

SEED = 500
host = subprocess.check_output('hostname').decode('utf-8').strip()
np.random.seed(SEED)

e = GymEnv('mjrl_swimmer-v0')

sizes = [2000 * 1, 2000 * 10, 2000 * 20, 2000 * 50, 2000*100, -1]
num_iters = [1, 10, 50, 100, 250]
num_cpus = [32]
num_trajs = [5, 25, 50, 100, 250]
size = 20 * 2000
ni = 50
train_iter = 1000
num_traj = 40
num_cpu = 16

times = []

# for ni in num_iters:
# for num_cpu in num_cpus:
# for size in sizes:
for num_traj in num_trajs:
    policy = MLP(e.spec)
    baseline = MLPBaseline(e.spec) # TODO: find best (function?) num_iters
    agent = NPG(e, policy, baseline, const_learn_rate=0.1, seed=SEED, save_logs=True)
    # agent = NPGOffPolicy(e, policy, baseline, max_dataset_size=num_traj*200 ,const_learn_rate=0.1, seed=SEED, save_logs=True)

    ts = timer.time()
    train_agent(job_name='exp_6_baseline_' + host + '_swimmer_num_traj_' + str(num_traj),
                agent=agent,
                seed=SEED,
                niter=train_iter,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=num_cpu,
                sample_mode='trajectories',
                num_traj=num_traj,      # samples = 40*25 = 1000
                save_freq=75,
                evaluation_rollouts=10)
    times.append((timer.time()-ts))
    print("time taken = %f" % (timer.time()-ts))

print(times)
