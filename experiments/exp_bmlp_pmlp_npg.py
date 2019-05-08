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

SEED = 500
host = subprocess.check_output('hostname').decode('utf-8').strip()
np.random.seed(SEED)

e = GymEnv('mjrl_point_mass-v0')

sizes = [2000 * 1, 2000 * 10, 2000 * 20, 2000 * 50]

num_traj = 40

times = []

for size in sizes:
    policy = MLP(e.spec)
    baseline = MLPBaseline(e.spec, num_iters=10) # TODO: find best (function?) num_iters
    agent = NPGOffPolicy(e, policy, baseline, max_dataset_size=2000*10 ,const_learn_rate=0.1, seed=SEED, save_logs=True)
    # agent = NPGOffPolicy(e, policy, baseline, max_dataset_size=num_traj*200 ,const_learn_rate=0.1, seed=SEED, save_logs=True)

    ts = timer.time()
    train_agent(job_name='exp_3_' + host + '_point_mass_size_' + str(size),
                agent=agent,
                seed=SEED,
                niter=1000,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=multiprocessing.cpu_count() // 2,
                sample_mode='trajectories',
                num_traj=num_traj,      # samples = 40*25 = 1000
                save_freq=250,
                evaluation_rollouts=10)
    times.append((timer.time()-ts))
    print("time taken = %f" % (timer.time()-ts))

print(times)