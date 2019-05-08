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

num_iters = [1, 10, 100, 500, 1000]



for ni in num_iters:
    policy = MLP(e.spec)
    baseline = MLPBaseline(e.spec, num_iters=ni) # TODO: find best (function?) num_iters
    agent = NPGOffPolicy(e, policy, baseline, max_dataset_size=45*10 ,const_learn_rate=0.1, seed=SEED, save_logs=True)

    ts = timer.time()
    train_agent(job_name='exp_' + host + '_point_mass_num_iters_' + str(ni),
                agent=agent,
                seed=SEED,
                niter=1000,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=multiprocessing.cpu_count() // 2,
                sample_mode='trajectories',
                num_traj=40,      # samples = 40*25 = 1000
                save_freq=25,
                evaluation_rollouts=10)
    print("time taken = %f" % (timer.time()-ts))
