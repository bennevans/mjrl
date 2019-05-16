from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.q_baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.linear_baseline import LinearBaseline
from mjrl.algos.npg_cg_off_policy import NPGOffPolicy
from mjrl.algos.batch_reinforce_off_policy import BatchREINFORCEOffPolicy
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer

import numpy as np

import datetime
import multiprocessing
import subprocess


SEED = 500

np.random.seed(SEED)

suffix = datetime.datetime.strftime(datetime.datetime.now(), '%b_%d_%H_%M_%S')
host = subprocess.check_output('hostname').decode('utf-8').strip()


e = GymEnv('mjrl_swimmer-v0')
policy = MLP(e.spec, seed=SEED)
# policy = LinearPolicy(e.spec)
baseline = MLPBaseline(e.spec, epochs=15, learn_rate=1e-3, batch_size=10000)
# agent = BatchREINFORCEOffPolicy(e, policy, baseline, learn_rate=1e-5, seed=SEED, save_logs=True)
agent = NPGOffPolicy(e, policy, baseline, seed=SEED, save_logs=True,
    fit_off_policy=True, fit_on_policy=False, max_dataset_size=1e6,
    kl_dist=0.01, epochs=1, batch_size=4096, use_batches=False)

ts = timer.time()
train_agent(job_name='debug_exp_{}_swimmer_baseline_q_off_mlp_{}'.format(host, suffix),
            agent=agent,
            seed=SEED,
            niter=250,
            gamma=0.95,
            gae_lambda=0.97,
            num_cpu=12,
            sample_mode='trajectories',
            num_traj=5,      # samples = 40*25 = 1000
            save_freq=25,
            evaluation_rollouts=10)

print("time taken = %f" % (timer.time()-ts))

