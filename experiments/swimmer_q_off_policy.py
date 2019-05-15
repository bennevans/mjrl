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
baseline = MLPBaseline(e.spec, num_iters=50)
agent = BatchREINFORCEOffPolicy(e, policy, baseline, learn_rate=1e-3, seed=SEED, save_logs=True)
# agent = NPGOffPolicy(e, policy, baseline, max_dataset_size=40000,
    # const_learn_rate=0.1, seed=SEED, save_logs=True, fit_off_policy=True, fit_on_policy=False)

ts = timer.time()
train_agent(job_name='debug_exp_{}_swimmer_baseline_q_off_mlp_{}'.format(host, suffix),
            agent=agent,
            seed=SEED,
            niter=25,
            gamma=0.95,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=40,      # samples = 40*25 = 1000
            save_freq=5,
            evaluation_rollouts=10)

print("time taken = %f" % (timer.time()-ts))

