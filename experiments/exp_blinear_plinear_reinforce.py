from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.algos.batch_reinforce import BatchREINFORCE
from mjrl.algos.batch_reinforce_off_policy import BatchREINFORCEOffPolicy
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
import mjrl.baselines.linear_baseline
import mjrl.q_baselines.linear_baseline

import numpy as np

e = GymEnv('mjrl_point_mass-v0')

n_exps = 10
niter = 2000
lr = 0.1

# using value function (with just previous trajectories) as baseline
"""
value_best = []
for i in range(n_exps):
    np.random.seed(i)
    policy = LinearPolicy(e.spec)
    baseline = mjrl.baselines.linear_baseline.LinearBaseline(e.spec)
    agent = BatchREINFORCE(e, policy, baseline, learn_rate=lr, seed=i, save_logs=True)

    best = train_agent(job_name='point_mass_v_' + str(i),
                agent=agent,
                seed=i,
                niter=niter,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=4,
                sample_mode='trajectories',
                num_traj=40,      # samples = 40*25 = 1000
                save_freq=25,
                evaluation_rollouts=10)

    value_best.append(best)
"""
# using q function (with all trajectories) as baseline
"""
q_best_off = []
for i in range(n_exps):
    np.random.seed(i)
    policy = LinearPolicy(e.spec)
    baseline = mjrl.q_baselines.linear_baseline.LinearBaseline(e.spec)
    agent = BatchREINFORCEOffPolicy(e, policy, baseline, max_dataset_size=45*20, learn_rate=lr, seed=i, save_logs=True)

    best = train_agent(job_name='point_mass_q_off_' + str(i),
                agent=agent,
                seed=i,
                niter=niter,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=4,
                sample_mode='trajectories',
                num_traj=40,      # samples = 40*25 = 1000
                save_freq=25,
                evaluation_rollouts=10)

    q_best_off.append(best)

# print(value_best)
print(q_best_off)
"""

q_best = []
for i in range(n_exps):
    np.random.seed(i)
    policy = LinearPolicy(e.spec)
    baseline = mjrl.q_baselines.linear_baseline.LinearBaseline(e.spec)
    agent = BatchREINFORCE(e, policy, baseline, learn_rate=lr, seed=i, save_logs=True)

    best = train_agent(job_name='point_mass_q_' + str(i),
                agent=agent,
                seed=i,
                niter=niter,
                gamma=0.95,
                gae_lambda=0.97,
                num_cpu=4,
                sample_mode='trajectories',
                num_traj=40,      # samples = 40*25 = 1000
                save_freq=25,
                evaluation_rollouts=10)

    q_best.append(best)