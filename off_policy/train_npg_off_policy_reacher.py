
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.q_baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_off_policy import NPGOffPolicy
from mjrl.utils.train_agent import train_agent
from mjrl.algos.batch_reinforce_off_policy import BatchREINFORCEOffPolicy
import mjrl.envs
import time as timer
import numpy as np
from mjrl.utils.replay_buffer import ReplayBuffer
import os 
from shutil import copyfile
from mjrl.samplers.core import sample_paths
import pickle

SEED = 1236

e = GymEnv('mjrl_reacher-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = MLPBaseline(e.spec, fit_iters=100, batch_size=512, hidden_sizes=(512, 512),
    epochs=10, learn_rate=1e-3, use_epochs=False, use_gpu=True)

gamma = 0.9867

on_pol_baseline = MLPBaseline(e.spec, fit_iters=1, batch_size=512, hidden_sizes=(512, 512),
    epochs=5, learn_rate=1e-3, use_epochs=True, use_gpu=True)

def fit_iter_fn(i):
    if i < 10:
        return 100
    return 25
    # return max(50 - i, 5)

# def fit_iter_fn(i):
    # return int(max(500 * np.exp(i / -5), 100))

fixed_dataset_mode = 'mixed'

if fixed_dataset_mode == 'random':
    # fill out fixed dataset with random policy
    explore_pol = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED + 1, init_log_std=0.0)
    N = 80
    horizon = 1000
    input_dict = dict(num_traj=N, env=e, policy=explore_pol, horizon=horizon,
                                base_seed=SEED+2, num_cpu=1, pool=None)
    paths = sample_paths(**input_dict)
    fixed_dataset = ReplayBuffer()

    fixed_dataset.update(paths)
elif fixed_dataset_mode == 'pretrained':
    fixed_dataset = pickle.load(open('reacher_data/replay_buf_500.pickle', 'rb'))
elif fixed_dataset_mode == 'mixed':
    explore_pol = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED + 1, init_log_std=0.0)
    N_random = 250
    horizon = 1000
    input_dict = dict(num_traj=N_random, env=e, policy=explore_pol, horizon=horizon,
                                base_seed=SEED+2, num_cpu=1, pool=None)
    paths = sample_paths(**input_dict)
    fixed_dataset = pickle.load(open('reacher_data/replay_buf_500.pickle', 'rb'))

    fixed_dataset.update(paths)


# adv_mode = 'q_phi'
adv_mode = 'mc'

agent = NPGOffPolicy(e, policy, baseline, on_pol_baseline, fixed_dataset, normalized_step_size=0.1,
    seed=SEED, save_logs=True, # fit_iter_fn=fit_iter_fn,
    max_dataset_size=20000, drop_mode=ReplayBuffer.DROP_MODE_OLDEST,
    num_policy_updates=1, num_update_actions=10, num_update_states=75, #40*75 // 5, # numtraj * reacher len
    normalize_mode=BatchREINFORCEOffPolicy.NORMALIZE_STD,
    fit_on_policy=False, fit_off_policy=True, non_uniform=False, advantage_mode=adv_mode, gamma=gamma, num_cpu=1)

ts = timer.time()

base_dir = '/home/ben/data/off_policy/reacher_debug/mc_single/'
# base_dir = '/home/ben/data/off_policy/reacher_debug/q_phi/'
suffix = 'off_policy_mcl_2'
try:
    os.makedirs(base_dir)
except:
    print('skipping makedirs')
job_name = os.path.join(base_dir, suffix)

# job_name = 'aravind_exp/reacher_off_pol_40_mcl_0'
cur_file = os.path.realpath(__file__)
os.mkdir(job_name)
copyfile(cur_file, os.path.join(job_name, 'source.py'))


train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=30,
            gamma=gamma,
            gae_lambda=0.97,
            num_cpu=8,
            sample_mode='trajectories',
            num_traj=40,      # samples = 40*25 = 1000
            save_freq=25,
            evaluation_rollouts=10,
            include_i=True)
print("time taken = %f" % (timer.time()-ts))


