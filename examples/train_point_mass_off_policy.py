from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.q_baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_off_policy import NPGOffPolicy
from mjrl.utils.train_agent import train_agent
from mjrl.utils.replay_buffer import ReplayBuffer

import mjrl.envs
import time as timer
SEED = 500

e = GymEnv('mjrl_point_mass-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
# baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
baseline = MLPBaseline(e.spec, epochs=2, fit_iters=50)
agent = NPGOffPolicy(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True,
                        max_dataset_size=10000, drop_mode=ReplayBuffer.DROP_MODE_RANDOM, num_update_actions=10,
                        num_update_states=256)

ts = timer.time()
train_agent(job_name='pm_off_policy_exp1',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=2,
            sample_mode='trajectories',
            num_traj=10,      # samples = 10*500 = 5000
            save_freq=5,
            evaluation_rollouts=None,
            plot_keys=['stoc_pol_mean', 'eval_score', 'running_score'])
print("time taken = %f" % (timer.time()-ts))
