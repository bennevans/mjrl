from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.q_baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg_off_policy import NPGOffPolicy
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
SEED = 500

e = GymEnv('mjrl_hopper-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)

baseline = MLPBaseline(e.spec, fit_iters=50, use_time=True)

def my_func(i):
    return max(500 - 100*i, 50)

agent = NPGOffPolicy(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True,
    num_update_actions=10, num_update_states=256, max_dataset_size=20000, fit_iter_fn=my_func)

ts = timer.time()
train_agent(job_name='hopper_off_policy_time',
            agent=agent,
            seed=SEED,
            niter=100,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=6,
            sample_mode='trajectories',
            num_traj=16,      # samples = 40*25 = 1000
            save_freq=5,
            evaluation_rollouts=10,
            include_i=True)
print("time taken = %f" % (timer.time()-ts))
