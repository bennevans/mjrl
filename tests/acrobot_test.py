from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
SEED = 500

e = GymEnv('mjrl_acrobot-v0')
policy = MLP(e.spec, hidden_sizes=(64,64), seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=5, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.05, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='acrobot_exp1',
            agent=agent,
            seed=SEED,
            niter=200,
            gamma=0.995,  
            gae_lambda=0.97,
            num_cpu=6,
            sample_mode='trajectories',
            num_traj=20,      # samples = 10*500 = 5000
            save_freq=5,
            evaluation_rollouts=5)
print("time taken = %f" % (timer.time()-ts))
