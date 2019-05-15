import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog


class BatchREINFORCEOffPolicy:
    def __init__(self, env, policy, baseline,
                max_dataset_size=-1,
                learn_rate=0.01,
                seed=None,
                save_logs=False,
                fit_off_policy=True,
                fit_on_policy=False,
                update_epochs=1,
                batch_size=64):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.replay_buffer = {}
        self.max_dataset_size = max_dataset_size
        self.fit_off_policy = fit_off_policy
        self.fit_on_policy = fit_on_policy
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        if save_logs: self.logger = DataLog()

    def CPI_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max'):
        
        print('train_step')
        ts_time = timer.time()
        
        # Clean up input arguments
        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        if sample_mode == 'trajectories':
            paths = trajectory_sampler.sample_paths_parallel(N, self.policy, T, env_name,
                                                             self.seed, num_cpu)
        elif sample_mode == 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)
        

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        # eval_statistics = self.train_from_paths(paths)
        # eval_statistics.append(N)

        # train from replay buffer
        print('update_replay_buffer')
        urb_t = timer.time()
        self.update_replay_buffer(paths)
        print('update_replay_buffer done', timer.time() - urb_t)

        print('train_from_replay_buffer')
        tfrb = timer.time()
        eval_statistics = self.train_from_replay_buffer(paths)
        print('train_from_replay_buffer done', timer.time() - tfrb)
        eval_statistics.append(N)

        print('fit_off_policy')
        fop = timer.time()
        if self.save_logs:
            ts = timer.time()
            # TODO combine error after? throwing away rn
            if self.fit_on_policy:
                self.baseline.fit(paths)
            if self.fit_off_policy:
                error_before, error_after = self.baseline.fit_off_policy(self.replay_buffer, self.policy, gamma, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
            self.logger.log_kv('dataset_size', len(self.replay_buffer['observations']))
            self.logger.log_kv('t', self.replay_buffer['t'])
        else:
            if self.fit_on_policy:
                self.baseline.fit(paths)
            if self.fit_off_policy:
                self.baseline.fit_off_policy(self.replay_buffer, self.policy, gamma)
        print('fit_off_policy done', timer.time() - fop)
        print('train_step done', timer.time() - ts_time)

        return eval_statistics

    def update_replay_buffer(self, paths):
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        l = observations.shape[0] - 1
        if 'observations' not in self.replay_buffer:
            self.replay_buffer['observations'] = observations[:-1]
            self.replay_buffer['observations_prime'] = observations[1:]
            self.replay_buffer['actions'] = actions[:-1]
            self.replay_buffer['rewards'] = rewards[:-1]
            self.replay_buffer['last_update'] = np.zeros(l)
            self.replay_buffer['t'] = 0
        else:
            self.replay_buffer['t'] += 1
            self.replay_buffer['observations'] = np.concatenate([self.replay_buffer['observations'], observations[:-1]])
            self.replay_buffer['observations_prime'] = np.concatenate([self.replay_buffer['observations_prime'], observations[1:]])
            self.replay_buffer['actions'] = np.concatenate([self.replay_buffer['actions'], actions[:-1]])
            self.replay_buffer['rewards'] = np.concatenate([self.replay_buffer['rewards'], rewards[:-1]])
            self.replay_buffer['last_update'] = np.concatenate([self.replay_buffer['last_update'], np.ones(l) * self.replay_buffer['t']])
        
                
        if self.max_dataset_size > 0 and len(self.replay_buffer['observations']) > self.max_dataset_size:
            self.replay_buffer['observations'] = self.replay_buffer['observations'][-self.max_dataset_size:, :]
            self.replay_buffer['observations_prime'] = self.replay_buffer['observations_prime'][-self.max_dataset_size:, :]
            self.replay_buffer['actions'] = self.replay_buffer['actions'][-self.max_dataset_size:, :]
            self.replay_buffer['rewards'] = self.replay_buffer['rewards'][-self.max_dataset_size:]
            self.replay_buffer['last_update'] = self.replay_buffer['last_update'][-self.max_dataset_size:]
            
    # TODO will calling this multiple times screw things up?
    def update_policy(self, observations, actions, weights, paths):
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, weights).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, weights)
        t_gLL += timer.time() - ts

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + self.alpha * vpg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, weights).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

    def train_from_replay_buffer(self, paths):
        # TODO cache baseline and only update when necessary?
        observations = self.replay_buffer['observations']

        actions = self.policy.get_action_batch(observations)

        predictions = self.baseline.predict({'observations': observations, 'actions': actions})

        # TODO multiple iterations / batches?
        # for epoch in range(self.update_epochs):

        print('update_policy')
        up = timer.time()
        self.update_policy(observations, actions, predictions, paths)
        print('update_policy done', timer.time() - up)

        print('path_returns')
        pr = timer.time()
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        print('path_returns done', timer.time() - pr)
        return base_stats


    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                            0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        self.update_policy(observations, actions, advantages, paths)

        return base_stats

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)
