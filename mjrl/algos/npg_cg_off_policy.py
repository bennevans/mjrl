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
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce_off_policy import BatchREINFORCEOffPolicy
from mjrl.utils.replay_buffer import ReplayBuffer


class NPGOffPolicy(BatchREINFORCEOffPolicy):
    def __init__(self, env, policy, baseline,
                    normalized_step_size=0.01,
                    const_learn_rate=None,
                    FIM_invert_args={'iters': 10, 'damping': 1e-4},
                    hvp_sample_frac=1.0,
                    seed=None,
                    save_logs=False,
                    kl_dist=None,
                    max_dataset_size=-1,
                    fit_off_policy=True,
                    fit_on_policy=False,
                    use_batches=False,
                    epochs=1,
                    batch_size=256,
                    fit_iter_fn=None,
                    drop_mode=ReplayBuffer.DROP_MODE_OLDEST,
                    pg_update_using_rb=True,
                    pg_update_using_advantage=True,
                    num_update_states=10,
                    num_update_actions=10,
                    num_policy_updates=1,
                    normalize_mode=BatchREINFORCEOffPolicy.NORMALIZE_STD,
                    non_uniform=False):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """
        super().__init__(env, policy, baseline, max_dataset_size=max_dataset_size,
            fit_off_policy=fit_off_policy, fit_on_policy=fit_on_policy,
            fit_iter_fn=fit_iter_fn, drop_mode=drop_mode, pg_update_using_rb=pg_update_using_rb,
            pg_update_using_advantage=pg_update_using_advantage, num_update_states=num_update_states,
            num_update_actions=num_update_actions, num_policy_updates=num_policy_updates, normalize_mode=normalize_mode)
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.use_batches = use_batches
        self.epochs = epochs
        self.batch_size = batch_size
        self.non_uniform = non_uniform
        if save_logs: self.logger = DataLog()

    def HVP(self, observations, actions, vector, regu_coef=None):
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        vec = Variable(torch.from_numpy(vector).float(), requires_grad=False)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist_info = self.policy.old_dist_info(obs, act)
        new_dist_info = self.policy.new_dist_info(obs, act)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, self.policy.trainable_params)
        hvp_flat = np.concatenate([g.contiguous().view(-1).data.numpy() for g in hvp])
        return hvp_flat + regu_coef*vector

    def build_Hvp_eval(self, inputs, regu_coef=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef]
            Hvp = self.HVP(*full_inp)
            return Hvp
        return eval
    
    def update_policy(self, observations, actions, weights):
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, weights)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)

        return alpha, n_step_size, t_gLL, t_FIM, new_params


    def train(self, paths):
        n = self.num_update_states # how many states to take from rb 
        m = self.num_update_actions # how many actions per state

        # TODO repeat this process k times?

        if self.pg_update_using_rb:
            samples, n = self.replay_buffer.sample(n, self.baseline, self.non_uniform) # TODO is it bad to reupdate n?
            observations = np.tile(samples['observations'], (m, 1))
            times = np.tile(samples['times'], (m))
            traj_length = np.tile(samples['traj_length'], (m))
            actions = self.policy.get_action_batch(observations)

            Qs = self.baseline.predict(
                {
                'observations': observations,
                'actions': actions,
                'times': times,
                'traj_length': traj_length
                }
            )

            if self.save_logs:
                self.logger.log_kv('Q_mean', np.mean(Qs))
                self.logger.log_kv('Q_max', np.max(Qs))
                self.logger.log_kv('Q_min', np.min(Qs))
                self.logger.log_kv('Q_std', np.std(Qs))
                self.logger.log_kv('rb_mean_reward', np.mean(samples['rewards']))
                self.logger.log_kv('rb_max_reward', np.max(samples['rewards']))
                self.logger.log_kv('rb_min_reward', np.min(samples['rewards']))
                self.logger.log_kv('rb_std_reward', np.std(samples['rewards']))
            
            weights = np.copy(Qs)

            

            # TODO do non-loop
            if self.pg_update_using_advantage:
                for i in range(n):
                    observations[i::n]
                    V = np.average(Qs[i::n])
                    weights[i::n] -= V
                
                if self.normalize_mode == BatchREINFORCEOffPolicy.NORMALIZE_STD:
                    weights = (weights - np.mean(weights)) / (np.std(weights) + 1e-6)
                elif self.normalize_mode == BatchREINFORCEOffPolicy.NORMALIZE_FIXED_RANGE:
                    min_weight = np.min(weights)
                    max_weight = np.max(weights)
                    new_min = -1
                    new_max = 1
                    weights = (weights-min_weight) / (max_weight - min_weight) * (new_max - new_min) + new_min
                elif self.normalize_mode == BatchREINFORCEOffPolicy.NORMALIZE_MIN_MAX:
                    min_weight = np.min(weights)
                    max_weight = np.max(weights)

                    norm_factor = max(np.abs(min_weight), np.abs(max_weight))

                    weights /= norm_factor

                if self.save_logs:
                    self.logger.log_kv('weights_mean', np.mean(weights))
                    self.logger.log_kv('weights_max', np.max(weights))
                    self.logger.log_kv('weights_min', np.min(weights))
                    self.logger.log_kv('weights_std', np.std(weights))
                    # TODO log vs?
        else:
            raise Exception('not implemented')

        surr_before = self.CPI_surrogate(observations, actions, weights).data.numpy().ravel()[0]
        alpha, n_step_size, t_gLL, t_FIM, new_params = self.update_policy(observations, actions, weights)
        
        surr_after = self.CPI_surrogate(observations, actions, weights).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            std = np.exp(self.policy.log_std.detach().numpy())
            for i, v in enumerate(std):
                self.logger.log_kv('std_{}'.format(i), v)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]

        return base_stats

    def train_from_replay_buffer(self, paths):
        
        observations = self.replay_buffer['observations']
        actions = self.policy.get_action_batch(observations)
        predictions = self.baseline.predict(
            {
                'observations': observations,
                'actions': actions,
                'times': self.replay_buffer['times'],
                'traj_length': self.replay_buffer['traj_length']
            }
        )
        
        if self.save_logs:
            self.logger.log_kv('Q_mean', np.mean(predictions))
            self.logger.log_kv('Q_max', np.max(predictions))
            self.logger.log_kv('Q_min', np.min(predictions))
            self.logger.log_kv('Q_std', np.std(predictions))


        surr_before = self.CPI_surrogate(observations, actions, predictions).data.numpy().ravel()[0]
        n = observations.shape[0]
        
        # TODO averaging of stats, not just last iter
        if self.use_batches:
            for ep in range(self.epochs):
                if ep > 0:
                    actions = self.policy.get_action_batch(observations)
                    predictions = self.baseline.predict({'observations': observations, 'actions': actions})
            
                rand_idx_all = np.random.permutation(n)
                for mb in range(max(n // self.batch_size - 1, 1)):
                    rand_idx = rand_idx_all[mb*self.batch_size:(mb+1)*self.batch_size]
                    o_batch = observations[rand_idx, :]
                    a_batch = actions[rand_idx, :]
                    p_batch = predictions[rand_idx]
                    
                    alpha, n_step_size, t_gLL, t_FIM, new_params = self.update_policy(o_batch, a_batch, p_batch)
        else:
            for ep in range(self.epochs):
                if ep > 0:
                    actions = self.policy.get_action_batch(observations)
                    predictions = self.baseline.predict({'observations': observations, 'actions': actions})
                alpha, n_step_size, t_gLL, t_FIM, new_params = self.update_policy(observations, actions, predictions)
        
        surr_after = self.CPI_surrogate(observations, actions, predictions).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
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

        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]

        return base_stats
    
    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling, 
        # but scaling can help with least squares

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

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
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

        return base_stats
