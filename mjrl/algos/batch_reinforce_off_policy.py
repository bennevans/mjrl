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

from mjrl.utils.replay_buffer import ReplayBuffer
from mjrl.utils.process_samples import discount_sum
from sklearn.linear_model import LinearRegression

def get_ith(path, i):
    new_path = {} 
    for k in path: 
        try: 
            new_path[k] = path[k][i:i+1] 
        except: 
            pass 
    return new_path 

def get_first(path):
    return get_ith(path, 0)

def get_last(path): 
    last_idx = len(path['observations']) - 1
    return get_ith(path, last_idx)

def evaluate(path, gamma, baseline):
    T = len(path['actions'])
    p0 = get_first(path)
    pl = get_last(path)
    pred = baseline.predict(p0)
    last = baseline.predict(pl)
    mc = discount_sum(path['rewards'], gamma)
    return pred, mc[0] + + gamma**T * last

def evaluate_idx(path, start_idx, end_idx, gamma, baseline):
    if start_idx >= end_idx:
        raise IndexError('start_idx should be < than end_idx')
    
    p0 = get_ith(path, start_idx)
    pl = get_ith(path, end_idx)
    pred = baseline.predict(p0)
    last = baseline.predict(pl)
    mc = discount_sum(path['rewards'][start_idx:end_idx], gamma)

    return pred, mc[0] + + gamma**(end_idx - start_idx) * last

def evaluate_start_end(gamma, paths, baseline):
    preds = []
    mc_terms = []
    for path in paths:
        pred, mc_term = evaluate(path, gamma, baseline)
        preds.append(pred[0])
        mc_terms.append(mc_term[0])
    return preds, mc_terms

def evaluate_n_step(n, gamma, paths, baseline):
    preds = []
    mc_terms = []
    for path in paths:
        T = len(path['observations'])
        for t in range(T-n):
            pred, mc_term = evaluate_idx(path, t, t+n, gamma, baseline)
            preds.append(pred[0])
            mc_terms.append(mc_term[0])
    return preds, mc_terms

def mse(pred, mc):
    pred = np.array(pred)
    mc = np.array(mc)
    n = len(mc)
    return np.sum((pred - mc)**2) / n

def line_fit(pred, mc):
    X = np.array(pred).reshape((-1,1))
    y = np.array(mc)
    model = LinearRegression()
    try:
        model.fit(X, y)
        r_sq = model.score(X, y)
        b = model.intercept_.tolist()
        m = model.coef_[0].tolist()
        return m, b, r_sq
    except Exception as e:
        print(str(e))
        return 0, 0, -1
class BatchREINFORCEOffPolicy:
    def __init__(self, env, policy, baseline,
                max_dataset_size=-1,
                learn_rate=0.01,
                seed=None,
                save_logs=False,
                fit_off_policy=True,
                fit_on_policy=False,
                update_epochs=1,
                batch_size=64,
                fit_iter_fn=None,
                drop_mode=ReplayBuffer.DROP_MODE_OLDEST,
                pg_update_using_rb=True,
                pg_update_using_advantage=True,
                num_update_states=10,
                num_update_actions=10,
                num_policy_updates=1,
                normalize_advanages=True):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.replay_buffer = ReplayBuffer(max_dataset_size=max_dataset_size, drop_mode=drop_mode)
        self.max_dataset_size = max_dataset_size
        self.fit_off_policy = fit_off_policy
        self.fit_on_policy = fit_on_policy
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.fit_iter_fn = fit_iter_fn
        self.pg_update_using_rb = pg_update_using_rb
        self.pg_update_using_advantage = pg_update_using_advantage
        self.num_update_states = num_update_states
        self.num_update_actions = num_update_actions
        self.num_policy_updates = num_policy_updates
        self.normalize_advanages = normalize_advanages

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
                    env=None,
                    sample_mode='trajectories',
                    horizon=1e6,
                    gamma=0.995,
                    gae_lambda=0.98,
                    num_cpu='max',
                    i=None):
        
        # Clean up input arguments
        env = self.env.env_id if env is None else env
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()

        
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu)
            paths = trajectory_sampler.sample_paths(**input_dict)
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu)
            paths = trajectory_sampler.sample_data_batch(**input_dict)
        
        
        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths

        # train from replay buffer
        self.replay_buffer.update(paths)

        # inner loop (Algorithm 1: line 6)
        # TODO logging
        for k in range(self.num_policy_updates):
            # update Q
            if self.fit_iter_fn is not None:
                if i is None:
                    raise Exception('must set include_i=True in train_agent')
                self.baseline.fit_iters = self.fit_iter_fn(i)
            
            error_before, error_after = self.baseline.fit_off_policy_many(self.replay_buffer, self.policy, gamma)

            # update policy
            eval_statistics = self.train(paths)

        if self.save_logs:
            ts = timer.time()
            # TODO combine error after? throwing away rn
            if self.fit_on_policy:
                error_before, error_after = self.baseline.fit(paths, return_errors=True)
            if self.fit_off_policy:
                pred_1, mc_1 = evaluate_n_step(1, gamma, paths, self.baseline)
                pred_start_end, mc_start_end = evaluate_start_end(gamma, paths, self.baseline)
                m_1, b_1, r_sq_1 = line_fit(pred_1, mc_1)
                m_end, b_end, r_sq_end = line_fit(pred_start_end, mc_start_end)
                self.logger.log_kv('MSE_1_before', mse(pred_1, mc_1))
                self.logger.log_kv('MSE_end_before', mse(pred_start_end, mc_start_end))
                self.logger.log_kv('m_1_before', m_1)
                self.logger.log_kv('b_1_before', b_1)
                self.logger.log_kv('r_sq_1_before', r_sq_1)
                self.logger.log_kv('m_end_before', m_end)
                self.logger.log_kv('b_end_before', b_end)
                self.logger.log_kv('r_sq_end_before', r_sq_end)

                if self.fit_iter_fn is not None:
                    if i is None:
                        raise Exception('must set include_i=True in train_agent')
                    self.baseline.fit_iters = self.fit_iter_fn(i)
                error_before, error_after = self.baseline.fit_off_policy_many(self.replay_buffer, self.policy, gamma)
                self.logger.log_kv('fit_iters', self.baseline.fit_iters)
                self.logger.log_kv('fit_epochs', self.baseline.epochs)

                pred_1, mc_1 = evaluate_n_step(1, gamma, paths, self.baseline)
                pred_start_end, mc_start_end = evaluate_start_end(gamma, paths, self.baseline)
                m_1, b_1, r_sq_1 = line_fit(pred_1, mc_1)
                m_end, b_end, r_sq_end = line_fit(pred_start_end, mc_start_end)
                self.logger.log_kv('MSE_1_after', mse(pred_1, mc_1))
                self.logger.log_kv('MSE_end_after', mse(pred_start_end, mc_start_end))
                self.logger.log_kv('m_1_after', m_1)
                self.logger.log_kv('b_1_after', b_1)
                self.logger.log_kv('r_sq_1_after', r_sq_1)
                self.logger.log_kv('m_end_after', m_end)
                self.logger.log_kv('b_end_after', b_end)
                self.logger.log_kv('r_sq_end_after', r_sq_end)

            self.logger.log_kv('rb_num_terminal', np.sum(self.replay_buffer['is_terminal']))
            self.logger.log_kv('rb_oldest', np.min(self.replay_buffer['iterations']))
            self.logger.log_kv('rb_newest', np.max(self.replay_buffer['iterations']))
            self.logger.log_kv('rb_avg_age', np.mean(self.replay_buffer['iterations']))

            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
            self.logger.log_kv('dataset_size', len(self.replay_buffer['observations']))
            self.logger.log_kv('t', self.replay_buffer['t'])

        else:
            if self.fit_on_policy:
                self.baseline.fit(paths)
            if self.fit_off_policy:
                self.baseline.fit_off_policy_many(self.replay_buffer, self.policy, gamma)

        # eval_statistics = self.train_from_replay_buffer(paths)
        eval_statistics = self.train(paths)
        eval_statistics.append(N)

        if self.save_logs:
            self.log_rollout_statistics(paths)

        return eval_statistics

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

    def train(self, paths):
        raise Exception('not implemented')
        return None

    def train_from_replay_buffer(self, paths):
        # TODO cache baseline and only update when necessary?
        observations = self.replay_buffer['observations']

        actions = self.policy.get_action_batch(observations)

        predictions = self.baseline.predict({'observations': observations, 'actions': actions})

        print(predictions)

        self.update_policy(observations, actions, predictions, paths)

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
        # advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        returns = np.concatenate([path["returns"] for path in paths])
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

        self.update_policy(observations, actions, returns, paths)

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

        self.running_score = mean_return if self.running_score is None else \
                            0.9*self.running_score + 0.1*mean_return
