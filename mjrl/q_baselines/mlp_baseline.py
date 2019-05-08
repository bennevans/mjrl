from os import environ
# Select GPU 0 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle

class MLPBaseline:
    def __init__(self, env_spec, obs_dim=None, learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, num_iters=1, use_gpu=False, hidden_sizes=[128,128]):
        self.d = env_spec.observation_dim + env_spec.action_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.num_iters = num_iters
        self.use_gpu = use_gpu

        modules = [nn.Linear(self.d + 4, hidden_sizes[0]), nn.ReLU()]

        for i in range(len(hidden_sizes) - 1):
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(hidden_sizes[-1], 1))

        self.model = nn.Sequential(*modules)

        if self.use_gpu:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = torch.nn.MSELoss()

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)/10.0

        a = np.clip(path["actions"], -10, 10) # assumes actions are up-to-date
        
        features = np.concatenate([o, a], axis=1)

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        N, n = features.shape
        num_feat = int( n + 4 )            # linear + time till pow 4
        feat_mat =  np.ones((N, num_feat)) # memory allocation

        # linear features
        feat_mat[:,:n] = features

        k = 0  # start from this row
        l = len(path["observations"])
        al = np.arange(l)/1000.0
        for j in range(4):
            feat_mat[k:k+l, -4+j] = al**(j+1)
        k += l
        return feat_mat


    def fit(self, paths, return_errors=False):

        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)
        featmat = featmat.astype('float32')
        returns = returns.astype('float32')
        num_samples = returns.shape[0]

        # Make variables with the above data
        if self.use_gpu:
            featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns).cuda(), requires_grad=False)
        else:
            featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns), requires_grad=False)

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)

        for ep in range(self.epochs):
            rand_idx = np.random.permutation(num_samples)
            for mb in range(int(num_samples / self.batch_size) - 1):
                if self.use_gpu:
                    data_idx = torch.LongTensor(rand_idx[mb*self.batch_size:(mb+1)*self.batch_size]).cuda()
                else:
                    data_idx = torch.LongTensor(rand_idx[mb*self.batch_size:(mb+1)*self.batch_size])
                batch_x = featmat_var[data_idx]
                batch_y = returns_var[data_idx]
                self.optimizer.zero_grad()
                yhat = self.model(batch_x)
                loss = self.loss_function(yhat, batch_y)
                loss.backward()
                self.optimizer.step()

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
            return error_before, error_after

    def fit_off_policy(self, replay_buffer, policy, gamma, return_errors=False):
        n = replay_buffer['observations'].shape[0]
        # only update actions if they are in the batch

        errors_before = []
        errors_after = []
        batches_x = []
        batches_y = []

        # pick all batches before-hand so we can compute pre and post errors
        for _ in range(self.num_iters):
            rand_idx = np.random.permutation(n)[:self.batch_size]

            replay_buffer['last_update'][rand_idx] = replay_buffer['t']

            observations_prime = replay_buffer['observations_prime'][rand_idx]
            actions_prime = np.stack([policy.get_action(obs)[0] for obs in observations_prime])
            
            path = {
                'observations': replay_buffer['observations'][rand_idx],
                'actions': replay_buffer['actions'][rand_idx]
            }

            path_prime = {
                'observations': observations_prime,
                'actions': actions_prime
            }

            Qs = self.predict(path_prime)
            targets = replay_buffer['rewards'][rand_idx] + gamma*Qs
            featmat = np.array(self._features(path))

            featmat = featmat.astype('float32')
            returns = targets.astype('float32')

            if self.use_gpu:
                featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
                returns_var = Variable(torch.from_numpy(returns).cuda(), requires_grad=False)
            else:
                featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
                returns_var = Variable(torch.from_numpy(returns), requires_grad=False)

            if return_errors:
                if self.use_gpu:
                    predictions = self.model(featmat_var).cpu().data.numpy().ravel()
                else:
                    predictions = self.model(featmat_var).data.numpy().ravel()
                errors = returns.ravel() - predictions
                errors_before.append(errors)

            batches_x.append(featmat_var)
            batches_y.append(returns_var)
        
        for batch_x, batch_y in zip(batches_x, batches_y):
            self.optimizer.zero_grad()
            yhat = self.model(batch_x)
            loss = self.loss_function(yhat, batch_y)
            loss.backward()
            self.optimizer.step()

            if return_errors:
                if self.use_gpu:
                    predictions = self.model(batch_x).cpu().data.numpy().ravel()
                else:
                    predictions = self.model(batch_x).data.numpy().ravel()
                errors = batch_y.cpu().data.numpy().ravel() - predictions
                errors_after.append(errors)
                
        
        if return_errors:
            before = np.concatenate(errors_before)
            after = np.concatenate(errors_after)
            returns = np .concatenate(batches_y)
            error_before = np.sum(before**2) / (np.sum(returns**2) + 1e-8)
            error_after = np.sum(after**2) / (np.sum(returns**2) + 1e-8)
            return error_before, error_after



    # def fit_off_policy(self, replay_buffer, policy, gamma, return_errors=False):
        
    #     # TODO: make this in baseline, so we don't rewrite the code
    #     # TODO: use fit() in fit_off_policy?
    #     states = []
    #     actions = []
    #     state_primes = []
    #     action_primes = []
    #     rewards = []
        
    #     # TODO optimize
    #     for path in replay_buffer:
    #         l = len(path["observations"])
    #         for i, (s, a, r) in enumerate(zip(path["observations"], path["actions"], path["rewards"])):
    #             if i == l - 1:
    #                 break
    #             sp = path["observations"][i+1]
    #             states.append(s)
    #             actions.append(a)
    #             state_primes.append(sp)
    #             ap, _ = policy.get_action(sp)
    #             action_primes.append(ap)
    #             rewards.append(r)
    #     rewards = np.array(rewards)
        
    #     faux_path = {
    #         'observations': np.stack(states),
    #         'actions': np.stack(actions)
    #     }
    #     faux_path_prime = {
    #         'observations': np.stack(state_primes),
    #         'actions': np.stack(action_primes)
    #     }

    #     Qs = self.predict(faux_path_prime)

    #     targets = rewards + gamma*Qs
    #     featmat = np.array(self._features(faux_path))

    #     featmat = featmat.astype('float32')
    #     returns = targets.astype('float32')
    #     num_samples = returns.shape[0]

    #     # Make variables with the above data
    #     if self.use_gpu:
    #         featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
    #         returns_var = Variable(torch.from_numpy(returns).cuda(), requires_grad=False)
    #     else:
    #         featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
    #         returns_var = Variable(torch.from_numpy(returns), requires_grad=False)

    #     if return_errors:
    #         if self.use_gpu:
    #             predictions = self.model(featmat_var).cpu().data.numpy().ravel()
    #         else:
    #             predictions = self.model(featmat_var).data.numpy().ravel()
    #         errors = returns.ravel() - predictions
    #         error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)

    #     for i in range(self.num_iters):
    #         rand_idx = np.random.permutation(num_samples)[:self.batch_size]

    #         batch_x = featmat_var[torch.from_numpy(rand_idx)]
    #         batch_y = returns_var[torch.from_numpy(rand_idx)]
    #         self.optimizer.zero_grad()
    #         yhat = self.model(batch_x)
    #         loss = self.loss_function(yhat, batch_y)
    #         loss.backward()
    #         self.optimizer.step()

    #     if return_errors:
    #         if self.use_gpu:
    #             predictions = self.model(featmat_var).cpu().data.numpy().ravel()
    #         else:
    #             predictions = self.model(featmat_var).data.numpy().ravel()
    #         errors = returns.ravel() - predictions
    #         error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
    #         return error_before, error_after




    def predict(self, path):
        featmat = self._features(path).astype('float32')
        if self.use_gpu:
            feat_var = Variable(torch.from_numpy(featmat).float().cuda(), requires_grad=False)
            prediction = self.model(feat_var).cpu().data.numpy().ravel()
        else:
            feat_var = Variable(torch.from_numpy(featmat).float(), requires_grad=False)
            prediction = self.model(feat_var).data.numpy().ravel()
        return prediction
