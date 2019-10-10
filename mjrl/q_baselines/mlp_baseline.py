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

from mjrl.utils.logger import DataLog

import pickle
from tqdm import tqdm
import time as timer

class MLPBaseline:
    def __init__(self, env_spec, obs_dim=None, learn_rate=1e-3, reg_coef=0.0,
                batch_size=64, epochs=1, fit_iters=1, use_gpu=False, hidden_sizes=[64,64], err_tol=1e-6,
                use_time=True, use_epochs=False, nonlin=nn.ReLU):
        self.d = env_spec.observation_dim + env_spec.action_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.fit_iters = fit_iters
        self.reg_coef = reg_coef
        self.use_gpu = use_gpu
        self.learn_rate = learn_rate
        
        self.logger = DataLog()

        modules = [nn.Linear(self.d + int(use_time) * 3, hidden_sizes[0]), nonlin()]

        for i in range(len(hidden_sizes) - 1):
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            modules.append(nonlin())

        modules.append(nn.Linear(hidden_sizes[-1], 1))

        self.model = nn.Sequential(*modules)
        self.model_old = copy.deepcopy(self.model)

        if self.use_gpu:
            self.model.cuda()
            self.model_old.cuda()

        self.use_time = use_time
        self.use_epochs = use_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef, momentum=0.9)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef, momentum=0.0)
        self.loss_function = torch.nn.MSELoss()
        # self.loss_function = torch.nn.SmoothL1Loss()
        # self.mse_loss = torch.nn.MSELoss()
        # self.lam = 1e5

    # def loss_function(self, input, target):
    #     diffs = []
    #     for curr, new in zip(self.model_old.parameters(), self.model.parameters()):
    #         diffs.append(((curr - new)**2).view(-1))
    #     diff_loss = torch.mean(torch.cat(diffs))
    #     mse_part = self.mse_loss(input, target)
    #     # print('losses', mse_part, diff_loss, self.lam * diff_loss)
    #     return mse_part + self.lam * diff_loss

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)/10.0
        # TODO this / 10 for the actions is new. attempt to prevent numerical instability
        a = np.clip(path["actions"], -10, 10) / 10.0 # assumes actions are up-to-date
        
        feats_list = [o, a]

        if self.use_time:
            t = np.expand_dims(path['times'] / path['traj_length'], axis=1)
            feats_list.append(t)
            feats_list.append(t**2)
            feats_list.append(t**3)
        features = np.concatenate(feats_list, axis=1)

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        return features


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

    def fit_off_policy_many(self, replay_buffer, policy, gamma):

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate, weight_decay=self.reg_coef)


        # print('fit_off_policy_many', self.fit_iters)

        # first_model = copy.deepcopy(self.model)
        copy_time = 0.0
        fit_off_policy_time = 0.0

        all_weights = torch.cat([torch.flatten(x) for x in self.model.parameters()])
        pre_norm = torch.norm(all_weights).item()
        errors = []

        for j in tqdm(range(self.fit_iters)):
            fit_off_policy_start = timer.time()

            # TODO testing this out
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate, weight_decay=self.reg_coef)

            self.model_old = copy.deepcopy(self.model)
            self.model_old.eval()
            copy_time_end = timer.time()
            errors += self.fit_off_policy(replay_buffer, policy, gamma, return_errors=False)

            fit_end = timer.time()
            copy_time += (copy_time_end - fit_off_policy_start)
            fit_off_policy_time += (fit_end - copy_time_end)

            all_weights_old = torch.cat([torch.flatten(x) for x in self.model_old.parameters()])
            norm_old = torch.norm(all_weights_old).item()

            all_weights = torch.cat([torch.flatten(x) for x in self.model.parameters()])
            norm_new = torch.norm(all_weights).item()

            if np.abs(norm_old - norm_new) > 100:
                print('big norm diff inner')
                print('old', norm_old, 'new', norm_new)
                import pdb; pdb.set_trace()
        
        all_weights = torch.cat([torch.flatten(x) for x in self.model.parameters()])
        post_norm = torch.norm(all_weights).item()

        print('pre:', pre_norm, 'post:', post_norm)

        if np.abs(pre_norm - post_norm) > 100:
            print('big norm diff')
            # import pdb; pdb.set_trace()

        # print('copy_time', copy_time)
        # print('fit_off_policy_time', fit_off_policy_time)
        return errors


    def fit_off_policy(self, replay_buffer, policy, gamma, return_errors=False):
        # print('fit_off_policy', torch.cuda.memory_allocated(0))

        n = replay_buffer['observations'].shape[0]        

        observations_prime = replay_buffer['observations_prime']
        actions_prime = policy.get_mean_action_batch(observations_prime)
        
        path = {
            'observations': replay_buffer['observations'],
            'actions': replay_buffer['actions'],
        }

        path_prime = {
            'observations': observations_prime,
            'actions': actions_prime
        }

        if(self.use_time):
            path['times'] = replay_buffer['times']
            path['traj_length'] = replay_buffer['traj_length']

            path_prime['times'] = replay_buffer['times'] + 1
            path_prime['traj_length'] = replay_buffer['traj_length']


        if self.use_epochs:

            for ep in range(self.epochs):

                # target_create_start = timer.time()

                # Qs = self.predict(path_prime) # TODO set flag for both?
                Qs = self.predict_old(path_prime)
                targets = (replay_buffer['rewards'] + gamma * Qs).astype('float32')
                terminal_states = np.argwhere(replay_buffer['is_terminal'] == 1)
                targets[terminal_states] = replay_buffer['rewards'][terminal_states]

                featmat = np.array(self._features(path)).astype('float32')

                if self.use_gpu:
                    featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
                    targets_var = Variable(torch.from_numpy(targets).cuda(), requires_grad=False)
                else:
                    featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
                    targets_var = Variable(torch.from_numpy(targets), requires_grad=False)

                # target_create_end = timer.time()

                rand_idx = np.random.permutation(n)
                for mb in range(n // self.batch_size + 1):
                    if self.use_gpu:
                        data_idx = rand_idx[mb*self.batch_size:(mb+1)*self.batch_size]
                    else:
                        data_idx = rand_idx[mb*self.batch_size:(mb+1)*self.batch_size]
                    batch_x = featmat_var[data_idx]
                    batch_y = targets_var[data_idx]
                    self.optimizer.zero_grad()
                    yhat = torch.squeeze(self.model(batch_x))
                    loss = self.loss_function(yhat, batch_y)
                    loss.backward()
                    self.optimizer.step()
                
                # mb_end = timer.time()

                # print('target_create', target_create_end - target_create_start)
                # print('mb time', mb_end - target_create_end)

            # self.model = copy.deepcopy(self.model_old)

        else:
            errors = []
            for ep in range(self.epochs): # use epochs as iterations for now
                batch_idx = np.random.permutation(n)[:self.batch_size]

                path = {
                    'observations': replay_buffer['observations'][batch_idx],
                    'actions': replay_buffer['actions'][batch_idx],
                }

                path_prime = {
                    'observations': observations_prime[batch_idx],
                    'actions': actions_prime[batch_idx]
                }

                if(self.use_time):
                    path['times'] = replay_buffer['times'][batch_idx]
                    path['traj_length'] = replay_buffer['traj_length'][batch_idx]

                    path_prime['times'] = replay_buffer['times'][batch_idx] + 1
                    path_prime['traj_length'] = replay_buffer['traj_length'][batch_idx]


                Qs = self.predict_old(path_prime)
                targets = (replay_buffer['rewards'][batch_idx] + gamma * Qs).astype('float32')
        
                if self.use_time:
                    terminal_states = np.argwhere(replay_buffer['is_terminal'][batch_idx] == 1)
                    targets[terminal_states] = replay_buffer['rewards'][terminal_states]
        
                featmat = np.array(self._features(path)).astype('float32')
        
                
                if self.use_gpu:
                    featmat_var = torch.from_numpy(featmat).cuda()
                    targets_var = torch.from_numpy(targets).cuda()
                else:
                    featmat_var = torch.from_numpy(featmat)
                    targets_var = torch.from_numpy(targets)

                batch_x = featmat_var
                batch_y = targets_var
                
                self.optimizer.zero_grad()
                yhat = torch.squeeze(self.model(batch_x))
                loss = self.loss_function(yhat, batch_y)
                # if loss.item() > 100000:
                #     print('fit_off_policy loss threshold', loss.item())
                    # import pdb; pdb.set_trace()
                errors.append(loss.item())
                loss.backward()
                self.optimizer.step()
                
                # mb_end = timer.time()

                # print('target_create', target_create_end - target_create_start)
                # print('mb time', mb_end - target_create_end)

            # self.model = copy.deepcopy(self.model_old)

        del featmat_var
        del targets_var

        return errors



    def predict(self, path):
        # print('predict', torch.cuda.memory_allocated(0))
        featmat = self._features(path).astype('float32')
        if self.use_gpu:
            feat_var = torch.from_numpy(featmat).float().cuda()
            # print(feat_var.shape)
            prediction = self.model(feat_var).cpu().data.numpy().ravel()
        else:
            feat_var = torch.from_numpy(featmat).float()
            prediction = self.model(feat_var).data.numpy().ravel()
        del featmat
        del feat_var
        return prediction

    def predict_old(self, path):
        featmat = self._features(path).astype('float32')
        if self.use_gpu:
            feat_var = torch.from_numpy(featmat).float().cuda()
            prediction = self.model_old(feat_var).cpu().data.numpy().ravel()
        else:
            feat_var = torch.from_numpy(featmat).float()
            prediction = self.model_old(feat_var).data.numpy().ravel()
        del featmat
        del feat_var
        return prediction
