
import numpy as np

class ReplayBuffer():

    DROP_MODE_OLDEST = 'oldest'
    DROP_MODE_RANDOM = 'random'

    def __init__(self, max_dataset_size=-1, drop_mode=DROP_MODE_OLDEST):
        self.replay_buffer = {}
        self.max_dataset_size = max_dataset_size
        self.drop_mode = drop_mode
    
    def update(self, paths):
        """
        adds new paths to the replay buffer
        params:
            paths - a dict with keys
                observations
                actions
                rewards
        """

        for path in paths:
            observations = path['observations']
            actions = path['actions']
            rewards = path['rewards']
            times = path['times']
            traj_length = path['traj_length']

            l = observations.shape[0] - 1
            
            if 'observations' not in self.replay_buffer:

                self.replay_buffer['observations'] = observations[:-1]
                self.replay_buffer['observations_prime'] = observations[1:]
                self.replay_buffer['actions'] = actions[:-1]
                self.replay_buffer['rewards'] = rewards[:-1]
                self.replay_buffer['last_update'] = np.zeros(l)
                self.replay_buffer['times'] = times[:-1]
                # need to do this ourselves, since we cutoff the last s,a
                self.replay_buffer['is_terminal'] = np.zeros(l)
                self.replay_buffer['is_terminal'][-1] = 1
                self.replay_buffer['traj_length'] = traj_length[:-1]
                self.replay_buffer['t'] = 0
                self.replay_buffer['iteration'] = 0
                self.replay_buffer['iterations'] = np.zeros(l)
            else:
                self.replay_buffer['t'] += 1
                self.replay_buffer['observations'] = np.concatenate([self.replay_buffer['observations'], observations[:-1]])
                self.replay_buffer['observations_prime'] = np.concatenate([self.replay_buffer['observations_prime'], observations[1:]])
                self.replay_buffer['actions'] = np.concatenate([self.replay_buffer['actions'], actions[:-1]])
                self.replay_buffer['rewards'] = np.concatenate([self.replay_buffer['rewards'], rewards[:-1]])
                self.replay_buffer['last_update'] = np.concatenate([self.replay_buffer['last_update'], np.ones(l) * self.replay_buffer['t']])
                self.replay_buffer['times'] = np.concatenate([self.replay_buffer['times'], times[:-1]])
                self.replay_buffer['is_terminal'] = np.concatenate([self.replay_buffer['is_terminal'], np.zeros(l)])
                self.replay_buffer['is_terminal'][-1] = 1
                self.replay_buffer['traj_length'] = np.concatenate([self.replay_buffer['traj_length'], traj_length[:-1]])
                self.replay_buffer['iterations'] = np.concatenate([self.replay_buffer['iterations'],  np.ones(l) * self.replay_buffer['iteration']])
        self.replay_buffer['iteration'] += 1

        cur_size = len(self.replay_buffer['observations'])
        if self.max_dataset_size > 0 and cur_size > self.max_dataset_size:
            if self.drop_mode == self.DROP_MODE_OLDEST:
                self.replay_buffer['observations'] = self.replay_buffer['observations'][-self.max_dataset_size:, :]
                self.replay_buffer['observations_prime'] = self.replay_buffer['observations_prime'][-self.max_dataset_size:, :]
                self.replay_buffer['actions'] = self.replay_buffer['actions'][-self.max_dataset_size:, :]
                self.replay_buffer['rewards'] = self.replay_buffer['rewards'][-self.max_dataset_size:]
                self.replay_buffer['last_update'] = self.replay_buffer['last_update'][-self.max_dataset_size:]
                self.replay_buffer['times'] = self.replay_buffer['times'][-self.max_dataset_size:]
                self.replay_buffer['is_terminal'] = self.replay_buffer['is_terminal'][-self.max_dataset_size:]
                self.replay_buffer['traj_length'] = self.replay_buffer['traj_length'][-self.max_dataset_size:]
                self.replay_buffer['iterations'] = self.replay_buffer['iterations'][-self.max_dataset_size:]
            elif self.drop_mode == self.DROP_MODE_RANDOM:
                keep_idx = np.random.permutation(np.arange(cur_size))[:self.max_dataset_size]
                self.replay_buffer['observations'] = self.replay_buffer['observations'][keep_idx]
                self.replay_buffer['observations_prime'] = self.replay_buffer['observations_prime'][keep_idx]
                self.replay_buffer['actions'] = self.replay_buffer['actions'][keep_idx]
                self.replay_buffer['rewards'] = self.replay_buffer['rewards'][keep_idx]
                self.replay_buffer['last_update'] = self.replay_buffer['last_update'][keep_idx]
                self.replay_buffer['times'] = self.replay_buffer['times'][keep_idx]
                self.replay_buffer['is_terminal'] = self.replay_buffer['is_terminal'][keep_idx]
                self.replay_buffer['traj_length'] = self.replay_buffer['traj_length'][keep_idx]
                self.replay_buffer['iterations'] = self.replay_buffer['iterations'][keep_idx]
            else:
                raise Exception("invalid drop mode: {}".format(self.drop_mode))

    def __getitem__(self, key):
        return self.replay_buffer[key]

    def sample(self, n):
        """
        returns n samples from the replay buffer at uniform random in dict form
        """
        cur_size = len(self.replay_buffer['observations'])
        sample_idx = np.random.permutation(np.arange(cur_size))[:n]

        # assert n < cur_size # TODO handle this case


        return {
            'observations': self['observations'][sample_idx],
            'observations_prime': self['observations_prime'][sample_idx],
            'actions': self['actions'][sample_idx],
            'rewards': self['rewards'][sample_idx],
            'last_update': self['last_update'][sample_idx],
            'times': self['times'][sample_idx],
            'is_terminal': self['is_terminal'][sample_idx],
            'traj_length': self['traj_length'][sample_idx],
            'iterations': self['iterations'][sample_idx],
        }, min(cur_size, n)



    def __str__(self):
        return "length: {}\n{}".format(len(self.replay_buffer['observations']), str(self.replay_buffer))