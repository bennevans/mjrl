
import numpy as np

class ReplayBuffer():

    def __init__(self, max_dataset_size=-1):
        self.replay_buffer = {}
        self.max_dataset_size = max_dataset_size
    
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
            if 'observations' not in self.replay_buffer:
                observations = path['observations']
                actions = path['actions']
                rewards = path['rewards']

                l = observations.shape[0] - 1

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
        
        # remove oldest data (for now, could drop randomly?)
        if self.max_dataset_size > 0 and len(self.replay_buffer['observations']) > self.max_dataset_size:
            self.replay_buffer['observations'] = self.replay_buffer['observations'][-self.max_dataset_size:, :]
            self.replay_buffer['observations_prime'] = self.replay_buffer['observations_prime'][-self.max_dataset_size:, :]
            self.replay_buffer['actions'] = self.replay_buffer['actions'][-self.max_dataset_size:, :]
            self.replay_buffer['rewards'] = self.replay_buffer['rewards'][-self.max_dataset_size:]
            self.replay_buffer['last_update'] = self.replay_buffer['last_update'][-self.max_dataset_size:]

    def __getitem__(self, key):
        return self.replay_buffer[key]