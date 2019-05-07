import numpy as np
import copy

class Baseline:
    def __init__(self, env_spec, **kwargs):
        self.d = env_spec.observation_dim + env_spec.action_dim    # number of states

    def fit(self, paths, return_errors=False):
        raise NotImplementedError("Baseline not implemented. Create a valid baseline")

    def predict(self, path):
        raise NotImplementedError("Baseline not implemented. Create a valid baseline")
