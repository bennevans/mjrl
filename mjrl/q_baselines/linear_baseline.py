import numpy as np
import copy

from mjrl.q_baselines.baseline import Baseline

class LinearBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5, epochs=1):
        super().__init__(env_spec)
        self._reg_coeff = reg_coeff
        self._coeffs = np.zeros(env_spec.observation_dim + env_spec.action_dim + 4)
        self.epochs = epochs

    def _features(self, path):
        # compute regression features for the path
        o = np.clip(path["observations"], -10, 10)
        a = np.clip(path['actions'], -10, 10) # assumes actions are up-to-date

        features = np.concatenate([o, a], axis=1)

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        
        l = len(path["observations"])
        al = np.arange(l).reshape(-1, 1) / 1000.0
        feat = np.concatenate([features, al, al**2, al**3, np.ones((l, 1))], axis=1)
        return feat

    def fit(self, paths, return_errors=False):

        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if return_errors:
            predictions = featmat.dot(self._coeffs) if self._coeffs is not None else np.zeros(returns.shape)
            errors = returns - predictions
            error_before = np.sum(errors**2)/np.sum(returns**2)

        reg_coeff = copy.deepcopy(self._reg_coeff)
        for _ in range(10):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

        if return_errors:
            predictions = featmat.dot(self._coeffs)
            errors = returns - predictions
            error_after = np.sum(errors**2)/np.sum(returns**2)
            return error_before, error_after

    def fit_off_policy(self, replay_buffer, policy, gamma, return_errors=False):
        # process trajectories one at a time to make sure we have valid s'
        n = replay_buffer['observations'].shape[0]        

        observations_prime = replay_buffer['observations_prime']
        actions_prime = policy.get_action_batch(observations_prime)
        path = {
            'observations': replay_buffer['observations'],
            'actions': replay_buffer['actions']
        }

        path_prime = {
            'observations': observations_prime,
            'actions': actions_prime
        }

        if return_errors:
            Qs = self.predict(path_prime)

            targets = (replay_buffer['rewards'] + gamma * Qs)
            featmat = np.array(self._features(path))
            predictions = featmat.dot(self._coeffs)
            errors = targets - predictions
            error_before = np.sum(errors**2)/np.sum(targets**2)

        for ep in range(self.epochs):

            Qs = self.predict(path_prime)

            targets = (replay_buffer['rewards'] + gamma * Qs)
            featmat = np.array(self._features(path))
            
            reg_coeff = copy.deepcopy(self._reg_coeff)
            Sigma = featmat.T.dot(featmat)
            for _ in range(10):
                self._coeffs = np.linalg.lstsq(
                    Sigma + reg_coeff * np.identity(featmat.shape[1]),
                    featmat.T.dot(targets)
                )[0]
                if not np.any(np.isnan(self._coeffs)):
                    break
                reg_coeff *= 10
                print('warning, reg_coeff too small, increasing')

        if return_errors:
            predictions = featmat.dot(self._coeffs)
            errors = targets - predictions
            error_after = np.sum(errors**2)/np.sum(targets**2)
            return error_before, error_after

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["observations"]))
        return self._features(path).dot(self._coeffs)
