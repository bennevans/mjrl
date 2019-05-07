import numpy as np
import copy

from mjrl.q_baselines.baseline import Baseline

class LinearBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        super().__init__(env_spec)
        self._reg_coeff = reg_coeff
        self._coeffs = None

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
        states = []
        actions = []
        state_primes = []
        action_primes = []
        rewards = []
        
        # TODO optimize
        for path in replay_buffer:
            l = len(path["observations"])
            for i, (s, a, r) in enumerate(zip(path["observations"], path["actions"], path["rewards"])):
                if i == l - 1:
                    break
                sp = path["observations"][i+1]
                states.append(s)
                actions.append(a)
                state_primes.append(sp)
                ap, _ = policy.get_action(sp)
                action_primes.append(ap)
                rewards.append(r)
        rewards = np.array(rewards)
        
        faux_path = {
            'observations': np.stack(states),
            'actions': np.stack(actions)
        }
        faux_path_prime = {
            'observations': np.stack(state_primes),
            'actions': np.stack(action_primes)
        }

        Qs = self.predict(faux_path_prime)

        targets = rewards + gamma*Qs
        featmat = np.array(self._features(faux_path))
        
        reg_coeff = copy.deepcopy(self._reg_coeff)
        for _ in range(10):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(targets)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

        if return_errors:
            predictions = featmat.dot(self._coeffs)
            errors = targets - predictions
            error_after = np.sum(errors**2)/np.sum(targets**2)
            return targets, error_after

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["observations"]))
        return self._features(path).dot(self._coeffs)
