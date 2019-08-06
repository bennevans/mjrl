import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

class HopperHeightEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    LAYING_QPOS=np.array([1.0, 0.07, 1.561, 0.0003, -0.5667, 0.7855])

    def __init__(self):
        self.torso_bid = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'hopper_height.xml', 4)
        self.torso_bid = self.sim.model.body_name2id('torso')

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos, height, angle = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]

        reward = 0.0
        upright_bonus = 1.0
        t_height = 0.85

        if height > t_height:
            reward += upright_bonus
            reward -= 0.1 * np.abs(pos)
        else:
            reward += height
        
        # reward -= 1e-3 * np.sum(a**2)

        return self._get_obs(), reward, False, dict(solved=1.0*(height > t_height))

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset_model(self):
        # randomize the agent and goal
        self.set_state(HopperHeightEnv.LAYING_QPOS, np.zeros_like(HopperHeightEnv.LAYING_QPOS))
        self.sim.forward()
        return self._get_obs()

    def evaluate_success(self, paths, logger=None):
        success = 0.0
        for p in paths:
            if np.mean(p['env_infos']['solved'][-4:]) > 0.0:
                success += 1.0
        success_rate = 100.0*success/len(paths)
        if logger is None:
            # nowhere to log so return the value
            return success_rate
        else:
            # log the success
            # can log multiple statistics here if needed
            logger.log_kv('success_rate', success_rate)
            return None

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()