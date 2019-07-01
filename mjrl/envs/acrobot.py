import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

class AcrobotEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'acrobot.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):

        self.do_simulation(a, self.frame_skip)
        tip_height = self.data.sensordata[6]
        ctrl_cost = 1e-3 * np.square(a).sum()
        reward = tip_height - ctrl_cost

        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos_init = self.init_qpos.copy()
        qpos_init[0] = self.np_random.uniform(low=-np.pi, high=np.pi)
        qpos_init[1] = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.set_state(qpos_init, self.init_qvel)
        self.sim.forward()
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*1.2