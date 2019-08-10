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
        # dense reward
        # ctrl_cost = 1e-3 * np.square(a).sum()
        # reward = tip_height - ctrl_cost
        

        # sparse reward
        # reward = 0.0

        # if tip_height > 1.9:
        #     reward += 1

        # shaped reward
        reward = tip_height

        if tip_height > 1.9:
            reward += 1.0

            # success_vel_pen = 1e-2 * np.abs(self.data.qvel[0])
            success_vel_pen = 5e-1 * np.linalg.norm(self.data.qvel)
            
            if tip_height > 1.95:
                reward += 2.0
                reward -= success_vel_pen
            # ctrl_cost = 1e-4 * np.square(a).sum()
            # reward -= ctrl_cost
        
        # vel_pen = 1e-3 * np.abs(self.data.qvel[0])
        # acc_pen = 1e-3 * np.linalg.norm(self.data.qacc)

        # reward -= vel_pen

        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        wrapped_to_pi = (self.data.qpos + np.pi) % (2 * np.pi) - np.pi

        return np.concatenate([
            wrapped_to_pi,
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