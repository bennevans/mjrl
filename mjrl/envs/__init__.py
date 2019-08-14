from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='mjrl_point_mass-v0',
    entry_point='mjrl.envs:PointMassEnv',
    max_episode_steps=25,
)

register(
    id='mjrl_hard_point_mass-v0',
    entry_point='mjrl.envs:HardPointMassEnv',
    max_episode_steps=250,
)

register(
    id='mjrl_swimmer-v0',
    entry_point='mjrl.envs:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id='mjrl_acrobot-v0',
    entry_point='mjrl.envs:AcrobotEnv',
    # max_episode_steps=2000,
    max_episode_steps=400,
)

register(
    id='mjrl_bike-v0',
    entry_point="mjrl.envs:BikeEnv",
    max_episode_steps=1500,
)

register(
    id='mjrl_hopper_height-v0',
    entry_point="mjrl.envs:HopperHeightEnv",
    max_episode_steps=400,
)

register(
    id='mjrl_hopper-v0',
    entry_point="mjrl.envs:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='mjrl_reacher-v0',
    entry_point='mjrl.envs:Reacher7DOFEnv',
    max_episode_steps=75,
)

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mjrl.envs.point_mass import PointMassEnv
from mjrl.envs.hard_point_mass import HardPointMassEnv
from mjrl.envs.swimmer import SwimmerEnv
from mjrl.envs.acrobot import AcrobotEnv
from mjrl.envs.bike import BikeEnv
from mjrl.envs.hopper import HopperEnv
from mjrl.envs.reacher_env import Reacher7DOFEnv