import os

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.reacher import ReacherEnv


class Reacher3JointEnv(ReacherEnv):
    def __init__(self):
        utils.EzPickle.__init__(self)
        FILE_PATH = os.getcwd() + '/advanced/reacher_3joint/reacher_3joint.xml'
        mujoco_env.MujocoEnv.__init__(self, FILE_PATH, 2)
