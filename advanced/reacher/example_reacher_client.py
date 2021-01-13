import gym, math, os
import numpy as np

from thoughtforge_client import BaseThoughtForgeClientSession


##################################################################
### Modify Reacher gym to not reset its arm position
from gym.envs.mujoco.reacher import ReacherEnv

last_qpos = None
last_qvel = None

def new_reset(self):
    global last_qpos, last_qvel
    last_qpos = np.copy(self.sim.data.qpos)
    last_qvel = np.copy(self.sim.data.qvel)
    self.sim.reset()
    ob = self.reset_model()
    return ob

ReacherEnv.reset = new_reset

def new_reset_model(self):
    # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
    qpos = np.copy(last_qpos)
    qvel = np.copy(last_qvel)
    while True:
        self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        if np.linalg.norm(self.goal) < 0.2:
            break
    qpos[-2:] = self.goal
    # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
    qvel[-2:] = 0
    self.set_state(qpos, qvel)
    return self._get_obs()

ReacherEnv.reset_model = new_reset_model
##################################################################


gym.register(
    id='long-reacher-v2',
    entry_point='gym.envs.mujoco.reacher:ReacherEnv',
    max_episode_steps=500,
)

def angle_between(v1, v2):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



class ExampleReacherSession(BaseThoughtForgeClientSession):

    def _reset_env(self):
        """ local helper function specific for openAI gym environments """
        self.last_observation = self.env.reset()
        self.score = 0
        # for tracking first couple derivatives of motion
        self.last_rotvel0 = 0
        self.last_rotvel1 = 0
        self.last_delta1_rotvel0 = 0
        self.last_delta1_rotvel1 = 0

    def sim_started_notification(self):
        """ On sim start, initialize environment """   
        print("Initializing long-reacher-v2...", end='')
        self.env = gym.make('long-reacher-v2')
        if self.env is not None:
            self._reset_env()
        print("Complete.")

    def sim_ended_notification(self):
        """ On sim start, destroy environment """
        if self.env is not None:
            self.env.close()
            self.env = None
        
    def update(self, motor_dict):
        """ advance the environment sim """
        # render the environment locally
        self.env.render()

        # extract action sent from server
        motor_value_0 = motor_dict['joint_1_torque'][0]
        motor_value_1 = motor_dict['joint_2_torque'][0]

        # step openAI gym env and get updated observation
        env_step_result = self.env.step([motor_value_0, motor_value_1])
        self.last_observation, reward, terminal, _ = env_step_result
        self.score += reward
        if terminal:
            print("End of episode. Score =", self.score)
            self._reset_env()
 
        # send updated environment data to server
        body0_pos = self.env.get_body_com("body0")
        body1_pos =self.env.get_body_com("body1")
        fingertip_pos = self.env.get_body_com("fingertip")
        target_pos = self.env.get_body_com("target")

        body0_to_target = target_pos - body0_pos
        body0_to_fingertip = fingertip_pos - body0_pos
        angle0_sign = np.sign(np.cross(body0_to_target, body0_to_fingertip)[2])
        angle0 = angle_between(body0_to_target, body0_to_fingertip)/np.pi
        # angle0_sensor
        angle0_sensor_val = angle0_sign * angle0
        
        clamped_abs_angle0 = angle0
        if angle0 < 0.25:
            clamped_abs_angle0 = 0.25        
        body0_xvelr = self.env.data.get_body_xvelr("body0")[2]
        body0_xvelr_scaled = body0_xvelr/10
        if abs(body0_xvelr_scaled) < 0.04:
            body0_xvelr_scaled = 0
        # rotvel0_sensor
        rotvel0_sensor_val = (body0_xvelr_scaled) / clamped_abs_angle0

        new_delta1_rotvel0 = self.last_rotvel0 - body0_xvelr
        # delta1_rotvel0_sensor
        delta1_rotvel0_sensor_val = (new_delta1_rotvel0 / clamped_abs_angle0)
        self.last_rotvel0 = body0_xvelr

        new_delta2_rotvel0 = self.last_delta1_rotvel0 - new_delta1_rotvel0
        # delta2_rotvel0_sensor
        delta2_rotvel0_sensor_val = (new_delta2_rotvel0  / clamped_abs_angle0)
        self.last_delta1_rotvel0 = new_delta1_rotvel0

        # radius_sensor
        radius_sensor_val = np.linalg.norm(body0_to_target) - np.linalg.norm(body0_to_fingertip)

        body1_xvelr = self.env.data.get_body_xvelr("body1")[2]
        # rotvel1_sensor
        rotvel1_sensor_val = body0_xvelr/20
        if abs(rotvel1_sensor_val) < 0.08:
            rotvel1_sensor_val = 0

        # delta1_rotvel1_sensor
        delta1_rotvel1_sensor_val = self.last_rotvel1 - body1_xvelr
        self.last_rotvel1 = body1_xvelr

        # delta2_rotvel1_sensor
        delta2_rotvel1_sensor_val = self.last_delta1_rotvel1 - delta1_rotvel1_sensor_val
        self.last_delta1_rotvel1 = delta1_rotvel1_sensor_val
        
        sensor_values = {
            "angle0_sensor1" : angle0_sensor_val,
            "angle0_sensor2" : angle0_sensor_val,
            "angle0_sensor3" : angle0_sensor_val,
            "angle0_sensor4" : angle0_sensor_val,
            'rotvel1_sensor': rotvel1_sensor_val,
            'delta1_rotvel1_sensor': delta1_rotvel1_sensor_val,
            'delta2_rotvel1_sensor': delta2_rotvel1_sensor_val,
            'radius_sensor1': radius_sensor_val,
            'radius_sensor2': radius_sensor_val,
            'rotvel0_sensor': rotvel0_sensor_val,
            'delta1_rotvel0_sensor': delta1_rotvel0_sensor_val,
            'delta2_rotvel0_sensor': delta2_rotvel0_sensor_val
        }
        return sensor_values


if __name__ == "__main__": 
    session = ExampleReacherSession.from_file('./advanced/reacher/example_reacher.params')
