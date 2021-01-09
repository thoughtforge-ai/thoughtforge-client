import os, gym
import numpy as np

from thoughtforge_client import BaseThoughtForgeClientSession


class ExampleAcrobotSession(BaseThoughtForgeClientSession):

    def _reset_env(self):
        """ local helper function specific for openAI gym environments """
        self.last_observation = self.env.reset()
        self.score = 0

    def sim_started_notification(self):
        """ On sim start, initialize environment """   
        print("Initializing Acrobot...", end='')
        self.env = gym.make('Acrobot-v1')
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

        # extract actions sent from server
        motor1_value = motor_dict['motor1']
        motor2_value = motor_dict['motor2']
        if (motor1_value > 0) and (motor2_value > 0):
            acrobot_action = 2  
        elif (motor1_value < 0) and (motor2_value < 0):
            acrobot_action = 0
        else:
            acrobot_action = 1
        
        # step openAI gym env and get updated observation
        env_step_result = self.env.step(acrobot_action)
        self.last_observation, reward, terminal, _ = env_step_result
        self.score += reward
        if terminal:
            print("End of episode. Score =", self.score)
            self._reset_env()
 
        # send updated environment data to server
        s0 = np.arctan2(self.last_observation[1], self.last_observation[0])
        s1 = np.arctan2(self.last_observation[3], self.last_observation[2])
        h = -np.cos(s0) - np.cos(s1 + s0)
        height_sensor_val = 2 - h
        # use the full sensor range by embedding direction into height
        if self.last_observation[1] < 0:
            height_sensor_val = -height_sensor_val
        sensor_values = {
            'height_sensor': height_sensor_val,
        }
        return sensor_values


if __name__ == "__main__": 
    session = ExampleAcrobotSession('./examples/acrobot/example_acrobot.params')
