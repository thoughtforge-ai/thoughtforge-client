import gym
import math 
import os
from dotenv import load_dotenv

from thoughtforge_client import BaseThoughtForgeClientSession


EPSILON = 0.000001


class ExampleMountainCarSession(BaseThoughtForgeClientSession):

    def _reset_env(self):
        """ local helper function specific for openAI gym environments """
        self.last_observation = self.env.reset()
        self.score = 0

    def sim_started_notification(self):
        """ On sim start, initialize environment """   
        print("Initializing MountainCarContinuous-v0...", end='')
        self.env = gym.make('MountainCarContinuous-v0')
        if self.env is not None:
            self._reset_env()
        print("Complete.")
        
    def update(self, motor_dict):
        """ advance the environment sim """
        # render the environment locally
        self.env.render()

        # extract action sent from server
        motor_value = motor_dict['force_motor']        
        
        # step openAI gym env and get updated observation
        env_step_result = self.env.step([motor_value])
        self.last_observation, reward, terminal, _ = env_step_result
        self.score += reward
        if terminal:
            print("End of episode. Score =", self.score)
            self._reset_env()
 
        # send updated environment data to server
        x_position = self.last_observation[0] - 0.5
        height = math.sin(3 * self.last_observation[0]) - 1
        height_vel = height / (self.last_observation[1] + EPSILON)
        sensor_values = {
            'pos_sensor': x_position,
            'height_vel_sensor': height_vel,
        }
        return sensor_values


if __name__ == "__main__": 
    host = '0.0.0.0' if not 'HOST' in os.environ else os.environ['HOST']
    port = 4343 if not 'PORT' in os.environ else int(os.environ['PORT'])
    load_dotenv()
    api_key = os.getenv("THOUGHTFORGE_API_KEY")
    session = ExampleMountainCarSession('./examples/mountaincar/example_mountaincar.params', host, port, api_key)
