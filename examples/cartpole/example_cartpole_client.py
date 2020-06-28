import gym
import os
from dotenv import load_dotenv

from thoughtforge_client import BaseThoughtForgeClientSession


# this is just a modification to the cartpole environment to extend it to 500 steps
gym.register(
    id='long-CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=195.0,
)

class ExampleCartpoleSession(BaseThoughtForgeClientSession):

    def _reset_env(self):
        """ local helper function specific for openAI gym environments """
        self.last_observation = self.env.reset()
        self.score = 0

    def sim_started_notification(self):
        """ On sim start, initialize environment """   
        print("Initializing Cartpole...", end='')
        self.env = gym.make('long-CartPole-v0')
        if self.env is not None:
            self._reset_env()
        print("Complete.")
        
    def update(self, motor_dict):
        """ advance the environment sim """
        # render the environment locally
        self.env.render()

        # extract action sent from server
        motor_value = motor_dict['motor']
        cartpole_action = 1 if motor_value >= 0.0 else 0
        
        # step openAI gym env and get updated observation
        env_step_result = self.env.step(cartpole_action)
        self.last_observation, reward, terminal, _ = env_step_result
        self.score += reward
        if terminal:
            print("End of episode. Score =", self.score)
            self._reset_env()
 
        # send updated environment data to server
        sensor_values = {
            'pos_sensor': self.last_observation[0],
            'vel_sensor': self.last_observation[1],
            'angle_sensor1': self.last_observation[2],
            'angle_sensor2': self.last_observation[2],
            'angle_vel_sensor1': self.last_observation[3],
            'angle_vel_sensor2': self.last_observation[3],
        }
        return sensor_values


if __name__ == "__main__": 
    host = '0.0.0.0' if not 'HOST' in os.environ else os.environ['HOST']
    port = 4343 if not 'PORT' in os.environ else int(os.environ['PORT'])
    load_dotenv()
    api_key = os.getenv("THOUGHTFORGE_API_KEY")
    session = ExampleCartpoleSession('./examples/cartpole/example_cartpole.params', host, port, api_key)
