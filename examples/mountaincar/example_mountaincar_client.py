import gym, math , os

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
            'pos_sensor1': x_position,
            'pos_sensor2': x_position,
            'height_vel_sensor1': height_vel,
            'height_vel_sensor2': height_vel,
            'height_vel_sensor3': height_vel,
        }
        return sensor_values


if __name__ == "__main__": 
    session = ExampleMountainCarSession.from_file('./examples/mountaincar/example_mountaincar.params')
