
import json, os, requests, traceback
from urllib.parse import urlencode, urlparse, urlunparse

from utils import safe_dict_get, load_client_params, CURRENT_CLIENT_PARAMS_VERSION


class BaseThoughtForgeClientSession():
    """ Base class for implementing a client for the ThoughtForge server API """
    def __init__(self, file_name, host, port, api_key, model_data=None):
        try:
            self.client_params = load_client_params(file_name)
            # check version and api key
            if ('version' not in self.client_params) or self.client_params['version'] != CURRENT_CLIENT_PARAMS_VERSION:
                print(self.client_params)
                print("Version not supported.")
                assert(False)
            if not api_key:
                print("ThoughtForge API Key required.")
                assert(False)

            self.session_id = None
            self.host = host
            self.port = port
            self.api_key = api_key
            self.model_data = model_data
            self.sensor_name_map = {}
            self.motor_name_map = {}
            self._stop_requested = False
            self.all_session_logs = []
            self._initialize_session()
            if self.session_id is not None and self.session_id >= 0:
                self._start_sim()
        except (KeyboardInterrupt, SystemExit):
            print("KeyboardInterrupt/SystemExit received.")
        except Exception as e:
            print(traceback.format_exc())
            print("Exception received:", e)
            self.close()
            # let all other exceptions pass through after we close the session
            raise
        finally:
            self.close()

    def _build_url(self, path, args_dict=None):
        # Returns a list in the structure of urlparse.ParseResult
        scheme = 'http'
        netloc = self.host + ':' + str(self.port)
        path = path
        params = urlencode(args_dict) if args_dict else '' 
        query = ''
        fragments = ''
        return urlunparse([scheme, netloc, path, params, query, fragments])

    def _initialize_session(self):
        # close old session
        if self.session_id != None:
            self.close()
        
        # iniitalize session
        model_data_to_send = None
        if self.model_data is not None:
            converted_model_data = {}
            weight_array_list = self.model_data['weights']
            weight_list_list = [weightarray.tolist() for weightarray in weight_array_list]
            converted_model_data['weights'] = weight_list_list
            converted_model_data['values'] = self.model_data['values'].tolist()
            model_data_to_send = json.dumps(converted_model_data).encode()

        initSession_params = {
            'version': self.client_params['version'],
            'internal_timescale': safe_dict_get(self.client_params, 'internal_timescale', 1), 
            'ticks_per_sensor_sample': safe_dict_get(self.client_params, 'ticks_per_sensor_sample', 1),
            'center_block_size_extra': safe_dict_get(self.client_params, 'center_block_size_extra', 0),
            'center_block_stride': safe_dict_get(self.client_params, 'center_block_stride', 1),
            'random_seed': safe_dict_get(self.client_params, 'random_seed', 42),
            'motors' : json.dumps(self.client_params['motors']),
            'sensors' : json.dumps(self.client_params['sensors']),
            }
        init_url = self._build_url('/initSession', initSession_params)
        headers = {"X-thoughtforge-key": self.api_key}
        response = requests.post(init_url, headers=headers, data=model_data_to_send)
        if response.ok:
            response_dict = response.json()
            self.session_id = response_dict['session_id']
            if self.session_id >= 0:
                self.motor_name_map = json.loads(response_dict['motor_ids'])
                self.sensor_name_map = json.loads(response_dict['sensor_ids'])
                print("Session", self.session_id, "has been initialized.")
            else:
                print("Session inialization failed.")
            session_log = json.loads(response_dict['session_log'])
            self.process_session_logs(session_log)
        else:
            print("Session inialization failed. Server returned", response)

    def _start_sim(self):
        """ Starts simulation of the agent and environment and triggers subsequent calls to update() """
        self.sim_started_notification()
        motor_ids = list(self.motor_name_map.values())
        next_motor_action_dict = {motor_name: 0.0 for motor_name in self.motor_name_map.keys()}
        print("Session", self.session_id, "starting simulation....")
        while not self._stop_requested:
            named_sensor_dict = self.update(next_motor_action_dict)
            sensor_dict = {self.sensor_name_map[key]:val for key, val in named_sensor_dict.items()}
            update_params = {
                'session_id': self.session_id,
                'sensor_dict': json.dumps(sensor_dict), 
                'motor_ids_requested': json.dumps(motor_ids)
            }
            update_url = self._build_url('/updateSim', update_params)
            headers = {"X-thoughtforge-key": self.api_key}
            response = requests.post(update_url, headers=headers)
            if not response.ok:
                print("Session update failed. Server returned", response)
            response_dict = response.json()
            motor_dict = response_dict['motor_dict']
            session_log = json.loads(response_dict['session_log'])
            self.process_session_logs(session_log)
            int_key_response_dict = {int(key):val for key, val in motor_dict.items()}
            next_motor_action_dict = {motor_name: safe_dict_get(int_key_response_dict, motor_id, 0.0) for motor_name, motor_id in self.motor_name_map.items()}

    def close(self):
        # shut down session
        if self.session_id is not None and self.session_id >= 0:
            shutdownSession_params = {'session_id': self.session_id}
            shutdown_url = self._build_url('/shutdownSession', shutdownSession_params)
            headers = {"X-thoughtforge-key": self.api_key}
            response = requests.post(shutdown_url, headers=headers)
            if response.ok:
                print("Session", self.session_id, "has been shut down.")
                response_dict = response.json()
                session_log = json.loads(response_dict['session_log'])
                self.process_session_logs(session_log)
            else:
                print("Session shutdown failed. Server returned", response)
            self.session_id = None
            self.sensor_name_map = {}
            self.motor_name_map = {}

    def get_num_motors(self):
        """ returns the number of motors that have been added to the session model """
        return len(self.motor_name_map)

    def get_num_sensors(self):
        """ returns the number of sensors that have been added to the session model """
        return len(self.sensor_name_map)

    def stop_sim(self):
        """ This function requests stopping of the simulation.  """
        self._stop_requested = True

    def sim_started_notification(self):
        """ This function can optionally be implemented by users to initialize 
        any simulation environment parameters """
        pass

    def update(self, motor_action_dict):
        """ Implement this function in client code to update environment state and return sensor data. """
        raise NotImplementedError
    
    def process_session_logs(self, session_logs):
        if len(session_logs) > 0:
            print("Server messages recieved:")
            for log_messsage in session_logs:
                print(" -", log_messsage)
            self.all_session_logs.extend(session_logs)