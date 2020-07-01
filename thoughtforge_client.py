
import json, os, requests, traceback
import numpy as np
from dotenv import load_dotenv
from typing import List
from urllib.parse import urlencode, urlparse, urlunparse

from utils import safe_dict_get, load_client_params, CURRENT_CLIENT_PARAMS_VERSION


class BaseThoughtForgeClientSession():
    """ BaseThoughtForgeClientSession
    
    This class is a base class for implementing client applications on the ThoughtForge 
    platform. It manages the protocol for talking to the server and is inteded to be inherited
    by users to implement simple interative simulations.

    .. seealso:: Please see the ./examples/cartpole/ directory for an example usage of this class.

    :param file_name: The parameter file for specifying sensors, motors and model configuration
    :type file_name: str
    :param host: Host address for the destination ThoughtForge server. Defaults to `None`. If left unset, will be populated from hte environment variable 'THOUGHTFORGE_HOST'
    :type host: str
    :param port: Host port for the destination ThoughtForge server. Defaults to `None`. If left unset, will be populated from hte environment variable 'THOUGHTFORGE_PORT'
    :type port: int
    :param model_data: Optional parameter for supplying saved model data at initialization of the sim.
    :type model_data: dict

    """
    def __init__(self, file_name, host=None, port=None, api_key=None, model_data=None):
        try:
            load_dotenv()
            if api_key is None:
                api_key = os.getenv("THOUGHTFORGE_API_KEY")
            if host is None:
                host = os.getenv("THOUGHTFORGE_HOST")
            if port is None:
                port = os.getenv("THOUGHTFORGE_PORT")

            self.client_params = load_client_params(file_name)

            # check version and api key
            if ('version' not in self.client_params) or self.client_params['version'] != CURRENT_CLIENT_PARAMS_VERSION:
                print(self.client_params)
                print("Version not supported.")
                assert(False)
            if not api_key:
                print("ThoughtForge API Key required.")
                assert(False)

            self.sim_t = 0
            self.session_id = None
            self.sensor_name_map = {}
            self.motor_name_map = {}
            self._stop_requested = False
            self.all_session_logs = []
            self.sensor_value_history = []
            self.motor_value_history = []
            self.debug_data_history = []

            self.host = host
            self.port = port
            self.api_key = api_key
            self.model_data = model_data
            self._initialize_session()
            if self.session_id is not None and self.session_id >= 0:
                self._start_sim()
        except (KeyboardInterrupt, SystemExit):
            print("KeyboardInterrupt/SystemExit received.")
        except Exception as e:
            print(traceback.format_exc())
            print("Exception received:", e)
            self._close_session()
            # let all other exceptions pass through after we close the session
            raise
        finally:
            self._close_session()

    def _build_url(self, path, args_dict=None):
        """ Helper function for generating request URLS """ 
        # Returns a list in the structure of urlparse.ParseResult
        scheme = 'http'
        netloc = self.host + ':' + str(self.port)
        path = path
        params = urlencode(args_dict) if args_dict else '' 
        query = ''
        fragments = ''
        return urlunparse([scheme, netloc, path, params, query, fragments])

    def _validate_sensors_motors(self):
        """ this function is called after receiving a successful response 
        during server initialzation to ensure all motors and sensors were
        successfully registered """
        validation_successful = True
        # for validation, check that we have the expected number of sensors
        total_sensors = 0
        for entry in self.client_params['sensors']:
            if isinstance(entry['name'], List):
                total_sensors += len(entry['name'])
            else:
                total_sensors += 1
        if len(self.sensor_name_map) != total_sensors:
            print("Some sensors failed registration. Found", len(self.sensor_name_map), "expected", total_sensors)
            validation_successful = False
        # for validation, check that we have the expected number of motors
        total_motors = 0
        for entry in self.client_params['motors']:
            if isinstance(entry['name'], List):
                total_motors += len(entry['name'])
            else:
                total_motors += 1
        if len(self.motor_name_map) != total_motors:
            print("Some motors failed registration. Found", len(self.sensor_name_map), "expected", total_sensors)
            validation_successful = False
        return validation_successful 

    def _initialize_session(self):
        """ Initializes a remote session on the ThoughtForge server """ 
        # close old session
        if self.session_id != None:
            self._close_session()
        
        # if initialized with model_data, include in the post request
        model_data_to_send = None
        if self.model_data is not None:
            converted_model_data = {}
            weight_array_list = self.model_data['weights']
            weight_list_list = [weightarray.tolist() for weightarray in weight_array_list]
            converted_model_data['weights'] = weight_list_list
            converted_model_data['values'] = self.model_data['values'].tolist()
            model_data_to_send = json.dumps(converted_model_data).encode()

        self.debug_enabled = safe_dict_get(self.client_params, 'enable_debug', False)
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
        initialization_failed = False
        if response.ok:
            response_dict = response.json()
            self.session_id = response_dict['session_id']
            self.motor_name_map = json.loads(response_dict['motor_ids'])
            self.sensor_name_map = json.loads(response_dict['sensor_ids'])
            self.block_name_map = json.loads(response_dict['block_ids'])
            session_log = json.loads(safe_dict_get(response_dict, 'session_log', []))
            self._process_session_logs(session_log)
            if self.session_id < 0 or not self._validate_sensors_motors():
                initialization_failed = True
        else:
            initialization_failed = True

        if initialization_failed:
            print("Session inialization failed.")
            self.session_id = -1
        else:
            print("Session", self.session_id, "has been initialized.")

    def _start_sim(self):
        """ Starts simulation of the agent and environment and triggers subsequent calls to update() """
        self.sim_started_notification()
        motor_ids = list(self.motor_name_map.values())
        next_motor_dict = {motor_name: 0.0 for motor_name in self.motor_name_map.keys()}
        print("Session", self.session_id, "starting simulation....")
        while not self._stop_requested:
            self.motor_value_history.append(next_motor_dict)
            # send motor data into client to update the environment
            named_sensor_dict = self.update(next_motor_dict)
            self.sensor_value_history.append(named_sensor_dict)
            sensor_dict = {self.sensor_name_map[key]:val for key, val in named_sensor_dict.items()}
            # sent sensor data to the server
            update_params = {
                'session_id': self.session_id,
                'sensor_dict': json.dumps(sensor_dict), 
                'motor_ids_requested': json.dumps(motor_ids),
                'collect_debug_data': self.debug_enabled
            }
            update_url = self._build_url('/updateSim', update_params)
            headers = {"X-thoughtforge-key": self.api_key}
            response = requests.post(update_url, headers=headers)
            if not response.ok:
                print("Session update failed. Server returned", response)
            response_dict = response.json()
            # retrieve motor responses from the server
            motor_dict = response_dict['motor_dict']
            int_key_response_dict = {int(key):val for key, val in motor_dict.items()}
            next_motor_dict = {motor_name: safe_dict_get(int_key_response_dict, motor_id, 0.0) for motor_name, motor_id in self.motor_name_map.items()}
            # process session logs and debugging data from server
            session_log = json.loads(safe_dict_get(response_dict, 'session_log', []))
            self._process_session_logs(session_log)
            debugging_data = json.loads(response_dict['debugging_data'])
            self._process_debugging_data(debugging_data)
            # update simulation time
            self.sim_t += 1
    
    def _process_session_logs(self, session_logs):
        """ Processes session logs as they are received from the server. """ 
        if len(session_logs) > 0:
            print("Server messages recieved:")
            for log_messsage in session_logs:
                print(" -", log_messsage)
            self.all_session_logs.extend(session_logs)

    def _process_debugging_data(self, debugging_data):
        """ Processes debugging data as it is received by the server. If client has enabled debugging,
        this function calls the debug_data_received_notification() callback. """ 
        if self.debug_enabled:
            global_stability = debugging_data['global_stability_rate']
            global_energy_estimate = debugging_data['global_energy_estimate']
            unmapped_block_stability = debugging_data['block_stability_rates']
            unmapped_block_energy = debugging_data['block_energy_estimates']
            unmapped_block_times = debugging_data['block_stable_times']
            # convert numeric block id to string for easy lookup (debugging data keys are strings from json decode)
            named_block_stability_rates = {
                name: unmapped_block_stability[str(block_id)]
                for name, block_id in self.block_name_map.items()}
            named_block_energy_estimates = {
                name: unmapped_block_energy[str(block_id)]
                for name, block_id in self.block_name_map.items()}
            named_block_stable_times = {
                name: unmapped_block_times[str(block_id)]
                for name, block_id in self.block_name_map.items()}

            debug_data_dict = {
                'global_stability_rate': global_stability,
                'global_energy_estimate': global_energy_estimate,
                'block_stability_rates' : named_block_stability_rates,
                'block_energy_estimates' : named_block_energy_estimates,
                'block_stable_times' : named_block_stable_times
            }
            self.debug_data_history.append(debug_data_dict)
            self.debug_data_received_notification(debug_data_dict)

    def _end_sim(self):
        """ Reports session information and clears out session-specific state """
        def _get_history_by_name(list_of_named_dicts, name):
            if len(list_of_named_dicts) > 0 and name in list_of_named_dicts[0]:
                value_history = [entry_dict[name] for entry_dict in list_of_named_dicts]
                return value_history
            else:
                print("Unable to find any values for name", name)
                return []
        print("-----------------------------------------------------------------------")
        print("Session", self.session_id, "Summary (sim time:", self.sim_t, "updates)")
        for motor_name in self.motor_name_map.keys():
            motor_history = _get_history_by_name(self.motor_value_history, motor_name)
            # note: example motor_history to debug further
            motor_min = np.min(motor_history)
            motor_max = np.max(motor_history)
            motor_median = np.median(motor_history)
            motor_mean = np.mean(motor_history)
            print("- Motor '" + motor_name + "'\tmin:", motor_min, "\tmax:", motor_max, "\tmedian:", motor_median, "\tmean:", motor_mean)
        for sensor_name in self.sensor_name_map.keys():
            sensor_history = _get_history_by_name(self.sensor_value_history, sensor_name)
            # note: example sensor_history to debug further
            sensor_min = np.min(sensor_history)
            sensor_max = np.max(sensor_history)
            sensor_median = np.median(sensor_history)
            sensor_mean = np.mean(sensor_history)
            print("- Sensor '" +  sensor_name + "'\tmin:", sensor_min, "\tmax:", sensor_max, "\tmedian:", sensor_median, "\tmean:", sensor_mean)

        if self.debug_enabled:
            final_state = self.debug_data_history[-1]
            # note: example debug_data_history to debug further
            print("Last debug state received:")
            for key, val in final_state.items():
                print("-", key, ":", val)
        else:
            print("Note: Stability/Energy history values not available unless 'enable_debug' is set to true in client .params settings. ")
        print("-----------------------------------------------------------------------")
        
        # cleanup session state
        self.sim_ended_notification()
        self.sim_t = 0
        self.session_id = None
        self.sensor_name_map = {}
        self.motor_name_map = {}
        self._stop_requested = False
        self.all_session_logs = []
        self.sensor_value_history = []
        self.motor_value_history = []
        self.debug_data_history = []

    def _close_session(self):
        """ Closes the remote ThoughtForge session """ 
        # shut down session
        if self.session_id is not None and self.session_id >= 0:
            shutdownSession_params = {'session_id': self.session_id}
            shutdown_url = self._build_url('/shutdownSession', shutdownSession_params)
            headers = {"X-thoughtforge-key": self.api_key}
            response = requests.post(shutdown_url, headers=headers)
            if response.ok:
                print("Session", self.session_id, "has been shut down.")
                response_dict = response.json()
                session_log = json.loads(safe_dict_get(response_dict, 'session_log', []))
                self._process_session_logs(session_log)
            else:
                print("Session shutdown failed. Server returned", response)
            self._end_sim()

    def get_num_motors(self):
        """ returns the number of motors that have been added to the session model 
        
        :return: The number of Motors.
        :rtype: int
        """
        return len(self.motor_name_map)

    def get_num_sensors(self):
        """ returns the number of sensors that have been added to the session model 
        
        :return: The number of sensors.
        :rtype: int
        """
        return len(self.sensor_name_map)

    def stop_sim(self):
        """ This function requests stopping of the simulation.  
        Client applications can call this to request shutdown of the simulation loop 
        """
        self._stop_requested = True

    def sim_started_notification(self):
        """ This function can optionally be implemented by users to initialize 
        any simulation environment parameters """
        pass

    def sim_ended_notification(self):
        """ This function can optionally be implemented by users to handle end-of-session
        needs or to report on results """
        pass

    def debug_data_received_notification(self, debug_data_dict):
        """ 
        This function is called automatically when debug data is being collected.
        Implement this function in a user client session to perform custom runtime debugging.
        
        .. note:: Debug mode can be enabled in a client params file by setting **"enable_debug": true**
        
        .. note:: Here are examples of things to look at:
        ::

            print("Global stability rate:", round(debug_data_dict['global_stability_rate'], 6))
            print("Global energy estimate", round(debug_data_dict['global_energy_estimate'], 6))
            print("Per-block stability rates:", debug_data_dict['block_stability_rates'])
            print("Per-block energy estimates:", debug_data_dict['block_energy_estimates'])
            print("Per-block stable times:", debug_data_dict['block_stable_times'])

        :param debug_data_dict: A dictionary of global and block-level debug statistics
        :type debug_data_dict: dict
        """ 
        pass

    def update(self, motor_action_dict):
        """ 
        Implement this function in client code to update environment state and return sensor data. 
        
        `update()` is called automatically during simulation and is intended to be the primary
        point of entry for users to define the interaction between the thoughtforge model and
        the specific simulation environment.
        
        :param motor_action_dict: A dictionary of motor names to motor values generated by the model
        :type motor_action_dict: dict
        :return: A dictionary of sensor names to sensor values to send to the model
        :rtype: dict
        """
        raise NotImplementedError