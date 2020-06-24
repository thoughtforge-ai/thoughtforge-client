import gym
import math, os, pickle
import numpy as np
from dotenv import load_dotenv

from thoughtforge_client import BaseThoughtForgeClientSession
from advanced.reacher_3joint.example_reacher3_client import ExampleReacher3Session


if __name__ == "__main__": 
    host = '0.0.0.0' if not 'HOST' in os.environ else os.environ['HOST']
    port = 4343 if not 'PORT' in os.environ else int(os.environ['PORT'])
    saved_network_directory = "./advanced/reacher_3joint/"
    filename = INSERT FILE NAME HERE
    file_location = os.path.join(saved_network_directory, filename)
    model_data = None
    with open(file_location, 'rb') as out_file:
        model_data = pickle.load(out_file)

    load_dotenv()
    api_key = os.getenv("THOUGHTFORGE_API_KEY")
    session = ExampleReacher3Session('./advanced/reacher_3joint/example_reacher3.params', host, port, api_key, model_data=model_data)
