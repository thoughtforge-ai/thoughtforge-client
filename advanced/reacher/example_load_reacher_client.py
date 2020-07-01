import gym
import math, os, pickle
import numpy as np

from thoughtforge_client import BaseThoughtForgeClientSession
from advanced.reacher.example_reacher_client import ExampleReacherSession


if __name__ == "__main__": 
    host = '0.0.0.0' if not 'HOST' in os.environ else os.environ['HOST']
    port = 4343 if not 'PORT' in os.environ else int(os.environ['PORT'])
    saved_network_directory = "./advanced/reacher/"
    filename = INSERT FILE NAME HERE
    file_location = os.path.join(saved_network_directory, filename)
    model_data = None
    with open(file_location, 'rb') as out_file:
        model_data = pickle.load(out_file)

    session = ExampleReacherSession('./advanced/reacher/example_reacher.params', model_data=model_data)
