import json
import os
from collections import namedtuple
from typing import List


CURRENT_CLIENT_PARAMS_VERSION = 0


def safe_dict_get(dict, keyName, default):
    """ Utility function for accessing dictionary values """
    return dict[keyName] if keyName in dict.keys() else default

def _load_enforced_type(file_name, enforced_type, use_named_tuple=False):
    print('load', file_name, enforced_type)
    def mapJSONToObject(dict):
        return namedtuple('X', dict.keys())(*dict.values())
    config = {}
    file_extension = os.path.splitext(file_name)[1]
    if(file_extension == enforced_type):
      with open(file_name) as f:
          if(use_named_tuple):
              config = json.load(f, object_hook=mapJSONToObject)
          else:
              config = json.load(f) #, object_hook=mapJSONToObject
    else:
      raise 'config files must be of type' + enforced_type + ' got ' + file_name
    return config  

def load_client_params(file_name):
    """ return json client configuration from the given filename """
    return _load_enforced_type(file_name, '.params')
