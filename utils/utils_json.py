"""
    File name: utils_json.py
    Description: Basic json utils for further operation

    Author: Botian Lan
"""

import json
import pathlib


def read_json(file_path):
    """
    Read json file function
    :param file_path:
    :return: a dict data
    """
    json_file_path = pathlib.Path(file_path)
    if json_file_path.is_file():
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    else:
        raise FileNotFoundError("Json file '{}' is not exist!".format(file_path))


def write_json(file_path, dic, indent):
    """
    Write dict data to json file in specific path
    :param file_path:
    :param dic:
    :param indent:
    :return: None
    """
    json_str = json.dumps(dic, indent=indent)
    with open(file_path, 'w') as json_file:
        json_file.write(json_str)
