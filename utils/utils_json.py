"""
    File name: utils_json.py
    Description: Basic json utils for further operation

    Author: Botian Lan
    Time: 2022/01/28
"""

import json


def read_json(file_path):
    json_file = open(file_path)
    return json.load(json_file)


def write_json(file_path, dic, indent):
    json_str = json.dumps(dic, indent=indent)
    with open(file_path, 'w') as json_file:
        json_file.write(json_str)
