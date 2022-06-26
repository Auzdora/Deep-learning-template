"""
    File name: config_parse.py
    Description: This file parses config.json and return the result outside of file.
                To complete basic initialization.

    Author: Botian Lan
"""
from utils_json import *


class _ConfigParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.json_data = self.reader()

        # Parsers
        self.my_dataset = False
        self.data_config = self.data_parser()
        self.model_config = self.model_parser()
        self.checkpoint_enable = self.checkpoint_parser()

    def reader(self):
        return read_json(self.file_path)

    def data_parser(self):
        """
            Parse data episode in config.json, it has two branches. If data_split is true, that means
        your data has been divided. Else if data_split is false, that means data is original data.

        :return: data_config, a dict data type
        """
        data_config = self.json_data['data']

        if data_config['data_split']:
            data_config = data_config['split_data']
            self.my_dataset = False
            return data_config
        else:
            data_config = data_config['original_data']
            self.my_dataset = True
            return data_config

    def model_parser(self):
        model_config = self.json_data['model_params']
        return model_config

    def checkpoint_parser(self):
        """
        Get json data "checkpoint_enable", logically, when it returns true,
        system will use check point to load last time epoch model information,
        when it returns false, it will train from the beginning.
        :return: true or false
        """
        checkpoint = self.json_data['checkpoint_enable']
        if checkpoint:
            return True
        return False
