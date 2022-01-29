"""
    File name: logger_parse.py
    Description: This file could read json logging file and config log system

    Author: Botian Lan
    Time: 2022/01/29
"""
import logging.config
import logging

from utils_json import *


def logger_parser(config_file_path):
    logger_config = read_json(config_file_path)
    logging.config.dictConfig(logger_config)


if __name__ == '__main__':
    logger_parser('log_config.json')
    logging.debug('I am yours')