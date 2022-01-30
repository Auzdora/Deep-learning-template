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
    return logger_config


def logger_packer(config_file_path):
    """
        By using logger_parser, this function can directly parse log_config.json and generate a list which
    consists of loggers that json file has.
    :param config_file_path:
    :return: exist loggers name cluster
    """
    names = []
    logger_json = logger_parser(config_file_path)
    loggers_cluster = logger_json['loggers']
    for name, value in loggers_cluster.items():
        names.append(name)
    return names


if __name__ == '__main__':
    loggers = logger_packer('log_config.json')
    print(loggers)
    console = logging.getLogger('console_loggers')
    a = 3
    console.info('I am yoursP{}'.format(a))