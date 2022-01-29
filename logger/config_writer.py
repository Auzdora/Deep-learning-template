from utils_json import *

data = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'train_formatter': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        },

        'model_info_formatter': {
            'class': 'logging.Formatter',
            'format': '%(levelname)s: %(message)s'
        }
    },

    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'train_formatter'
        },

        'log_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'train_formatter'
        }
    },

    'root': {
        'level': 'INFO',
        'handlers': ['console', 'log_file']
    }
}

if __name__ == '__main__':
    data = write_json('log_config.json', data, indent=4)