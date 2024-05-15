import json
import pathlib
import logging.config
from logging import Logger


def get_logger(logger_name: str, json_formatter_path:str, filter_path:str) -> Logger:

    config_file = pathlib.Path("logger/logging_config.json").resolve()

    with open(config_file) as f_in:
        config = json.load(f_in)
        config['handlers']['file']['filename'] = pathlib.Path("logs/project_logs.log.jsonl").resolve()
        config['formatters']['json']['()'] = json_formatter_path
        config['filters']['no_errors']['()'] = filter_path
    
    logging.config.dictConfig(config)

    return logging.getLogger(logger_name)  

if __name__ == "__main__":
    pass
    # logger = get_logger("my_project")
    # logger.info('testing')

    