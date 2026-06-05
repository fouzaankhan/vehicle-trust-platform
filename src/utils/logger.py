import logging
import yaml
import os

def setup_logger(name: str) -> logging.Logger:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    log_level = config["logging"]["level"]
    log_format = config["logging"]["format"]

    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(name)
    return logger