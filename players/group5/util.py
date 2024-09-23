import logging
import os


def setup_file_logger(logger: logging.Logger, name: str, log_dir: str="./log"):
    logger.setLevel(logging.DEBUG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    return logger