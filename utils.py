"""Utility functions, not matching elsewhere."""
import yaml
from loguru import logger

def init_logger(config_path="loguru.yml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.configure(**cfg)

# call once at startup
init_logger()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
