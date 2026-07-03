import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path='config/logging.yaml'):
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

logger = setup_logging()