import os
import json
import yaml
from pathlib import Path

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path