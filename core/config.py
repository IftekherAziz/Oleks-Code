import yaml

def load_experiment_configs(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)