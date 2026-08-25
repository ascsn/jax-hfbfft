import yaml

def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"The file '{file_path}' does not contain valid YAML data.")

    return config
