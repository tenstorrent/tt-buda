import yaml

def read_netlist_value(file_path, key_path):
    """
    Reads a netlist value from a YAML file based on the given key path.

    Args:
        file_path (str): The path to the YAML file.
        key_path (str): The key path to the desired value in the YAML file.
                        Keys are separated by slashes ("/").

    Returns:
        The value corresponding to the given key path in the YAML file, or None if the key path is not found.
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    key_path = [element for element in key_path.split("/") if element]
    for key in key_path:
        if isinstance(data, dict):
            if key not in data:
                raise KeyError(f"Key '{key}' not found in the YAML file.")
        elif isinstance(data, list):
            try:
                key = int(key)
            except ValueError:
                raise KeyError(f"Key '{key}' is not an integer.")
            if key >= len(data):
                raise IndexError(f"Index '{key}' out of range.")
        data = data[key]
        if data is None:
            return None
    return data
