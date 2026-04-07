import os
import yaml

def get_config_path() -> str:
    """Returns the absolute path to the config.yaml file."""
    # current file is in utils/, config is in ../config/config.yaml
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "config", "config.yaml")

def load_config() -> dict:
    """Loads the config.yaml file."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config.yaml: {e}")

def get_assets_root() -> str:
    """Returns the root path for assets."""
    config = load_config()
    path = config.get("paths", {}).get("assets_root", "")
    if not path:
        raise ValueError("assets_root not defined in config.yaml")
    return path

def get_data_collection_root() -> str:
    """Returns the default root path for data collection."""
    config = load_config()
    path = config.get("paths", {}).get("data_collection_root", "~/data_collected_oceansim")
    return os.path.expanduser(path)

def get_scene_path(scene_key: str) -> str:
    """Returns the full path for a specific scene asset.
    
    Args:
        scene_key: The key in config.yaml under paths.scenes (e.g., 'mhl_water', 'robot')
    """
    config = load_config()
    rel_path = config.get("paths", {}).get("scenes", {}).get(scene_key, "")
    if not rel_path:
        raise ValueError(f"Scene path for '{scene_key}' not defined in config.yaml")
        
    return os.path.join(get_assets_root(), rel_path)

def get_waypoints_default_path() -> str:
    """Returns the full path to the default waypoints file."""
    config = load_config()
    # Logic: if path is absolute, return as is. If relative, assume relative to extension root.
    # Extension root is parent of 'isaacsim' package folder in this structure logic?
    # Let's inspect where this file is relative to extension root.
    # File: .../isaacsim/oceansim/utils/assets_utils.py
    # Extension Root (setup.py location usually): .../
    # Actually, let's keep it simple. If it looks relative, make it relative to the extension root.
    # How to find extension root reliably? 
    # In `ui_builder.py` they use `get_extension_path(self._ext_id)`.
    # Here we don't have ext_id without context.
    # But we know the file structure.
    # .../OceanSim-NTNU/isaacsim/oceansim/utils/assets_utils.py
    # .../OceanSim-NTNU/ is the extension root.
    
    rel_path = config.get("paths", {}).get("waypoints_default", "")
    if os.path.isabs(rel_path):
        return rel_path
        
    return os.path.join(_get_extension_root(), rel_path)

def get_map_config_path() -> str:
    """Returns the full path to the map.yaml file."""
    config = load_config()
    rel_path = config.get("paths", {}).get("configs", {}).get("map", "")
    if not rel_path:
        # Fallback if not defined
        return os.path.join(_get_extension_root(), "isaacsim/oceansim/config/map.yaml")
        
    if os.path.isabs(rel_path):
        return rel_path
    
    return os.path.join(_get_extension_root(), rel_path)

def get_imu_config_path() -> str:
    """Returns the full path to the imu_config.yaml file."""
    config = load_config()
    rel_path = config.get("paths", {}).get("configs", {}).get("imu", "")
    if not rel_path:
         # Fallback
         return os.path.join(_get_extension_root(), "isaacsim/oceansim/config/imu_config.yaml")

    if os.path.isabs(rel_path):
        return rel_path

    return os.path.join(_get_extension_root(), rel_path)

def _get_extension_root() -> str:
    """Returns the extension root directory."""
    # utils -> oceansim -> isaacsim -> OceanSim-NTNU
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../../"))

# Backwards compatibility alias if needed, but we should refactor usages.
def get_oceansim_assets_path() -> str:
    return get_assets_root()

def get_map_image_filename() -> str:
    """Returns the filename for the map image (e.g. map.png)."""
    config = load_config()
    return config.get("filenames", {}).get("map_image", "map.png")

def get_map_yaml_filename() -> str:
    """Returns the filename for the map yaml (e.g. map.yaml)."""
    config = load_config()
    return config.get("filenames", {}).get("map_yaml", "map.yaml")

def get_imu_metadata_filename() -> str:
    """Returns the filename for the IMU metadata (e.g. imu_metadata.yaml)."""
    config = load_config()
    return config.get("filenames", {}).get("imu_metadata", "imu_metadata.yaml")

def get_dynamics_config_dir() -> str:
    """Returns the absolute path to the dynamics config directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "config", "dynamics")

def list_dynamics_configs() -> list:
    """Returns a list of available dynamics config names (without .yaml extension), sorted."""
    config_dir = get_dynamics_config_dir()
    if not os.path.isdir(config_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(config_dir)
        if f.endswith('.yaml')
    )

def get_dynamics_config_path(name: str) -> str:
    """Returns the full path to a dynamics config YAML by name (without extension)."""
    path = os.path.join(get_dynamics_config_dir(), f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dynamics config not found: {path}")
    return path

def load_dynamics_config(name: str) -> dict:
    """Loads a dynamics config YAML by name and returns the parsed dict."""
    path = get_dynamics_config_path(name)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("Config Path:", get_config_path())
    print("Assets Root:", get_assets_root())
    try:
        print("Robot Path:", get_scene_path("robot"))
    except Exception as e:
        print(e)
