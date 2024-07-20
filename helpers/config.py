
import json
import os
from typing import Any, Dict, Optional, Union

from .metaclasses import Singleton


class ComfyExtensionConfig(metaclass=Singleton):
    """
    A simple class to load the comfy extension config from config.json or config.user.json into
    memory and provide a method to access it.
    """
    def __init__(self) -> None:
        self.config = None
    
    @classmethod
    def get_model_from_name(cls, model_name: str) -> Union[str, None]:
        """
        Get the real model name from the display name
        """
        from .logger import ComfyLogger
        models = [k for k, v in cls().get(property=f"models").items() if model_name.lower() == v["name"].lower()]
        if len(models) == 0:
            ComfyLogger().log(f"Model entry: {model_name} not found in config", type="WARNING", always=True)
            return None
        return models[0]
    
    @classmethod
    def get_tags_from_name(cls, tag_name: str) -> Union[str, None]:
        """
        Get the real tag name from the display name
        """
        from .logger import ComfyLogger
        tags = [k for k, v in cls().get(property=f"tags").items() if tag_name.lower() == v["name"].lower()]
        if len(tags) == 0:
            ComfyLogger().log(f"Tag entry: {tag_name} not found in config", type="WARNING", always=True)
            return None
        return tags[0]
    
    @classmethod
    def get(cls, reload: bool = False, property: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the extension config from the config.json or config.user.json file
        
        Args:
            reload (bool): Reload the config file
            property (Optional[str]): A property to get from the config file
        """
        from .logger import ComfyLogger
        if not reload and cls().config is not None:
            config = cls().config
        else:
            from .extension import ComfyExtension
            config_path = ComfyExtension().extension_dir("config.user.json")
            if not os.path.exists(config_path):
                config_path = ComfyExtension().extension_dir("config.json")
            if not os.path.exists(config_path):
                ComfyLogger().log("Missing config.json and config.user.json, this extension may not work correctly. Please reinstall the extension.", type="ERROR", always=True)
                print(f"Extension path: {ComfyExtension().extension_dir()}")
                return {"name": "Unknown", "version": -1}
            config = {}
            with open(config_path, "r") as f:
                config = json.loads(f.read())
            cls().config = config
        if property is not None:
            if "." not in property:
                return config[property]
            for prop in property.split("."):
                if isinstance(config, dict) and prop in config:
                    config = config[prop]
                elif isinstance(config, list) and prop.isdigit():
                    config = config[int(prop)]
                elif isinstance(config, str):
                    return config
                elif isinstance(config, object):
                    try:
                        config = getattr(config, prop)
                    except AttributeError:
                        ComfyLogger().log(f"Failed to cast {prop} in config", type="WARNING", always=True)
                        return {"name": "Unknown", "version": -1}
                else:
                    ComfyLogger().log(f"Property {prop} not found in config", type="WARNING", always=True)
                    return {"name": "Unknown", "version": -1}
            return config
        return config