
import json
import os
from typing import Any, Dict

from .metaclasses import Singleton


class ComfyExtensionConfig(metaclass=Singleton):
    """
    A simple class to load the comfy extension config from config.json or config.user.json into
    memory and provide a method to access it.
    """
    def __init__(self) -> None:
        self.config = None
    
    @classmethod
    def get(cls, reload: bool = False) -> Dict[str, Any]:
        from .logger import ComfyLogger
        if not reload and cls().config is not None:
            return cls().config
        from .extension import ComfyExtension
        config_path = ComfyExtension().extension_dir("config.user.json")
        if not os.path.exists(config_path):
            config_path = ComfyExtension().extension_dir("config.json")
        if not os.path.exists(config_path):
            ComfyLogger().log("Missing config.json and config.user.json, this extension may not work correctly. Please reinstall the extension.", type="ERROR", always=True)
            print(f"Extension path: {ComfyExtension().extension_dir()}")
            return {"name": "Unknown", "version": -1}
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        return config