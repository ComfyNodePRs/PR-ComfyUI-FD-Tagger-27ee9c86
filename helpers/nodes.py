from importlib.util import spec_from_file_location, module_from_spec, spec_from_loader
import inspect
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
from server import PromptServer

from .metaclasses import Singleton


class ComfyNode(metaclass=Singleton):
    """
    A singleton class to provide additional node utility functions for a comfy extension.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def update_node_status(cls, client_id: Optional[str], node: str, text: str, progress: Optional[float] = None) -> None:
        if client_id is None:
            client_id = PromptServer.instance.client_id
        if client_id is None:
            return
        PromptServer.instance.send_sync("furrydiffusion/update_status", {
            "node": node,
            "progress": progress,
            "text": text
        }, client_id)

    @classmethod
    async def update_node_status_async(cls, client_id: Optional[str], node: str, text: str, progress: Optional[float] = None) -> None:
        if client_id is None:
            client_id = PromptServer.instance.client_id
        if client_id is None:
            return
        await PromptServer.instance.send("furrydiffusion/update_status", {
            "node": node,
            "progress": progress,
            "text": text
		}, client_id)
    
    @classmethod
    def get_module_vars(cls, module_path):
        module_dir, module_file = os.path.split(module_path)
        module_name, _ = os.path.splitext(module_file)
        abs_module_dir = os.path.abspath(module_dir)
        sys.path.insert(0, abs_module_dir)
        
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        package_parts = module_dir.split(os.sep)
        module.__package__ = '.'.join(package_parts[-2:])

        try:
            spec.loader.exec_module(module)
            module_vars = {name: value for name, value in vars(module).items() if not name.startswith('__') and not inspect.ismodule(value) and not inspect.isclass(value) and not inspect.isfunction(value)}
        finally:
            del sys.modules[module_name]
            sys.path = [p for p in sys.path if not p.startswith(abs_module_dir)]
        return module_name, module_vars

    @classmethod
    def get_node_vars(cls) -> Dict[str, Any]:
        from .extension import ComfyExtension
        from .logger import ComfyLogger
        source_path = ComfyExtension().extension_dir("nodes", mkdir=False)
        vars = {}
        for file in os.listdir(source_path):
            if file.endswith(".py") and not file.startswith("__"):
                module_name, module_vars = cls.get_module_vars(os.path.join(source_path, file))
                for name, obj in module_vars.items():
                    if name.isupper() and name.isidentifier():
                        if isinstance(obj, str):
                            ComfyLogger().log(f"Loaded str({name})", type="DEBUG", always=True)
                            if name not in vars:
                                vars.update({name: obj})
                            else:
                                vars.update({name: [o for o in obj if obj not in vars[name]]})
                        elif isinstance(obj, dict):
                            ComfyLogger().log(f"Loaded dict({name}) in {module_name}", type="DEBUG", always=True)
                            for key, value in obj.items():
                                if name not in vars:
                                    vars.update({name: {key: value}})
                                else:
                                    vars[name].update({key: value})	
        return vars
