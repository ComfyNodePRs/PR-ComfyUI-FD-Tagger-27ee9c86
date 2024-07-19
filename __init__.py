import os
from .helpers.extension import ComfyExtension
from .helpers.nodes import ComfyNode

if init(check_imports=["torch", "torchvision", "PIL", "aiohttp", "requests", "numpy"]):
    from .fdtagger import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    WEB_DIRECTORY = "./web"
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
