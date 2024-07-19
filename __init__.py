import os
from .helpers.extension import ComfyExtension
from .helpers.nodes import ComfyNode

if not ComfyExtension().init(check_imports=["torch", "torchvision", "PIL", "aiohttp", "requests", "numpy"]):
    raise ImportError("ComfyExtension failed to initialize")

vars = ComfyNode().get_node_vars(os.path.dirname(__file__))    
for name, obj in vars.items():
    globals()[name] = obj
__all__ = list(vars.keys())