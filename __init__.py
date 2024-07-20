import os
from .helpers.extension import ComfyExtension
from .helpers.nodes import ComfyNode
from .helpers.logger import ComfyLogger

if not ComfyExtension().init(check_imports=["torch", "torchvision", "PIL", "aiohttp", "requests", "numpy", "colorama"]):
    raise ImportError("ComfyExtension failed to initialize")

vars = ComfyNode().get_node_vars()
for name, obj in vars.items():
    globals()[name] = obj
__all__ = list(vars.keys())