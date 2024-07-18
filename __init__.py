from .comfynode import init

if init(check_imports=["pytorch", "torchvision", 'PIL']):
    from .fdtagger import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    WEB_DIRECTORY = "./web"
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
