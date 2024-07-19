import os
import shutil
import inspect
from typing import List, Optional
from server import PromptServer

from .metaclasses import Singleton


class ComfyExtension(metaclass=Singleton):
    """
    A singleton class to provide utility functions for a comfy extension.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def name(cls) -> str:
        """
        Get the name of the extension
        """
        from config import ComfyExtensionConfig
        return str(ComfyExtensionConfig().get()["name"]).lower().replace(" ", "_").replace('/', ".")
    
    @classmethod
    def extension_dir(cls, subpath: Optional[str] = None, mkdir: bool = False) -> str:
        """
        Get the directory the extension is installed in
        """
        dir = os.path.dirname(__file__).partition("helpers")[0]
        if subpath is not None:
            dir = os.path.join(dir, subpath)
        dir = os.path.abspath(dir)
        if mkdir and not os.path.exists(dir):
            from .logger import ComfyLogger
            ComfyLogger().log(f"Directory {dir} does not exist, it is being created", type="WARNING", always=True)
            os.makedirs(dir)
        return dir 

    @classmethod
    def comfy_dir(cls, subpath: Optional[str] = None) -> str:
        """
        Get the directory the extension is installed in
        """
        dir = os.path.dirname(inspect.getfile(PromptServer))
        if subpath is not None:
            dir = os.path.join(dir, subpath)
        dir = os.path.abspath(dir)
        return dir
    
    @classmethod
    def web_extension_dir(cls) -> str:
        """
        Get the web extension directory
        """
        from .config import ComfyExtensionConfig
        from .logger import ComfyLogger
        names = ComfyExtensionConfig().get()["name"].split('.')
        dir = os.path.join(cls.comfy_dir(), "web", "extensions", names[0])
        if not os.path.exists(dir):
            ComfyLogger().log(f"Web extension directory {dir} does not exist, it is being created", type="WARNING", always=True)
            os.makedirs(dir)
        dir += os.sep + (os.sep.join(names[1:]))
        return dir
    
    @classmethod
    def client_id(cls) -> str:
        """
        Get the client id of the extension
        """
        return 

    @classmethod
    def install_js(cls) -> None:
        from .logger import ComfyLogger
        from .files import ComfyFiles
        src_dir = cls().extension_dir(f"web{os.sep}js")
        if not os.path.exists(src_dir):
            ComfyLogger().log(f"js installation skipped, source directory {src_dir} does not exist", type="WARNING", always=True)
            return
        should_install = cls().should_install_js()
        if should_install:
            ComfyLogger().log("It looks like you're running an old version of ComfyUI that requires manual setup of web files, it is recommended you update your installation.", "WARNING", True)
        dst_dir = cls().web_extension_dir()
        linked = ComfyFiles().is_symlink(dst_dir)
        if linked or os.path.exists(dst_dir):
            if linked:
                if should_install:
                    ComfyLogger().log("JS already linked, PromptServer will serve extension", level='INFO')
                else:
                    os.unlink(dst_dir)
                    ComfyLogger().log("JS unlinked, PromptServer will serve extension", level='INFO')
            elif not should_install:
                shutil.rmtree(dst_dir)
                ComfyLogger().log("JS deleted, PromptServer will serve extension", level='INFO')
            return
        if not should_install:
            ComfyLogger().log("JS skipped, PromptServer will serve extension", level='WARNING')
            return
        if ComfyFiles().link_item(src_dir, dst_dir):
            ComfyLogger().log("JS linked, extension will be served by JavaScript", level='INFO')
            return
        ComfyLogger().log("Installing JS files", level='INFO')
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    @classmethod
    def should_install_js(cls) -> bool:
        return not hasattr(PromptServer.instance, "supports") or "custom_nodes_from_web" not in PromptServer.instance.supports

    @classmethod
    def init(cls, check_imports: Optional[List[str]]) -> bool:
        from .logger import ComfyLogger
        ComfyLogger().log("Initializing...", type="DEBUG", always=True)
        if check_imports is not None:
            import importlib.util
            for imp in check_imports:
                spec = importlib.util.find_spec(imp)
                if spec is None:
                    ComfyLogger().log(f"{imp} is required, please ensure that the nessecary requirements are installed.",
                        type="ERROR", always=True)
                    return False
        cls.install_js()
        return True
