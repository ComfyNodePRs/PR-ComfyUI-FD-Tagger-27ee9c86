import asyncio
import gc
import os
from typing import Callable, Dict, List, Optional
from aiohttp import web
import aiohttp
import msgspec

from helpers.http import ComfyHTTP
from helpers.config import ComfyExtensionConfig
from helpers.logger import ComfyLogger
from helpers.metaclasses import Singleton


class JtpTagManager(metaclass=Singleton):
    """
    The JTP Tag Manager class is a singleton class that manages the loading, unloading, downloading, and installation of JTP Vision Transformer tags.
    """
    def __init__(self, tags_basepath: str, download_progress_callback: Callable[[int, int], None], download_complete_callback: Optional[Callable[[str], None]] = None) -> None:
        self.tags_basepath = tags_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        self.data = {}

    def __del__(self) -> None:
        for tags_name in self.data.keys():
            if self.is_loaded(tags_name):
                _ = self.download(tags_name)
        self.data.clear()
        del self.data()
        gc.collect()
    
    @classmethod
    def is_loaded(cls, tags_name: str) -> bool:
        return tags_name in cls().data.keys() and cls().data[tags_name]["tags"] is not None
    
    @classmethod
    def load(cls, tags_name: str) -> bool:
        tags_path = os.path.join(cls().tags_basepath, f"{tags_name}.json")
        if cls().is_loaded(tags_name):
            ComfyLogger().log(f"Tags for model {tags_name} already loaded", "WARNING", True)
            return True
        if not os.path.exists(tags_path):
            ComfyLogger.log(f"Tags for model {tags_name} not found in path: {tags_path}", "ERROR", True)
            return False
        
        with open(tags_path, "r") as file:
            cls().data[tags_name]["tags"] = msgspec.json.decode(file.read(), type=Dict[str, float], strict=False)
        return True
    
    @classmethod
    def unload(cls, tags_name: str) -> bool:
        if not cls().is_loaded(tags_name):
            ComfyLogger().log(f"Tags for model {tags_name} not loaded, nothing to do here", "WARNING", True)
            return True
        del cls().data[tags_name]["tags"]
        cls().data[tags_name]["tags"] = None
        gc.collect()
        return True

    @classmethod
    def is_installed(cls, tags_name: str) -> bool:
        """
        Check if a tags file is installed in a directory
        """
        return any(tags_name + ".json" in s for s in cls().list_installed())
    
    @classmethod
    def list_installed(cls) -> List[str]:
        """
        Get a list of installed tags files in a directory
        """
        tags_path = os.path.abspath(cls().tags_basepath)
        if not os.path.exists(tags_path):
            ComfyLogger().log(f"Tags path {tags_path} does not exist, it is being created", "WARN", True)
            os.makedirs(os.path.abspath(tags_path))
            return []
        tags = list(filter(
            lambda x: x.endswith(".json"), os.listdir(tags_path)))
        return tags
    
    @classmethod
    async def download(cls, tags_name: str) -> web.Response:
        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config["huggingface_endpoint"]
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        if hf_endpoint.endswith("/"):
            hf_endpoint = hf_endpoint.rstrip("/")
        
        tags_path = os.path.join(cls().tags_basepath, f"{tags_name}.json")
        
        url: str = config["models"][tags_name]["url"]
        url = url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not url.endswith("/"):
            url += "/"
        
        ComfyLogger().log(f"Downloading tags {tags_name} from {url}", "INFO", True)
        async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
            try:
                await ComfyHTTP().download_to_file(f"{url}{tags_name}.json", tags_path, cls().download_progress_callback, session=session)
            except aiohttp.client_exceptions.ClientConnectorError as err:
                ComfyLogger().log("Unable to download tags. Download files manually or try using a HF mirror/proxy in your config.json", "ERROR", True)
                raise
            cls().download_complete_callback(tags_name)
        return web.Response(status=200)