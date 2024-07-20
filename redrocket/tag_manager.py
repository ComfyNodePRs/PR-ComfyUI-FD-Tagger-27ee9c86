import asyncio
import gc
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
from aiohttp import web
import aiohttp
import msgspec
import torch

from ..helpers.metaclasses import Singleton


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
        del self.data
        gc.collect()
    
    @classmethod
    def is_loaded(cls, tags_name: str) -> bool:
        """
        Check if tags are loaded into memory
        """
        return tags_name in cls().data.keys() and cls().data[tags_name]["tags"] is not None
    
    @classmethod
    def load(cls, tags_name: str) -> bool:
        """
        Mount the tags for a model into memory
        """
        from ..helpers.logger import ComfyLogger
        
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
        """
        Unmount the tags for a model from memory
        """
        from ..helpers.logger import ComfyLogger
        
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
        from ..helpers.logger import ComfyLogger
        
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
        """
        Load tags for a model from a URL and save them to a file.
        """
        from ..helpers.http import ComfyHTTP
        from ..helpers.config import ComfyExtensionConfig
        from ..helpers.logger import ComfyLogger
        
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

    @classmethod
    def process_tags(cls, model_name: str, indices: Union[torch.Tensor, None], values: Union[torch.Tensor, None], exclude_tags: str, replace_underscore: bool, threshold: float, trailing_comma: bool) -> Tuple[str, Dict[str, float]]:
        """
        Process the tags for a model based on the indices and values from the model output
        """
        corrected_excluded_tags = [tag.replace("_", " ").strip() for tag in exclude_tags.split(",") if not tag.isspace()]
        tag_score = {cls().data[model_name]["tags"][indices[i]]: values[i].item() for i in range(indices.size(0)) if cls().data[model_name]["tags"][indices[i]] not in corrected_excluded_tags}
        if not replace_underscore:
            tag_score = {key.replace(" ", "_"): value for key, value in tag_score.items()}
        tag_score = dict(sorted(tag_score.items(), key=lambda item: item[1], reverse=True))
        tag_score = {key: value for key, value in tag_score.items() if value > threshold}
        return ", ".join(tag_score.keys()) + ("," if trailing_comma else ""), tag_score