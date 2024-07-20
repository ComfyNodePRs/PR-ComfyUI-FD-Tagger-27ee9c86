import asyncio
import gc
import os
from typing import Callable, List, Optional
from aiohttp import web
import aiohttp
import torch
import timm
import safetensors.torch

from ..helpers.http import ComfyHTTP
from ..helpers.config import ComfyExtensionConfig
from ..helpers.logger import ComfyLogger
from ..helpers.metaclasses import Singleton


class V2GatedHead(torch.nn.Module):
    def __init__(self,
        num_features: int,
        num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes * 2)
        self.act = torch.nn.Sigmoid()
        self.gate = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.act(x[:, :self.num_classes]) * self.gate(x[:, self.num_classes:])
        return x


class JtpModelManager(metaclass=Singleton):
    """
    The RedRocket JTP Model Manager class is a singleton class that manages the loading, unloading, downloading, and installation of JTP Vision Transformer models.
    """
    def __init__(self, model_basepath: str, download_progress_callback: Callable[[int, int], None], download_complete_callback: Optional[Callable[[str], None]] = None) -> None:
        self.model_basepath = model_basepath
        self.download_progress_callback = download_progress_callback
        self.download_complete_callback = download_complete_callback
        self.data = {}

    def __del__(self) -> None:
        for model_name in self.data.keys():
            if self.is_loaded(model_name):
                _ = self.unload(model_name)
        self.data.clear()
        del self.data
        gc.collect()
    
    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        """
        Check if a RedRocket JTP Vision Transformer model is loaded into memory
        """
        return model_name in cls().data.keys() and cls().data[model_name]["model"] is not None
    
    @classmethod
    def load(cls, model_name: str, version: int = 1, device: torch.device = torch.cpu) -> bool:
        """
        Load a RedRocket JTP Vision Transformer model into memory
        """
        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")
        if cls().is_loaded(model_name):
            ComfyLogger().log(f"Model {model_name} already loaded", "WARNING", True)
            return True
        if not os.path.exists(model_path):
            ComfyLogger().log(f"Model {model_name} not found in path: {model_path}", "ERROR", True)
            return False
        
        ComfyLogger().log(f"Loading model {model_name} (version: {version} from {model_path}...", "INFO", True)
        cls().data[model_name]["model"] = timm.create_model("vit_so400m_patch14_siglip_384.webli", pretrained=False, num_classes=9083)
        if f"{version}" == "2":
            cls().data[model_name]["model"].head = V2GatedHead(min(cls().data[model_name]["model"].head.weight.shape), 9083)
        safetensors.torch.load_model(model=cls().data[model_name]["model"], filename=model_path)
        if torch.cuda.is_available() is True and device.type == "cuda":
            cls().data[model_name]["model"].cuda()
            cls().data[model_name]["model"].to(dtype=torch.float16, memory_format=torch.channels_last)
        cls().data[model_name]["model"].eval()
        ComfyLogger().log(f"Model {model_name} loaded successfully", "INFO", True)
        return True

    @classmethod
    def switch_device(cls, model_name: str, device: torch.device) -> bool:
        """
        Switch the device of a RedRocket JTP Vision Transformer model
        """
        if not cls().is_loaded(model_name):
            ComfyLogger().log(f"Model {model_name} not loaded, nothing to do here", "WARNING", True)
            return False
        if device.type == "cuda" and torch.cuda.is_available() is False:
            ComfyLogger().log("CUDA is not available, cannot switch to GPU", "ERROR", True)
            return False
        
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
            cls().data[model_name]["model"].cuda()
            cls().data[model_name]["model"] = cls().data[model_name]["model"].to(dtype=torch.float16, memory_format=torch.channels_last)
            ComfyLogger().log("Switched to GPU with mixed precision", "INFO", True)
        elif device.type == "cuda" and torch.cuda.get_device_capability()[0] < 7:
            cls().data[model_name]["model"].cuda()
            cls().data[model_name]["model"].to(device)
            ComfyLogger().log("Switched to GPU without mixed precision", "WARNING", True)
        else:
            cls().data[model_name]["model"].cpu()
            cls().data[model_name]["model"].to(device)
            ComfyLogger().log("Switched to CPU", "INFO", True)
        return True
    
    @classmethod
    def unload(cls, model_name: str) -> bool:
        """
        Unload a RedRocket JTP Vision Transformer model from memory
        """
        if not cls().is_loaded(model_name):
            ComfyLogger().log(f"Model {model_name} not loaded, nothing to do here", "WARNING", True)
            return True
        del cls().data[model_name]["model"]
        cls().data[model_name]["model"] = None
        gc.collect()
        if torch.cuda.is_available() is True:
            torch.cuda.empty_cache()
        return True

    @classmethod
    def is_installed(cls, model_name: str) -> bool:
        """
        Check if a vision transformer model is installed in a directory
        """
        return any(model_name + ".safetensors" in s for s in cls().list_installed())
    
    @classmethod
    def list_installed(cls) -> List[str]:
        """
        Get a list of installed vision transformer models in a directory
        """
        model_path = os.path.abspath(cls().model_basepath)
        if not os.path.exists(model_path):
            ComfyLogger().log(f"Model path {model_path} does not exist, it is being created", "WARN", True)
            os.makedirs(os.path.abspath(model_path))
            return []
        models = list(filter(
            lambda x: x.endswith(".safetensors"), os.listdir(model_path)))
        return models
    
    @classmethod
    async def download(cls, model_name: str) -> web.Response:
        """
        Download a RedRocket JTP Vision Transformer model from a URL
        """
        config = ComfyExtensionConfig().get()
        hf_endpoint: str = config["huggingface_endpoint"]
        if not hf_endpoint.startswith("https://"):
            hf_endpoint = f"https://{hf_endpoint}"
        if hf_endpoint.endswith("/"):
            hf_endpoint = hf_endpoint.rstrip("/")
        
        model_path = os.path.join(cls().model_basepath, f"{model_name}.safetensors")
        
        url: str = config["models"][model_name]["url"]
        url = url.replace("{HF_ENDPOINT}", hf_endpoint)
        if not url.endswith("/"):
            url += "/"
        
        ComfyLogger().log(f"Downloading model {model_name} from {url}", "INFO", True)
        async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
            try:
                await ComfyHTTP().download_to_file(f"{url}{model_name}.safetensors", model_path, cls().download_progress_callback, session=session)
            except aiohttp.client_exceptions.ClientConnectorError as err:
                ComfyLogger().log("Unable to download model. Download files manually or try using a HF mirror/proxy in your config.json", "ERROR", True)
                raise
            cls().download_complete_callback(model_name)
        return web.Response(status=200)