import enum
import torch
import numpy as np
import os
from PIL import Image
from aiohttp import web
from typing import List, Union, Tuple, Dict, Any

from ....comfy import utils
from ....server import PromptServer

from ..helpers.config import ComfyExtensionConfig
from ..helpers.extension import ComfyExtension
from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager

class ModelDevice(enum):
    CPU = "cpu"
    GPU = "cuda"

    def cast_to_device(self) -> torch.device:
        return torch.device(self.value)

    def __all__(self) -> List[str]:
        return [self.CPU, self.GPU]


async def classify_tags(image: np.ndarray, model_name: str, device: torch.device, steps: float = 0.35, threshold: float = 0.35, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False) -> Tuple[str, Dict[str, float]]:
    """
    Classify e621 tags for an image using RedRocket JTP Vision Transformer model
    """
    from redrocket.classifier import JtpInference
    version: int = ComfyExtensionConfig().get()["models"][model_name]["version"]
    tag_string, tag_scores = JtpInference(device=device).run_classifier(model_name=model_name, device=device, version=version, image=image, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
    return tag_string, tag_scores


async def download_progress_callback(perc: int, file_name: str, client_id: Union[str, None], node: Union[str, None], api_endpoint: Union[str, None]) -> None:
    """
    Callback function for download progress updates
    """
    from ..helpers.nodes import ComfyNode
    if client_id is None:
        client_id = ComfyExtension().client_id()
    if client_id is None:
        raise ValueError("Client ID is not set")
    if api_endpoint is None:
        api_endpoint = ComfyExtension().api_endpoint()
    if api_endpoint is None:
        raise ValueError("API endpoint is not set")
    message: str = ""
    if perc < 100:
        message = "[{0:d1}%] Downloading {1}...".format(perc, file_name)
    else:
        message = "Download {0} complete!".format(file_name)
    ComfyNode().update_node_status(client_id=client_id, node=node, api_endpoint=api_endpoint, text=message, progress=perc)


async def download_complete_callback(file_name: str, client_id: Union[str, None], node: Union[str, None], api_endpoint: Union[str, None]) -> None:
    """
    Callback function for download completion updates
    """
    from ..helpers.nodes import ComfyNode
    if client_id is None:
        client_id = ComfyExtension().client_id()
    if client_id is None:
        raise ValueError("Client ID is not set")
    ComfyNode().update_node_status(client_id=client_id, node=node, api_endpoint=api_endpoint)


@PromptServer.instance.routes.get("/furrydiffusion/fdtagger/tag")
async def get_tags(request: web.Request) -> web.Response:
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)
    type: str = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)
    target_dir: str = ComfyExtension().comfy_dir(type)
    image_path: str = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)
    if not os.path.isfile(image_path):
        return web.Response(status=404)
    image: np.ndarray = np.array(Image.open(image_path).convert("RGBA"))
    models: List[str] = JtpModelManager().list_installed()
    default: str = ComfyExtensionConfig().get()["settings"]["model"]
    model: str = default if default in models else models[0]
    steps: int = int(request.query.get("steps", ComfyExtensionConfig().get()["settings"]["steps"]))
    threshold: float = float(request.query.get("threshold", ComfyExtensionConfig().get()["settings"]["threshold"]))
    exclude_tags: str = request.query.get("exclude_tags", ComfyExtensionConfig().get()["settings"]["exclude_tags"])
    replace_underscore: bool = request.query.get("replace_underscore", ComfyExtensionConfig().get()["settings"]["replace_underscore"]) == "true"
    trailing_comma: bool = request.query.get("trailing_comma", ComfyExtensionConfig().get()["settings"]["trailing_comma"]) == "true"
    device: ModelDevice = ModelDevice(request.query.get("device", "cpu"))
    client_id: str = request.rel_url.query.get("clientId", None)
    node: str = request.rel_url.query.get("node", None)
    return web.json_response(await classify_tags(image=image, model_name=model, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma, client_id=client_id, node=node))


class FDTagger():
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        models: List[str] = [o["name"] for o in ComfyExtensionConfig().get()["models"].keys()]
        return {"required": {
            "image": ("IMAGE", ),
            "model": (models, {"default": ComfyExtensionConfig().get()["settings"]["model"]}),
            "steps": ("INTEGER", {"default": ComfyExtensionConfig().get()["settings"]["steps"], "min": 1, "max": 500}),
            "threshold": ("FLOAT", {"default": ComfyExtensionConfig().get()["settings"]["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": ComfyExtensionConfig().get()["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": ComfyExtensionConfig().get()["settings"]["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": ComfyExtensionConfig().get()["settings"]["exclude_tags"]}),
            "device": ("STRING", {"default": "cpu", "options": ["cpu", "cuda"]})
        }}

    RETURN_TYPES: Tuple[str] = ("STRING",)
    OUTPUT_IS_LIST: Tuple[bool] = (True,)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "image"

    def tag(self, image: np.ndarray, model: str, steps: int, threshold: float, exclude_tags: str = "", replace_underscore: bool = False, trailing_comma: bool = False) -> Dict[str, Any]:
        from ..helpers.multithreading import ComfyThreading
        tensor = np.array(image * steps, dtype=np.uint8)
        pbar = utils.ProgressBar(tensor.shape[0])
        tags: List[str] = []
        for i in range(tensor.shape[0]):
            tags.append(ComfyThreading().wait_for_async(lambda: classify_tags(image=tensor[i], model_name=model, threshold=threshold,
                        exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}

JtpModelManager(model_basepath=ComfyExtension().extension_dir("models", mkdir=True), download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpTagManager(tags_basepath=ComfyExtension().extension_dir("tags", mkdir=True), download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "FD_FDTagger|fdtagger": FDTagger,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "FD_FDTagger|fdtagger": "FurryDiffusion Tagger 🐺",
}
