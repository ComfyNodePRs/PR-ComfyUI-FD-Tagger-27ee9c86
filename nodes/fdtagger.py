import torch
import numpy as np
import os
from PIL import Image
from aiohttp import web
from typing import List, Union, Tuple, Dict, Any

import comfy.utils
from server import PromptServer


from ..helpers.config import ComfyExtensionConfig
config = ComfyExtensionConfig().get()
defaults: Dict[str, Any] = {
    "model": "JTP_PILOT-e4-vit_so400m_patch14_siglip_384",
    "threshold": 0.35,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "huggingface_endpoint": "https://huggingface.co",
}
defaults.update(config.get("settings", {}))

from ..helpers.extension import ComfyExtension
from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager

async def classify_tags(image: np.ndarray, model_name: str, threshold: float = 0.35, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False, client_id: Union[str, None] = None, node: Union[str, None] = None) -> str:
    from redrocket.classifier import JtpInference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    version: int = config["models"][model_name]["version"]
    tag_string, _ = JtpInference(device=device).run_classifier(model_name=model_name, version=version, image=image, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
    return tag_string

async def download_progress_callback(perc: int, model_name: str, node: Union[str, None]) -> None:
    from ..helpers.nodes import ComfyNode
    client_id = PromptServer.instance.client_id
    message: str = ""
    if perc < 100:
        message = "[{0:d1}%] Downloading {1}...".format(perc, model_name)
    else:
        message = "Download {0} complete!".format(model_name)
    ComfyNode().update_node_status(client_id=client_id, node=node, text=message, progress=perc)


async def download_complete_callback(node: Union[str, None]) -> None:
    from ..helpers.nodes import ComfyNode
    client_id = PromptServer.instance.client_id
    ComfyNode().update_node_status(client_id, node, None)

JtpModelManager(model_basepath=ComfyExtension().extension_dir("models", mkdir=True), download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)
JtpTagManager(tags_basepath=ComfyExtension().extension_dir("tags", mkdir=True), download_progress_callback=download_progress_callback, download_complete_callback=download_complete_callback)

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
    default: str = defaults["model"] + ".safetensors"
    model: str = default if default in models else models[0]
    threshold: float = float(request.query.get("threshold", defaults["threshold"]))
    exclude_tags: str = request.query.get("exclude_tags", defaults["exclude_tags"])
    replace_underscore: bool = request.query.get("replace_underscore", defaults["replace_underscore"]) == "true"
    trailing_comma: bool = request.query.get("trailing_comma", defaults["trailing_comma"]) == "true"
    client_id: str = request.rel_url.query.get("clientId", "")
    node: str = request.rel_url.query.get("node", "")
    return web.json_response(await classify_tags(image=image, model_name=model, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma, client_id=client_id, node=node))


class FDTagger():
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        models: List[str] = ComfyExtensionConfig().get()["models"].keys()
        return {"required": {
            "image": ("IMAGE", ),
            "model": (models, {"default": defaults["model"]}),
            "steps": ("INTEGER", {"default": 255, "min": 1, "max": 500}),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES: Tuple[str] = ("STRING",)
    OUTPUT_IS_LIST: Tuple[bool] = (True,)
    FUNCTION: str = "tag"
    OUTPUT_NODE: bool = True
    CATEGORY: str = "image"

    def tag(self, image: np.ndarray, model: str, steps: int, threshold: float, exclude_tags: str = "", replace_underscore: bool = False, trailing_comma: bool = False) -> Dict[str, Any]:
        from ..helpers.multithreading import ComfyThreading
        tensor = np.array(image * steps, dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags: List[str] = []
        for i in range(tensor.shape[0]):
            tags.append(ComfyThreading().wait_for_async(lambda: classify_tags(image=tensor[i], model_name=model, threshold=threshold,
                        exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "FD_FDTagger|fdtagger": FDTagger,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "FD_FDTagger|fdtagger": "FurryDiffusion Tagger 🐺",
}
