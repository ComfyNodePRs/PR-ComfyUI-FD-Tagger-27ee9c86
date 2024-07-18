import torch
import comfy.utils
import asyncio
import aiohttp
import numpy as np
import os
from PIL import Image
from aiohttp import web
from typing import List, Union, Tuple, Dict, Any
import folder_paths
from .comfynode import get_ext_dir, get_comfy_dir, download_to_file, update_node_status, wait_for_async, get_extension_config, log
from .inference import JtpInference
from server import PromptServer

config = get_extension_config()
defaults: Dict[str, Any] = {
    "model": "JTP_PILOT-e4-vit_so400m_patch14_siglip_384",
    "threshold": 0.35,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "HF_ENDPOINT": "https://huggingface.co",
}
defaults.update(config.get("settings", {}))

if "fd_tagger" in folder_paths.folder_names_and_paths:
    models_dir: str = folder_paths.get_folder_paths("fd_tagger")[0]
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
else:
    models_dir = get_ext_dir("models", mkdir=True)
known_models: List[str] = list(config["models"].keys())


def get_installed_models() -> List[str]:
    models = filter(lambda x: x.endswith(
        ".safetensors"), os.listdir(models_dir))
    models = [m for m in models if os.path.exists(
        os.path.join(models_dir, os.path.splitext(m)[0] + ".json"))]
    return models


async def tag(image: np.ndarray, model_name: str, threshold: float = 0.35, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False, client_id: Union[str, None] = None, node: Union[str, None] = None) -> str:
    if model_name.endswith(".safetensors"):
        model_name = model_name[0:-5]
    installed = list(get_installed_models())
    if not any(model_name + ".safetensors" in s for s in installed):
        await download_model(model_name, client_id, node)

    model_path: str = os.path.join(models_dir, model_name + ".safetensors")
    tags_path: str = os.path.join(models_dir, model_name + ".json")
    device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    version: int = config["models"][model_name]["version"]
    classifier = JtpInference(model_path, tags_path, device, version)
    input_image = Image.fromarray(image).convert('RGBA')
    tag_string, _ = classifier.run_classifier(input_image, threshold)
    return tag_string


async def download_model(model: str, client_id: Union[str, None], node: Union[str, None]) -> web.Response:
    hf_endpoint: str = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url: str = config["models"][model]["url"]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc: int) -> None:
            nonlocal client_id
            message: str = ""
            if perc < 100:
                message = f"Downloading {model}"
            update_node_status(client_id, node, message, perc)
        try:
            await download_to_file(f"{url}{model}.safetensors", os.path.join(models_dir, f"{model}.safetensors"), update_callback, session=session)
            await download_to_file(f"{url}tags.json", os.path.join(models_dir, f"{model}.json"), update_callback, session=session)
        except aiohttp.client_exceptions.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise
        update_node_status(client_id, node, None)
    return web.Response(status=200)


@PromptServer.instance.routes.get("/furrydiffusion/fdtagger/tag")
async def get_tags(request: web.Request) -> web.Response:
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)
    type: str = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)
    target_dir: str = get_comfy_dir(type)
    image_path: str = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)
    if not os.path.isfile(image_path):
        return web.Response(status=404)
    image: Image.Image = Image.open(image_path)
    models: List[str] = get_installed_models()
    default: str = defaults["model"] + ".safetensors"
    model: str = default if default in models else models[0]
    threshold: float = float(request.query.get("threshold", defaults["threshold"]))
    exclude_tags: str = request.query.get("exclude_tags", defaults["exclude_tags"])
    replace_underscore: bool = request.query.get("replace_underscore", defaults["replace_underscore"]) == "true"
    trailing_comma: bool = request.query.get("trailing_comma", defaults["trailing_comma"]) == "true"
    client_id: str = request.rel_url.query.get("clientId", "")
    node: str = request.rel_url.query.get("node", "")
    return web.json_response(await tag(image=np.array(image), model_name=model, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma, client_id=client_id, node=node))


class FDTagger():
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        extra: List[str] = [name for name, _ in (os.path.splitext(
            m) for m in get_installed_models()) if name not in known_models]
        models: List[str] = known_models + extra
        return {"required": {
            "image": ("IMAGE", ),
            "model": (models, {"default": defaults["model"]}),
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

    def tag(self, image: np.ndarray, model: str, threshold: float, exclude_tags: str = "", replace_underscore: bool = False, trailing_comma: bool = False) -> Dict[str, Any]:
        tensor: np.ndarray = image * 255
        tensor = np.array(tensor, dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags: List[str] = []
        for i in range(tensor.shape[0]):
            img: Image.Image = Image.fromarray(tensor[i])
            tags.append(wait_for_async(lambda: tag(image=np.array(img), model_name=model, threshold=threshold,
                        exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "FD_FDTagger|fdtagger": FDTagger,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "FD_FDTagger|fdtagger": "FurryDiffusion Tagger 🐺",
}
