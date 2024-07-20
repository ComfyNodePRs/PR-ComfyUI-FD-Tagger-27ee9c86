from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Union, Dict, Any

from .image_manager import JtpImageManager

from ..redrocket.tag_manager import JtpTagManager
from ..redrocket.model_manager import JtpModelManager
from ..helpers.metaclasses import Singleton


class JtpInference(metaclass=Singleton):
    """
    A Clip Vision Classifier by RedRocket (inference code made rubust by deitydurg)
    """
    def __init__(self, device: Optional[torch.device] = torch.cpu) -> None:
        torch.set_grad_enabled(False)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def run_classifier(cls, model_name: str, version: int, device: torch.device, image: Union[np.ndarray, Path], steps: float, threshold: float, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False) -> Tuple[str, Dict[str, float]]:
        from ..helpers.logger import ComfyLogger

        # Load all the things
        if JtpModelManager().is_installed(model_name) is False:
            if not JtpModelManager().download(model_name):
                ComfyLogger().log(f"Model {model_name} could not be downloaded", "ERROR", True)
                return "", {}
        if JtpTagManager().is_installed(model_name) is False:
            if not JtpTagManager().download(model_name):
                ComfyLogger().log(f"Tags for model {model_name} could not be downloaded", "ERROR", True)
                return "", {}            
        if JtpModelManager().is_loaded(model_name) is False:
            if not JtpModelManager().load(model_name, version=version, device=device):
                ComfyLogger().log(f"Model {model_name} could not be loaded", "ERROR", True)
                return "", {}
        if JtpTagManager().is_loaded(model_name) is False:
            if not JtpTagManager().load(model_name):
                ComfyLogger().log(f"Tags for model {model_name} could not be loaded", "ERROR", True)
                return "", {}
        if cls().device != device:
            JtpModelManager().switch_device(model_name, device)
            cls().device = device
            
        tensor = JtpImageManager().load(image)
        if tensor is None:
            ComfyLogger().log("Image could not be loaded", "ERROR", True)
            return "", {}
        elif isinstance(tensor, Tuple):
            return tensor
        
        # Run inference
        with torch.no_grad():
            if f"{version}" == "1":
                logits = JtpModelManager().data[model_name]["model"](tensor)
                probits = torch.nn.functional.sigmoid(logits[0]).cpu()
                values, indices = probits.topk(250)
                del logits
            elif f"{version}" == "2":
                probits = JtpModelManager().data[model_name]["model"](tensor)[0].cpu()
                values, indices = probits.topk(250)
            else:
                raise ValueError(f"Invalid model version: {cls().version}")
        
        tags_str, tag_scores  = JtpTagManager().process_tags(model_name=model_name, values=values, indices=indices, steps=steps, threshold=threshold, exclude_tags=exclude_tags, replace_underscore=replace_underscore, trailing_comma=trailing_comma)
        
        # CLear memory for next run
        del tensor, probits, values, indices
        import gc
        gc.collect()
        
        # Commit the result to the cache
        JtpImageManager().commit_cache(image, (tags_str, tag_scores))
        return tags_str, tag_scores