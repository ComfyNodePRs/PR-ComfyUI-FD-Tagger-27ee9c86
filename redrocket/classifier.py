import numpy as np
import torch
from typing import Optional, Tuple, Union, Dict, Any
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from redrocket.tag_manager import JtpTagManager
from redrocket.model_manager import JtpModelManager
from helpers.metaclasses import Singleton


class Fit(torch.nn.Module):
    def __init__(self, bounds: Union[Tuple[int, int], int], interpolation: InterpolationMode = InterpolationMode.LANCZOS, grow: bool = True, pad: Union[float, None] = None) -> None:
        super().__init__()
        self.bounds = (bounds, bounds) if isinstance(bounds, int) else bounds
        self.interpolation = interpolation
        self.grow = grow
        self.pad = pad

    def forward(self, img) -> Any:
        wimg, himg = img.size
        hbound, wbound = self.bounds
        hscale = hbound / himg
        wscale = wbound / wimg
        if not self.grow:
            hscale = min(hscale, 1.0)
            wscale = min(wscale, 1.0)
        scale = min(hscale, wscale)
        if scale == 1.0:
            return img
        hnew = min(round(himg * scale), hbound)
        wnew = min(round(wimg * scale), wbound)
        img = TF.resize(img, (hnew, wnew), self.interpolation)
        if self.pad is None:
            return img
        hpad = hbound - hnew
        wpad = wbound - wnew
        tpad = hpad // 2
        bpad = hpad - tpad
        lpad = wpad // 2
        rpad = wpad - lpad
        return TF.pad(img, (lpad, tpad, rpad, bpad), self.pad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"bounds={self.bounds}, " +
            f"interpolation={self.interpolation.value}, " +
            f"grow={self.grow}, " +
            f"pad={self.pad})"
        )


class CompositeAlpha(torch.nn.Module):
    def __init__(self, background: Union[Tuple[float, float, float], float]) -> None:
        super().__init__()
        self.background = (background, background, background) if isinstance(
            background, float) else background
        self.background = torch.tensor(
            self.background).unsqueeze(1).unsqueeze(2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-3] == 3:
            return img
        alpha = img[..., 3, None, :, :]
        img[..., :3, :, :] *= alpha
        background = self.background.expand(-1, img.shape[-2], img.shape[-1])
        if background.ndim == 1:
            background = background[:, None, None]
        elif background.ndim == 2:
            background = background[None, :, :]
        img[..., :3, :, :] += (1.0 - alpha) * background
        return img[..., :3, :, :]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"background={self.background})"
        )


class GatedHead(torch.nn.Module):
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


class JtpInference(metaclass=Singleton):
    """
    A Clip Vision Classifier by RedRocket (inference code made rubust by deitydurg)
    """
    def __init__(self, device: Optional[torch.device] = torch.cpu) -> None:
        torch.set_grad_enabled(False)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = {}
        self.transform = self.get_transform()

    @classmethod
    def get_transform(cls) -> transforms.Compose:
        return transforms.Compose([
            Fit((384, 384)),
            transforms.ToTensor(),
            CompositeAlpha(0.5),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                                 0.5, 0.5, 0.5], inplace=True),
            transforms.CenterCrop((384, 384)),
        ])

    @classmethod
    def run_classifier(cls, model_name: str, version: int, image: np.ndarray, threshold: float, exclude_tags: str = "", replace_underscore: bool = True, trailing_comma: bool = False) -> Tuple[str, Dict[str, float]]:
        # Install and load model and tags
        if JtpModelManager().is_installed(model_name) is False:
            JtpModelManager().download(model_name)
        if JtpTagManager().is_installed(model_name) is False:
            JtpTagManager().download(model_name)
        if JtpModelManager().is_loaded(model_name) is False:
            JtpModelManager().load(model_name)
        if JtpTagManager().is_loaded(model_name) is False:
            JtpTagManager().load(model_name)
            
        # Load image to device
        tensor = cls().transform(image).unsqueeze(0)
        if cls().device.type == 'cuda' and torch.cuda.is_available():
            tensor.cuda()
            if torch.cuda.get_device_capability()[0] >= 7:
                tensor = tensor.to(dtype=torch.float16, memory_format=torch.channels_last)
        else:
            tensor.cpu()
        
        # Run inference
        with torch.no_grad():
            if f"{version}" == "1":
                logits = JtpModelManager().data[model_name]["model"](tensor)
                probits = torch.nn.functional.sigmoid(logits[0]).cpu()
                values, indices = probits.topk(250)
            elif f"{version}" == "2":
                probits = JtpModelManager().data[model_name]["model"](tensor)[0].cpu()
                values, indices = probits.topk(250)
            else:
                raise ValueError(f"Invalid model version: {cls().version}")
        
        corrected_excuded_tags = [tag.replace("_", " ").lstrip().rstrip() for tag in exclude_tags.split(",") if tag.isspace() is False]
        tag_score = {JtpTagManager().data[model_name]["tags"][indices[i]]: values[i].item() for i in range(indices.size(0)) if JtpTagManager().data[model_name]["tags"][indices[i]] not in corrected_excuded_tags}
        if replace_underscore is False:
            tag_score = {key.replace(" ", "_"): value for key, value in tag_score.items()}
        tag_score = dict(sorted(tag_score.items(), key=lambda item: item[1], reverse=True))
        tag_score = {key: value for key, value in tag_score.items() if value > threshold}
        return ", ".join(tag_score.keys()) + ("," if trailing_comma is True else ""), tag_score