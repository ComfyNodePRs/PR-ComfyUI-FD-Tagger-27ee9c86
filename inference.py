import json
import os
import torch
import timm
from PIL import Image
from typing import Tuple, Union, List, Dict, Any
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import safetensors.torch


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


class JtpInference:
    def __init__(self, model_path: str, tags_path: str, device: Union[torch.device, None] = None, version: int = 1) -> None:
        torch.set_grad_enabled(False)
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.version = version
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.allowed_tags = self._load_tags(tags_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model_name = model_path.split(os.sep)[-1].split(".")[0]
        model = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli", pretrained=False, num_classes=9083)
        if self.version == 1:
            model.head = torch.nn.Sequential(
                torch.nn.Linear(model.head.in_features,
                                model.head.out_features * 2),
                torch.nn.Sigmoid(),
                torch.nn.Linear(model.head.out_features * 2, 9083)
            )
            model.load_state_dict(torch.load(
                filename=model_path, map_location=self.device))
        if self.version == 2:
            model.head = GatedHead(min(model.head.weight.shape), 9083)
            safetensors.torch.load_model(
                model=model, filename=model_path, device=self.device)
        else:
            raise ValueError(f"Invalid model version: {self.version}")
        model.eval()
        model.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 7:
            model.to(dtype=torch.float16, memory_format=torch.channels_last)
        return model

    def _get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            Fit((384, 384)),
            transforms.ToTensor(),
            CompositeAlpha(0.5),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                                 0.5, 0.5, 0.5], inplace=True),
            transforms.CenterCrop((384, 384)),
        ])

    def _load_tags(self, tags_path: str) -> List[str]:
        with open(tags_path, "r") as file:
            tags = json.load(file)
        return [tag.replace("_", " ") for tag in tags.keys()]

    def _create_tags(self, tag_score: Dict[str, float], threshold: float) -> Tuple[str, Dict[str, float]]:
        filtered_tag_score = {key: value for key,
                              value in tag_score.items() if value > threshold}
        return ", ".join(filtered_tag_score.keys()), filtered_tag_score

    def run_classifier(self, image: Image, threshold: float) -> Tuple[str, Dict[str, float]]:
        img = image.convert('RGBA')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 7:
            tensor = tensor.to(dtype=torch.float16,
                               memory_format=torch.channels_last)
        with torch.no_grad():
            logits = self.model(tensor)
            probits = torch.nn.functional.sigmoid(logits[0]).cpu()
            values, indices = probits.topk(250)
        tag_score = {self.allowed_tags[indices[i]]: values[i].item(
        ) for i in range(indices.size(0))}
        sorted_tag_score = dict(
            sorted(tag_score.items(), key=lambda item: item[1], reverse=True))
        return self._create_tags(sorted_tag_score, threshold)
