import enum
import hashlib
import os
from pathlib import Path
import time
from PIL import Image
from typing import Any, Dict, Tuple, Union
import numpy
import torch
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from ..helpers.metaclasses import Singleton


class Fit(torch.nn.Module):
    """
    Resize an image to fit within the given bounds while maintaining aspect ratio
    """
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
    """
    Composite an image with an alpha channel onto a background color
    """
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


class ImageCacheType(enum):
	"""
	An enumeration of the types of image caches
	"""
	ROUNDROBIN = "roundrobin"
	LEAST_RECENTLY_USED = "least_recently_used"


class JtpImageManager(metaclass=Singleton):
	def __init__(self, cache_maxsize: int = 10, cache_method: ImageCacheType = ImageCacheType.ROUNDROBIN) -> None:
		self.data = {}
		self.data["images"] = {}
		self.cache_maxsize = cache_maxsize if cache_maxsize > 0 else 10
		self.cache_method = cache_method if cache_method in ImageCacheType else ImageCacheType.ROUNDROBIN
  
	def __del__(self) -> None:
		self.data.clear()
		del self.data
		import gc
		gc.collect()

	@classmethod
	def is_cached(cls, image_name: str) -> bool:
		"""
		Check if an image is loaded into memory
		"""
		return image_name in cls().data.keys() and cls().data[image_name] is not None and cls().data[image_name]["input"] is not None

	@classmethod
	def is_done(cls, image_name: str) -> bool:
		"""
		Check if an image is done processing
		"""
		return cls().is_cached(image_name) and image_name in cls().data.keys() and cls().data[image_name] is not None and cls().data[image_name]["output"] is not None

	@classmethod
	def do_transform(cls, image: numpy.ndarray, width: int, height: int, interpolation, grow, pad, background: Tuple[int, int, int]) -> torch.Tensor:
		"""
		Perform transformations on an image
  		"""
		return transforms.Compose([
            Fit(bounds=(width, height), interpolation=interpolation, grow=grow, pad=pad),
            transforms.ToTensor(),
            CompositeAlpha(background=background),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.CenterCrop(size=(width, height)),
        ])(image).unsqueeze(0)
  
	@classmethod
	def cache_roundrobin(cls) -> None:
		"""
		Remove the oldest image from the cache
		"""
		if len(cls().data.keys()) >= cls().cache_maxsize:
			oldest = min(cls().data, key=lambda k: cls().data[k]["timestamp"])
			_ = cls().data.pop(oldest)
   
	@classmethod
	def cache_lastrecentlyused(cls) -> None:
		"""
		Remove the last recently used image from the cache
		"""
		if len(cls().data.keys()) >= cls().cache_maxsize:
			lastused = min(cls().data, key=lambda k: cls().data[k]["used_timestamp"])
			_ = cls().data.pop(lastused)
   
	@classmethod
	def load(cls, image: Union[Path, numpy.ndarray, None]) -> Union[Tuple[str, Dict[str, Any]], torch.Tensor, None]:
		"""
		Load an image into memory
  
		- Return None if no image is provided or there was an error loading it
		- Return a loaded tensor if we need to perform inference on it
		- Return a tuple containg tags and tag:score dict if we are already through with it.
		"""
		from ..helpers.logger import ComfyLogger
		if image is not None and isinstance(image, Path):
			# Image is a path to an image, so load it with PIL, then stuff into a numpy array
			image = str(image)
			if cls().is_done(image):
				ComfyLogger().log(f"Image {image} already processed, using from cache", "WARNING", True)
				return cls().data[image]["output"]
			if cls().is_cached(image):
				ComfyLogger().log(f"Image {image} already loaded, using from cache", "WARNING", True)
				cls().data[image]["used_timestamp"] = time.time()
				image_input = cls().data[image]["input"]
			else:
				if not os.path.exists(image):
					ComfyLogger.log(f"Image {image} not found in path: {image}", "ERROR", True)
					return None
				cache_func = {
					ImageCacheType.ROUNDROBIN: cls().cache_roundrobin,
					ImageCacheType.LEAST_RECENTLY_USED: cls().cache_lastrecentlyused
				}
				cache_func[cls().cache_method]()
				cls().data[image] = {
					"input": numpy.array(Image.open(image).convert("RGBA")).reshape([0]),
					"timestamp": time.time(),
					"used_timestamp": time.time(),
					"output": None
				}
				ComfyLogger().log(f"Image: {image} loaded into cache", "DEBUG", True)
				image_input = cls().data[image]["input"]
		elif image is not None and isinstance(image, numpy.ndarray):
			# Image is a numpy array, so sha256 it and look for it in the cache
			image_hash = hashlib.sha256(image.tobytes(), usedforsecurity=False).hexdigest()
			if cls().is_done(image):
				ComfyLogger().log(f"Image {image} already processed, using from cache", "WARNING", True)
				return cls().data[image]["output"]
			if cls().is_cached(image_hash):
				ComfyLogger().log(f"Image {image_hash} already loaded, using from cache", "WARNING", True)
				cls().data[image_hash]["used_timestamp"] = time.time()
				image_input = cls().data[image_hash]["input"]
			else:
				cache_func = {
					ImageCacheType.ROUNDROBIN: cls().cache_roundrobin,
					ImageCacheType.LEAST_RECENTLY_USED: cls().cache_lastrecentlyused
				}
				cache_func[cls().cache_method]()
				cls().data[image_hash] = {
					"input": image.reshape([0]),
					"timestamp": time.time(),
					"used_timestamp": time.time(),
					"output": None
				}
				ComfyLogger().log(f"Image {image_hash} loaded into cache", "DEBUG", True)
				image_input = cls().data[image_hash]["input"]
		else:
			ComfyLogger().log("No image provided to load", "ERROR", True)
			return None

		# If we reach this codepath, we have to perform inference on the image
		tensor = cls().do_transform(
      		image=image_input,
			width=384,
			height=384,
			interpolation=InterpolationMode.LANCZOS,
			grow=True,
			pad=None,
			background=0.5,
		)
		if cls().device.type == 'cuda' and torch.cuda.is_available():
			tensor.cuda()
			if torch.cuda.get_device_capability()[0] >= 7:
				tensor = tensor.to(dtype=torch.float16, memory_format=torch.channels_last)
				ComfyLogger().log("Image loaded to GPU with mixed precision", "INFO", True)
			else:
				ComfyLogger().log("Image loaded to older GPU without mixed precision", "WARNING", True)
		else:
			tensor.cpu()
		return tensor

	@classmethod
	def unload_image(cls, image: Union[Path, numpy.ndarray, None]) -> bool:
		"""
		Unload an image from memory
		"""
		if image is not None and isinstance(image, Path):
			image = str(image)
			if cls().is_cached(image):
				_ = cls().data.pop(image)
				return True
		elif image is not None and isinstance(image, numpy.ndarray):
			image_hash = hashlib.sha256(image.tobytes(), usedforsecurity=False).hexdigest()
			if cls().is_cached(image_hash):
				_ = cls().data.pop(image_hash)
				return True
		return False

	@classmethod
	def commit_cache(cls, image, output: Tuple[str, Dict[str, Any]]) -> bool:
		"""
		Commit the output of an image to the cache
		"""
		if image is not None and isinstance(image, Path):
			image = str(image)
			if cls().is_cached(image):
				cls().data[image]["output"] = output
				return True
		elif image is not None and isinstance(image, numpy.ndarray):
			image_hash = hashlib.sha256(image.tobytes(), usedforsecurity=False).hexdigest()
			if cls().is_cached(image_hash):
				cls().data[image_hash]["output"] = output
				return True
		return False	
  
  