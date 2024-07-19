import importlib
import inspect
import os
from typing import Any, Dict, Optional, Union
from server import PromptServer

from .metaclasses import Singleton


class ComfyNode(metaclass=Singleton):
	"""
	A singleton class to provide additional node utility functions for a comfy extension.
	"""
	def __init__(self) -> None:
		pass

	@classmethod
	def update_node_status(cls, client_id: Optional[str], node: str, text: str, progress: Optional[float] = None) -> None:
		if client_id is None:
			client_id = PromptServer.instance.client_id
		if client_id is None:
			return
		PromptServer.instance.send_sync("furrydiffusion/update_status", {
			"node": node,
			"progress": progress,
			"text": text
		}, client_id)

	@classmethod
	async def update_node_status_async(cls, client_id: Optional[str], node: str, text: str, progress: Optional[float] = None) -> None:
		if client_id is None:
			client_id = PromptServer.instance.client_id
		if client_id is None:
			return
		await PromptServer.instance.send("furrydiffusion/update_status", {
			"node": node,
			"progress": progress,
			"text": text
		}, client_id)
  
	@classmethod
	def get_node_vars(cls, source_path) -> Dict[str, Any]:
		vars = {}
		for file in os.listdir(source_path):
			if file.endswith(".py"):
				abs_path = os.path.abspath(os.path.join(source_path, file))
				module = importlib.import_module(abs_path)
				for name, obj in inspect.getmembers(module):
					if isinstance(obj, Union[Dict[str, Any], str]) and name in ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", 'WEB_DIRECTORY']:
						if isinstance(obj, str):
							vars[name] = [o for o in obj if obj not in vars[name]]
						elif isinstance(obj, dict):
							for key, value in obj.items():
								vars[name][key] = value
				del module
		return vars
            