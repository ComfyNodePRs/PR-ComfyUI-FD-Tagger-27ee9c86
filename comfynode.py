import asyncio
import os
import json
import shutil
import inspect
import aiohttp
from typing import List, Dict, Any, Callable, Optional
from aiohttp import ClientSession
from tqdm import tqdm
from server import PromptServer

config: Optional[Dict[str, Any]] = None


def is_logging_enabled() -> bool:
    config = get_extension_config()
    if "logging" not in config:
        return False
    return config["logging"]


def log(message: str, type: Optional[str] = None, always: bool = False) -> None:
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    name = get_extension_config()["name"]

    print(f"fd_tagger:{name}) {message}")


def get_ext_dir(subpath: Optional[str] = None, mkdir: bool = False) -> str:
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_comfy_dir(subpath: Optional[str] = None) -> str:
    dir = os.path.dirname(inspect.getfile(PromptServer))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    return dir


def get_web_ext_dir() -> str:
    config = get_extension_config()
    name = config["name"]
    dir = get_comfy_dir("web/extensions/furrydiffusion")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir += "/" + name
    return dir


def get_extension_config(reload: bool = False) -> Dict[str, Any]:
    global config
    if not reload and config is not None:
        return config

    config_path = get_ext_dir("config.user.json")
    if not os.path.exists(config_path):
        config_path = get_ext_dir("config.json")

    if not os.path.exists(config_path):
        log("Missing config.json and config.user.json, this extension may not work correctly. Please reinstall the extension.", type="ERROR", always=True)
        print(f"Extension path: {get_ext_dir()}")
        return {"name": "Unknown", "version": -1}
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config


def link_js(src: str, dst: str) -> bool:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass
    try:
        os.symlink(src, dst)
        return True
    except:
        import logging
        logging.exception('')
        return False


def is_junction(path: str) -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False


def install_js() -> None:
    src_dir = get_ext_dir("web/js")
    if not os.path.exists(src_dir):
        log("No JS")
        return

    should_install = should_install_js()
    if should_install:
        log("It looks like you're running an old version of ComfyUI that requires manual setup of web files, it is recommended you update your installation.", "warning", True)
    dst_dir = get_web_ext_dir()
    linked = os.path.islink(dst_dir) or is_junction(dst_dir)
    if linked or os.path.exists(dst_dir):
        if linked:
            if should_install:
                log("JS already linked")
            else:
                os.unlink(dst_dir)
                log("JS unlinked, PromptServer will serve extension")
        elif not should_install:
            shutil.rmtree(dst_dir)
            log("JS deleted, PromptServer will serve extension")
        return

    if not should_install:
        log("JS skipped, PromptServer will serve extension")
        return

    if link_js(src_dir, dst_dir):
        log("JS linked")
        return

    log("Copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def should_install_js() -> bool:
    return not hasattr(PromptServer.instance, "supports") or "custom_nodes_from_web" not in PromptServer.instance.supports


def init(check_imports: Optional[List[str]]) -> bool:
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.",
                    type="ERROR", always=True)
                return False

    install_js()
    return True


async def download_to_file(url: str, destination: str, update_callback: Optional[Callable[[int], None]], is_ext_subpath: bool = True, session: Optional[ClientSession] = None) -> None:
    close_session = False
    if session is None:
        close_session = True
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        session = aiohttp.ClientSession(loop=loop)
    if is_ext_subpath:
        destination = get_ext_dir(destination)
    try:
        async with session.get(url) as response:
            size = int(response.headers.get('content-length', 0)) or None

            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                with open(destination, mode='wb') as f:
                    perc = 0
                    async for chunk in response.content.iter_chunked(2048):
                        f.write(chunk)
                        progressbar.update(len(chunk))
                        if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
                            last = perc
                            perc = round(progressbar.n / progressbar.total, 2)
                            if perc != last:
                                last = perc
                                await update_callback(perc)
    finally:
        if close_session and session is not None:
            await session.close()


def wait_for_async(async_fn: Callable[[], Any], loop: Optional[asyncio.AbstractEventLoop] = None) -> Any:
    res: List[Any] = []

    async def run_async() -> None:
        r = await async_fn()
        res.append(r)

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    loop.run_until_complete(run_async())

    return res[0]


def update_node_status(client_id: Optional[str], node: str, text: str, progress: Optional[float] = None) -> None:
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    PromptServer.instance.send_sync("furrydiffusion/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)


async def update_node_status_async(client_id: Optional[str], node: str, text: str, progress: Optional[float] = None) -> None:
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    await PromptServer.instance.send("furrydiffusion/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)
