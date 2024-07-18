# ComfyUI WD 1.4 Tagger (Furry Diffusion Branch)

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension allowing the interrogation of Furry Diffusion tags from images using JTP tag inference.

> **NSFW Content Warning**: This ConfyUI extension can be used to classify or may mistakenly classify content as NSFW (obscene) contnet.  Please refrain from using this extension if you are below the general age of consent in your jurisdication.

**Based on:**
- [RedRocket/JointTaggerProject](https://huggingface.co/RedRocket/JointTaggerProject) - The model this custom extension runs on.
- [pythongosssss/ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) - Where this repository was forked from.

**All models created by:**
- [RedRocket](https://huggingface.co/RedRocket)

## Installation

1. `git clone https://github.com/loopyd/ComfyUI-FD-Tagger` into the `custom_nodes` folder 
    - e.g. `custom_nodes\ComfyUI-FD-Tagger`  
2. Open a Command Prompt or Terminal Application of your choice.
3. Change to the `custom_nodes\ComfyUI-FD-Tagger` folder you just created 
    - e.g. `cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FD-Tagger` or wherever you have it installed
4. Install python packages
    - **Windows Standalone installation** (embedded python):
        `../../../python_embeded/python.exe -s -m pip install -r requirements.txt`  
    - **Manual/non-Windows installation**
        `pip install -r requirements.txt`

## Usage

Add the node via `Furry Diffusion` -> `FDTagger|deitydurg`
Connect your inputs, and outputs.  
Models are automatically downloaded at runtime if missing.
The node supports tagging and outputting multiple batched inputs.

# Node Inputs

- **model**: The interrogation model to use. You can try them out here [WaifuDiffusion v1.4 Tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags). The newest model (as of writing) is `MOAT` and the most popular is `ConvNextV2`.  
- **threshold**: The score for the tag to be considered valid
- **exclude_tags** A comma separated list of tags that should not be included in the results

Quick interrogation of images is also available on any node that is displaying an image, e.g. a `LoadImage`, `SaveImage`, `PreviewImage` node.  

Simply right click on the node (or if displaying multiple images, on the image you want to interrogate) and select `FD Tagger` from the menu.

Default settings used for this are in the `settings` section of `config.json`.

# Node Outputs

- **tags** - A comma-seperated list of e621 tags.  This prompt can be used to recreate the image on FluffyRock, or any of the compatible Furry Diffusion models that use the same tags (results may vary)

## Requirements

`pytorch` for running the models on CPU or GPU.

## Changelog
- 2024-06-18 - Move to own repo/branch, fork for Furry Diffusion community
