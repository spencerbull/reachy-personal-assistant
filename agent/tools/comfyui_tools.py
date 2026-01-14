"""
ComfyUI Tools for image generation.

Provides LangChain-compatible tools for generating images via ComfyUI.
Returns URLs to generated images that can be clicked in the chat.
"""

import asyncio
import base64
import json
import os
import uuid
from typing import Optional

import aiohttp
from langchain_core.tools import tool
from loguru import logger

# ComfyUI server configuration
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://kyber:8188")

# Path to workflow template
WORKFLOW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "comfyui-workflows",
    "IF-Gemini-Image.json"
)


async def upload_image_to_comfyui(image_bytes: bytes, filename: str = "reachy_input.png") -> Optional[str]:
    """
    Upload an image to ComfyUI's input folder.
    
    Returns:
        The filename as stored on the server, or None if failed
    """
    url = f"{COMFYUI_URL}/upload/image"
    
    try:
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field(
                'image',
                image_bytes,
                filename=filename,
                content_type='image/png'
            )
            form.add_field('overwrite', 'true')
            
            logger.info(f"Uploading image to ComfyUI: {url}")
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    result = await response.json()
                    server_filename = result.get("name", filename)
                    logger.info(f"Image uploaded successfully as: {server_filename}")
                    return server_filename
                else:
                    error = await response.text()
                    logger.error(f"Failed to upload image: {response.status} - {error}")
                    return None
    except Exception as e:
        logger.error(f"Error uploading image to ComfyUI: {e}")
        return None


async def queue_prompt(workflow: dict) -> Optional[str]:
    """
    Queue a prompt on ComfyUI.
    
    Returns:
        The prompt_id if successful, None otherwise
    """
    url = f"{COMFYUI_URL}/prompt"
    
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": workflow}
            
            logger.info(f"Queueing prompt on ComfyUI: {url}")
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    prompt_id = result.get("prompt_id")
                    logger.info(f"Prompt queued with ID: {prompt_id}")
                    return prompt_id
                else:
                    error = await response.text()
                    logger.error(f"Failed to queue prompt: {response.status} - {error}")
                    return None
    except Exception as e:
        logger.error(f"Error queueing prompt: {e}")
        return None


async def wait_for_completion(prompt_id: str, timeout: int = 120) -> Optional[dict]:
    """
    Wait for a prompt to complete and return the output info.
    
    Returns:
        Dict with output info (filename, subfolder, type) or None
    """
    url = f"{COMFYUI_URL}/history/{prompt_id}"
    start_time = asyncio.get_event_loop().time()
    
    logger.info(f"Waiting for prompt {prompt_id} to complete...")
    
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    logger.error(f"Timeout waiting for prompt {prompt_id}")
                    return None
                
                async with session.get(url) as response:
                    if response.status == 200:
                        history = await response.json()
                        
                        if prompt_id in history:
                            prompt_data = history[prompt_id]
                            
                            if prompt_data.get("status", {}).get("completed", False):
                                logger.info(f"Prompt {prompt_id} completed successfully")
                                
                                # Extract output image info
                                outputs = prompt_data.get("outputs", {})
                                for node_id, node_output in outputs.items():
                                    if "images" in node_output:
                                        for img in node_output["images"]:
                                            return {
                                                "filename": img.get("filename"),
                                                "subfolder": img.get("subfolder", ""),
                                                "type": img.get("type", "temp")
                                            }
                                
                                logger.warning("No images in output")
                                return None
                
                # Log progress every 10 seconds
                if int(elapsed) % 10 == 0 and elapsed > 0:
                    logger.info(f"Still waiting... ({int(elapsed)}s elapsed)")
                
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"Error waiting for completion: {e}")
        return None


def build_image_url(filename: str, subfolder: str = "", img_type: str = "temp") -> str:
    """
    Build a URL to view an image from ComfyUI.
    """
    url = f"{COMFYUI_URL}/view?filename={filename}&type={img_type}"
    if subfolder:
        url += f"&subfolder={subfolder}"
    return url


@tool
async def generate_image_tool(prompt: str, input_image_b64: str) -> dict:
    """
    Generate a transformed image using ComfyUI.
    
    This tool takes a text prompt describing the desired transformation
    and an input image (base64 encoded), then generates a new image
    using the IF-Gemini workflow.
    
    Args:
        prompt: Text description of the desired image transformation
        input_image_b64: Base64-encoded input image (with or without data URL prefix)
    
    Returns:
        Dict with success status and image_url (clickable link) or error message
    """
    try:
        # Decode the input image
        if input_image_b64.startswith("data:"):
            # Extract base64 from data URL
            _, b64_data = input_image_b64.split(",", 1)
        else:
            b64_data = input_image_b64
        
        image_bytes = base64.b64decode(b64_data)
        logger.info(f"Decoded input image: {len(image_bytes)} bytes")
        
        # Upload input image to ComfyUI
        input_filename = f"reachy_input_{uuid.uuid4().hex[:8]}.png"
        server_filename = await upload_image_to_comfyui(image_bytes, input_filename)
        
        if not server_filename:
            return {
                "success": False,
                "image_url": None,
                "message": "Failed to upload input image to ComfyUI"
            }
        
        # Load workflow template
        if not os.path.exists(WORKFLOW_PATH):
            return {
                "success": False,
                "image_url": None,
                "message": f"Workflow file not found: {WORKFLOW_PATH}"
            }
        
        with open(WORKFLOW_PATH, 'r') as f:
            workflow = json.load(f)
        
        # Configure workflow with input image and prompt
        prompt_set = False
        image_set = False
        
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            
            # Set input image on LoadImage node
            if class_type == "LoadImage":
                node["inputs"]["image"] = server_filename
                image_set = True
                logger.info(f"Set LoadImage node {node_id} to use: {server_filename}")
            
            # Set prompt on IFGeminiNode (the main generation node in IF-Gemini workflow)
            if class_type == "IFGeminiNode":
                if "prompt" in inputs:
                    node["inputs"]["prompt"] = prompt
                    prompt_set = True
                    logger.info(f"Set prompt in IFGeminiNode (node {node_id}): {prompt[:80]}...")
            
            # Also handle other common prompt node types
            if class_type in ["IF_PromptMkr", "CLIPTextEncode", "PromptExpansion"]:
                if "prompt" in inputs:
                    node["inputs"]["prompt"] = prompt
                    prompt_set = True
                    logger.info(f"Set prompt in {class_type} (node {node_id})")
                elif "text" in inputs:
                    node["inputs"]["text"] = prompt
                    prompt_set = True
                    logger.info(f"Set text in {class_type} (node {node_id})")
        
        if not prompt_set:
            logger.warning(f"Could not find prompt node in workflow! Prompt may not be applied.")
        if not image_set:
            logger.warning(f"Could not find LoadImage node in workflow! Image may not be applied.")
        
        logger.info(f"Workflow configured - prompt_set: {prompt_set}, image_set: {image_set}")
        
        # Queue the prompt
        prompt_id = await queue_prompt(workflow)
        
        if not prompt_id:
            return {
                "success": False,
                "image_url": None,
                "message": "Failed to queue prompt on ComfyUI"
            }
        
        # Wait for completion
        output_info = await wait_for_completion(prompt_id, timeout=120)
        
        if not output_info:
            return {
                "success": False,
                "image_url": None,
                "message": "Image generation timed out or failed"
            }
        
        # Build the URL to view the image
        image_url = build_image_url(
            output_info["filename"],
            output_info.get("subfolder", ""),
            output_info.get("type", "temp")
        )
        
        logger.info(f"Image generation completed! URL: {image_url}")
        
        return {
            "success": True,
            "image_url": image_url,
            "message": f"Image generated successfully. View it here: {image_url}"
        }
        
    except Exception as e:
        logger.error(f"Error in generate_image_tool: {e}", exc_info=True)
        return {
            "success": False,
            "image_url": None,
            "message": f"Error generating image: {str(e)}"
        }


def get_all_comfyui_tools() -> list:
    """Return all ComfyUI tools."""
    return [generate_image_tool]
