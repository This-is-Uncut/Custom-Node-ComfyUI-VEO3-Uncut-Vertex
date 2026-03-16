import os
import requests
import time
import random
import io
import torch
import numpy as np
import tempfile
import json
import sys
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials

# --- Fix for Windows asyncio ConnectionResetError ---
if sys.platform == 'win32':
    try:
        import asyncio
        from asyncio.proactor_events import _ProactorBasePipeTransport
        # Store original method
        if not hasattr(_ProactorBasePipeTransport, '_original_call_connection_lost'):
            _ProactorBasePipeTransport._original_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        def _silence_connection_lost(self, exc):
            try:
                self._original_call_connection_lost(exc)
            except ConnectionResetError:
                pass
        _ProactorBasePipeTransport._call_connection_lost = _silence_connection_lost
    except ImportError:
        pass

# --- Configuration Logic ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "uncut_vertex_config.json")

TOKEN_ENDPOINT_PATH = "/get_vertex_token"
DEFAULT_PORT = "9191"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception: pass
    return {"project_id": "your-project-id", "location": "us-central1", "server_ip": ""}

def save_config(project_id, location, server_ip=""):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"project_id": project_id, "location": location, "server_ip": server_ip}, f)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

def fetch_vertex_token(server_ip):
    host = server_ip.strip()
    if host.startswith("http://"):
        host = host[7:]
    elif host.startswith("https://"):
        host = host[8:]
        
    if ":" not in host:
        host = f"{host}:{DEFAULT_PORT}"
    url = f"http://{host}{TOKEN_ENDPOINT_PATH}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise Exception(f"No access_token in response: {data}")
        return token
    except Exception as e:
        raise Exception(f"Failed to fetch Vertex token from {url}: {e}")

try:
    from comfy_api.latest import InputImpl
except ImportError:
    class InputImpl:
        @staticmethod
        def VideoFromFile(data): return data

def tensor_to_veo_image(img_tensor):
    img_np = (255. * img_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    if len(img_np.shape) == 4: 
        img_np = img_np[0]
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    # Return the base Image type for wrapping
    return types.Image(image_bytes=buffer.getvalue(), mime_type="image/png")

class Veo31GeneratorVertex:
    @classmethod
    def INPUT_TYPES(cls):
        current_config = load_config()
        return {
            "required": {
                "model": (["veo-3.1-generate-001", "veo-3.1-fast-generate-001", "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview"],),
                "mode": (["Start Image", "Start+End Images", "Reference Images"],),
                "audio_generation": (["Video Only", "Video + Audio"],),
                "project_id": ("STRING", {"default": current_config.get("project_id", "your-project-id")}),
                "location": ("STRING", {"default": current_config.get("location", "us-central1")}),
                "server_ip": ("STRING", {"default": current_config.get("server_ip", "")}),
                "aspect_ratio": (["16:9", "9:16"],),
                "resolution": (["720p", "1080p", "2160p (4k)"],),
                "duration": (["4", "6", "8"], {"default": "8"}),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VIDEO", "INT")
    RETURN_NAMES = ("video", "used_seed")
    FUNCTION = "generate"
    CATEGORY = "UncutNodes"

    def generate(self, model, mode, audio_generation, project_id, location, server_ip, aspect_ratio, resolution, duration, 
                 positive_prompt, negative_prompt, seed, **kwargs):
        
        save_config(project_id, location, server_ip)

        client_kwargs = dict(vertexai=True, project=project_id, location=location)

        if server_ip.strip():
            token = fetch_vertex_token(server_ip)
            client_kwargs["credentials"] = Credentials(token=token)

        client = genai.Client(**client_kwargs)
        actual_seed = seed if seed > 0 else random.randint(1, 0xFFFFFFFF)
        
        res_val = "4k" if "4k" in resolution else resolution.split(" ")[0]
        # Multi-modal modes (Interpolation/Reference) require 8s
        dur_val = 8 if (res_val in ["1080p", "4k"] or mode in ["Start+End Images", "Reference Images"]) else int(duration)
        
        config_dict = {
            "aspect_ratio": aspect_ratio,
            "resolution": res_val,
            "duration_seconds": dur_val,
            "person_generation": "allow_adult",
            "seed": actual_seed,
            "generate_audio": audio_generation == "Video + Audio"
        }

        # Handle multimodal logic
        start_frame_arg = None
        
        if mode == "Start Image" and "start_image" in kwargs:
            start_frame_arg = tensor_to_veo_image(kwargs["start_image"])
            if negative_prompt.strip():
                config_dict["negative_prompt"] = negative_prompt
            
        elif mode == "Start+End Images":
            if "start_image" in kwargs:
                start_frame_arg = tensor_to_veo_image(kwargs["start_image"])
            if "end_image" in kwargs:
                config_dict["last_frame"] = tensor_to_veo_image(kwargs["end_image"])
            if negative_prompt.strip():
                config_dict["negative_prompt"] = negative_prompt
                
        elif mode == "Reference Images":
            # 1. Negative prompts are NOT supported here
            # 2. Start frame is NOT used here
            start_frame_arg = None 
            
            # 3. Build the Reference Image list using the proper SDK Class
            ref_list = []
            for i in range(1, 4):
                ref_key = f"ref_image_{i}"
                if ref_key in kwargs and kwargs[ref_key] is not None:
                    ref_obj = types.VideoGenerationReferenceImage(
                        image=tensor_to_veo_image(kwargs[ref_key]),
                        reference_type="asset" # Tells the model these are 'ingredients'
                    )
                    ref_list.append(ref_obj)
            
            config_dict["reference_images"] = ref_list

        print(f"Submitting {model}: {mode} | Res: {res_val} | Audio: {config_dict['generate_audio']} | Seed: {actual_seed}")

        def run_call(use_seed=True):
            current_config = config_dict.copy()
            if not use_seed: current_config.pop("seed", None)
            
            return client.models.generate_videos(
                model=model,
                prompt=positive_prompt,
                image=start_frame_arg,
                config=types.GenerateVideosConfig(**current_config)
            )

        # Catch local SDK ValueErrors (Seeds) and API ClientErrors
        try:
            operation = run_call(use_seed=True)
        except (ValueError, Exception) as e:
            if "seed" in str(e).lower():
                print("⚠️ Retrying without seed due to SDK/API limitation...")
                operation = run_call(use_seed=False)
            else:
                raise e

        # Polling...
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        if operation.error: 
            raise Exception(f"Veo API Error: {operation.error}")

        # Safety & Stream handling
        response = operation.response
        if (getattr(response, "rai_media_filtered_count", 0) or 0) > 0:
            raise Exception(f"RAI Blocked: {getattr(response, 'rai_media_filtered_reasons', ['Safety'])[0]}")

        if response.generated_videos:
            video_info = response.generated_videos[0]
            video_obj = video_info.video
            
            # Fetch video_bytes using files.download if they aren't provided inline (Developer API)
            if not getattr(video_obj, "video_bytes", None):
                try:
                    client.files.download(file=video_obj)
                except ValueError as e:
                    if "only supported in the Gemini Developer client" in str(e):
                        if getattr(video_obj, "uri", None):
                            raise Exception(f"Vertex AI returned a URI ({video_obj.uri}) without inline bytes. Please ensure your configuration allows inline video bytes or handles GCS buckets.") from e
                        raise
                    raise

            # We can directly use video_bytes if available, avoiding disk I/O
            if getattr(video_obj, "video_bytes", None):
                video_data = video_obj.video_bytes
                if isinstance(video_data, str):
                    import base64
                    video_data = base64.b64decode(video_data)
                video_stream = io.BytesIO(video_data)
                video_stream.seek(0)
                if hasattr(client, "close"): client.close()
                return (InputImpl.VideoFromFile(video_stream), actual_seed)
                
            # Fallback if video_bytes is still not populated but save() works somehow
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                video_obj.save(tmp_path)
                with open(tmp_path, 'rb') as f:
                    video_stream = io.BytesIO(f.read())
                video_stream.seek(0)
                if hasattr(client, "close"): client.close()
                return (InputImpl.VideoFromFile(video_stream), actual_seed)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
            
        if hasattr(client, "close"): client.close()
        raise Exception("No video returned.")

NODE_CLASS_MAPPINGS = { "Veo31GeneratorVertex": Veo31GeneratorVertex }
NODE_DISPLAY_NAME_MAPPINGS = {"Veo31GeneratorVertex": "Veo 3.1 - Vertex- Uncut"}