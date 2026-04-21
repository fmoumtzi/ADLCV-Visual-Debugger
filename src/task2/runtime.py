import os
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = os.path.join("models", "Qwen2.5-VL-3B-Instruct")


def resolve_runtime() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_load_config(device: torch.device, for_training: bool) -> Dict:
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            import flash_attn  # noqa: F401

            config["attn_implementation"] = "flash_attention_2"
        except ImportError:
            config["attn_implementation"] = "sdpa"

        if not for_training:
            config["device_map"] = "auto"

        return config

    if device.type == "mps":
        return {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

    return {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }


def resolve_model_path(model_path: str) -> str:
    if model_path and os.path.isdir(model_path):
        return model_path
    if os.path.isdir(DEFAULT_LOCAL_MODEL_PATH):
        return DEFAULT_LOCAL_MODEL_PATH
    return model_path or MODEL_ID


def load_qwen_vl(model_path: str, for_training: bool = False) -> Tuple:
    device = resolve_runtime()
    resolved_path = resolve_model_path(model_path)
    config = resolve_model_load_config(device, for_training=for_training)
    if device.type == "cpu":
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = "1"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        resolved_path,
        **config,
    )

    if "device_map" not in config:
        model.to(device)
    model.train(mode=for_training)
    if not for_training:
        model.eval()

    processor = AutoProcessor.from_pretrained(resolved_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor, device, resolved_path


def open_rgb(image_path: str) -> Image.Image:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing image: {image_path}")
    return Image.open(image_path).convert("RGB")
