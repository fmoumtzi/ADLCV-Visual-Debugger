import os
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = os.path.join("models", "Qwen2-VL-2B-Instruct")

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


def resolve_runtime() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path(model_path: str) -> str:
    if model_path and os.path.isdir(model_path):
        return model_path
    if os.path.isdir(DEFAULT_LOCAL_MODEL_PATH):
        return DEFAULT_LOCAL_MODEL_PATH
    return model_path or MODEL_ID


def build_base_load_kwargs(device: torch.device, for_training: bool) -> Dict:
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"

        if not for_training:
            kwargs["device_map"] = "auto"
        return kwargs

    if device.type == "mps":
        return {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

    return {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }


def load_qwen_vl(
    model_path: str,
    for_training: bool = False,
    use_lora: bool = False,
    use_4bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    device = resolve_runtime()
    resolved_path = resolve_model_path(model_path)

    if device.type == "cpu":
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = "1"

    load_kwargs = build_base_load_kwargs(device, for_training=for_training)

    if use_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit loading is only recommended on CUDA.")
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs.pop("torch_dtype", None)
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        resolved_path,
        **load_kwargs,
    )

    processor = AutoProcessor.from_pretrained(resolved_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if use_lora:
        from peft import LoraConfig, get_peft_model

        if use_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        model = get_peft_model(model, peft_config)

        # IMPORTANT: move to device unless using device_map
        if "device_map" not in load_kwargs:
            model.to(device)

        model.train(mode=for_training)
        if not for_training:
            model.eval()

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        return model, processor, device, resolved_path

    if "device_map" not in load_kwargs:
        model.to(device)

    model.train(mode=for_training)
    if not for_training:
        model.eval()

    return model, processor, device, resolved_path


def open_rgb(image_path: str) -> Image.Image:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing image: {image_path}")
    return Image.open(image_path).convert("RGB")