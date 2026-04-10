import argparse
import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.2
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL on an image + prompt.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="",
        help="Optional prompt to ask about each image",
    )
    return parser.parse_args()


def find_images():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    image_paths = []
    for name in sorted(os.listdir(IMAGES_DIR)):
        path = os.path.join(IMAGES_DIR, name)
        _, ext = os.path.splitext(name)
        if os.path.isfile(path) and ext.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {IMAGES_DIR}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    return image_paths


def resolve_runtime():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_load_config(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            import flash_attn 

            return {
                "torch_dtype": dtype,
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            }
        except ImportError:
            return {
                "torch_dtype": dtype,
                "attn_implementation": "sdpa",
                "device_map": "auto",
            }

    if device.type == "mps":
        return {"torch_dtype": torch.float16}

    return {"torch_dtype": torch.float32}


def main():
    image_paths = find_images()
    device = resolve_runtime()
    model_load_config = resolve_model_load_config(device)

    print(f"Loading model: {MODEL_ID} on {device}")
    print(
        "Model settings:",
        f"dtype={model_load_config['torch_dtype']},",
        f"attention={model_load_config.get('attn_implementation', 'default')}",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        **model_load_config,
    )

    if "device_map" not in model_load_config:
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    args = parse_args()
    prompt = args.prompt.strip() or "Describe the image in detail."

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"\n=== RESPONSE: {os.path.basename(image_path)} ===")
        print(output_text)


if __name__ == "__main__":
    main()
