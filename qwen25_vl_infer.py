import argparse
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
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
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.bfloat16


def main():
    args = parse_args()
    prompt = args.prompt.strip() or "Describe the image in detail."
    image_paths = find_images()
    device, torch_dtype = resolve_runtime()

    print(f"Loading model: {MODEL_ID} on {device} with {torch_dtype}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"\n=== RESPONSE: {os.path.basename(image_path)} ===")
        print(output_text)


if __name__ == "__main__":
    main()
