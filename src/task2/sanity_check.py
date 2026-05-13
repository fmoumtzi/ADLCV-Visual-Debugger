import argparse
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from .io_utils import read_jsonl
except ImportError:
    from io_utils import read_jsonl


def load_rows(jsonl_path, split, max_examples):
    rows = []
    for row in read_jsonl(jsonl_path):
        if split and row.get("split") != split:
            continue
        if row.get("image_path"):
            rows.append(row)
    return rows[:max_examples]


def build_description_messages(mode):
    prompt = (
        "Describe this image in one short sentence. "
        "Mention the main objects and scene. Do not answer TRUE or FALSE."
    )

    if mode == "trl_current":
        # This mimics your current TRL dataset prompt style:
        # prompt is pure text, image is provided separately to processor.
        content = prompt

    elif mode == "explicit_image_token":
        # This is the explicit Qwen-style multimodal chat format.
        content = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return [{"role": "user", "content": content}]


def generate(model, processor, messages, image, device, max_new_tokens=64):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model_path", default="models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["trl_current", "explicit_image_token"],
        default="trl_current",
    )
    args = parser.parse_args()

    rows = load_rows(args.jsonl, args.split, args.max_examples)
    if len(rows) < 2:
        raise RuntimeError("Need at least 2 rows.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print("Device:", device)
    print("Dtype:", dtype)
    print("Rows:", len(rows))
    print("Mode:", args.mode)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    same_count = 0

    for i, row in enumerate(rows):
        wrong_row = rows[(i + 1) % len(rows)]

        correct_image = Image.open(row["image_path"]).convert("RGB")
        wrong_image = Image.open(wrong_row["image_path"]).convert("RGB")

        messages = build_description_messages(args.mode)

        out_correct = generate(
            model=model,
            processor=processor,
            messages=messages,
            image=correct_image,
            device=device,
        )

        out_wrong = generate(
            model=model,
            processor=processor,
            messages=messages,
            image=wrong_image,
            device=device,
        )

        if out_correct == out_wrong:
            same_count += 1

        print("=" * 80)
        print(f"Example {i}")
        print("Correct image:", row["image_path"])
        print("Wrong image:", wrong_row["image_path"])
        print("Description with CORRECT image:", repr(out_correct))
        print("Description with WRONG image:  ", repr(out_wrong))
        print("Same output?", out_correct == out_wrong)

        correct_image.close()
        wrong_image.close()

    print("=" * 80)
    print(f"Same outputs: {same_count}/{len(rows)}")

    if same_count == len(rows):
        print("WARNING: descriptions were identical for every image swap.")
        print("This suggests the image may not be connected correctly.")
    else:
        print("Good sign: descriptions changed when the image changed.")


if __name__ == "__main__":
    main()