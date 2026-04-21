import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_DIR = os.path.join("models", "Qwen2.5-VL-3B-Instruct")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 32
SUPPORTED_SPLITS = {"train", "val"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ask Qwen2.5-VL VQA v2 questions and save the VLM answers next to "
            "the target multiple_choice_answer annotations."
        )
    )
    parser.add_argument(
        "--split",
        choices=sorted(SUPPORTED_SPLITS),
        default="val",
        help="VQA/COCO 2014 split to run. Defaults to val.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of VQA questions to process.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Question index to start from before applying num_samples.",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory created by setup_and_download.sh.",
    )
    parser.add_argument(
        "--model_path",
        default=MODEL_DIR,
        help=(
            "Local Qwen model directory. If it does not exist, the script falls "
            "back to the Hugging Face model id."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/qwen25_vl_vqa_mc_answers.jsonl",
        help="JSONL file where answers will be written.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of new tokens Qwen can generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic greedy decoding.",
    )
    parser.add_argument(
        "--skip_missing_images",
        action="store_true",
        help="Skip samples whose COCO image file is missing instead of failing.",
    )
    return parser.parse_args()


def split_prefix(split):
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    return f"{split}2014"


def resolve_vqa_paths(data_dir, split):
    prefix = split_prefix(split)
    vqa_dir = os.path.join(data_dir, "vqa")
    coco_dir = os.path.join(data_dir, "coco", prefix)

    return {
        "multiple_choice_questions": os.path.join(
            vqa_dir,
            f"v2_MultipleChoice_mscoco_{prefix}_questions.json",
        ),
        "open_ended_questions": os.path.join(
            vqa_dir,
            f"v2_OpenEnded_mscoco_{prefix}_questions.json",
        ),
        "annotations": os.path.join(vqa_dir, f"v2_mscoco_{prefix}_annotations.json"),
        "images": coco_dir,
    }


def require_file(path, setup_hint):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}\n{setup_hint}")


def require_dir(path, setup_hint):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing directory: {path}\n{setup_hint}")


def resolve_question_file(paths, setup_hint):
    candidates = [
        ("multiple_choice", paths["multiple_choice_questions"]),
        ("open_ended", paths["open_ended_questions"]),
    ]
    for question_file_type, path in candidates:
        if os.path.isfile(path):
            return path, question_file_type

    expected_paths = "\n".join(f"- {path}" for _, path in candidates)
    raise FileNotFoundError(
        f"Missing VQA question file. Expected one of:\n{expected_paths}\n{setup_hint}"
    )


def load_vqa_dataset(data_dir, split, num_samples, start_index, skip_missing_images):
    if num_samples <= 0:
        raise ValueError("--num_samples must be greater than 0")
    if start_index < 0:
        raise ValueError("--start_index must be 0 or greater")

    paths = resolve_vqa_paths(data_dir, split)
    setup_hint = "Run setup_and_download.sh to download the VQA v2 and COCO 2014 files."
    question_path, question_file_type = resolve_question_file(paths, setup_hint)

    require_file(paths["annotations"], setup_hint)
    require_dir(paths["images"], setup_hint)

    with open(question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(paths["annotations"], "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    annotations_by_question_id = {ann["question_id"]: ann for ann in annotations}
    selected_questions = questions[start_index:]
    data = []
    prefix = split_prefix(split)

    for question in selected_questions:
        question_id = question["question_id"]
        image_id = question["image_id"]
        image_path = os.path.join(paths["images"], f"COCO_{prefix}_{image_id:012d}.jpg")

        if not os.path.isfile(image_path):
            if skip_missing_images:
                continue
            raise FileNotFoundError(f"Missing image for question {question_id}: {image_path}")

        annotation = annotations_by_question_id.get(question_id, {})
        target_answer = annotation.get("multiple_choice_answer", "")
        annotator_answers = [
            answer.get("answer", "")
            for answer in annotation.get("answers", [])
            if answer.get("answer")
        ]

        data.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "image_path": image_path,
                "question": question["question"],
                "choices": question.get("multiple_choices", []),
                "target_answer": target_answer,
                "annotator_answers": annotator_answers,
                "answer_type": annotation.get("answer_type", ""),
                "question_type": annotation.get("question_type", ""),
                "question_file_type": question_file_type,
            }
        )

        if len(data) >= num_samples:
            break

    if not data:
        raise RuntimeError("No VQA samples were loaded. Check the split, start index, and image files.")

    return data


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

            attention = "flash_attention_2"
        except ImportError:
            attention = "sdpa"

        return {
            "torch_dtype": dtype,
            "attn_implementation": attention,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

    if device.type == "mps":
        return {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

    return {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }


def resolve_model_path(model_path):
    if os.path.isdir(model_path):
        return model_path
    return MODEL_ID


def load_model(model_path):
    device = resolve_runtime()
    resolved_model_path = resolve_model_path(model_path)
    model_load_config = resolve_model_load_config(device)
    if device.type == "cpu":
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = "1"

    print(f"Loading model: {resolved_model_path} on {device}")
    print(
        "Model settings:",
        f"dtype={model_load_config['torch_dtype']},",
        f"attention={model_load_config.get('attn_implementation', 'default')}",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        resolved_model_path,
        **model_load_config,
    )

    if "device_map" not in model_load_config:
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(resolved_model_path)
    return model, processor, device, resolved_model_path


def build_prompt(question, choices):
    if choices:
        formatted_choices = "\n".join(f"- {choice}" for choice in choices)
        return (
            "Answer the visual multiple-choice question using the image. "
            "Choose exactly one option and reply with only that option text.\n"
            f"Question: {question}\n"
            f"Options:\n{formatted_choices}"
        )

    return (
        "Answer the visual question using the image. "
        "Reply with only the short answer, without explanation.\n"
        f"Question: {question}"
    )


def ask_qwen(model, processor, device, image_path, prompt, max_new_tokens, temperature):
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
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    image.close()
    return output_text.strip()


def main():
    args = parse_args()
    dataset = load_vqa_dataset(
        args.data_dir,
        args.split,
        args.num_samples,
        args.start_index,
        args.skip_missing_images,
    )
    model, processor, device, resolved_model_path = load_model(args.model_path)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded {len(dataset)} VQA {args.split} samples.")
    print(f"Writing answers to {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Answering VQA questions"):
            prompt = build_prompt(item["question"], item["choices"])
            vlm_answer = ask_qwen(
                model,
                processor,
                device,
                item["image_path"],
                prompt,
                args.max_new_tokens,
                args.temperature,
            )
            
            hallucination = True
            if (vlm_answer.lower() == item["target_answer"].lower() or vlm_answer.lower() in [a.lower() for a in item["annotator_answers"]]):
                hallucination = False

            result = {
                "question_id": item["question_id"],
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "split": args.split,
                "question": item["question"],
                "choices": item["choices"],
                "prompt": prompt,
                "vlm_answer": vlm_answer,
                "target_answer": item["target_answer"],
                "annotator_answers": item["annotator_answers"],
                "answer_type": item["answer_type"],
                "question_type": item["question_type"],
                "question_file_type": item["question_file_type"],
                "model": resolved_model_path,
                "hallucination": hallucination,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Done. Saved {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
