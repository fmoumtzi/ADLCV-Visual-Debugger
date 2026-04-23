import argparse
import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def setup_experiment_dir(exp_name):
    base_dir = os.path.join("results", exp_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_dataset(dataset_name, num_samples):
    print(f"Loading up to {num_samples} samples from the {dataset_name.upper()} validation split...")
    data = []
    img_dir = "data/coco/val2014"
    
    if dataset_name == "vqa":
        q_path = "data/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
        a_path = "data/vqa/v2_mscoco_val2014_annotations.json"
        
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']
        with open(a_path, 'r') as f:
            annotations = json.load(f)['annotations']
            
        answers_dict = {ann['question_id']: ann['multiple_choice_answer'] for ann in annotations}
        
        for q in questions:
            q_id = q['question_id']
            img_id = q['image_id']
            img_filename = f"COCO_val2014_{img_id:012d}.jpg" 
            img_path = os.path.join(img_dir, img_filename)
            
            data.append({
                "image_id": img_id,
                "image_path": img_path,
                "question": q['question'],
                "ground_truth": answers_dict.get(q_id, "")
            })
            if len(data) >= num_samples:
                break
                
    elif dataset_name == "coco":
        c_path = "data/coco/annotations/captions_val2014.json"
        with open(c_path, 'r') as f:
            annotations = json.load(f)['annotations']
            
        for ann in annotations:
            img_id = ann['image_id']
            img_filename = f"COCO_val2014_{img_id:012d}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            
            data.append({
                "image_id": img_id,
                "image_path": img_path,
                "question": "Describe this image in detail.",
                "ground_truth": ann['caption']
            })
            if len(data) >= num_samples:
                break
                
    return data

def load_model(model_name):
    print(f"Loading {model_name} into VRAM...")
    local_model_path = os.path.join(".", "models", model_name)
    
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Error: Could not find '{local_model_path}'.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Bulletproof hardware check
    compute_dtype = torch.float16
    attention_mech = "sdpa"
    
    if device == "cuda":
        gpu_major_version, _ = torch.cuda.get_device_capability()
        if gpu_major_version >= 8:
            try:
                import flash_attn
                print("Enabling bfloat16 and Flash Attention.")
                compute_dtype = torch.bfloat16
                attention_mech = "flash_attention_2"
            except ImportError:
                print("Falling back to float16 and SDPA.")
        else:
            print(f"Falling back to float16 and SDPA.")

    # Determine the correct model class based on the name
    ModelClass = Qwen2_5_VLForConditionalGeneration if "Qwen" in model_name else AutoModelForImageTextToText

    model = ModelClass.from_pretrained(
        local_model_path,
        torch_dtype=compute_dtype,
        attn_implementation=attention_mech,
        device_map="cuda" 
    )
    processor = AutoProcessor.from_pretrained(local_model_path)
    
    return model, processor, device

def main():
    parser = argparse.ArgumentParser(description="Phase 1: VLM Generation")
    parser.add_argument("--vlm_model", type=str, default="Qwen2.5-VL-3B-Instruct", help="Folder name of the VLM to use")
    parser.add_argument("--dataset", type=str, choices=["vqa", "coco"], required=True, help="Dataset to evaluate")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images to process")
    parser.add_argument("--exp_name", type=str, required=True, help="Unique name for this experiment run")
    parser.add_argument("--prompt_type", type=str, choices=["structured", "direct"], default="structured", 
                        help="'structured' for taxonomy, 'direct' for base question.")
    
    args = parser.parse_args()

    output_dir = setup_experiment_dir(args.exp_name)
    output_file = os.path.join(output_dir, "generations.jsonl")
    
    print(f"=== Starting Experiment: {args.exp_name} ===")
    dataset = load_dataset(args.dataset, args.num_samples)
    model, processor, device = load_model(args.vlm_model)

    results = []
    structured_prompt = """Analyze this image and provide a breakdown in the following format:
1. Main objects present
2. Their colors and materials
3. Their spatial layout
4. The exact count of key objects
5. Any text visible
6. The main actions taking place."""

    for item in tqdm(dataset, desc="Generating Descriptions"):
        # Image loading and prompt construction
        raw_image = Image.open(item['image_path']).convert("RGB")
        
        if args.prompt_type == "structured":
            final_prompt = f"{structured_prompt}\n\nAdditionally, answer this specific question: {item['question']}" if args.dataset == "vqa" else structured_prompt
        else:
            final_prompt = item['question']

        # DYNAMIC ROUTING: Qwen vs SmolVLM
        if "Qwen" in args.vlm_model:
            messages = [
                {"role": "user", "content": [{"type": "image", "image": item['image_path']}, {"type": "text", "text": final_prompt}]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
            
        else: # SmolVLM
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_prompt}]}
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=[raw_image], return_tensors="pt").to(device)

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        results.append({
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "dataset": args.dataset,
            "prompt_type": args.prompt_type,
            "vqa_question": item["question"],
            "vqa_ground_truth": item["ground_truth"],
            "vlm_generation": output_text
        })
        torch.cuda.empty_cache()

    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Phase 1 complete. Saved {len(results)} generations to {output_file}")

if __name__ == "__main__":
    main()