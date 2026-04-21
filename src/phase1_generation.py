import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def setup_experiment_dir(exp_name):
    """Creates the results directory if it doesn't exist."""
    base_dir = os.path.join("results", exp_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_dataset(dataset_name, num_samples):
    """Parses the actual VQA or COCO JSON files and maps them to image paths."""
    print(f"Loading up to {num_samples} samples from the {dataset_name.upper()} validation split...")
    data = []
    
    img_dir = "data/coco/val2014"
    
    if dataset_name == "vqa":
        # Paths to VQA v2 files
        q_path = "data/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
        a_path = "data/vqa/v2_mscoco_val2014_annotations.json"
        
        with open(q_path, 'r') as f:
            questions = json.load(f)['questions']
        with open(a_path, 'r') as f:
            annotations = json.load(f)['annotations']
            
        # Create a fast lookup dictionary for answers based on question_id
        answers_dict = {ann['question_id']: ann['multiple_choice_answer'] for ann in annotations}
        
        for q in questions:
            q_id = q['question_id']
            img_id = q['image_id']
            # COCO images are padded with 12 zeros
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
        # Path to COCO captions
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
                "question": "Describe this image in detail.", # Base prompt for pure captioning
                "ground_truth": ann['caption']
            })
            if len(data) >= num_samples:
                break
                
    return data

def load_model(model_name):
    """Loads the VLM into the A100 GPU using bfloat16 and Flash Attention."""
    print(f"Loading {model_name} into VRAM...")
    
    # Point to the local directory where your script downloaded the weights
    local_model_path = os.path.join(".", "models", model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- DYNAMIC HARDWARE CHECK ---
    if device == "cuda" and torch.cuda.is_bf16_supported():
        print("GPU (A100/H100). Enabling bfloat16 and Flash Attention.")
        compute_dtype = torch.bfloat16
        attention_mech = "flash_attention_2"
    else:
        print("GPU (V100) or CPU detected. Falling back to float16 and SDPA.")
        compute_dtype = torch.float16
        attention_mech = "sdpa"

    # Load the model with the dynamic variables
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=compute_dtype,
        attn_implementation=attention_mech,
        device_map="cuda" # 'cuda' automatically binds to the visible device
    )
    
    processor = AutoProcessor.from_pretrained(local_model_path)
    
    return model, processor, device

def main():
    parser = argparse.ArgumentParser(description="Phase 1: VLM Generation")
    parser.add_argument("--vlm_model", type=str, default="Qwen2.5-VL-3B-Instruct", help="Folder name of the VLM to use")
    parser.add_argument("--dataset", type=str, choices=["vqa", "coco"], required=True, help="Dataset to evaluate")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images to process")
    parser.add_argument("--exp_name", type=str, required=True, help="Unique name for this experiment run")
    
    # NEW ARGUMENT: Prompt Type
    parser.add_argument("--prompt_type", type=str, choices=["structured", "direct"], default="structured", 
                        help="Choose 'structured' for the taxonomy breakdown, or 'direct' for just the dataset question.")
    
    args = parser.parse_args()

    # 1. Set up output directory
    output_dir = setup_experiment_dir(args.exp_name)
    output_file = os.path.join(output_dir, "generations.jsonl")
    
    print(f"=== Starting Experiment: {args.exp_name} ===")
    print(f"Prompt Strategy: {args.prompt_type.upper()}")
    
    # 2. Load Data
    dataset = load_dataset(args.dataset, args.num_samples)

    # 3. Load Model
    model, processor, device = load_model(args.vlm_model)

    # 4. Generation Loop
    results = []
    
    structured_prompt = """Analyze this image and provide a breakdown in the following format:
1. Main objects present
2. Their colors and materials
3. Their spatial layout
4. The exact count of key objects
5. Any text visible
6. The main actions taking place."""

    for item in tqdm(dataset, desc="Generating Descriptions"):
        
        if args.prompt_type == "structured":
            if args.dataset == "vqa":
                final_prompt = f"{structured_prompt}\n\nAdditionally, answer this specific question: {item['question']}"
            else:
                final_prompt = structured_prompt
        elif args.prompt_type == "direct":
            # Just ask the VQA question directly (or the base COCO caption prompt)
            final_prompt = item['question']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item['image_path']},
                    {"type": "text", "text": final_prompt},
                ],
            }
        ]

        # Prepare inputs using Qwen's specific processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
            # Trim the prompt tokens from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # Save the result
        results.append({
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "dataset": args.dataset,
            "prompt_type": args.prompt_type, # Saving this so you know which prompt was used
            "vqa_question": item["question"],
            "vqa_ground_truth": item["ground_truth"],
            "vlm_generation": output_text
        })
        
        # Free memory during the loop
        torch.cuda.empty_cache()

    # 5. Save to the specific experiment folder
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Phase 1 complete. Saved {len(results)} generations to {output_file}")

if __name__ == "__main__":
    main()