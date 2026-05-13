import os
from PIL import Image
import json
import argparse
from tqdm import tqdm
from transformers import AutoProcessor
import torch

def main(args):
    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # 2. Foolproof Architecture Selection
    if "SmolVLM" in args.model_path:
        from transformers import Idefics3ForConditionalGeneration as VLMClass
    elif "Qwen" in args.model_path:
        from transformers import Qwen2VLForConditionalGeneration as VLMClass
    else:
        from transformers import AutoModelForCausalLM as VLMClass

    # 3. Load Model
    model = VLMClass.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Load Local Dataset
    with open(args.input_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Slice to 15k
    data = data[:15000]
    
    results = []
    
    # Generation Loop (Simplified for single-item iteration, batching requires custom collators for VLMs)
    for item in tqdm(data, desc="Generating Responses"):
        image_path = f"/dtu/blackhole/0d/223821/visual_debugger_project/ADLCV-Visual-Debugger/{item['image_path']}"
        question = item['vqa_question']
        
        # 1. Load the image explicitly using PIL
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path} due to image load error: {e}")
            continue
        
        # 2. Standardize the message format (just specify "type": "image")
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, 
                {"type": "text", "text": question}
            ]}
        ]
        
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # 3. Pass the PIL Image object (raw_image) instead of the string path
        inputs = processor(images=[raw_image], text=[text_prompt], return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
            
        generated_text = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Structure Output
        result_item = {
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "dataset": "vqa",
            "prompt_type": "structured",
            "vqa_question": question,
            "vqa_ground_truth": item["vqa_ground_truth"], # Assumed to be a list of 10 strings now
            "vlm_generation": generated_text
        }
        results.append(result_item)
        
        # Stream write to prevent data loss
        with open(args.output_jsonl, 'a') as f:
            f.write(json.dumps(result_item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)