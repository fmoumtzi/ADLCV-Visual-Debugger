import argparse
import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_judge_model(model_name):
    """Loads the pure-text LLM Judge dynamically."""
    print(f"Loading Judge Model '{model_name}' into VRAM...")
    
    local_model_path = os.path.join(".", "models", model_name)
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Error: Could not find '{local_model_path}'.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=compute_dtype,
        attn_implementation=attention_mech,
        device_map="cuda"
    )
    
    # Set pad token if not set (fixes some Llama/Prometheus warnings)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer, device

def build_evaluation_prompt(ground_truth, vlm_generation, prompt_type):
    """Constructs the prompt based on the selected strategy."""
    
    if prompt_type == "taxonomy_json":
        system_prompt = """You are a strict, objective AI evaluator. You output ONLY valid JSON. You never output conversational text, preambles, or markdown formatting outside of the JSON block."""
        user_prompt = f"""
Ground Truth Annotation: "{ground_truth}"
VLM Generation: "{vlm_generation}"

Task: Analyze the VLM Generation for hallucinations or errors compared to the Ground Truth. 
Evaluate across these 6 categories:
1. Object: Did it mention an object that isn't there, or miss a core object?
2. Attribute: Did it get a color, material, or state wrong?
3. Spatial: Did it hallucinate the position or layout?
4. Counting: Did it count something incorrectly?
5. Text: Did it misread written text?
6. Action: Did it hallucinate an action or movement?

You MUST output ONLY a valid JSON object with a short rationale and a boolean dictionary where 'true' means an error occurred. Do not include any other text.
Format:
{{
  "rationale": "short explanation here",
  "errors": {{
    "object": false,
    "attribute": false,
    "spatial": false,
    "counting": false,
    "text": false,
    "action": false
  }}
}}"""
    
    elif prompt_type == "binary":
        system_prompt = "You are a strict evaluator. Reply with ONLY 'YES' or 'NO'."
        user_prompt = f"""
Ground Truth: {ground_truth}
VLM Output: {vlm_generation}

Is there anything in the VLM output that contradicts the Ground Truth? Answer ONLY YES or NO."""

    return system_prompt, user_prompt

def extract_json_from_text(text):
    """Safely extracts JSON from LLM output, even if it adds conversational filler."""
    try:
        # Find everything between the first { and the last }
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Phase 2: LLM Judge Evaluation")
    parser.add_argument("--judge_model", type=str, default="Prometheus-2-7B", help="Folder name of the Judge LLM")
    parser.add_argument("--exp_name", type=str, required=True, help="The experiment folder to read from and write to")
    parser.add_argument("--eval_prompt_type", type=str, choices=["taxonomy_json", "binary"], default="taxonomy_json")
    
    args = parser.parse_args()

    # Define paths
    exp_dir = os.path.join("results", args.exp_name)
    input_file = os.path.join(exp_dir, "generations.jsonl")
    output_file = os.path.join(exp_dir, "evaluations.jsonl")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find {input_file}. Run Phase 1 first!")

    print(f"=== Starting Evaluation for: {args.exp_name} ===")
    
    # Load Model
    model, tokenizer, device = load_judge_model(args.judge_model)

    # Read Generations
    with open(input_file, 'r') as f:
        generations = [json.loads(line) for line in f]

    results = []
    
    # Evaluation Loop
    for item in tqdm(generations, desc="Judging Responses"):
        gt = item.get('vqa_ground_truth', '')
        gen = item.get('vlm_generation', '')
        
        sys_p, usr_p = build_evaluation_prompt(gt, gen, args.eval_prompt_type)
        
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
            output_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            judge_response = tokenizer.batch_decode(output_ids_trimmed, skip_special_tokens=True)[0]

        # Process output based on prompt type
        parsed_evaluation = judge_response # Default fallback
        if args.eval_prompt_type == "taxonomy_json":
            extracted_json = extract_json_from_text(judge_response)
            if extracted_json:
                parsed_evaluation = extracted_json
            else:
                print(f"Failed to parse JSON for image {item['image_id']}")
                parsed_evaluation = {"error": "Failed to parse LLM output", "raw": judge_response}

        # Append evaluation to the original item data
        evaluated_item = item.copy()
        evaluated_item["judge_model"] = args.judge_model
        evaluated_item["evaluation_type"] = args.eval_prompt_type
        evaluated_item["evaluation"] = parsed_evaluation
        
        results.append(evaluated_item)
        
    # Save Evaluations
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Phase 2 complete. Saved evaluations to {output_file}")

if __name__ == "__main__":
    main()