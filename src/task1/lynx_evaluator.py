import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def main(args):
    # 1. Load Tokenizer (Forcing left-padding for batched inference)
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  

    # 2. 4-bit Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 3. Load Model (Using SDPA to prevent attention bugs)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation="sdpa"
    )
    
    # 4. Load Data and Inject Traceability IDs
    generations = []
    with open(args.input_jsonl, 'r') as f:
        for row_index, line in enumerate(f):
            item = json.loads(line)
            # Ensure full traceability even if question_id is missing
            if "question_id" not in item:
                item["trace_id"] = f"row_{row_index}"
            generations.append(item)
            
    batch_size = 32  # You can safely increase this to 64 since we aren't loading images!
    
    # 5. Batched Evaluation Loop
    for i in tqdm(range(0, len(generations), batch_size), desc=f"Evaluating with {os.path.basename(args.judge_model_path)}"):
        batch = generations[i:i + batch_size]
        text_prompts = []
        
        for item in batch:
            # The Exhaustive Search Prompt
            system_instruction = (
                "You are an expert text-evaluator judge. You will be provided with a Question, a Vision-Language Model's (VLM) Answer, and a list of Ground Truth answers.\n\n"
                "CONTEXT: The Ground Truth list contains answers from multiple human annotators. Humans often disagree! You will frequently see conflicting answers like both 'yes' and 'no' in the same list.\n\n"
                "Your task is to determine if the VLM Answer is CORRECT.\n"
                "Rules for correctness:\n"
                "1. It is CORRECT if its core meaning matches ANY single answer in the Ground Truths list, even if it contradicts other answers in the list.\n"
                "2. Map spelled-out numbers to digits (e.g., 'one' matches '1').\n"
                "3. Do NOT penalize the VLM if it answers in a full sentence. Extract the core claim.\n"
                "4. It is INCORRECT (a hallucination) ONLY if you have searched the ENTIRE list and found zero matches.\n\n"
                "Return STRICTLY valid JSON matching this exact format:\n"
                "{\n"
                "  \"reasoning\": \"<step-by-step logic: 1. Identify core claim. 2. Search the ENTIRE GT list for this claim. 3. Explicitly state if the claim is found ANYWHERE in the list. 4. Declare correct/incorrect.>\",\n"
                "  \"is_correct\": false\n"
                "}\n"
            )
            
            user_content = f"Question: {item['vqa_question']}\nVLM Generation: {item['vlm_generation']}\nGround Truths: {item['vqa_ground_truth']}"
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ]
            
            text_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            text_prompts.append(text_prompt)
            
        # Tokenize the whole batch text
        inputs = tokenizer(text_prompts, return_tensors="pt", padding=True).to(judge_model.device)
        
        with torch.no_grad():
            outputs = judge_model.generate(**inputs, max_new_tokens=150,pad_token_id=tokenizer.eos_token_id)
            
        # Decode only the newly generated tokens
        generated_texts = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 6. Clean JSON and Save Results
        with open(args.output_jsonl, 'a') as f:
            for item, response in zip(batch, generated_texts):
                clean_response = response.strip()
                
                # 1. Isolate the JSON block (Ignores conversational leading text like "Here is the JSON:")
                start_idx = clean_response.find('{')
                if start_idx != -1:
                    clean_response = clean_response[start_idx:]
                
                
                # 2. Fix Python boolean quirks (Do NOT replace quotes for Llama-3!)
                clean_response = clean_response.replace("True", "true").replace("False", "false")
                
                # 3. Auto-complete missing closing braces (Handles cutoff generations)
                open_braces = clean_response.count('{')
                close_braces = clean_response.count('}')
                if open_braces > close_braces:
                    clean_response += '}' * (open_braces - close_braces)
                    
                # 4. Strip trailing garbage (Removes trailing markdown or conversational text)
                end_idx = clean_response.rfind('}')
                if end_idx != -1:
                    clean_response = clean_response[:end_idx+1]
                
                try:
                    evaluation_json = json.loads(clean_response)
                except json.JSONDecodeError:
                    print(f"\n[WARNING] JSON Decode Failed. Raw Output: {clean_response}")
                    # Safe binary fallback
                    evaluation_json = {"reasoning": "Failed to parse JSON output.", "is_correct": False}
                    
                item["judge_model"] = os.path.basename(args.judge_model_path)
                item["evaluation_type"] = "taxonomy_json"
                item["evaluation"] = evaluation_json
                
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--judge_model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)