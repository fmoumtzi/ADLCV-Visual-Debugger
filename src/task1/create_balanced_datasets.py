import json
import argparse
import random
import os

def main(args):
    # Set a random seed so your "random" sampling is perfectly reproducible 
    # if you ever need to run this exact script again.
    random.seed(42)
    
    correct_items = []
    hallucinated_items = []
    
    print(f"Reading from {args.input_jsonl}...")
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            eval_data = item.get("evaluation", {})
            reasoning = eval_data.get("reasoning", "")
            
            # 1. Skip failed parses
            if "Failed to parse JSON output" in reasoning:
                continue
                
            # 2. Sort into buckets based on correctness
            is_correct = False
            if "is_correct" in eval_data:
                is_correct = eval_data["is_correct"]
            elif "errors" in eval_data:
                has_error = any(eval_data["errors"].values())
                is_correct = not has_error
            else:
                continue 
            
            if is_correct:
                correct_items.append(item)
            else:
                hallucinated_items.append(item)
                
    print(f"Found {len(correct_items)} Correct and {len(hallucinated_items)} Hallucinated examples.")
    
    # 3. Determine the bottleneck and sample
    target_size = min(len(correct_items), len(hallucinated_items))
    print(f"Sampling {target_size} from each class for a 50/50 split...")
    
    sampled_correct = random.sample(correct_items, target_size)
    sampled_hallucinated = random.sample(hallucinated_items, target_size)
    
    # 4. Combine and thoroughly shuffle the final dataset
    balanced_dataset = sampled_correct + sampled_hallucinated
    random.shuffle(balanced_dataset)
    
    # 5. Save to the new file
    print(f"Saving {len(balanced_dataset)} perfectly balanced rows to {args.output_jsonl}...")
    with open(args.output_jsonl, 'w') as f:
        for item in balanced_dataset:
            f.write(json.dumps(item) + "\n")
            
    print("Done!\n" + "="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to your evaluated generations file")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the new balanced dataset")
    args = parser.parse_args()
    main(args)