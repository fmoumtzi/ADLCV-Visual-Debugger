import json
import argparse
import os

def main(args):
    correct_count = 0
    hallucinated_count = 0
    failed_count = 0
    total_lines = 0
    
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            total_lines += 1
            item = json.loads(line)
            eval_data = item.get("evaluation", {})
            reasoning = eval_data.get("reasoning", "")
            
            # 1. Filter out failed parses
            if "Failed to parse JSON output" in reasoning:
                failed_count += 1
                continue
                
            # 2. Determine correctness based on JSON format
            if "is_correct" in eval_data:
                # New Binary Format
                if eval_data["is_correct"]:
                    correct_count += 1
                else:
                    hallucinated_count += 1
            elif "errors" in eval_data:
                # Old Taxonomy Format (If ANY error is true, it's a hallucination)
                has_error = any(eval_data["errors"].values())
                if has_error:
                    hallucinated_count += 1
                else:
                    correct_count += 1
            else:
                # Unrecognized format
                failed_count += 1 
                
    valid_total = correct_count + hallucinated_count
    
    print("=" * 45)
    print(f" Dataset Analysis: {os.path.basename(args.input_jsonl)}")
    print("=" * 45)
    print(f"Total lines processed : {total_lines}")
    print(f"Failed to Parse       : {failed_count} ({(failed_count/max(total_lines, 1))*100:.1f}%)")
    print(f"Valid Evaluations     : {valid_total} ({(valid_total/max(total_lines, 1))*100:.1f}%)")
    print("-" * 45)
    
    if valid_total > 0:
        correct_pct = (correct_count / valid_total) * 100
        hallucinated_pct = (hallucinated_count / valid_total) * 100
        print(f"Correct Answers       : {correct_count} ({correct_pct:.1f}%)")
        print(f"Hallucinations        : {hallucinated_count} ({hallucinated_pct:.1f}%)")
        print("-" * 45)
        
        # SFT/GRPO split analysis
        target_size = min(correct_count, hallucinated_count)
        print(f"\n[Training Split Analysis]")
        print(f"To create a perfectly balanced 50/50 dataset, you can sample:")
        print(f"- {target_size} Correct examples")
        print(f"- {target_size} Hallucinated examples")
        print(f"Total perfectly balanced dataset size: {target_size * 2} rows")
        
        if valid_total < 5000:
            print("\nNote: You might want to wait until the evaluation finishes before sampling!")
    else:
        print("No valid evaluations found!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to your evaluations file")
    args = parser.parse_args()
    main(args)