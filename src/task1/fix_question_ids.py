import json
import argparse

def main(args):
    print(f"Loading raw questions from {args.raw_questions}...")
    
    # 1. Load the official VQA questions file
    with open(args.raw_questions, 'r') as f:
        data = json.load(f)
        questions_list = data["questions"]
        
    # 2. Build the lookup dictionary
    # Key: (image_id, question_text) -> Value: question_id
    question_map = {}
    for q in questions_list:
        key = (str(q["image_id"]), q["question"].strip())
        question_map[key] = q["question_id"]

    print(f"Built lookup table with {len(question_map)} unique questions.")
    print(f"Injecting IDs into {args.output_data}...")
    
    fixed_count = 0
    missing_count = 0
    
    # 3. Read generations, inject ID, write to new test file
    with open(args.generated_data, 'r') as infile, open(args.output_data, 'w') as outfile:
        for line in infile:
            item = json.loads(line)
            
            image_id = str(item["image_id"])
            question_text = item["vqa_question"].strip()
            key = (image_id, question_text)
            
            if key in question_map:
                item["question_id"] = question_map[key]
                fixed_count += 1
            else:
                missing_count += 1
                
            outfile.write(json.dumps(item) + "\n")

    print("-" * 30)
    print(f"Success! Fixed {fixed_count} lines.")
    if missing_count > 0:
        print(f"[WARNING] Could not find IDs for {missing_count} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_questions", type=str, required=True, help="Path to v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--generated_data", type=str, required=True, help="Path to your VLM generations file")
    parser.add_argument("--output_data", type=str, required=True, help="Path to save the updated test file")
    args = parser.parse_args()
    main(args)