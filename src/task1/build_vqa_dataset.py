import json
import os
from tqdm import tqdm
import argparse

def main(args):
    print(f"Loading Questions from {args.questions_path}...")
    with open(args.questions_path, 'r') as f:
        questions_data = json.load(f)['questions']
        
    print(f"Loading Annotations from {args.annotations_path}...")
    with open(args.annotations_path, 'r') as f:
        annotations_data = json.load(f)['annotations']

    # Create a quick lookup dictionary mapping question_id -> annotation object
    print("Mapping annotations...")
    anno_dict = {anno['question_id']: anno for anno in annotations_data}

    print(f"Building combined JSONL at {args.output_path}...")
    with open(args.output_path, 'w') as out_file:
        for q in tqdm(questions_data, desc="Processing VQA"):
            q_id = q['question_id']
            img_id = q['image_id']
            question_text = q['question']
            
            # Get the matching annotation
            anno = anno_dict.get(q_id)
            if not anno:
                continue
                
            # Extract all annotator answers (usually exactly 10)
            ground_truth_list = [ans['answer'] for ans in anno['answers']]
            
            # Format the COCO image path (COCO uses 12-digit zero-padded IDs)
            # Example: COCO_val2014_000000393226.jpg
            img_filename = f"COCO_val2014_{img_id:012d}.jpg"
            img_path = f"data/coco/val2014/{img_filename}"
            
            # Build the exact schema the generate script expects
            record = {
                "image_id": img_id,
                "image_path": img_path,
                "dataset": "vqa",
                "prompt_type": "structured",
                "vqa_question": question_text,
                "vqa_ground_truth": ground_truth_list
            }
            
            out_file.write(json.dumps(record) + "\n")

    print("Dataset successfully built!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_path", type=str, default="data/vqa/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--annotations_path", type=str, default="data/vqa/v2_mscoco_val2014_annotations.json")
    parser.add_argument("--output_path", type=str, default="data/vqa_local_dataset.jsonl")
    args = parser.parse_args()
    main(args)