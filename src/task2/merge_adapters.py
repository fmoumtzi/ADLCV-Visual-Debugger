import argparse
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to base model")
    parser.add_argument("--lora_model", required=True, help="Path to SFT LoRA adapter")
    parser.add_argument("--output_dir", required=True, help="Where to save merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )

    print(f"Loading LoRA adapter from: {args.lora_model}")
    model = PeftModel.from_pretrained(model, args.lora_model)

    print("Fusing weights... (this takes a minute)")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(args.output_dir)
    
    print("Merge complete!")

if __name__ == "__main__":
    main()