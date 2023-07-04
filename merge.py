"""
Merge LoRA weights with the base model.

Usage: 
    python3 merge.py \
        --base_model_name_or_path BASE_MODEL_NAME_OR_PATH \
        --peft_model_path PEFT_MODEL_PATH \
        --output_dir OUTPUT_DIR \
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_model_path, adapter_path, output_dir):
    print(f"Loading base model: {base_model_path}")
    device_arg = {"device_map": "auto"}
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **device_arg,
    )

    print(f"Loading PEFT: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, trust_remote_code=True)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Loading tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Saving to {output_dir}")
    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    merge(args.base_model_name_or_path, args.peft_model_path, args.output_dir)
