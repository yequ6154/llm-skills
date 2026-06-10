#!/usr/bin/env python3
"""
Script to modify model keys in quant_model_description.json
1. Replace 'model.language_model.*' with 'model.*'
2. Replace 'model.vision_tower.vision_model.*' with 'vision_tower.vision_model.*'
"""

import json
import re
import argparse
from pathlib import Path

def transform_keys(data):
    """
    Transform dictionary keys according to the rules:
    - model.language_model.* -> model.*
    - model.vision_tower.vision_model.* -> vision_tower.vision_model.*
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_key = key
            
            # Rule 1: Replace 'model.language_model.' with 'model.'
            if key.startswith('model.language_model.'):
                new_key = key.replace('model.language_model.', 'model.', 1)
                print(f"Transform: {key} -> {new_key}")
            
            # Rule 2: Replace 'model.vision_tower.vision_model.' with 'vision_tower.vision_model.'
            elif key.startswith('model.vision_tower.vision_model.'):
                new_key = key.replace('model.vision_tower.vision_model.', 'vision_tower.vision_model.', 1)
                print(f"Transform: {key} -> {new_key}")
            
            # Recursively process nested dictionaries
            new_dict[new_key] = transform_keys(value)
        return new_dict
    elif isinstance(data, list):
        return [transform_keys(item) for item in data]
    else:
        return data

def transform_keys_regex(data):
    """
    Alternative implementation using regular expressions
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_key = key
            
            # Using regex for more precise replacement
            # Replace 'model.language_model.' with 'model.' (only at the beginning)
            new_key = re.sub(r'^model\.language_model\.', 'model.', new_key)
            
            # Replace 'model.vision_tower.vision_model.' with 'vision_tower.vision_model.' (only at the beginning)
            new_key = re.sub(r'^model\.vision_tower\.vision_model\.', 'vision_tower.vision_model.', new_key)
            
            if new_key != key:
                print(f"Transform: {key} -> {new_key}")
            
            new_dict[new_key] = transform_keys_regex(value)
        return new_dict
    elif isinstance(data, list):
        return [transform_keys_regex(item) for item in data]
    else:
        return data

def main():
    parser = argparse.ArgumentParser(description='Transform model keys in JSON file')
    parser.add_argument('input_file', type=str, help='Input JSON file path')
    parser.add_argument('-o', '--output_file', type=str, default=None, 
                        help='Output JSON file path (default: input_file_transformed.json)')
    parser.add_argument('--in-place', action='store_true',
                        help='Modify the file in place')
    parser.add_argument('--use-regex', action='store_true',
                        help='Use regex-based transformation (alternative implementation)')
    
    args = parser.parse_args()
    
    # Read input JSON file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return
    
    print(f"Processing {args.input_file}...")
    print("-" * 50)
    
    # Transform keys
    if args.use_regex:
        transformed_data = transform_keys_regex(data)
    else:
        transformed_data = transform_keys(data)
    
    print("-" * 50)
    
    # Determine output file path
    if args.in_place:
        output_path = input_path
        print(f"Modifying file in place: {output_path}")
    elif args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / f"{input_path.stem}_transformed.json"
    
    # Write transformed JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nSuccess! Transformed file saved to: {output_path}")
    
    # Show statistics
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    original_keys = list(data.keys())
    new_keys = list(transformed_data.keys())
    
    model_language_count = sum(1 for k in original_keys if k.startswith('model.language_model.'))
    model_vision_count = sum(1 for k in original_keys if k.startswith('model.vision_tower.vision_model.'))
    
    print(f"Keys starting with 'model.language_model.*': {model_language_count}")
    print(f"Keys starting with 'model.vision_tower.vision_model.*': {model_vision_count}")
    print(f"Total keys transformed: {model_language_count + model_vision_count}")
    print(f"Total keys in output: {len(new_keys)}")

if __name__ == "__main__":
    main()