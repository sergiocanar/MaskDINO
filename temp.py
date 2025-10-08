import os 
import json
import argparse
from utils import load_json
from os.path import join as path_join

def indent_json(input_json_path, output_json_path):
    """
    Reformat a JSON file with indentation for better readability.

    Parameters:
    - input_json_path: path to the input JSON file.
    - output_json_path: path to save the reformatted JSON file.
    """
    # Load the JSON data
    data = load_json(input_json_path)

    # Save the JSON data with indentation
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the reformatted JSON file.")
    args = parser.parse_args()

    indent_json(args.input_json, args.output_json)
    
    print(f"Reformatted JSON saved to {args.output_json}")