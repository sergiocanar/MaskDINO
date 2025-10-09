import os 
import json
import argparse
from utils import load_json, save_json
from glob import glob
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

def join_coco_json(json_path_lt: list, save_dir: str):
    
    final_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for json_path_file in json_path_lt:
        json_data = load_json(coco_json_path=json_path_file)
        
        final_dict["images"].extend(json_data["images"])
        final_dict["annotations"].extend(json_data["annotations"])
        final_dict["categories"] = json_data["categories"]       
    
    save_dir = path_join(save_dir, 'endoscapes201_bbox.json')
    save_json(data_dict=final_dict,
              save_path=save_dir)
    
    print(f'Joined json to path: {save_dir}')    
    
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the reformatted JSON file.")
    args = parser.parse_args()

    indent_json(args.input_json, args.output_json)
    
    print(f"Reformatted JSON saved to {args.output_json}")
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # annot_dir = path_join(this_dir, '201_annotations')
    # output_path = path_join(this_dir, 'data', 'endoscapes', 'annotations_201')
    # path_lt = glob(path_join(annot_dir, "*.json"))
    # join_coco_json(json_path_lt=path_lt, save_dir=output_path)