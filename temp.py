import os 
import json
import argparse
from utils import load_json, save_json
from glob import glob
from os.path import join as path_join

def make_combined_trained_json(all_json: dict, train_json: dict, val_json: dict, test_json: dict):
    # Collect all image file names in each split
    val_files = {img["file_name"] for img in val_json["images"]}
    test_files = {img["file_name"] for img in test_json["images"]}
    train_files = {img["file_name"] for img in train_json["images"]}

    final_dict = {
        "images": [],
        "annotations": [],
        "categories": all_json["categories"]
    }

    # Add all training images and annotations
    final_dict["images"].extend(train_json["images"])
    final_dict["annotations"].extend(train_json["annotations"])

    # Add remaining images (those not in val or test)
    val_test_files = val_files.union(test_files)

    for img_dict in all_json["images"]:
        fname = img_dict["file_name"]
        if fname not in val_test_files and fname not in train_files:
            final_dict["images"].append(img_dict)
            # Add corresponding annotations from all_json
            img_id = img_dict["id"]
            anns = [a for a in all_json["annotations"] if a["image_id"] == img_id]
            final_dict["annotations"].extend(anns)

    return final_dict



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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    # parser.add_argument("--output_json", type=str, required=True, help="Path to save the reformatted JSON file.")
    # args = parser.parse_args()

    # indent_json(args.input_json, args.output_json)
    
    # print(f"Reformatted JSON saved to {args.output_json}")
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # annot_dir = path_join(this_dir, '201_annotations')
    # output_path = path_join(this_dir, 'data', 'endoscapes', 'annotations_201')
    # path_lt = glob(path_join(annot_dir, "*.json"))
    # join_coco_json(json_path_lt=path_lt, save_dir=output_path)
    all_json = load_json('data/endoscapes/calculate_masks/all_seg_201.json')
    train_json = load_json('/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations/train_annotation_coco.json')
    val_json = load_json('/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations/val_annotation_coco.json')
    test_json = load_json('/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations/test_annotation_coco.json')
    
    final_train_json = make_combined_trained_json(all_json, train_json, val_json, test_json)
    save_json(final_train_json, '/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations_201/train_annotation_coco.json')