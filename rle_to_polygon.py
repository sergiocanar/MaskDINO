import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_json
from pycocotools import mask

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

def rle_to_polygon(coco_json_path, output_json_path):
    """
    Convert RLE annotations to polygon format in a COCO JSON.

    Parameters:
    - coco_json_path: path to the input COCO JSON with RLE annotations.
    - output_json_path: path to save the modified COCO JSON with polygon annotations.
    """
    # Load the COCO JSON
    data = load_json(coco_json_path)

    valid_annotations = []
    # Iterate through the annotations
    for ann in tqdm(data['annotations']):
        if 'segmentation' in ann and type(ann['segmentation']) is not list:
            
            # Decode RLE to binary mask
            binary_mask = mask.decode(ann['segmentation'])
            
            # Convert binary mask to contours using OpenCV
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []
            for contour in contours:
                contour = contour.squeeze().ravel().tolist()

                # Exclude invalid polygons
                if len(contour)%2 == 0 and len(contour) >= 6:
                    segmentation.append(contour)
            
            if len(segmentation) == 0:
                continue
            else:                
                # Replace RLE with polygon
                ann['segmentation'] = segmentation
                valid_annotations.append(ann)
    
    
    data["annotations"] = valid_annotations
    
    # Save the modified COCO JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Parse for Endoscapes2023 annotation files
    for split in ['train','val','test']:
        input_coco_json = os.path.join(args.data_path, f"{split}_annotation_coco.json")
        output_coco_json = os.path.join(args.data_path, f"{split}_annotation_coco_polygon.json")
        rle_to_polygon(input_coco_json, output_coco_json)
