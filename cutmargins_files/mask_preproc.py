import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from os.path import join as path_join
from pycocotools import mask as mask_utils
from cut_black_margins import filter_black_endo2023
from utils import load_json, create_directory_if_not_exists

parser = argparse.ArgumentParser()

parser.add_argument("--annots_dir", type=str, default="annotations")
parser.add_argument("--split", type=str, default="train")

args = parser.parse_args()


def prepocess_masks(base_dir: str, cutted_dir: str,coco_json_dict: dict, coco_obj: COCO, output_path: str = None):
        
    #Get images information list
    img_info_lt = coco_obj.dataset['images']
    new_mask_dict = {
        "images": [],
        "annotations": [],
        "categories": coco_json_dict['categories']
    }
        
    with tqdm(total=len(img_info_lt), desc='Processing frames...', unit='frame') as pbar:

        #Iterate over info dict
        for img_info in img_info_lt:
            
            file_name = img_info['file_name']
            
            if 'video_id' not in list(img_info.keys()):
                video_name = int(file_name.split('_')[0])
            else:
                video_name = img_info['video_id']            
            #Get relevant information
            h,w = img_info['height'], img_info['width']
            
            new_img_info = {
                "id": img_info['id'],
                "file_name": file_name,
                "height": None,
                "width": None,
                "video_name": video_name,
                "frame_id": img_info['id']
            }
            
            #Key for dictionary
            img_path = path_join(base_dir, f'{file_name}')
            cutted_img_path = path_join(cutted_dir, f'{file_name}')
            
            if not os.path.exists(cutted_img_path):
                print(f'Skipping: {cutted_img_path}. Eliminated img.')
                pbar.update(1)
                continue
            
            
            #Load image
            img = cv2.imread(img_path)
            cutted_img = cv2.imread(cutted_img_path)
            h_targ, w_targ, _ = cutted_img.shape
            
            new_img_info["height"] = h_targ
            new_img_info["width"] = w_targ
            
            #Add img to new_dict
            new_mask_dict['images'].append(new_img_info)
            
            #load image annotations
            ann_ids = coco_obj.getAnnIds(imgIds=img_info['id'])
            anns = coco_obj.loadAnns(ann_ids)
            
            if len(anns) == 0:
                print(f'Skipping:{cutted_img_path}. No annotations.')
                pbar.update(1)
                continue
            
            for ann in anns:
                
                if "iscrowd" not in list(ann.keys()):
                    is_crowd = 0
                else: 
                    is_crowd = ann['iscrowd']
                    
                new_ann = {
                    "id": ann['id'],
                    "image_id": ann['image_id'],
                    "segmentation": None,
                    "iscrowd": is_crowd,
                    "bbox": None,
                    "area": None,
                    "category_id": ann['category_id']
                }
                
                #Initizalize mask 
                segm_mask = coco_obj.annToMask(ann)
                mask = np.zeros_like(segm_mask, dtype=np.uint8)
                if segm_mask.shape != mask.shape:
                    print(f"[Shape mismatch] Image ID: {img_info['id']}, Annotation ID: {ann['id']}")
                    print(f"segm_mask.shape: {segm_mask.shape}, mask.shape: {mask.shape}")
                    # breakpoint()
                
                mask[segm_mask == 1] = ann['category_id'] 
                                
                _, coords = filter_black_endo2023(image=img)
                
                #Cut the mask
                if coords is not None:
                    l, r, b, t = coords
                    new_mask = mask[l:r, b:t]
                else:
                    print(f'Mask not cutted for frame: {file_name}')
                    new_mask = segm_mask
                                
                if new_mask is None or new_mask.size == 0:
                    print(f'Invalid mask for: Annotation:{ann['id']} in Image: {ann['image_id']}')
                    continue 
                                
                if new_mask.shape != (h_targ, w_targ):
                    # breakpoint()
                    print(f"Shape mismatch for image {img_info['file_name']} (ID: {img_info['id']}): "f"new_mask shape = {new_mask.shape}, cutted_img shape = {(h_targ, w_targ)}")
                    new_mask = cv2.resize(new_mask, (w_targ, h_targ))
                
                assert new_mask.shape == (h_targ, w_targ), (f"Shape mismatch for image {img_info['file_name']} (ID: {img_info['id']}): "f"new_mask shape = {new_mask.shape}, cutted_img shape = {(h_targ, w_targ)}")                
                binary_mask = (new_mask == ann['category_id']).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary_mask))  # Must be Fortran-contiguous
                # Get bbox and area
                bbox = mask_utils.toBbox(rle).tolist()  # [x, y, width, height]
                area = mask_utils.area(rle).item()      # scalar
                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON compatibility
                    
                new_ann['segmentation'] = rle
                new_ann['bbox'] = bbox
                new_ann['area'] = area

                new_mask_dict['annotations'].append(new_ann)
                     
            pbar.update(1)
            
    json_final_path_masks = path_join(output_path, f'{args.split}_annotation_coco.json')
    with open(json_final_path_masks, 'w') as out_f:
        json.dump(new_mask_dict, out_f, indent=4)
        
    print(f'Cutted frames json saved to: {json_final_path_masks}')
    

if __name__ == "__main__":

    #Relevant paths...
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    endoscapes_cutmargins_dir = path_join(data_dir, 'endoscapes_cutmargins')
    
    frames_dir = path_join(endoscapes_dir, 'frames')
    frames_cutmargins_dir = path_join(endoscapes_cutmargins_dir, 'frames')
        
    #json paths
    annots_path = path_join(endoscapes_dir, args.annots_dir)
    coco_json_path = path_join(annots_path, f'{args.split}_annotation_coco.json')

    #Load coco annotations
    coco_dict = load_json(coco_json_path)
    coco_obj = COCO(coco_json_path)

    output_dir = path_join(endoscapes_cutmargins_dir, args.annots_dir)            
    create_directory_if_not_exists(output_dir)

    #Preprocess masks
    prepocess_masks(base_dir=frames_dir, 
                cutted_dir=frames_cutmargins_dir,
                coco_json_dict= coco_dict,
                coco_obj= coco_obj,
                output_path=output_dir)
    
    

    print("Cuts Done")
    