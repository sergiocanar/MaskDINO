import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_json
from pycocotools.coco import COCO
from os.path import join as path_join
from pycocotools import mask as mask_utils

parser = argparse.ArgumentParser()

parser.add_argument("--annots_dir", type=str, default="annotations")
parser.add_argument("--split", type=str, default="train")

args = parser.parse_args()

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def prepocess_masks(base_dir: str, cutted_dir: str,coco_json_dict: dict, coco_obj: COCO, output_path: str = None):
        
    #Get images information list
    img_info_lt = coco_obj.dataset['images']
    new_mask_dict = {
        "images": [],
        "annotations": [],
        "categories": coco_json_dict['categories']
    }
    
    diff_dict = {}
    
    with tqdm(total=len(img_info_lt), desc='Processing frames...', unit='frame') as pbar:

        #Iterate over info dict
        for img_info in img_info_lt:
            
            #Get relevant information
            video_name = img_info['video_id']            
            h,w = img_info['height'], img_info['width']
            file_name = img_info['file_name']
            
            new_img_info = {
                "id": img_info['id'],
                "file_name": file_name,
                "height": None,
                "width": None,
                "video_name": video_name,
                "frame_id": img_info['frame_id']
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
                
                
                new_ann = {
                    "id": ann['id'],
                    "image_id": ann['image_id'],
                    "segmentation": None,
                    "iscrowd": ann['iscrowd'],
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
                
                # breakpoint()
                
                # -------------- CHALLENGE DATA ---------------
                if not video_name in ['video_138', 'video_197', 'video_227']:  #General case
                    new_frame, l, w, b, h = filter_black(image=img) 
                    #Cut the mask
                    new_mask = mask[l:l+w, b:b+h]
                else:
                    new_frame, top, bottom, x_start, x_end = filter_black_color(image=img)
                    new_mask = mask[top:bottom, x_start:x_end]       
                
                
                if new_mask is None or new_mask.size == 0:
                    print(f'Invalid mask for: Annotation:{ann['id']} in Image: {ann['image_id']}')
                    continue  # skip invalid mask
                
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
                # print(new_ann)
                # breakpoint()
                new_mask_dict['annotations'].append(new_ann)
            
            #Get shapes for comparison
            new_h, new_w = new_mask.shape            
                      

            
            diff_h = abs(new_h - h_targ)
            diff_w = abs(new_w - w_targ)
            
            diff_dict[file_name] = [diff_h, diff_w]    
            pbar.update(1)
            
    json_final_path_masks = path_join(output_path, f'{args.split}_annotation_coco.json')
    with open(json_final_path_masks, 'w') as out_f:
        json.dump(new_mask_dict, out_f, indent=4)
    
def filter_black(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)
    right = max(edges_x)
    width = right - left
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    pre1_picture = image[left : left + width, bottom : bottom + height]
    return pre1_picture, left, width, bottom, height


def filter_black_color(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)

    h, w = binary_image2.shape
    edges_x = []
    edges_y = []

    for i in range(h):
        for j in range(10, w - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    top = min(edges_x)
    bottom = max(edges_x)
    left = min(edges_y)
    right = max(edges_y)

    left_margin_width = left
    x_start = left_margin_width
    x_end = w - left_margin_width

    if x_start >= x_end:
        print("Recorte inválido: margen izquierdo muy ancho.")
        return image

    cropped_image = image[top:bottom, x_start:x_end]
    return cropped_image, top, bottom, x_start, x_end


def margin_or_circle(image, black_threshold=15, white_threshold=230, max_columns_color=50, pixel_ratio_color=0.1, white_circle_min_pixels=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Detectar margen de color
    start_col = max(w - max_columns_color, 0)
    for col in range(w - 1, start_col - 1, -1):
        col_pixels = gray[:, col]
        count_non_black = (col_pixels > black_threshold).sum()
        if (count_non_black / h) > pixel_ratio_color:
            return True

    # 2. Detectar círculo blanco (zona derecha, centrada verticalmente)
    crop_top = int(h * 0.25)
    crop_bottom = int(h * 0.75)
    crop_left = int(w - 150)
    crop_right = w

    region = gray[crop_top:crop_bottom, crop_left:crop_right]
    white_pixels = (region > white_threshold).sum()

    if white_pixels > white_circle_min_pixels:
        return True

    return False


def process_image(image_source: str, frame_id: str, frame_dict: dict, image_save: str = None):
    frame = cv2.imread(image_source)
        
    if frame is None:
        print(f"Warning: no se pudo cargar la imagen {image_source}")
        return
    
    is_margin = False
    
    if margin_or_circle(frame):
        _, y0, y1, x0, x1 = filter_black_color(frame)
        is_margin = True
    else:
        _, x0, x1, y1, y0 = filter_black(frame)
        is_margin = False
        
    frame_dict[frame_id]['crop'] = [x0, x1, y0, y1]
    frame_dict[frame_id]['is_margin'] = is_margin
        
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
    