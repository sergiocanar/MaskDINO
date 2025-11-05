# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021.
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import join as path_join
from skimage.morphology import binary_erosion, disk
from utils import load_json, create_directory_if_not_exists

def change_size(image):
    
    coords = None
    
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    
    if not edges_x:
        return image, coords

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left  
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom  

    pre1_picture = image[left:left + width, bottom:bottom + height]  

    #print(pre1_picture.shape) 
    
    coords = (left, width, bottom, height)
    
    return pre1_picture, coords



def filter_black_endo2023(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 31, 5)
    inv_img = cv2.bitwise_not(binary_image)
    
    #Mathematical morphology
    disk_3 = disk(3)
    binary_image2 = binary_erosion(inv_img, disk_3) 
    
    x, y = binary_image2.shape[:2]

    edges_x, edges_y = [], []

    for i in range(x):
        for j in range(y):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    crop_coords = None
    pre1_picture = None
    
    if not edges_x:
        return image, None
    else:
        left, right = min(edges_x), max(edges_x)
        bottom, top = min(edges_y), max(edges_y)
        crop_coords = (left, right, bottom, top)
        pre1_picture = image[left:right, bottom:top]
        
        edges_x = np.array(edges_x)
        edges_y = np.array(edges_y)
                    
    return pre1_picture, crop_coords

def process_image(frame_src: str, save_path: str):
    file_name = os.path.basename(frame_src)
    frame_out_path = os.path.join(save_path, file_name)
    frame = cv2.imread(frame_src)
    # frame, _ = filter_black_endo2023(frame)
    frame, _ = change_size(frame)
    cv2.imwrite(frame_out_path, frame)

if __name__ == "__main__":
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    output_dir = path_join(data_dir, 'endoscapes_cutmargins')
    save_path = path_join(output_dir, 'frames')
    
    train_dict = load_json("data/endoscapes/annotations_201/train_annotation_coco.json")
    val_dict = load_json("data/endoscapes/annotations_201/val_annotation_coco.json")
    test_dict = load_json("data/endoscapes/annotations_201/test_annotation_coco.json")
    
    train_files = [img["file_name"] for img in train_dict["images"]]
    val_files = [img["file_name"] for img in val_dict["images"]]
    test_files = [img["file_name"] for img in test_dict["images"]]
        
    frames_dir = path_join(endoscapes_dir, 'frames')

    img_lt = train_files + val_files + test_files
        
    debug_counter = 0
    
    debug_dir = path_join(parent_dir, 'debug')
    create_directory_if_not_exists(debug_dir)
        
    with tqdm(total=len(img_lt), desc='Cutting margins...', unit='frames') as pbar:            
        for img_file in img_lt:
            
            source_path = path_join(frames_dir, img_file)
            
            create_directory_if_not_exists(save_path)
            
            process_image(frame_src=source_path,
                        save_path=save_path)
            debug_counter += 1
            pbar.update(1)
    
    # src_dir = '/home/scanar/endovis/Datasets/endoscapes/frames'
    # dst_dir = '/home/scanar/endovis/Datasets/endoscapes/cutmargins_frames'
    # os.makedirs(dst_dir, exist_ok=True)
    # video_lt = os.listdir(src_dir)
    
    # for video in tqdm(video_lt, desc="Videos", unit="video"):
    #     frames_in_video_lt = glob(os.path.join(src_dir, video, '*.jpg'))
    #     dst_video_dir = os.path.join(dst_dir, video)
    #     os.makedirs(dst_video_dir, exist_ok=True)
        
    #     for img_path in tqdm(frames_in_video_lt, desc=f"{video}", leave=False, unit="frame"):
    #         process_image(frame_src=img_path, save_path=dst_video_dir)
