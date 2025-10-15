# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021.
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from tqdm import tqdm
from os.path import join as path_join
from utils import load_json, create_directory_if_not_exists

def filter_black(image, debug_dir: str = None, debug_counter: int = 0):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)
    x, y = binary_image2.shape[:2]

    edges_x, edges_y = [], []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        print("Empty edge list → skipping crop")
        final_path = path_join(debug_dir, f'debug_empty_{debug_counter}.png')
        cv2.imwrite(final_path, image)
        return image

    left, right = min(edges_x), max(edges_x)
    bottom, top = min(edges_y), max(edges_y)

    print(f"Detected crop → rows: {left}-{right}, cols: {bottom}-{top}, "
          f"height: {right-left}, width: {top-bottom}")

    pre1_picture = image[left:right, bottom:top]
    return pre1_picture



def process_image(frame_src: str, save_path: str, debug_dir: str = None, debug_counter: int = 0):
    file_name = os.path.basename(frame_src)
    frame_out_path = os.path.join(save_path, file_name)
    frame = cv2.imread(frame_src)
    frame = filter_black(frame, debug_dir=debug_dir, debug_counter=debug_counter)
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
                        save_path=save_path,
                        debug_dir=debug_dir,
                        debug_counter=debug_counter)
            debug_counter += 1
            pbar.update(1)