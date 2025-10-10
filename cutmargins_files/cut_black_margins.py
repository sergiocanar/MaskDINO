# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021.
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------

import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from utils import load_json
from functools import partial
from os.path import join as path_join
from multiprocessing import Pool, cpu_count

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_black(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(
        binary_image2, 19
    )  # filter the noise, need to adjust the parameter based on the dataset
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

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom

    pre1_picture = image[left : left + width, bottom : bottom + height]

    return pre1_picture


def process_image(image_source, image_save):
    frame = cv2.imread(image_source)
    
    #dim = (int(frame.shape[1] / frame.shape[0] * 300), 300)
    #frame = cv2.resize(frame, dim)
    frame = filter_black(frame)
    #img_result = cv2.resize(frame, (250, 250))
    #if frame.shape[0] < 224 or frame.shape[1] < 224:
        #raise ValueError(f"The shape of the resulting frame is lower than 224. Frame path: {image_source}")

    cv2.imwrite(image_save, frame)


def process_video(video_id, video_source, video_save):
    create_directory_if_not_exists(video_save)

    for image_id in sorted(os.listdir(video_source)):
        if image_id == ".DS_Store":
            continue
        image_source = os.path.join(video_source, image_id)
        image_save = os.path.join(video_save, image_id)

        process_image(image_source, image_save)


def select_video(csv_path: str, preprocessing_types: str):
    """
    This function loads the csv file with all videos info and return the list with just the video_ids that
    follows the desired preprocessing conditions
    """

    df = pd.read_csv(csv_path)
    df = df[df['Preprocessing type'].isin(preprocessing_types)]
    return df['Video Id'].tolist()


if __name__ == "__main__":
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    output_dir = path_join(data_dir, 'endoscapes_cutmargins')
    
    train_dict = load_json("data/endoscapes/annotations_201/train_annotation_coco.json")
    val_dict = load_json("data/endoscapes/annotations_201/val_annotation_coco.json")
    test_dict = load_json("data/endoscapes/annotations_201/test_annotation_coco.json")
    
    train_files = [img["file_name"] for img in train_dict["images"]]
    val_files = [img["file_name"] for img in val_dict["images"]]
    test_files = [img["file_name"] for img in test_dict["images"]]
        
    frames_dir = path_join(endoscapes_dir, 'frames')

    img_lt = train_files + val_files + test_files

    def process_frame(frame_src, save_path):
        file_name = os.path.basename(frame_src)
        frame_out_path = os.path.join(save_path, file_name)
        process_image(image_source=frame_src, image_save=frame_out_path)

    with tqdm(total=len(img_lt), desc='Cutting margins...', unit='frames') as pbar:            
        for img_file in img_lt:
            
            source_path = path_join(frames_dir, img_file)
            save_path = path_join(output_dir, 'frames')
            
            create_directory_if_not_exists(save_path)
            
            process_frame(frame_src=source_path,
                        save_path=save_path)
            pbar.update(1)