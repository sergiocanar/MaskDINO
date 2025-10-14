import os 
import sys
import cv2 
import json
import logging
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from utils import save_json
from collections import Counter
from os.path import join as path_join

# Create log file in the same directory
this_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(this_dir, 'frame_check.log')

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler(sys.stdout)  
    ]
)


def is_black_frame(frame: np.ndarray, threshold: float = 0.10, black_ratio_threshold: float = 0.90) -> bool:
    """
    Determine if a frame is predominantly black using a hybrid approach:
    - First, check the mean pixel intensity.
    - Then, if dark enough, apply Otsu thresholding to confirm.
    
    Parameters:
    - frame (np.ndarray): The input BGR frame.
    - threshold (float): Initial grayscale mean threshold (default=0.10).
    - black_ratio_threshold (float): Proportion of black pixels required after Otsu (default=0.98).
    
    Returns:
    - bool: True if the frame is considered black, False otherwise.
    """
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Quick early exit using mean intensity
    frame_mean = np.mean(gray_frame)
    if frame_mean > threshold * 255:
        return False

    # Apply Otsu thresholding
    _, otsu_thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Compute ratio of black pixels
    black_pixel_ratio = np.sum(otsu_thresh == 0) / otsu_thresh.size

    return black_pixel_ratio > black_ratio_threshold

def is_green_frame(frame: np.ndarray):
    
    channels = cv2.split(frame)
    blue_channel = channels[0]
    green_channel = channels[1]
    red_channel = channels[2]
    
    # Calculate the mean of each channel
    mean_blue = np.mean(blue_channel)
    mean_green = np.mean(green_channel)
    mean_red = np.mean(red_channel)
    
    # Check if the green channel is significantly higher than the others
    if mean_green > mean_blue * 1.5 and mean_green > mean_red * 1.5:
        return True
    return False
    
def is_blue_frame(frame: np.ndarray):
    channels = cv2.split(frame)
    blue_channel = channels[0]
    green_channel = channels[1]
    red_channel = channels[2]
    
    # Calculate the mean of each channel
    mean_blue = np.mean(blue_channel)
    mean_green = np.mean(green_channel)
    mean_red = np.mean(red_channel)
    
    # Check if the blue channel is significantly higher than the others
    if mean_blue > mean_green * 1.5 and mean_blue > mean_red * 1.5:
        return True
    return False

def is_weird_shape(frame_shape_dict: dict, tolerance: int = 100) -> list:
    """
    Detects frames whose shape differs from the dominant shape by more than the given tolerance.

    Parameters:
    - frame_shape_dict: dict mapping frame_name to {'height': h, 'width': w}
    - tolerance: minimum difference in pixels (either height or width) to consider shape "weird"

    Returns:
    - List of frame names with shape significantly different from the majority shape.
    """
    # Obtener todas las formas
    shapes = [(v['height'], v['width']) for v in frame_shape_dict.values()]
        
    # Identificar la forma más común
    most_common_shape = Counter(shapes).most_common(1)[0][0]
    h_ref, w_ref = most_common_shape
    
    # Detectar frames cuya diferencia sea significativa
    weird_frames = []
    for frame_name, shape in frame_shape_dict.items():
        h, w = shape['height'], shape['width']
        if abs(h - h_ref) >= tolerance or abs(w - w_ref) >= tolerance:
            weird_frames.append(frame_name)
            logging.info(f"Frame {frame_name} has weird shape: {shape}, expected: {most_common_shape}")
    
    return weird_frames
    
    
def check_frames(frames_dir: str, frames_lt: list,output_path: str = None):
    
    #Video dictionary to store black frames paths
    frames_checker_dict = {}
    frames_checker_dict['black_frames'] = []
    frames_checker_dict['weird_shapes'] = []
    frames_checker_dict['green_frames'] = []
    frames_checker_dict['blue_frames'] = []
    
    black_counter = 0
    blue_counter = 0
    green_counter = 0
    
    #Get the subdirectories in the frames directory
    with tqdm(total=len(frames_lt), desc=f"Checking frames...", unit='frame') as pbar: 
        # Iterate through each video directory
        for frame_path in frames_lt:
            frame_name = frame_path.split('/')[-1]
            frame_name = frame_name.split('.')[0]
                                        
            frame = cv2.imread(frame_path)
            h, w, _ = frame.shape
        
            frame_name = os.path.basename(frame_path)
            frame_name = frame_name.split('.')[0]
            
            if is_blue_frame(frame=frame):
                logging.info(f'Frame {frame_path} in video {frame_name} is blue!')
                # Append the blue frame path to the video dictionary
                frames_checker_dict['blue_frames'].append(frame_name)
                blue_counter += 1
            
            if is_green_frame(frame=frame):
                logging.info(f'Frame {frame_path} in video {frame_name} is green!')
                # Append the green frame path to the video dictionary
                frames_checker_dict['green_frames'].append(frame_name)
                green_counter += 1
            
            # Check if the frame is black
            if is_black_frame(frame=frame):
                logging.info(f'Frame {frame_path} in video {frame_name} is black!')
                # Append the black frame path to the video dictionary
                frames_checker_dict['black_frames'].append(frame_name)
                black_counter += 1

                
            pbar.update(1)
            
    print(f'Total frames analyzed: {len(frames_lt)}')
    print(f'Black frames: {black_counter}')
    print(f'Green frames: {green_counter}')
    print(f'Blue frames: {blue_counter}')
    
    json_path = path_join(output_path, 'frames2check.json')
    save_json(data_dict=frames_checker_dict, save_path=json_path)
        
if __name__ == "__main__":
    
    # Relevant paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes_cutmargins')
    frames_dir = path_join(endoscapes_dir, 'frames')
    output_path = path_join(this_dir, 'frames_checker')
    os.makedirs(output_path, exist_ok=True)
    
    frames_lt = glob(path_join(frames_dir, '*.jpg'))
    
    # Check black frames
    check_frames(frames_dir= frames_dir, 
                       frames_lt=frames_lt,
                       output_path=output_path)
    
    logging.info(f"Black frames check completed. Results saved to {output_path}")