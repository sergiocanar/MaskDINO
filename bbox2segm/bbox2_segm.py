import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from os.path import join as path_join

#Parser for paths!
def parser_bbox2seg():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path",
                        type=str,
                        help="Path to config file for loading the model parameters. Needs to be relative to the main directory",
                        required=True)
    
    parser.add_argument("--weights_path",
                        type=str,
                        help="Path to weights paths. Needs to be absolute to the main directory",
                        required=True)
    
    args = parser.parse_args()
    
    return args

class MaskGenerator():
    
    """
    Mask generator from Bboxes using MaskDINO
    
    
    """
    
    def __init__(self):
        pass    
    
    

if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = path_join(this_dir)
    
    args = parser_bbox2seg()
    
    