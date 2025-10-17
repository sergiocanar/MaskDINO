import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from os.path import join as path_join

def parser_bbox2seg():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_dir",
                        type=str,
                        help="Config file for loading the model parameters. Needs to be relative to the main directory")


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = path_join(this_dir)