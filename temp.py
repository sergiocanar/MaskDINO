import os
from glob import glob
from os.path import join as path_join
from utils import join_coco_jsons, load_json, create_symlink

def create_sym_links_endo2023(file_name_lt: list, src_path: str, dest_path: str):
    """
    Create symbolic links for a list of frame filenames.

    Args:
        file_name_lt (list): list of file names (e.g. ["120_69225.jpg", ...])
        src_path (str): directory where original frames are located
        dest_path (str): directory where symbolic links will be created
    """
    os.makedirs(dest_path, exist_ok=True)

    for file_ in file_name_lt:
        src_frame = path_join(src_path, file_)
        dst_frame = path_join(dest_path, file_)

        if not os.path.exists(src_frame):
            print(f"Source not found: {src_frame}")
            continue

        try:
            if not os.path.exists(dst_frame):
                create_symlink(frame_name=file_, src_path=src_frame, dst_dir=dest_path)
                print(f"Linked: {file_}")
            else:
                print(f"Exists: {file_}")
        except Exception as e:
            print(f"Error linking {file_}: {e}")
if __name__ == "__main__":
    
    #Relevant paths and list with 
    this_dir = os.path.dirname(os.path.abspath(__file__))
    endoscapes_dataset_dir = "/home/scanar/endovis/Datasets/endoscapes/"
    all_frames_dir = path_join(endoscapes_dataset_dir, 'all')
    frame_lt = glob(path_join(all_frames_dir, "*.jpg"))
    
    data_dir = path_join(this_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    frames_endo2023 = path_join(endoscapes_dir, 'frames')
    og_jsons_dir = path_join(data_dir, 'cvs_annotations')
    og_jsons_lt = glob(path_join(og_jsons_dir, '*.json'))
    
    output_json_path = path_join(og_jsons_dir, 'bbox201_annotation_coco.json')
    
    # join_coco_jsons(json_path_lt=og_jsons_lt,
    #            output_path=output_json_path)
    
    endoscapes201_json = load_json(coco_json_path=output_json_path)
    endoscapes201_files = [f['file_name'] for f in endoscapes201_json['images']]
    
    create_sym_links_endo2023(file_name_lt=endoscapes201_files,
                              src_path=all_frames_dir, 
                              dest_path=frames_endo2023)
    