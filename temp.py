import os
from tqdm import tqdm
from glob import glob
from os.path import join as path_join
from utils import join_coco_jsons, load_json, create_symlink, remove_black_frames, create_directory_if_not_exists, save_json

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
        
        if ".jpg" not in file_:
            file_ = f'{file_}.jpg'
        else:
            pass
        
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
            

def rm_black_frame_from_json(src_dir: str, black_frames_lt: list, splits_lt: list, output_dir: str):
    
    #Iterate over each split 
    for split in splits_lt:
        
        #Counter for each removed img and annots. 
        #In EndoscapesBbox201 it should be 81 frames.
        img_counter = 0
        annos_counter = 0
        
        json_path = path_join(src_dir, f'{split}_annotation_coco.json')
        
        if os.path.exists(json_path):
            data = load_json(coco_json_path=json_path)
        else:
            raise FileNotFoundError('Papi revisa tu path')
        
        images_info = data["images"]
        annos_info = data["annotations"]
        img_ids_lt = []
        
        new_data = {
            "images": [],
            "annotations": [],
            "categories":  data["categories"]
        }
        
        for img_dict in images_info:
            
            new_img_info = {
                'file_name': img_dict["file_name"],
                "height": img_dict["height"],
                "width": img_dict["width"],
                "id": img_dict["id"],
                "video_id": img_dict["video_id"]
            }
                            
            file_name = img_dict["file_name"]
            file_name = file_name.split('.')[0]
            img_id = img_dict["id"]
                        
            if file_name in black_frames_lt:
                img_counter += 1
                img_ids_lt.append(img_id)    
            else:
                new_data['images'].append(new_img_info)   
        
        print(f'Removed {img_counter} frames from {split} split')
        
        for anno in annos_info:
            
            img_id_in_anno = anno["image_id"]
            
            if img_id_in_anno in img_ids_lt:
                annos_counter += 1
            else:
                new_data["annotations"].append(anno) 
            
        print(f'Removed {annos_counter} annotations from {split} split')
        
        
        output_path = path_join(output_dir, f'{split}_annotation_coco.json')
        save_json(data_dict=new_data,
                  save_path=output_path)
        
        print(f'Saved {split} cleaned json file to: {output_path}')
    
    print('All jsons were updated and saved!')
        
def get201_train_json(all_seg_data: dict, seg50jsons_lt: list, save_dir: str = None):
    
    new_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    path_lt = []
    
    for json_path in seg50jsons_lt:
        json_f = load_json(json_path)
        files = [f['file_name'] for f in json_f["images"]]
        path_lt += files
        
    image_info_lt = all_seg_data["images"]
    annos_info_lt = all_seg_data["annotations"]
    
    img_ids_lt = []
    
    for img_info in image_info_lt:
        
        file_name = img_info["file_name"]
        img_id = img_info["id"]
        
        if file_name not in path_lt:
            new_dict["images"].append(img_info)
            img_ids_lt.append(img_id)
        else: 
            continue
    
    for annos_info in annos_info_lt:
        
        img_ann_id = annos_info["image_id"]
        
        if img_ann_id in img_ids_lt:
            new_dict["annotations"].append(annos_info)
        else:
            continue
        
    for json in seg50jsons_lt:
        if 'train' in json:
            train_json = load_json(json)
            
            new_dict["images"] += train_json["images"]            
            new_dict["annotations"] += train_json["annotations"]            
            
    new_dict["categories"] = all_seg_data["categories"]
    
    final_path = path_join(save_dir, 'train_annotation_coco.json')
    
    save_json(data_dict=new_dict,
              save_path=final_path)
    
    print(f'Saved 201 annotation json to: {final_path}')    
    
if __name__ == "__main__":
    
    #Relevant paths and list with 
    this_dir = os.path.dirname(os.path.abspath(__file__))
    endoscapes_dataset_dir = "/home/scanar/endovis/Datasets/endoscapes/"
    all_frames_dir = path_join(endoscapes_dataset_dir, 'all')
    frame_lt = glob(path_join(all_frames_dir, "*.jpg"))
    
    data_dir = path_join(this_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    frames_endo2023 = path_join(endoscapes_dir, 'frames')
    og_jsons_dir = path_join(data_dir, 'endoscapesbbox201')
    og_jsons_lt = glob(path_join(og_jsons_dir, '*.json'))
    
    output_json_path = path_join(og_jsons_dir, 'bbox201_annotation_coco.json')
    
    
    # join_coco_jsons(json_path_lt=og_jsons_lt,
    #            output_path=output_json_path)
    
    endoscapes201_json = load_json(coco_json_path=output_json_path)
    endoscapes201_files = [f['file_name'] for f in endoscapes201_json['images']]
    
    # create_sym_links_endo2023(file_name_lt=endoscapes201_files,
    #                           src_path=all_frames_dir, 
    #                           dest_path=frames_endo2023)
    
    annots_endo2023_dir = path_join(endoscapes_dir, 'annotations')
    seg50_json_lt = glob(path_join(annots_endo2023_dir, '*_coco.json'))
    calculate_masks_dir = path_join(endoscapes_dir, 'calculate_masks')
    all_seg_201_json = path_join(calculate_masks_dir, 'all_seg_201.json')
    all_seg_201_data = load_json(all_seg_201_json)
    
    output_201_dir = path_join(endoscapes_dir, 'annotations_201')
    
    
    
    get201_train_json(all_seg_data=all_seg_201_data,
                      seg50jsons_lt=seg50_json_lt,
                      save_dir=output_201_dir)
    
    # check_dir = path_join(this_dir, 'frames_checker', 'endoscapes2023')
    # frames2check_path = path_join(check_dir, 'frames2check.json')
    # frames2check_dict = load_json(frames2check_path)
    # black_frames_lt = frames2check_dict['black_frames']
    
    # # create_sym_links_endo2023(file_name_lt=black_frames_lt,
    # #                         src_path=all_frames_dir, 
    # #                         dest_path=check_dir)

    # # remove_black_frames(src_dir=frames_endo2023,
    # #                     frame_lt=black_frames_lt)
    
    # output_dir_clean_jsons = path_join(data_dir, 'endoscapesseg50_w_o_black_frames')
    # create_directory_if_not_exists(output_dir_clean_jsons)
    
    
    # rm_black_frame_from_json(src_dir=og_jsons_dir,
    #                          black_frames_lt=black_frames_lt,
    #                          splits_lt=['train','val', 'test'],
    #                          output_dir=output_dir_clean_jsons)