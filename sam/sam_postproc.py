import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from os.path import join as path_join
from pycocotools import mask as maskUtils
from utils import create_directory_if_not_exists, load_json, save_json, decode_rle_to_mask, encode_mask_to_rle

def warp_mask(mask, flow):
    """Warp mask according to optical flow (img1->img2)."""
    h, w = mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST)
    return (warped > 0.5).astype(np.uint8)

def compute_flow(img1, img2):
    """Compute dense optical flow from img1 -> img2."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

def warp_instances(inst_masks_lt: list, flow: np.ndarray, direction: str = 'to_t'):
    """
    Warp and stabilize a list of instance masks using optical flow.

    Args:
        inst_masks_lt (list): List of COCO-style annotations for one frame (each with 'segmentation' in RLE).
        flow (np.ndarray): Optical flow array of shape (H, W, 2).
            - If direction == 'to_t', flow should describe motion from source -> target (e.g., t-1 → t).
            - If direction == 'to_neighbor', flow should describe motion from target -> neighbor (e.g., t → t+1).
        direction (str): Either 'to_t' (warp source to current keyframe)
                         or 'to_neighbor' (warp keyframe to neighboring frame).
    
    Returns:
        list: Updated list of annotations with stabilized RLE masks.
    """
    assert direction in ['to_t', 'to_neighbor'], \
        f"Invalid direction: {direction}. Must be 'to_t' or 'to_neighbor'."

    new_inst_masks_lt = []

    for anno_info in inst_masks_lt:
        if 'segmentation' not in anno_info:
            print(f"[WARN] Segmentation not found for Img ID {anno_info.get('image_id')} "
                  f"Ann ID {anno_info.get('id')}")
            continue

        segm_info = anno_info['segmentation']
        mask = decode_rle_to_mask(rle=segm_info)

        # Determine correct flow direction
        if direction == 'to_t':
            flow_used = -flow                  # e.g., t−1 → t  (no sign inversion)
        elif direction == 'to_neighbor':
            flow_used = flow                  # e.g., t → t+1  (forward)
        else:
            raise ValueError(f"Unexpected direction: {direction}")

        # Warp the mask
        new_mask = warp_mask(mask=mask, flow=flow_used)        
        assert mask.shape == new_mask.shape, "Warped mask shape mismatch."

        # Temporal stabilization: blend original + warped
        stabilized_mask = ((0.7 * mask + 0.3 * new_mask) > 0.5).astype(np.uint8)

        # Encode back to RLE
        new_rle = encode_mask_to_rle(mask=stabilized_mask)
        anno_info['segmentation'] = new_rle

        new_inst_masks_lt.append(anno_info)

    return new_inst_masks_lt


# fixed color map (same used in your segmentation work)
color_map = {
    "cystic_plate":   (248/255, 231/255,  28/255),
    "calot_triangle": ( 74/255, 144/255, 226/255),
    "cystic_artery":  (218/255,  13/255,  15/255),
    "cystic_duct":    ( 65/255, 117/255,   6/255),
    "gallbladder":    (126/255, 211/255,  33/255),
    "tool":           (245/255, 166/255,  35/255),
    "background":     (0.0, 0.0, 0.0),
}

color_map = {
    "cystic_plate":   (248/255, 231/255,  28/255),
    "calot_triangle": ( 74/255, 144/255, 226/255),
    "cystic_artery":  (218/255,  13/255,  15/255),
    "cystic_duct":    ( 65/255, 117/255,   6/255),
    "gallbladder":    (126/255, 211/255,  33/255),
    "tool":           (245/255, 166/255,  35/255),
    "background":     (0.0, 0.0, 0.0),
}

def paint_masks(overlay, anns):
    for ann in anns:
        cat = ann.get("category_name", str(ann["category_id"]))
        color = color_map.get(cat, (1, 1, 1))
        mask = maskUtils.decode(ann["segmentation"]).astype(bool)
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array(color)
    return overlay

def visualize_triplet_masks_with_original(img_tm1, img_t, img_tp1,
                                          anns_tm1, anns_t, anns_tp1,
                                          warped_tm1, warped_tp1,
                                          title="Debug flow matching",
                                          output_path: str = None):
    """
    2-row visualization:
        Row 1: Original SAM masks (t−1, t, t+1)
        Row 2: Warped versions (t−1→t, t, t→t+1)
    """
    img_tm1, img_t, img_tp1 = img_tm1 / 255.0, img_t / 255.0, img_tp1 / 255.0

    # --- Originals (SAM) ---
    orig_tm1 = paint_masks(img_tm1.copy(), anns_tm1)
    orig_t   = paint_masks(img_t.copy(),   anns_t)
    orig_tp1 = paint_masks(img_tp1.copy(), anns_tp1)

    # --- Warped (flow post-processed) ---
    warp_tm1 = paint_masks(img_tm1.copy(), warped_tm1)
    warp_t   = paint_masks(img_t.copy(),   anns_t)  # keyframe stays the same
    warp_tp1 = paint_masks(img_tp1.copy(), warped_tp1)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    top_titles = ["t-1 (SAM)", "t (keyframe SAM)", "t+1 (SAM)"]
    bottom_titles = ["t-1→t (warped)", "t (keyframe)", "t→t+1 (warped)"]

    # Row 1: original SAM
    for i, (im, title_txt) in enumerate(zip([orig_tm1, orig_t, orig_tp1], top_titles)):
        ax[0, i].imshow(im)
        ax[0, i].set_title(title_txt)
        ax[0, i].axis("off")

    # Row 2: warped
    for i, (im, title_txt) in enumerate(zip([warp_tm1, warp_t, warp_tp1], bottom_titles)):
        ax[1, i].imshow(im)
        ax[1, i].set_title(title_txt)
        ax[1, i].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f'Image saved to: {output_path}')


def post_proc_sam_annots(base_frame_path: str, endo2023_seg50_json_path: str, sam_annots_json_path: str, output_dir = None):
    
    og_endo2023_seg50_info = load_json(coco_json_path=endo2023_seg50_json_path)
    og_imgs_info_lt = og_endo2023_seg50_info['images']
    keyframe_paths_lt = [img_info['file_name'] for img_info in og_imgs_info_lt]
    
    sam_json_info = load_json(coco_json_path=sam_annots_json_path)
    imgs_info_sam_lt = sam_json_info['images']
    
    sam_coco = COCO(sam_annots_json_path)
        
    final_dict = {
        "images": [],
        "annotations": [],
        "categories": sam_json_info['categories']
    }
    
    per_video_dict = {}
    
    with tqdm(total=len(imgs_info_sam_lt), desc='Re-arrenging info for flow matching...', unit='frame') as pbar:
        for img_dict in imgs_info_sam_lt:
            
            
            actual_keys = list(per_video_dict.keys())
            video_id = img_dict['video_id']
            file_name = img_dict['file_name']
            
            if video_id not in actual_keys:
                per_video_dict[video_id] = [file_name]
            else:
                per_video_dict[video_id].append(file_name)
                per_video_dict[video_id] = sorted(per_video_dict[video_id])
    
            final_dict['images'].append(img_dict)
            
            pbar.update(1)
             
    with tqdm(total=len(keyframe_paths_lt), desc='Computing flow matching for keyframes', unit='frame') as pbar:
        for keyframe_path in keyframe_paths_lt:
            video_id = int(keyframe_path.split('_')[0])
            sam_frames_lt = per_video_dict[video_id]
            
            if keyframe_path in sam_frames_lt:
                pos = sam_frames_lt.index(keyframe_path)
            else:
                print(f'Keyframe not found is a black frame: {path_join(base_frame_path, keyframe_path)}')
                continue
            t_minus1_frame_path = sam_frames_lt[pos-1]
            t_plus1_frame_path = sam_frames_lt[pos+1]
            
            final_path_kf = path_join(base_frame_path, keyframe_path)
            final_t_min1 = path_join(base_frame_path, t_minus1_frame_path)
            final_t_plus1 = path_join(base_frame_path, t_plus1_frame_path)
            
            
            kf = cv2.imread(final_path_kf)
            t_min1 = cv2.imread(final_t_min1)
            t_plus1 = cv2.imread(final_t_plus1)
            
            flow_tm1_t = compute_flow(img1=t_min1,
                                      img2=kf)
            flow_t_tplus1 = compute_flow(img1=kf,
                                         img2=t_plus1)
                        
            info_img_t_minus1 = [img_info for img_info in imgs_info_sam_lt if img_info['file_name'] == t_minus1_frame_path][0]
            info_img_t_plus1 = [img_info for img_info in imgs_info_sam_lt if img_info['file_name'] == t_plus1_frame_path][0]
            info_img_kf = [img_info for img_info in imgs_info_sam_lt if img_info['file_name'] == keyframe_path][0]
            
            anns_ids_kf = sam_coco.getAnnIds(imgIds=info_img_kf['id'])
            anns_ids_t_minus1 = sam_coco.getAnnIds(imgIds=[info_img_t_minus1['id']])
            anns_ids_t_plus1 = sam_coco.getAnnIds(imgIds=[info_img_t_plus1['id']])
            
            anns_kf = sam_coco.loadAnns(anns_ids_kf)
            anns_t_minus1_info = sam_coco.loadAnns(anns_ids_t_minus1)
            anns_t_plus1_info = sam_coco.loadAnns(anns_ids_t_plus1)
            
            
            # (t−1 → t) uses flow_tm1_t directly (no inversion)
            new_anns_t_minus1_info = warp_instances(
                inst_masks_lt=anns_t_minus1_info,
                flow=flow_tm1_t,
                direction='to_t'
            )

            # (t → t+1) uses flow_t_tplus1 directly
            new_anns_t_plus1_info = warp_instances(
                inst_masks_lt=anns_t_plus1_info,
                flow=flow_t_tplus1,
                direction='to_neighbor'
            )
                        
            added_ids = set()
            for ann in anns_kf + new_anns_t_minus1_info + new_anns_t_plus1_info:
                if ann["id"] not in added_ids:
                    final_dict["annotations"].append(ann)
                    added_ids.add(ann["id"])

            pbar.update(1)

    final_json_path = path_join(output_dir, 'train_annotation_coco.json')
    
    save_json(data_dict=final_dict,
              save_path=final_json_path)
    
    print(f'Final json saved to: {final_json_path}')
    
            
if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    
    data_dir = path_join(parent_dir, 'data')
    endo2023_dir = path_join(data_dir, 'endoscapes')
    frames_path = path_join(endo2023_dir, 'frames')
    sam_annots_dir = path_join(endo2023_dir, 'annotations_sam_extended')
    train_json_path = path_join(sam_annots_dir, 'train_annotation_coco.json')
    og_endoseg50_path = path_join(endo2023_dir, 'annotations')
    og_endoseg50_json = path_join(og_endoseg50_path, 'train_annotation_coco.json')
    
    output_dir = path_join(endo2023_dir, 'annotations_sam_postproc')
    create_directory_if_not_exists(output_dir)    
    
    post_proc_sam_annots(base_frame_path=frames_path,
                         endo2023_seg50_json_path=og_endoseg50_json,
                         sam_annots_json_path=train_json_path,
                         output_dir=output_dir)
    
    