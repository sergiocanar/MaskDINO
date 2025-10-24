import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from collections import defaultdict
from os.path import join as path_join
from matplotlib.patches import Rectangle
from pycocotools import mask as maskUtils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gt_folder",
    type=str,
    default="annotations"
)

parser.add_argument(
    "--run_name_md",
    type=str
)

parser.add_argument(
    "--run_name_pseudo_md",
    type=str
)

parser.add_argument(
    "--run_name_sam",
    type=str
)


args = parser.parse_args()

def load_predictions(json_path):
    """
    Load Detectron2 predictions (list of dicts) and organize by image_id.
    Returns: dict mapping image_id -> list of prediction dicts
    """
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    # Group predictions by image_id
    preds_by_image = defaultdict(list)
    for pred in predictions:
        img_id = pred['image_id']
        preds_by_image[img_id].append(pred)
    
    return preds_by_image

def plot_preds_comparison(coco_gt_path: str, coco_json_paths: list, images_dir: str, output_dir: str):
    """
    Plot predictions from multiple models on all ground truth images.
    Compatible with Detectron2 COCO format output (list of prediction dicts).
    """
    assert len(coco_json_paths) > 0, "At least one COCO JSON path must be provided."
    
    # Load ground truth
    coco_gt = COCO(coco_gt_path)
    
    # Color map for categories
    color_map = {
        "cystic_plate":   (248/255, 231/255,  28/255),   # bright yellow
        "calot_triangle": ( 74/255, 144/255, 226/255),   # blue
        "cystic_artery":  (218/255,  13/255,  15/255),   # red
        "cystic_duct":    ( 65/255, 117/255,   6/255),   # dark green
        "gallbladder":    (126/255, 211/255,  33/255),   # light green
        "tool":           (245/255, 166/255,  35/255),   # orange
        "background":     (0.0, 0.0, 0.0),
    }
    
    # Load predictions - Detectron2 format (list of prediction dicts)
    dict_preds = {}
    for json_path in coco_json_paths:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
        preds_by_image = load_predictions(json_path)
        dict_preds[run_name] = preds_by_image
    
    # Get all image IDs from ground truth
    img_ids = coco_gt.getImgIds()
    
    # Plot all images
    with tqdm(total=len(img_ids), desc="Plotting predictions", unit='frame') as pbar:
        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(images_dir, file_name)
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            img = plt.imread(img_path)
            
            # Create subplots: GT + all predictions
            num_cols = 1 + len(dict_preds)
            fig, axs = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
            
            if num_cols == 2:
                axs = [axs[0], axs[1]]
            
            # Plot ground truth
            ax_gt = axs[0]
            ann_ids_gt = coco_gt.getAnnIds(imgIds=img_id)
            anns_gt = coco_gt.loadAnns(ann_ids_gt)
            
            ax_gt.imshow(img)
            ax_gt.set_title(f"Ground Truth ({len(anns_gt)} objects)", fontsize=12)
            
            for ann in anns_gt:
                cat_id = ann["category_id"]
                cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                color = color_map.get(cat_name, np.random.rand(3))
                if "bbox" in ann:
                    x, y, w_box, h_box = ann["bbox"]
                    rect = Rectangle((x, y), w_box, h_box, linewidth=2,
                                    edgecolor=color, facecolor="none")
                    ax_gt.add_patch(rect)
                    ax_gt.text(x, y - 3, cat_name, color=color, fontsize=10, weight='bold')
            ax_gt.axis("off")
            
            # Plot predictions from each model
            for idx, (run_name, preds_by_image) in enumerate(dict_preds.items()):
                ax = axs[idx + 1]
                
                # Get predictions for this image
                preds = preds_by_image.get(img_id, [])
                
                ax.imshow(img)
                ax.set_title(f"{run_name} ({len(preds)} objects)", fontsize=12)
                
                for pred in preds:
                    cat_id = pred["category_id"]
                    cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                    color = color_map.get(cat_name, np.random.rand(3))
                    
                    if "bbox" in pred:
                        x, y, w_box, h_box = pred["bbox"]
                        rect = Rectangle((x, y), w_box, h_box, linewidth=2,
                                        edgecolor=color, facecolor="none")
                        ax.add_patch(rect)
                        
                        # Optionally show score
                        score = pred.get("score", 1.0)
                        label = f"{cat_name} {score:.2f}"
                        ax.text(x, y - 3, label, color=color, fontsize=10, weight='bold')
                
                ax.axis("off")
            
            plt.tight_layout()
            
            # Save with image name
            img_base_name = os.path.splitext(file_name)[0]
            save_path = path_join(output_dir, f"{img_base_name}_comparison.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            pbar.update(1)
if __name__ == "__main__":
    
    #Basic directories 
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    endoscapes_data_dir = path_join(parent_dir, 'data', 'endoscapes')
    frames_dir = path_join(endoscapes_data_dir, 'frames')
    output_preds_dir = path_join(parent_dir, 'outputs')
    
    #Gt and preds directories
    gt_annots_dir = path_join(endoscapes_data_dir, args.gt_folder)
    ouputs_cut_margins_dir = path_join(output_preds_dir, 'endoscapes2023_cutmargins')
    
    # Ground truth JSON path
    gt_json_path = path_join(gt_annots_dir, 'test_annotation_coco.json')  # Adjust filename as needed
    
    json_lt = []
    
    if args.run_name_md is not None:
        run_dir_maskdino = path_join(ouputs_cut_margins_dir, args.run_name_md)
        maskdino_json = path_join(run_dir_maskdino, 'inference', 'coco_instances_results.json')
        json_lt.append(maskdino_json)
    else:
        run_dir_maskdino = None
        
    if args.run_name_pseudo_md is not None:
        run_dir_pseudo_md = path_join(ouputs_cut_margins_dir, args.run_name_pseudo_md)
        pseudo_md_json = path_join(run_dir_pseudo_md, 'inference', 'coco_instances_results.json')
        json_lt.append(pseudo_md_json)
    else:
        run_dir_pseudo_md = None
        
    if args.run_name_sam is not None:
        run_dir_sam = path_join(ouputs_cut_margins_dir, args.run_name_sam)
        sam_inference_json = path_join(run_dir_sam, 'inference', 'coco_instances_results.json')
        json_lt.append(sam_inference_json)
    else:
        run_dir_sam = None
    
    if args.run_name_md is None and args.run_name_pseudo_md is None and args.run_name_sam is None:
        raise ValueError("At least one of --run_name_md, --run_name_pseudo_md, or --run_name_sam must be provided.")
    
    output_dir = path_join(this_dir, 'visualizations', 'endoscapes201_compare_preds')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_preds_comparison(
        coco_gt_path=gt_json_path,
        coco_json_paths=json_lt,
        images_dir=frames_dir,
        output_dir=output_dir
    )