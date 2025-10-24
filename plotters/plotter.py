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
    Compatible with Detectron2 COCO-style outputs (list of dicts with RLEs or polygons).
    Creates one subfolder per image and generates:
    - General mask plot
    - Plots per confidence interval (0.0–0.1, …, 0.9–1.0)
    """
    assert len(coco_json_paths) > 0, "At least one COCO JSON path must be provided."
    
    coco_gt = COCO(coco_gt_path)

    color_map = {
        "cystic_plate":   (248/255, 231/255,  28/255),
        "calot_triangle": ( 74/255, 144/255, 226/255),
        "cystic_artery":  (218/255,  13/255,  15/255),
        "cystic_duct":    ( 65/255, 117/255,   6/255),
        "gallbladder":    (126/255, 211/255,  33/255),
        "tool":           (245/255, 166/255,  35/255),
        "background":     (0.0, 0.0, 0.0),
    }

    # Load predictions for each run
    dict_preds = {}
    for json_path in coco_json_paths:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
        preds_by_image = load_predictions(json_path)
        dict_preds[run_name] = preds_by_image

    img_ids = coco_gt.getImgIds()
    conf_ranges = [(round(i, 1), round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1)]

    with tqdm(total=len(img_ids), desc="Plotting predictions", unit="frame") as pbar:
        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(images_dir, file_name)

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            img = plt.imread(img_path)
            h, w = img.shape[:2]
            img_name = os.path.splitext(file_name)[0]
            img_output_dir = path_join(output_dir, img_name)
            os.makedirs(img_output_dir, exist_ok=True)

            # Load GT annotations
            ann_ids_gt = coco_gt.getAnnIds(imgIds=img_id)
            anns_gt = coco_gt.loadAnns(ann_ids_gt)

            def plot_for_preds(preds_dict, title_suffix, save_name):
                num_cols = 1 + len(preds_dict)
                fig, axs = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
                if num_cols == 2:
                    axs = [axs[0], axs[1]]

                # ---- Ground truth masks ----
                ax_gt = axs[0]
                gt_canvas = np.zeros((h, w, 3), dtype=np.float32)
                for ann in anns_gt:
                    cat_name = coco_gt.loadCats(ann["category_id"])[0]["name"]
                    color = color_map.get(cat_name, np.random.rand(3))
                    segm = ann.get("segmentation", None)
                    if segm:
                        mask = maskUtils.decode(segm) if isinstance(segm, dict) else maskUtils.decode(maskUtils.frPyObjects(segm, h, w))
                        for c in range(3):
                            gt_canvas[..., c] = np.where(mask == 1, color[c], gt_canvas[..., c])
                ax_gt.imshow(gt_canvas)
                ax_gt.set_title(f"Ground Truth ({len(anns_gt)} objects)", fontsize=12)
                ax_gt.axis("off")

                # ---- Predicted masks ----
                for idx, (run_name, preds_by_image) in enumerate(preds_dict.items()):
                    ax = axs[idx + 1]
                    preds = preds_by_image.get(img_id, [])
                    mask_canvas = np.zeros((h, w, 3), dtype=np.float32)
                    for pred in preds:
                        cat_id = pred["category_id"]
                        cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                        color = color_map.get(cat_name, np.random.rand(3))
                        segm = pred.get("segmentation", None)
                        if segm:
                            try:
                                mask = maskUtils.decode(segm) if isinstance(segm, dict) else maskUtils.decode(maskUtils.frPyObjects(segm, h, w))
                            except Exception:
                                continue
                            for c in range(3):
                                mask_canvas[..., c] = np.where(mask == 1, color[c], mask_canvas[..., c])
                    ax.imshow(mask_canvas)
                    ax.set_title(f"{run_name} {title_suffix} ({len(preds)} objects)", fontsize=12)
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(path_join(img_output_dir, save_name), bbox_inches="tight", dpi=150)
                plt.close(fig)

            # -------- General (all scores) --------
            plot_for_preds(dict_preds, "(all scores)", "general.png")

            # -------- Confidence thresholds --------
            for low, high in conf_ranges:
                filtered_preds = {}
                for run_name, preds_by_image in dict_preds.items():
                    preds_in_range = [p for p in preds_by_image.get(img_id, []) if low <= p.get("score", 0) < high]
                    filtered_preds[run_name] = {img_id: preds_in_range}
                plot_for_preds(filtered_preds, f"[{low}-{high})", f"conf_{low}_{high}.png")

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