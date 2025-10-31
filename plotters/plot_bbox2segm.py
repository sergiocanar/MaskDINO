import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from os.path import join as path_join
from matplotlib.patches import Rectangle
from pycocotools import mask as maskUtils
from tqdm import tqdm

# === Custom colormap ===
color_map = {
    "cystic_plate":   (248/255, 231/255,  28/255),
    "calot_triangle": ( 74/255, 144/255, 226/255),
    "cystic_artery":  (218/255,  13/255,  15/255),
    "cystic_duct":    ( 65/255, 117/255,   6/255),
    "gallbladder":    (126/255, 211/255,  33/255),
    "tool":           (245/255, 166/255,  35/255),
    "background":     (0.0, 0.0, 0.0),
}

def decode_mask(segm, h, w):
    """Decode polygon or RLE COCO segmentation."""
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        rle = segm
    return maskUtils.decode(rle)

def plot_image_comparison(img_path, coco_gt_path, coco_pred_path, img_id, output_dir=None):
    coco_gt = COCO(coco_gt_path)
    coco_dt = COCO(coco_pred_path)

    img_info = coco_gt.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    img_path = path_join(img_path, file_name)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
    pred_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    titles = ["GT BBOXES", "Pred from bbox", "Segmentation Masks"]
    
    # ========== GT BOXES ==========
    axes[0].imshow(img)
    axes[0].set_title(titles[0])
    for ann in gt_anns:
        cat_name = coco_gt.loadCats(ann["category_id"])[0]["name"]
        color = color_map.get(cat_name, (1,1,1))
        x, y, bw, bh = ann["bbox"]
        rect = Rectangle((x, y), bw, bh, linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y-5, cat_name, color=color, fontsize=8, fontweight='bold')

    # ========== PRED BOXES ==========
    axes[1].imshow(img)
    axes[1].set_title(titles[1])
    for ann in pred_anns:
        cat_name = coco_gt.loadCats(ann["category_id"])[0]["name"]
        color = color_map.get(cat_name, (1,1,1))
        x, y, bw, bh = ann["bbox"]
        rect = Rectangle((x, y), bw, bh, linewidth=2, edgecolor=color, facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x, y-5, cat_name, color=color, fontsize=8, fontweight='bold')

    # ========== SEGMENTATION MASK ==========
    mask_vis = np.zeros((h, w, 3))
    for ann in pred_anns:
        cat_name = coco_gt.loadCats(ann["category_id"])[0]["name"]
        color = np.array(color_map.get(cat_name, (1,1,1)))
        m = decode_mask(ann["segmentation"], h, w)
        mask_vis[m > 0] = color
        
    for ann in gt_anns:
        cat_name = coco_gt.loadCats(ann["category_id"])[0]["name"]
        color = np.array(color_map.get(cat_name, (1,1,1)))
        x, y, bw, bh = ann["bbox"]
        rect = Rectangle((x, y), bw, bh, linewidth=2, edgecolor=color, facecolor='none')
        mask_vis = cv2.rectangle(mask_vis, (int(x), int(y)), (int(x+bw), int(y+bh)), color.tolist(), 2)
        mask_vis = cv2.putText(mask_vis, cat_name, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1, cv2.LINE_AA)    
        
    axes[2].imshow(mask_vis)
    axes[2].set_title(titles[2])
    
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    final_path = os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}_comparison.png") if output_dir else None
    plt.savefig(final_path)
    plt.close()
# Example usage:


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)    
    data_dir = path_join(parent_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    frames_dir = path_join(endoscapes_dir, 'frames')
    output_dir = path_join(this_dir, 'visualizations', 'bbox2segm_comparisons')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    coco_obj = COCO("/home/scanar/endovis/models/MaskDINO/data/endoscapes/calculate_masks/train_annotation_coco.json")
    img_ids = coco_obj.getImgIds()
    
    problematic_classes = [1, 2, 3, 4]
    
    with tqdm(total=len(img_ids), desc="Plotting bbox2segm comparisons") as pbar:
        for img_id in img_ids:
            
            anns = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=[img_id]))
            cats_names = [ann['category_id'] for ann in anns]
            
            if not any(cat in problematic_classes for cat in cats_names):
                continue
            else:            
                plot_image_comparison(
                        img_path=frames_dir,
                        coco_gt_path="/home/scanar/endovis/models/MaskDINO/data/endoscapes/calculate_masks/train_annotation_coco.json",
                        coco_pred_path="/home/scanar/endovis/models/MaskDINO/data/endoscapes/calculate_masks/bbox2segm_results.json",
                        img_id=img_id,
                        output_dir=output_dir)
            pbar.update(1)    
    print(f"Saved comparisons to {output_dir}")