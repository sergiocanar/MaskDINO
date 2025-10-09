import os
import random
import json
from os.path import join as path_join
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from matplotlib.patches import Rectangle

def decode_segmentation(segm, h, w):
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm, dict) and "counts" in segm:
        rle = segm
    else:
        return np.zeros((h, w), dtype=np.uint8)
    return maskUtils.decode(rle)

def plot_random_samples(
    coco_json_path: str,
    coco_bbox_json_path: str,
    images_dir: str,
    output_dir: str,
    num_samples: int = 10
):
    coco_gt = COCO(coco_json_path)
    coco_bbox = COCO(coco_bbox_json_path)

    img_ids = coco_gt.getImgIds()
    random.shuffle(img_ids)
    selected_ids = img_ids[:num_samples]

    color_map = {
        "gallbladder": (0.0, 1.0, 1.0),
        "cystic_duct": (0.0, 1.0, 0.0),
        "cystic_artery": (1.0, 0.0, 1.0),
        "cystic_plate": (1.0, 1.0, 0.0),
        "tool": (0.2, 0.2, 1.0),
        "background": (0.0, 0.0, 0.0),
    }

    for img_id in selected_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        img = plt.imread(img_path)
        h, w = img.shape[:2]

        ann_ids_gt = coco_gt.getAnnIds(imgIds=img_id)
        anns_gt = coco_gt.loadAnns(ann_ids_gt)

        ann_ids_bbox = coco_bbox.getAnnIds(imgIds=img_id)
        anns_bbox = coco_bbox.loadAnns(ann_ids_bbox)

        # --- Three subplots ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        ax_bbox, ax_gt, ax_mask = axes

        # --- LEFT: Image + bboxes from secondary JSON ---
        ax_bbox.imshow(img)
        ax_bbox.set_title("GT BBOXES", fontsize=12)
        for ann in anns_bbox:
            cat_id = ann["category_id"]
            cat_name = coco_bbox.loadCats(cat_id)[0]["name"]
            color = color_map.get(cat_name, np.random.rand(3))
            if "bbox" in ann:
                x, y, w_box, h_box = ann["bbox"]
                rect = Rectangle((x, y), w_box, h_box, linewidth=2,
                                 edgecolor=color, facecolor="none")
                ax_bbox.add_patch(rect)
                ax_bbox.text(x, y - 3, cat_name, color=color, fontsize=10, weight='bold')
        ax_bbox.axis("off")

        # --- MIDDLE: Ground-truth bboxes ---
        ax_gt.imshow(img)
        ax_gt.set_title(f"Pred from bbox ({len(anns_gt)} objects)", fontsize=12)
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

        # --- RIGHT: Masks on black background ---
        mask_canvas = np.zeros((h, w, 3), dtype=np.float32)
        for ann in anns_gt:
            cat_id = ann["category_id"]
            cat_name = coco_gt.loadCats(cat_id)[0]["name"]
            color = color_map.get(cat_name, np.random.rand(3))
            if "segmentation" in ann and ann["segmentation"]:
                mask = decode_segmentation(ann["segmentation"], h, w)
                if mask.sum() == 0:
                    continue
                for c in range(3):
                    mask_canvas[..., c] = np.where(mask == 1, color[c], mask_canvas[..., c])
        ax_mask.imshow(mask_canvas)
        ax_mask.set_title("Segmentation Masks", fontsize=12)
        ax_mask.axis("off")

        plt.tight_layout()
        save_path = path_join(output_dir, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, 'data', 'endoscapes')
    frames_dir = path_join(data_dir, 'frames')
    annots_dir = path_join(data_dir, 'annotations_201')
    annots2_dir = path_join(data_dir, '201_annotations')
    coco_path = path_join(annots_dir, 'all_seg_201.json')  # GT masks
    coco_bbox_path = path_join(annots2_dir, 'train_annotation_coco.json')  # secondary bboxes
    output_dir = path_join(this_dir, 'visualizations', 'endoscapes201_compare_new')
    os.makedirs(output_dir, exist_ok=True)

    plot_random_samples(coco_path, coco_bbox_path, frames_dir, output_dir, num_samples=10)
