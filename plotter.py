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
    """
    Convert COCO segmentation (polygon, RLE, or uncompressed RLE)
    into a binary mask of shape [H, W].
    """
    if isinstance(segm, list):
        # Polygon -- a single object can consist of multiple parts
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm, dict) and "counts" in segm:
        # RLE
        rle = segm
    else:
        return np.zeros((h, w), dtype=np.uint8)

    mask = maskUtils.decode(rle)
    return mask

def plot_random_samples(coco_json_path: str, images_dir: str, output_dir: str, num_samples: int = 10):
    coco = COCO(coco_json_path)
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)
    selected_ids = img_ids[:num_samples]

    # Fixed color per class for consistency
    color_map = {
        "gallbladder": (0.0, 1.0, 1.0),     # cyan
        "cystic_duct": (0.0, 1.0, 0.0),     # green
        "cystic_artery": (1.0, 0.0, 1.0),   # magenta
        "cystic_plate": (1.0, 1.0, 0.0),    # yellow
        "tool": (0.2, 0.2, 1.0),            # blue
        "background": (0.0, 0.0, 0.0),      # black (for fallback)
    }

    for img_id in selected_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        img = plt.imread(img_path)
        h, w = img.shape[:2]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # --- Create figure with two subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        ax_img, ax_mask = axes

        # --- LEFT: Image + Bounding Boxes ---
        ax_img.imshow(img)
        ax_img.set_title(f"Image ID: {img_id} ({len(anns)} objects)", fontsize=12)

        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = coco.loadCats(cat_id)[0]["name"]
            color = color_map.get(cat_name, np.random.rand(3))

            # Draw bbox
            if "bbox" in ann:
                x, y, w_box, h_box = ann["bbox"]
                rect = Rectangle((x, y), w_box, h_box, linewidth=2,
                                 edgecolor=color, facecolor="none")
                ax_img.add_patch(rect)
                ax_img.text(x, y - 3, cat_name, color=color, fontsize=10, weight='bold')

        ax_img.axis("off")

        # --- RIGHT: Masks over black background ---
        mask_canvas = np.zeros((h, w, 3), dtype=np.float32)  # RGB only (black background)

        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = coco.loadCats(cat_id)[0]["name"]
            color = color_map.get(cat_name, np.random.rand(3))

            if "segmentation" in ann and ann["segmentation"]:
                mask = decode_segmentation(ann["segmentation"], h, w)
                if mask.sum() == 0:
                    continue

                # Paint mask pixels with the class color
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
    coco_path = path_join(annots_dir, 'all_seg_201.json')
    output_dir = path_join(this_dir, 'visualizations', 'endoscapes201')
    os.makedirs(output_dir, exist_ok=True)
    
    
    plot_random_samples(coco_path,
                        frames_dir, 
                        output_dir=output_dir,
                        num_samples=10)
