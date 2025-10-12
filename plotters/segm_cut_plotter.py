import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image


def decode_segmentation(segm, h, w):
    """
    Convert COCO segmentation (polygon or RLE) into a binary mask of shape [H, W].
    """
    if isinstance(segm, list):
        # Polygon — can have multiple parts
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm, dict) and "counts" in segm:
        # RLE
        rle = segm
    else:
        return np.zeros((h, w), dtype=np.uint8)
    mask = maskUtils.decode(rle)
    if len(mask.shape) == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def combine_masks(coco, img_id):
    """
    Combine all instance masks for a given image ID into one binary mask.
    """
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None

    img_info = coco.loadImgs(img_id)[0]
    h, w = img_info["height"], img_info["width"]

    combined = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        segm = ann.get("segmentation", None)
        if segm is not None:
            mask = decode_segmentation(segm, h, w)
            combined = np.maximum(combined, mask)
    return combined


def plot_random_samples_from_coco(
    frame_dir, cut_frame_dir,
    coco_json_path, cut_coco_json_path,
    output_dir, num_samples=5
):
    # --- Load COCOs ---
    coco = COCO(coco_json_path)
    cut_coco = COCO(cut_coco_json_path)

    os.makedirs(output_dir, exist_ok=True)

    # --- Get image IDs ---
    all_img_ids = coco.getImgIds()
    random.shuffle(all_img_ids)
    selected_ids = all_img_ids[:num_samples]
    print(f"🎲 Selected {len(selected_ids)} random samples")

    for img_id in selected_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        frame_path = os.path.join(frame_dir, file_name)
        cut_frame_path = os.path.join(cut_frame_dir, file_name)

        if not os.path.exists(frame_path) or not os.path.exists(cut_frame_path):
            print(f"⚠️ Missing frame or cut frame for {file_name}")
            continue

        frame = np.array(Image.open(frame_path).convert("RGB"))
        cut_frame = np.array(Image.open(cut_frame_path).convert("RGB"))

        # --- Combine masks ---
        mask_combined = combine_masks(coco, img_id)
        cut_img_ids = cut_coco.getImgIds(imgIds=[img_id]) or cut_coco.getImgIds()
        # Try to find matching by filename
        cut_id = None
        for cid in cut_img_ids:
            info = cut_coco.loadImgs(cid)[0]
            if os.path.basename(info["file_name"]) == os.path.basename(file_name):
                cut_id = cid
                break

        if cut_id is None:
            print(f"⚠️ No matching cut image for {file_name}")
            continue
        cut_mask_combined = combine_masks(cut_coco, cut_id)

        # --- Black background RGB masks ---
        def mask_to_rgb(mask):
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask == 1] = [255, 255, 255]
            return mask_rgb

        mask_rgb = mask_to_rgb(mask_combined) if mask_combined is not None else np.zeros_like(frame)
        cut_mask_rgb = mask_to_rgb(cut_mask_combined) if cut_mask_combined is not None else np.zeros_like(cut_frame)

        # --- Plot ---
        fig, axs = plt.subplots(1, 4, figsize=(20, 8))
        axs[0].imshow(frame)
        axs[0].set_title("Original Frame")
        axs[0].axis("off")

        axs[1].imshow(mask_rgb)
        axs[1].set_title("Original Segmentation (Black BG)")
        axs[1].axis("off")

        axs[2].imshow(cut_frame)
        axs[2].set_title("Cropped Frame")
        axs[2].axis("off")

        axs[3].imshow(cut_mask_rgb)
        axs[3].set_title("Cropped Segmentation (Black BG)")
        axs[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {save_path}")

# Example usage
plot_random_samples_from_coco(
    frame_dir="/home/scanar/endovis/models/MaskDINO/data/endoscapes/frames",
    cut_frame_dir="/home/scanar/endovis/models/MaskDINO/data/endoscapes_cutmargins/frames",
    coco_json_path="/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations/train_annotation_coco.json",
    cut_coco_json_path="/home/scanar/endovis/models/MaskDINO/data/endoscapes_cutmargins/annotations/train_annotation_coco.json",
    output_dir="/home/scanar/endovis/models/MaskDINO/visualizations/compare_segms",
    num_samples=5
)
