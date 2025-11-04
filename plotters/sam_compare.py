import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# =========================
# CONFIG
# =========================
FRAMES_DIR = "/home/scanar/endovis/models/MaskDINO/data/endoscapes/frames"
ORIG_JSON  = "/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations_sam_extended/train_annotation_coco.json"
POST_JSON  = "/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations_sam_postproc/train_annotation_coco.json"
OUT_DIR    = "/home/scanar/endovis/models/MaskDINO/plotters/visualizations/sam_post_proc"
N_SAMPLES  = 10   # number of random frames to visualize

os.makedirs(OUT_DIR, exist_ok=True)

# category → RGB color map
COLOR_MAP = {
    1: (248, 231, 28),    # cystic_plate   (yellow)
    2: (74, 144, 226),    # calot_triangle (blue)
    3: (218, 13, 15),     # cystic_artery  (red)
    4: (65, 117, 6),      # cystic_duct    (dark green)
    5: (126, 211, 33),    # gallbladder    (light green)
    6: (245, 166, 35),    # tool           (orange)
}

# =========================
# FUNCTIONS
# =========================
def decode_rle_robust(rle):
    """Decode a COCO RLE mask into (H, W) uint8, safely handling edge cases."""
    try:
        mask = maskUtils.decode(rle)
    except Exception:
        return None
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    elif mask.ndim == 1:
        if "size" in rle and len(rle["size"]) == 2:
            h, w = rle["size"]
            if h * w == mask.size:
                mask = mask.reshape((h, w))
            else:
                return None
        else:
            return None
    return (mask > 0).astype(np.uint8)


def overlay_mask(img, mask, color, alpha=0.5):
    """Overlay a binary mask on RGB image."""
    if mask is None or mask.sum() == 0:
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert mask to boolean and expand to 3 channels
    mask_bool = mask.astype(bool)
    
    # Create color overlay
    color_arr = np.array(color, dtype=np.float32)
    
    # Apply overlay where mask is True
    out = img.copy().astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.where(mask_bool, 
                                 (1 - alpha) * img[:, :, c] + alpha * color_arr[c],
                                 img[:, :, c])
    
    return out.astype(np.uint8)

def render_image(coco, img_info, frame_dir, color_map, alpha=0.5):
    """Draw all masks for one frame."""
    path = os.path.join(frame_dir, img_info["file_name"])
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ann_ids = coco.getAnnIds(imgIds=[img_info["id"]])
    anns = coco.loadAnns(ann_ids)

    overlay = img.copy()
    for ann in anns:
        mask = decode_rle_robust(ann["segmentation"])
        if mask is None:
            print(f"⚠️  Bad RLE for ann {ann['id']} → skipped")
            continue
        try:
            overlay = overlay_mask(overlay, mask, color_map.get(ann["category_id"], (255, 255, 255)), alpha)
        except Exception as e:
            print(f"⚠️ Skipped ann {ann['id']} due to error: {e}")
            continue
    return overlay


def plot_and_save_comparison(img_info, img_orig, img_post, out_dir):
    """Save side-by-side comparison figure."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{img_info['file_name']} | Image ID {img_info['id']}", fontsize=11)

    axs[0].imshow(img_orig)
    axs[0].set_title("Original SAM")
    axs[0].axis("off")

    axs[1].imshow(img_post)
    axs[1].set_title("Post-Processed SAM")
    axs[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"compare_{img_info['file_name'].replace('/', '_')}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    coco_orig = COCO(ORIG_JSON)
    coco_post = COCO(POST_JSON)

    img_ids = coco_orig.getImgIds()
    random.shuffle(img_ids)

    for img_id in img_ids[:N_SAMPLES]:
        img_info = coco_orig.loadImgs([img_id])[0]
        img_post_info = coco_post.loadImgs([img_id])[0]

        try:
            img_orig = render_image(coco_orig, img_info, FRAMES_DIR, COLOR_MAP)
            img_post = render_image(coco_post, img_post_info, FRAMES_DIR, COLOR_MAP)
            plot_and_save_comparison(img_info, img_orig, img_post, OUT_DIR)
        except Exception as e:
            print(f"⚠️ Skipped {img_info['file_name']} due to error: {e}")

    print(f"\n🎉 All comparison plots saved to:\n{OUT_DIR}")
