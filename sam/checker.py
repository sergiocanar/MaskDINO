import json
from collections import Counter
from pycocotools import mask as maskUtils
import numpy as np
import cv2

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def decode_rle(rle):
    return maskUtils.decode(rle).astype(np.uint8)

def compare_coco_jsons(orig_path, post_path, check_iou=False):
    print(f"🔍 Comparing:\n - Original: {orig_path}\n - Post-proc: {post_path}\n")

    orig = load_json(orig_path)
    post = load_json(post_path)

    # === Basic counts ===
    print("📊 Basic stats")
    print(f"Images:       orig={len(orig['images'])}  post={len(post['images'])}")
    print(f"Annotations:  orig={len(orig['annotations'])}  post={len(post['annotations'])}")
    print(f"Categories:   orig={len(orig['categories'])}  post={len(post['categories'])}\n")

    # === Check unique IDs ===
    orig_ids = [a["id"] for a in orig["annotations"]]
    post_ids = [a["id"] for a in post["annotations"]]
    print(f"Unique ann IDs: orig={len(set(orig_ids))}, post={len(set(post_ids))}")

    dup_orig = len(orig_ids) - len(set(orig_ids))
    dup_post = len(post_ids) - len(set(post_ids))
    print(f"Duplicates:    orig={dup_orig}, post={dup_post}\n")

    # === Check category distribution ===
    orig_cats = Counter([a["category_id"] for a in orig["annotations"]])
    post_cats = Counter([a["category_id"] for a in post["annotations"]])

    print("📈 Category counts (orig → post):")
    all_cats = sorted(set(orig_cats.keys()) | set(post_cats.keys()))
    for cid in all_cats:
        print(f"  Cat {cid:>2}: {orig_cats[cid]:>5} → {post_cats[cid]:>5}")

    # === Optional IoU comparison (sample 5 random annotations per category) ===
    if check_iou:
        import random
        print("\n🧮 IoU sample comparison (first 5 per category)")
        post_anns_by_id = {a["id"]: a for a in post["annotations"]}
        for cid in all_cats:
            cat_anns = [a for a in orig["annotations"] if a["category_id"] == cid]
            for ann in random.sample(cat_anns, min(5, len(cat_anns))):
                ann_id = ann["id"]
                if ann_id not in post_anns_by_id:
                    continue
                mask1 = decode_rle(ann["segmentation"])
                mask2 = decode_rle(post_anns_by_id[ann_id]["segmentation"])
                inter = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                iou = inter / (union + 1e-6)
                print(f"  Cat {cid} | Ann {ann_id}: IoU={iou:.3f}")

    print("\n✅ Comparison finished.")

if __name__ == "__main__":
    # Example usage
    orig_json = "/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations_sam_extended/train_annotation_coco.json"
    post_json = "/home/scanar/endovis/models/MaskDINO/data/endoscapes/annotations_sam_postproc/train_annotation_coco.json"
    compare_coco_jsons(orig_json, post_json, check_iou=True)
