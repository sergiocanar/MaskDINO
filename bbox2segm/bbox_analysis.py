import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from os.path import join as path_join
from utils import create_directory_if_not_exists
from scipy.optimize import linear_sum_assignment

def bbox_iou_matrix(gt_boxes, pred_boxes):
    """
    Compute IoU matrix between two sets of COCO-style bboxes [x, y, w, h].
    Uses the same logic as COCOeval for 'bbox' mode.
    """
    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    pred_boxes = np.array(pred_boxes, dtype=np.float32)

    # Convert to [x1, y1, x2, y2]
    gt_xyxy = gt_boxes.copy()
    gt_xyxy[:, 2:] += gt_xyxy[:, :2]
    pred_xyxy = pred_boxes.copy()
    pred_xyxy[:, 2:] += pred_xyxy[:, :2]

    ious = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for i, g in enumerate(gt_xyxy):
        gx1, gy1, gx2, gy2 = g
        g_area = (gx2 - gx1) * (gy2 - gy1)
        for j, p in enumerate(pred_xyxy):
            px1, py1, px2, py2 = p
            p_area = (px2 - px1) * (py2 - py1)

            ix1, iy1 = max(gx1, px1), max(gy1, py1)
            ix2, iy2 = min(gx2, px2), min(gy2, py2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            union = g_area + p_area - inter
            ious[i, j] = inter / union if union > 0 else 0
    return ious

def match_and_compute_iou(gt_coco, pred_coco):
    """Hungarian matching per image & class using COCO bbox IoU logic."""
    all_ious = []

    for img_id in gt_coco.getImgIds():
        gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=[img_id]))
        pred_anns = pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=[img_id]))
        if not gt_anns or not pred_anns:
            continue

        # group by category
        gt_by_cls, pred_by_cls = {}, {}
        for ann in gt_anns:
            gt_by_cls.setdefault(ann["category_id"], []).append(ann["bbox"])
        for ann in pred_anns:
            pred_by_cls.setdefault(ann["category_id"], []).append(ann["bbox"])

        # match per class
        for cat_id, gt_boxes in gt_by_cls.items():
            pred_boxes = pred_by_cls.get(cat_id, [])
            if not pred_boxes:
                continue

            ious = bbox_iou_matrix(gt_boxes, pred_boxes)
            cost = 1 - ious
            row_ind, col_ind = linear_sum_assignment(cost)
            all_ious.extend(ious[row_ind, col_ind])

    return np.array(all_ious)


def analyze_bbox_alignment(gt_json, pred_json, output_hist=None):
    
    final_hist_path = path_join(output_hist, 'bbox_hist.png')
    
    gt_coco = COCO(gt_json)
    pred_coco = COCO(pred_json)

    ious = match_and_compute_iou(gt_coco, pred_coco)
    print(f"Matched pairs: {len(ious)}")
    print(f"Mean IoU: {ious.mean():.3f} | Median IoU: {np.median(ious):.3f}")

    plt.figure(figsize=(8, 5))
    plt.hist(ious, bins=20, color="royalblue", edgecolor="black")
    plt.title("Distribution of IoU Between GT and Predicted BBoxes")
    plt.xlabel("IoU (Jaccard Index)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(final_hist_path)
    plt.close()
    print(f"Histogram saved to {output_hist}")
    return ious


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, 'data')
    endoscapes_dir = path_join(data_dir, 'endoscapes')
    endo2023_bbox201_dir = path_join(data_dir, 'endoscapesbbox201')
    gt_json = path_join(endo2023_bbox201_dir, 'bbox201_annotation_coco.json')
    pred_json = os.path.join(endoscapes_dir, 'calculate_masks', 'all_seg_201.json')
    
    output_dir = path_join(this_dir, 'visualizations')
    create_directory_if_not_exists(output_dir)    
    
    analyze_bbox_alignment(gt_json, pred_json, output_hist=output_dir)
