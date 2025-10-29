import os
import sys

import json
import numpy as np
from pycocotools.coco import COCO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from os.path import join as path_join
from utils import create_directory_if_not_exists, save_json
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


def filter_predictions_by_iou(gt_json, pred_json, output_json):
    gt_coco = COCO(gt_json)
    pred_coco = COCO(pred_json)

    # --- Compute IoUs per detection ---
    iou_map = {}  # (img_id, ann_id) → IoU
    all_ious = []

    for img_id in gt_coco.getImgIds():
        gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=[img_id]))
        pred_anns = pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=[img_id]))
        if not gt_anns or not pred_anns:
            continue

        gt_by_cls, pred_by_cls = {}, {}
        for ann in gt_anns:
            gt_by_cls.setdefault(ann["category_id"], []).append(ann["bbox"])
        for ann in pred_anns:
            pred_by_cls.setdefault(ann["category_id"], []).append((ann["id"], ann["bbox"]))

        for cat_id, gt_boxes in gt_by_cls.items():
            preds = pred_by_cls.get(cat_id, [])
            if not preds:
                continue

            # Compute IoU matrix
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            pred_boxes = np.array([p[1] for p in preds], dtype=np.float32)
            gt_boxes[:, 2:] += gt_boxes[:, :2]
            pred_boxes[:, 2:] += pred_boxes[:, :2]

            ious = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
            for i, g in enumerate(gt_boxes):
                gx1, gy1, gx2, gy2 = g
                g_area = (gx2 - gx1) * (gy2 - gy1)
                for j, p in enumerate(pred_boxes):
                    px1, py1, px2, py2 = p
                    p_area = (px2 - px1) * (py2 - py1)
                    ix1, iy1 = max(gx1, px1), max(gy1, py1)
                    ix2, iy2 = min(gx2, px2), min(gy2, py2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    union = g_area + p_area - inter
                    iou = inter / union if union > 0 else 0
                    ious[i, j] = iou

            # Hungarian assignment
            from scipy.optimize import linear_sum_assignment
            cost = 1 - ious
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                matched_pred_id = preds[c][0]
                iou_val = ious[r, c]
                iou_map[(img_id, matched_pred_id)] = iou_val
                all_ious.append(iou_val)

    # --- Filter above mean IoU ---
    mean_iou = np.mean(all_ious)
    print(f"Mean IoU threshold: {mean_iou:.3f}")

    with open(pred_json, "r") as f:
        pred_data = json.load(f)

    filtered_anns = [
        ann for ann in pred_data["annotations"]
        if iou_map.get((ann["image_id"], ann["id"]), 0) >= 0.75
    ]

    filtered_json = pred_data.copy()
    filtered_json["annotations"] = filtered_anns
    
    save_json(filtered_json, output_json)
    
    print(f"Saved filtered predictions ({len(filtered_anns)} remain) → {output_json}")
    return mean_iou, len(filtered_anns)

def bbox_iou_matrix(gt_boxes, pred_boxes):
    """Compute IoU matrix between two sets of [x,y,w,h] boxes."""
    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    pred_boxes = np.array(pred_boxes, dtype=np.float32)

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


def compute_iou_per_class(gt_json, pred_json, out_dir="iou_per_class"):
    os.makedirs(out_dir, exist_ok=True)

    gt_coco = COCO(gt_json)
    pred_coco = COCO(pred_json)

    class_ious = {cat["id"]: [] for cat in gt_coco.loadCats(gt_coco.getCatIds())}
    class_names = {cat["id"]: cat["name"] for cat in gt_coco.loadCats(gt_coco.getCatIds())}

    for img_id in gt_coco.getImgIds():
        gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=[img_id]))
        pred_anns = pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=[img_id]))
        if not gt_anns or not pred_anns:
            continue

        gt_by_cls, pred_by_cls = {}, {}
        for ann in gt_anns:
            gt_by_cls.setdefault(ann["category_id"], []).append(ann["bbox"])
        for ann in pred_anns:
            pred_by_cls.setdefault(ann["category_id"], []).append(ann["bbox"])

        for cat_id, gt_boxes in gt_by_cls.items():
            preds = pred_by_cls.get(cat_id, [])
            if not preds:
                continue

            ious = bbox_iou_matrix(gt_boxes, preds)
            cost = 1 - ious
            row_ind, col_ind = linear_sum_assignment(cost)
            class_ious[cat_id].extend(ious[row_ind, col_ind])

    # --- Plot histograms per class ---
    summary = {}
    for cat_id, ious in class_ious.items():
        if len(ious) == 0:
            continue

        ious = np.array(ious)
        mean_iou = ious.mean()
        median_iou = np.median(ious)
        summary[class_names[cat_id]] = (mean_iou, median_iou, len(ious))

        plt.figure(figsize=(6,4))
        plt.hist(ious, bins=20, color="steelblue", edgecolor="black")
        plt.title(f"{class_names[cat_id]} — IoU Distribution\nMean={mean_iou:.2f}, Median={median_iou:.2f}")
        plt.xlabel("IoU (Jaccard Index)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{class_names[cat_id]}_iou_hist.png"))
        plt.close()

    # --- Combined summary plot ---
    if summary:
        labels, means = zip(*[(k, v[0]) for k, v in summary.items()])
        plt.figure(figsize=(8,4))
        plt.bar(labels, means, color="royalblue", edgecolor="black")
        plt.ylabel("Mean IoU")
        plt.title("Mean IoU per Class")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mean_iou_summary.png"))
        plt.close()

    print("Saved per-class IoU histograms and summary to", out_dir)
    return summary


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
    
    filtered_output_json = path_join(output_dir, 'filtered_predictions.json')
    filter_predictions_by_iou(gt_json, pred_json, filtered_output_json)
    
    stats_dir = path_join(output_dir, 'iou_per_class')
    create_directory_if_not_exists(stats_dir)
    stats = compute_iou_per_class(gt_json, pred_json, out_dir=stats_dir)
    for class_name, (mean_iou, median_iou, count) in stats.items():
        print(f"Class: {class_name} | Mean IoU: {mean_iou:.3f} | Median IoU: {median_iou:.3f} | Count: {count}")
