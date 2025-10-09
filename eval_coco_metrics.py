import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute COCO-style mAP between predictions and ground truth JSONs')
    parser.add_argument('--gt', required=True, help='Path to COCO-format ground truth JSON file')
    parser.add_argument('--pred', required=True, help='Path to COCO-format predictions JSON file')
    parser.add_argument('--iou-thresholds', nargs='+', type=float, default=[0.5],
            help='List of IoU thresholds to evaluate, defaulting to 0.5 (primary metric)')
    parser.add_argument('--max-dets', nargs=3, type=int, default=[1, 10, 100],
                        help='Per-image max detections as three values, e.g., 1 10 100')
    return parser.parse_args()

def load_predictions(pred_path):
    """
    Load a predictions file that may contain full COCO dataset keys, extracting only the 'annotations' list.
    Supported formats:
      - A list of detection dicts
      - A dict with keys 'images', 'annotations', and 'categories'
    """
    with open(pred_path, 'r') as f:
        data = json.load(f)
    # If file is full COCO-format dataset, extract annotations
    if isinstance(data, dict) and 'annotations' in data:
        preds = data['annotations']
    elif isinstance(data, list):
        preds = data
    else:
        raise ValueError(
            "Unsupported prediction file format: must be a list or contain 'annotations' key.")
    return preds

def main():
    args = parse_args()

    # Load COCO ground truth
    coco_gt = COCO(args.gt)

    # Load and preprocess predictions
    pred_list = load_predictions(args.pred)

    coco_dt = coco_gt.loadRes(pred_list)

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    # Set IoU thresholds and max detections
    # coco_eval.params.iouThrs = np.array(args.iou_thresholds)
    coco_eval.params.maxDets = args.max_dets

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    main()
