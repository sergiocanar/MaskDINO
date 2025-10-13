import itertools
import os
import os.path as osp
import json
import numpy as np
import sklearn
import argparse
import pandas as pd
from copy import deepcopy

from .classification_eval import eval_classification, eval_presence
from .detection_eval import eval_detection
from .semantic_segmentation_eval import eval_segmentation as eval_sem_segmentation
from .instance_segmentation_eval import eval_segmentation as eval_inst_segmentation
from .utils import load_json, save_json


def eval_segmentation(
    task,
    coco_anns,
    preds,
    img_ann_dict,
    mask_path,
    main_metric="mAP@0.5IoU_segm",
    **kwargs,
):
    inst_seg_results, aux_inst_seg = eval_inst_segmentation(
        task, coco_anns, preds, img_ann_dict
    )
    print("{} task mAP@0.5IoU_segm: {}".format(task, round(inst_seg_results, 8)))

    sem_seg_results, aux_sem_seg = eval_sem_segmentation(
        task, coco_anns, preds, img_ann_dict, mask_path=mask_path
    )
    print("{} task mIoU: {}".format(task, round(sem_seg_results, 8)))

    if main_metric == "mAP@0.5IoU_segm":
        main_metric = inst_seg_results
        aux_metrics = {"mIoU": sem_seg_results}
        aux_metrics.update(aux_sem_seg)
        aux_metrics.update(aux_inst_seg)
    elif main_metric == "mIoU":
        main_metric = sem_seg_results
        aux_metrics = {"mAP@0.5IoU_segm": inst_seg_results}
        aux_metrics.update(aux_sem_seg)
        aux_metrics.update(aux_inst_seg)
    return main_metric, aux_metrics


METRIC_DICT = {
    "mAP": eval_classification,
    "mAP@0.5IoU_box": eval_detection,
    "mAP@0.5IoU_segm": eval_inst_segmentation,
    "mIoU": eval_sem_segmentation,
    "mIoU_mAP@0.5": eval_segmentation,
    "mAP_pres": eval_presence,
    "classification": eval_classification,
    "detection": eval_detection,
    "inst_segmentation": eval_inst_segmentation,
    "sem_segmentation": eval_sem_segmentation,
    "segmentation": eval_segmentation,
    "presence": eval_presence,
}


def get_img_ann_dict(coco_anns, task):
    img_ann_dict = {}
    for img in coco_anns["images"]:
        img_ann_dict[img["file_name"]] = []

    if not "image_name" in coco_anns["annotations"][0].keys():
        id2image_name = {}
        for img in coco_anns["images"]:
            id2image_name[img["id"]] = img["file_name"]
        for ann in coco_anns["annotations"]:
            ann["image_name"] = id2image_name[ann["image_id"]]

    for idx, ann in enumerate(coco_anns["annotations"]):
        if (
            (task == "instruments" and "category_id" in ann) and ann["category_id"] >= 0
        ) or (
            task in ann
            and (ann[task] >= 0 if type(ann[task]) is int else min(ann[task]) >= 0)
        ):
            img_ann_dict[ann["image_name"]].append(idx)

    return img_ann_dict


def eval_task(task, metric, coco_anns, preds, masks_path):
    try:
        metric_funct = METRIC_DICT[metric]
    except KeyError:
        raise NotImplementedError(f"Metric {metric} is not supported")

    datasets = [img.get("dataset") for img in coco_anns["images"]]
    if not any(dataset is None for dataset in datasets):
        set_datasets = sorted(list(set(datasets)), key=lambda x: x.lower())
    else:
        set_datasets = []
    if len(set_datasets) >= 1:
        img_names_per_dataset = [
            [
                img["file_name"]
                for img in coco_anns["images"]
                if img["dataset"] == dataset
            ]
            for dataset in set_datasets
        ]
        coco_anns_per_dataset = [
            create_coco_anns_per_dataset(coco_anns, dataset) for dataset in set_datasets
        ]
        preds_per_dataset = [
            {
                img_name: preds[img_name]
                for img_name in preds
                if img_name in img_names_per_dataset[idx]
            }
            for idx in range(len(set_datasets))
        ]
        main_metric = []
        aux_metrics = {}
        for idx, dataset in enumerate(set_datasets):
            coco_anns = coco_anns_per_dataset[idx]
            preds = preds_per_dataset[idx]
            img_ann_dict = get_img_ann_dict(coco_anns, task)

            main_metric_ind, aux_metrics_ind = metric_funct(
                task,
                coco_anns,
                preds,
                img_ann_dict=img_ann_dict,
                main_metric="mAP@0.5IoU_segm",
                mask_path=masks_path,
            )
            main_metric.append(main_metric_ind)
            aux_metrics["mAP@0.5IoU_segm-" + dataset] = main_metric_ind
            for key, value in aux_metrics_ind.items():
                aux_metrics[key + "-" + dataset] = value
        main_metric = float(np.mean(main_metric))
        return main_metric, aux_metrics

    else:
        # If there is only one dataset or no dataset information
        img_ann_dict = get_img_ann_dict(coco_anns, task)

        main_metric, aux_metrics = metric_funct(
            task, coco_anns, preds, img_ann_dict=img_ann_dict, mask_path=masks_path
        )
        return main_metric, aux_metrics


def create_coco_anns_per_dataset(coco_anns, dataset):
    coco_anns_per_dataset = {
        "images": [],
        "annotations": [],
        "categories": coco_anns["categories"],
    }
    img_names = []
    for img in coco_anns["images"]:
        if img["dataset"] == dataset:
            coco_anns_per_dataset["images"].append(deepcopy(img))
            img_names.append(img["file_name"])

    for ann in coco_anns["annotations"]:
        if ann["image_name"] in img_names:
            coco_anns_per_dataset["annotations"].append(deepcopy(ann))

    return coco_anns_per_dataset


def main_per_task(coco_ann_path, pred_path, task, metric, masks_path=None):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path) if type(pred_path) == str else pred_path

    task_eval, aux_metrics = eval_task(task, metric, coco_anns, preds, masks_path)
    aux_metrics = dict(
        zip(aux_metrics.keys(), map(lambda x: round(x, 8), aux_metrics.values()))
    )
    print("{} task {}: {} {}".format(task, metric, round(task_eval, 8), aux_metrics))

    final_metrics = {metric: round(task_eval, 8)}
    final_metrics.update(aux_metrics)
    return final_metrics
