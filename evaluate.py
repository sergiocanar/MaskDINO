import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from pycocotools.coco import COCO
from os.path import join as path_join
from utils import load_json, save_json
from evaluate.main_eval import eval_task
from pycocotools.cocoeval import COCOeval
from evaluate.utils import load_json, read_detectron2_output

def eval_parser():
    
    parser = argparse.ArgumentParser(description="Evaluation parser")
    parser.add_argument(
        "--coco-ann-path",
        default=None,
        type=str,
        required=True,
        help="Path to coco anotations",
    )
    parser.add_argument(
        "--pred-path", default=None, type=str, required=True, help="Path to predictions"
    )
    parser.add_argument("--filter", action="store_true", help="Filter predictions")
    parser.add_argument("--coco", action="store_true", help="Compute COCO metrics")
    parser.add_argument(
        "--tasks", nargs="+", help="Tasks to be evaluated", default=None, required=True
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to be evaluated",
        choices=[
            "mAP",  # Classification mean Average Precision
            "mAP@0.5IoU_box",  # Detection mean Average Precision with 0.5 bounding box IoU threshold
            "mAP@0.5IoU_segm",  # Instance Segmentation mean Average Precision with 0.5 mask IoU threshold
            "mIoU",  # Semantic segmentation mean Intersection over Union
            "mIoU_mAP@0.5",  # Semantic segmentation IoU and instance segmentation mean Average Precision with a 0.5 mask threshold
            "classification",  # Same as 'mAP' (Classification mean Average Precision)
            "detection",  # Same as 'mAP@IoU_box' (Detection mean Average Precision with 0.5 bounding box IoU threshold)
            "inst_segmentation",  # Same as 'mAP@0.5IoU_segm' (Instance Segmentation mean Average Precision with 0.5 mask IoU threshold)
            "sem_segmentation",  # Same as 'mIoU' (Semantic segmentation mean Intersection over Union)
            "segmentation",  # Same as 'mIoU_mAP@0.5' (Semantic segmentation IoU and instance segmentation mean Average Precision with a 0.5 mask threshold)
        ],
        default=None,
        required=True,
    )
    parser.add_argument(
        "--masks-path",
        type=str,
        required=False,
        help="Path to semantic segmentation ground truth images",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="thresh",
        choices=[
            "thresh",  # General threshold filtering
            "topk",  # General top k filtering
            "topk_thresh",  # Threshold and top k filtering
            "cls_thresh",  # Per-class threshold filtering
            "cls_topk",  # Per-class top k filtering
            "cls_topk_thresh",  # Per-class top k and and threshold filtering
            "all",  # No filtering
        ],
        required=False,
        help="Prediction filtering method",
    )
    parser.add_argument(
        "--selection_info",
        help="Hypermarameters to perform filtering",
        required=False,
        default=0.75,
    )
    parser.add_argument(
        "--output_path", default=None, type=str, help="Output directory"
    )

    args = parser.parse_args()
    print(args)
    
    return args
def run_coco_eval(gt_path, preds, eval_type='segm'):
    coco_gt = COCO(gt_path)

    if isinstance(preds, dict) and "annotations" in preds:
        preds = preds["annotations"]

    # Clean unnecessary fields
    for p in preds:
        for k in ["decoder_out", "score_dist", "mask_embd", "global_ft"]:
            p.pop(k, None)
        if eval_type == "segm":
            p.pop("bbox", None)
    
    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, eval_type)
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Global metrics
    names = [
        "AP@[.5:.95]", "AP@0.5", "AP@0.75",
        "AP_small", "AP_medium", "AP_large",
        "AR@[.5:.95]_1", "AR@[.5:.95]_10", "AR@[.5:.95]_100",
        "AR_small", "AR_medium", "AR_large"
    ]
    metrics = {name: round(float(val) * 100, 3) for name, val in zip(names, coco_eval.stats)}

    # --- Per-class AP computation ---
    per_class_ap = {}
    for catId in coco_gt.getCatIds():
        nm = coco_gt.loadCats(catId)[0]['name']
        # Compute mean precision over IoUs and recalls for this category
        precision = coco_eval.eval['precision'][:, :, catId-1, 0, -1]
        precision = precision[precision > -1]
        if precision.size:
            per_class_ap[nm] = round(np.mean(precision) * 100, 3)
        else:
            per_class_ap[nm] = float('nan')
    
    metrics['per_class_AP'] = per_class_ap
        
    return metrics


def format_horizontal_table(metrics_dict, n_cols=6, default_category="Overall"):
    assert n_cols % 3 == 0, "n_cols debe ser múltiplo de 3"

    formatted = []
    for k, v in metrics_dict.items():
        parts = k.rsplit("-", 2)
        if len(parts) == 3:
            category, metric, _ = parts
        elif len(parts) == 2:
            metric, _ = parts
            category = default_category
        else:
            category = default_category
            metric = k.replace(f"-{default_category}", "")
        formatted.append([category, metric, v])

    row_len = n_cols // 3
    results_2d = [formatted[i : i + row_len] for i in range(0, len(formatted), row_len)]
    results_2d = [sum(row, []) for row in results_2d]
    headers = ["Category", "Metric", "Value"] * (n_cols // 3)

    return tabulate(
        results_2d, headers=headers, tablefmt="pipe", floatfmt=".3f", numalign="left"
    )


def main(coco_ann_path: str, pred_path:str, compute_coco: bool, preds: dict, tasks: list, metrics: list, output_dir: str, sufix: str, masks_path: str = None):
    
    '''
    Main function for evaluation different segmentation tasks. 
    
    This function takes a COCO annotation path. Computes COCO bbox or segmentation metrics as user defined.
    
    Arguments:
    - coco_ann_path (str): Path to the COCO annotation path. Needs to be .json
    - pred_path (str): Path to the COCO predictions path. Needs to be .json. In this case Im using MaskDINO outputs
    - preds (dict): This preds are a dictionary extracted from the Detectron2output. This is necessary to have compatibility for metrics calculation
    - tasks (list): List of tasks to be completed.
    - metrics (list): List of metrics to be completed.
    - output_dir (str): Output directory to save metrics files.
    - sufix (str): Parameter specific for metrics calculation.
    - masks_path (str): Path to masks if available.
    
    '''
    
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds_raw = load_json(pred_path) if type(pred_path) == str else pred_path
    all_metrics = {}
    
    if compute_coco:        
        if "mAP@0.5IoU_box" in metrics or "detection" in metrics:
                coco_type = "bbox"
        elif "mAP@0.5IoU_segm" in metrics or "inst_segmentation" in metrics or "segmentation" in metrics:
                coco_type = "segm"
        else:
                coco_type = None

        if coco_type:
            print(f"\n Running official COCO evaluation ({coco_type})...")
            print(f"--------------------------------------------------------")
            coco_metrics = run_coco_eval(coco_ann_path, preds_raw, coco_type)
            all_metrics[f"COCO_{coco_type}"] = coco_metrics
    else:
        print('Skipping COCO metrics computation...')
    
    datasets = [img.get("dataset") for img in coco_anns["images"]]
    
    if not any(dataset is None for dataset in datasets):
        datasets = sorted(list(set(datasets)), key=lambda x: x.lower())
    else:
        datasets = False
    for task, metric in zip(tasks, metrics):

        # Evaluate task, in this case segmentation in general.
        task_eval, aux_metrics = eval_task(task, metric, coco_anns, preds, masks_path)
        aux_metrics = dict(
            zip(aux_metrics.keys(), map(lambda x: round(x, 8), aux_metrics.values()))
        )
        if datasets:
            print("{} task {}: {}".format(task, metric, round(task_eval, 8)))
            for dataset in datasets:
                dataset_metrics = {k: v for k, v in aux_metrics.items() if dataset in k}
                general_metrics = {}
                category_metrics = {}

                modified_dataset = (
                    dataset.replace("-", "_") if "-" in dataset else dataset
                )

                for k, v in dataset_metrics.items():
                    prefix = k.replace(f"-{dataset}", "")
                    if modified_dataset != dataset:
                        k = k.replace(dataset, modified_dataset)
                    if prefix.count("-") >= 1:
                        category_metrics[k] = v
                    else:
                        general_metrics[k] = v

                if general_metrics:
                    print(f"\nMétricas generales para dataset: {dataset}")
                    print(
                        format_horizontal_table(
                            general_metrics, n_cols=6, default_category="Overall"
                        )
                    )

                if category_metrics:
                    print(f"\nMétricas por categoría para dataset: {dataset}")
                    print(format_horizontal_table(category_metrics, n_cols=6))
        else:
            print(
                "{} task {}: {} {}".format(
                    task, metric, round(task_eval, 8), aux_metrics
                )
            )
        final_metrics = {metric: round(task_eval, 8)}
        final_metrics.update(aux_metrics)
        all_metrics[task] = final_metrics

        if output_dir is not None and sufix is not None:
            if metric in ["mAP@0.5IoU_box", "detection"]:
                met_suf = "det"
            elif metric in ["mAP@0.5IoU_segm", "inst_segmentation"]:
                met_suf = "ins_seg"
            elif metric in ["mIoU", "sem_segmentation"]:
                met_suf = "sem_seg"
            elif metric == "mIoU_mAP@0.5":
                met_suf = "seg"
            else:
                met_suf = "class"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(os.path.join(output_dir, f"metrics_{met_suf}.json")):
                file_name = path_join(output_dir, f"metrics_{met_suf}.json")
                save_json_data = load_json(file_name)
                save_json_data[sufix] = all_metrics[task]
                if compute_coco:
                    save_json_data["COCO_segm"] = all_metrics.get('COCO_segm', {})
            else:
                save_json_data = {
                    sufix: all_metrics[task]
                }
                if compute_coco:
                    save_json_data["COCO_segm"] = all_metrics.get('COCO_segm', {})
                    
            save_json_path = path_join(output_dir, f"metrics_{met_suf}.json")
            save_json(save_json_data, save_json_path)
        
        
            excel_file = os.path.join(output_dir, f"metrics_{met_suf}.xlsx")
            if os.path.exists(excel_file):
                existing_df = pd.read_excel(excel_file)
            else:
                existing_df = pd.DataFrame()

            new_row = {
                k: v
                for k, v in [("Experiments", sufix)] + list(all_metrics[task].items())
                if k == "Experiments" or v > 0
            }
            
            if compute_coco:
                for k, v in all_metrics.get('COCO_segm', {}).items():
                    
                    if k != 'per_class_AP':
                        new_row[f"COCO_segm-{k}"] = v
                    else:
                        for class_name, class_ap in v.items():
                            new_row[f"COCO_segm-per_class_AP-{class_name}"] = class_ap
                                                
            new_df = pd.DataFrame([new_row])

            # updated_df = existing_df.append(new_df, ignore_index=True)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
                updated_df.to_excel(writer, index=False)    
                
            print(f"Metrics saved to {save_json_path} and {excel_file}")


if __name__ == "__main__":
    
    #Arguments used for evaluation
    args = eval_parser()

    assert len(args.tasks) == len(args.metrics), f"{args.tasks} {args.metrics}. Tasks and metrics must have the same length." 
    
    # Filter predictions if is requiered
    if args.filter:
        assert len(args.metrics) == 1, args.metrics
        assert args.tasks == ["instruments"], args.tasks
        
        #Boolean to know if we're evaluating segmentation
        segmentation = ("segmentation" in args.metrics[0] or "seg" in args.metrics[0]or "mIoU" in args.metrics[0])
        
        #Select filter method. In the argument parser selection can be a tuple. the first value is topk and the second is threshold
        if args.selection == "thresh":
            selection_params = [None, float(args.selection_info)]
        elif args.selection == "topk":
            selection_params = [int(args.selection_info), None]
        elif args.selection == "topk_thresh":
            assert (
                type(args.selection_info) == str
                and "," in args.selection_info
                and len(args.selection_info.split(",")) == 2
            )
            selection_params = args.selection_info.split(",")
            selection_params[0] = int(selection_params[0])
            selection_params[1] = float(selection_params[1])
        elif "cls" in args.selection:
            assert type(args.selection_info) == str
            assert os.path.isfile(args.selectrion_info)
            selection_params = load_json(args.selection_info)
        else:
            raise ValueError(f"Incorrect selection type {args.selection}")
        preds_dict = read_detectron2_output(
            coco_anns_path=args.coco_ann_path,
            preds_path=args.pred_path,
            selection=args.selection,
            selection_params=selection_params,
            segmentation=segmentation
        )

    output_dir = None
    sufix = None
    if args.output_path is not None:
        if args.selection in ["thresh", "topk", "topk_thresh"]:
            sufix = f"{args.selection}_{args.selection_info}"
        elif "cls" in args.selection:
            sufix = args.selection_info.split("/")[-1].replace(".json", "")
        elif "all" == args.selection:
            sufix = "all"
        else:
            breakpoint()
        output_dir = args.output_path
    main(
        coco_ann_path=args.coco_ann_path,
        pred_path=args.pred_path,
        compute_coco=args.coco,
        preds=preds_dict,
        tasks=args.tasks,
        metrics=args.metrics,
        output_dir=output_dir,
        sufix=sufix,
        masks_path=args.masks_path,
    )