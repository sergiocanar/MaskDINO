import argparse
import os
import json
import pandas as pd
import numpy as np
from evaluate.main_eval import eval_task
from evaluate.utils import load_json, read_detectron2_output
from tabulate import tabulate


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


def main(coco_ann_path, pred_path, tasks, metrics, output_dir, sufix, masks_path):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path) if type(pred_path) == str else pred_path
    datasets = [img.get("dataset") for img in coco_anns["images"]]
    if not any(dataset is None for dataset in datasets):
        datasets = sorted(list(set(datasets)), key=lambda x: x.lower())
    else:
        datasets = False
    all_metrics = {}
    for task, metric in zip(tasks, metrics):
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
                with open(
                    os.path.join(output_dir, f"metrics_{met_suf}.json"), "r"
                ) as f:
                    save_json = json.load(f)
                    save_json[sufix] = all_metrics[task]
            else:
                save_json = {sufix: all_metrics[task]}
            with open(os.path.join(output_dir, f"metrics_{met_suf}.json"), "w") as f:
                json.dump(save_json, f, indent=4)

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
            new_df = pd.DataFrame([new_row])

            # updated_df = existing_df.append(new_df, ignore_index=True)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
                updated_df.to_excel(writer, index=False)
    overall_metric = np.mean(
        [v[m] for v, m in zip(list(all_metrics.values()), metrics)]
    )
    print("Overall Metric: {}".format(overall_metric))
    return overall_metric


if __name__ == "__main__":
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

    assert len(args.tasks) == len(args.metrics), f"{args.tasks} {args.metrics}"
    preds = args.pred_path
    if args.filter:
        assert len(args.metrics) == 1, args.metrics
        assert args.tasks == ["instruments"], args.tasks
        segmentation = (
            "segmentation" in args.metrics[0]
            or "seg" in args.metrics[0]
            or "mIoU" in args.metrics[0]
        )
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
            with open(args.selection_info, "r") as f:
                selection_params = json.load(f)
        else:
            raise ValueError(f"Incorrect selection type {args.selection}")
        preds = read_detectron2_output(
            args.coco_ann_path, preds, args.selection, selection_params, segmentation
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
        args.coco_ann_path,
        preds,
        args.tasks,
        args.metrics,
        output_dir,
        sufix,
        args.masks_path,
    )