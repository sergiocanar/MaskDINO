# Endoscapes2023
# python match_annots_n_preds.py --coco_anns_path data/endoscapes/features/test_annotation_coco_polygon.json \
# --preds_path outputs/endoscapes2023_features/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path outputs/endoscapes2023_features/train_features/inference/coco_instances_results.json \
# --out_coco_preds_path outputs/endoscapes2023_features/train_features/train_ft.json \
# --out_features_path outputs/endoscapes2023_features/train_features/train_ft.pth \
# --selection all \
# --segmentation \
# --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path data/endoscapes/annotations/train_annotation_coco_polygon.json \
--preds_path outputs/endoscapes2023_features/tapis_features/train/inference/instances_predictions.pth \
--out_coco_anns_path outputs/endoscapes2023_features/tapis_features/train/inference/coco_instances_results.json \
--out_coco_preds_path outputs/endoscapes2023_features/tapis_features/train/train.json \
--out_features_path outputs/endoscapes2023_features/tapis_features/train/train.pth \
--selection all \
--segmentation \
--features_key decoder_out

python match_annots_n_preds.py --coco_anns_path data/endoscapes/annotations/val_annotation_coco_polygon.json \
--preds_path outputs/endoscapes2023_features/tapis_features/val/inference/instances_predictions.pth \
--out_coco_anns_path outputs/endoscapes2023_features/tapis_features/val/inference/coco_instances_results.json \
--out_coco_preds_path outputs/endoscapes2023_features/tapis_features/val/val.json \
--out_features_path outputs/endoscapes2023_features/tapis_features/val/val.pth \
--selection topk_thresh \
--selection_info 5,0.1 \
--validation \
--segmentation \
--features_key decoder_out

python match_annots_n_preds.py --coco_anns_path data/endoscapes/annotations/test_annotation_coco_polygon.json \
--preds_path outputs/endoscapes2023_features/tapis_features/test/inference/instances_predictions.pth \
--out_coco_anns_path outputs/endoscapes2023_features/tapis_features/test/inference/coco_instances_results.json \
--out_coco_preds_path outputs/endoscapes2023_features/tapis_features/test/test.json \
--out_features_path outputs/endoscapes2023_features/tapis_features/test/test.pth \
--selection topk_thresh \
--selection_info 5,0.1 \
--validation \
--segmentation \
--features_key decoder_out
