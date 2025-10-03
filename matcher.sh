# Endoscapes2023
python match_annots_n_preds.py --coco_anns_path data/endoscapes/annotations/test_annotation_coco_polygon.json \
--preds_path outputs/endoscapes2023/train_features/inference/instances_predictions.pth \
--out_coco_anns_path outputs/endoscapes2023/train_features/inference/coco_instances_results.json \
--out_coco_preds_path /home/scanar/endovis/models/MaskDINO/features/endoscapes2023/train_ft.json \
--out_features_path /home/scanar/endovis/models/MaskDINO/features/endoscapes2023/train_ft.pth \
--selection all --features_key decoder_out