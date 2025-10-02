# Endoscapes2023
python match_annots_n_preds.py --coco_anns_path data/endoscapes/annotations/test_annotation_coco_polygon.json \
--preds_path /media/SSD1/scanar/endovis/models/LEMIS-challenge/region_proposals/outputs/endoscapes2023/train_features/inference/instances_predictions.pth \
--out_coco_anns_path /media/SSD1/scanar/endovis/models/LEMIS-challenge/region_proposals/outputs/endoscapes2023/train_features/inference/coco_instances_results.json \
--out_coco_preds_path /media/SSD1/scanar/endovis/models/LEMIS-challenge/region_proposals/outputs/endoscapes2023/features/train_ft.json \
--out_features_path /media/SSD1/scanar/endovis/models/LEMIS-challenge/region_proposals/outputs/endoscapes2023/features/train_ft.pth \
--selection all --features_key decoder_out