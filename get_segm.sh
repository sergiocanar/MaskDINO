CUDA_VISIBLE_DEVICES=3 python bbox2segm/get_segm.py --config_path configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--weights_path outputs/endoscapes2023_cutmargins/annotations_lr_0_0001_train/model_best.pth \
--dataset_name endoscapes_train \
--save_dir data/endoscapes/calculate_masks