CUDA_VISIBLE_DEVICES=3 python train_net.py --eval-only --num-gpus 1 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
OUTPUT_DIR outputs/Endoscapes2023_cutmargins_annotations_sam_extended_lr_0_00001_test \
MODEL.WEIGHTS outputs/Endoscapes2023_cutmargins_annotations_sam_extended_lr_0_00001_train/model_best.pth