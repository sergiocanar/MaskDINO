CUDA_VISIBLE_DEVICES=0,1,2 python train_net.py --num-gpus 3 --eval-only \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--annots_dir annotations \
MODEL.WEIGHTS outputs/endoscapes2023_cutmargins/annotations_lr_0_0001_train/model_best.pth \
OUTPUT_DIR outputs/endoscapes2023_features/tapis_features/train \
TEST.DETECTIONS_PER_IMAGE 10
