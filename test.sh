CUDA_VISIBLE_DEVICES=3 python train_net.py --eval-only --num-gpus 1 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
OUTPUT_DIR outputs/Endoscapes2023_best_w_cutmargins \
MODEL.WEIGHTS weights/best_endoscapes_cutmargins.pth