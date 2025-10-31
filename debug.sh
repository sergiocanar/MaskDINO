CUDA_VISIBLE_DEVICES=3 python train_net.py --num-gpus 1 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
OUTPUT_DIR outputs/debug \
MODEL.WEIGHTS weights/best_endoscapes.pth