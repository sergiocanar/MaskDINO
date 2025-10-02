CUDA_VISIBLE_DEVICES=7 python train_net.py --eval-only --num-gpus 1 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
OUTPUT_DIR outputs/Endoscapes2023_test_finale \
MODEL.WEIGHTS /home/scanar/endovis/models/MaskDINO_Challenge/outputs/Endoscapes2023_train_og/model_best.pth
