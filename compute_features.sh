CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 python train_net.py --num-gpus 7 --eval-only \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
MODEL.WEIGHTS /home/scanar/endovis/models/MaskDINO_Challenge/outputs/Endoscapes2023_train_og/model_best.pth \
OUTPUT_DIR outputs/endoscapes2023/train_features \
