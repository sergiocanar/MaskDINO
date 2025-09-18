# python train_net.py --num-gpus 3 \
# --config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
# OUTPUT_DIR outputs/endoscapes_7 \
# MODEL.WEIGHTS maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

CUDA_VISIBLE_DEVICES=0,1,6,7 python train_net.py --num-gpus 4 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
OUTPUT_DIR outputs/Endoscapes2023_train_og \
MODEL.WEIGHTS weights/maskdino_swinL.pth
