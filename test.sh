CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --eval-only --num-gpus 4 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--annots_dir annotations \
OUTPUT_DIR outputs/endoscapes2023_cutmargins/annotations_201_new_weights_lr_0_00005_test \
MODEL.WEIGHTS /home/scanar/endovis/models/MaskDINO/outputs/endoscapes2023_cutmargins/annotations_201_new_weights_lr_0_00005_train/model_best.pth