CUDA_VISIBLE_DEVICES=2 python train_net.py --eval-only --num-gpus 1 \
--config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--annots_dir annotations \
OUTPUT_DIR outputs/Endoscapes2023_new_cutmargins_annotations_201_test \
MODEL.WEIGHTS outputs/Endoscapes2023_new_cutmargins_annotations_201_train/model_best.pth