CUDA_VISIBLE_DEVICES=1,3 python train_net.py --num-gpus 2 \
--config-file configs/endoscapes-cutted/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--annots_dir annotations_201 \
OUTPUT_DIR outputs/Endoscapes2023_cutmargins_annotations_201_lr_0_00001_train \
MODEL.WEIGHTS weights/maskdino_swinl_COCO.pth

