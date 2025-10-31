CUDA_VISIBLE_DEVICES=2,3 python train_net.py --num-gpus 2 \
--config-file configs/endoscapes-cutted/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--annots_dir annotations \
OUTPUT_DIR outputs/endoscapes2023_cutmargins/annotations_lr_0_00001_steps_as_paper_train \
MODEL.WEIGHTS weights/maskdino_swinl_COCO.pth