CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 \
--config-file configs/endoscapes-cutted/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
--annots_dir annotations_201 \
OUTPUT_DIR outputs/endoscapes2023_cutmargins/annotations_201_lr_0_00005_loss_cls_6 \
WANDB.NAME annotations_201_lr_0_00005_loss_cls_6 \
MODEL.WEIGHTS weights/maskdino_swinl_COCO.pth