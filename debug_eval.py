# debug_eval.py
import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from train_net import register_surgical_dataset

# -------------------------------
# 🔧 1. Load config + weights
# -------------------------------
from maskdino import add_maskdino_config   # adapt if your add_config fn is elsewhere
from detectron2.projects.deeplab import add_deeplab_config

def setup_cfg(cfg_path, weights_path, device="cuda"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    register_surgical_dataset(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg

# ⚠️ EDIT these paths
CFG_PATH = "configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
WEIGHTS_PATH = "/home/scanar/endovis/models/MaskDINO_Challenge/outputs/Endoscapes2023_train_og/model_best.pth"

cfg = setup_cfg(CFG_PATH, WEIGHTS_PATH)

print("\n========== CONFIG ==========")
print("Eval datasets:", cfg.DATASETS.TEST)
print("Num classes:", cfg.MODEL.ROI_HEADS.NUM_CLASSES)
print("Score threshold:", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
print("============================\n")

# -------------------------------
# 🔍 2. Dataset check
# -------------------------------
if len(cfg.DATASETS.TEST) == 0:
    raise RuntimeError("⚠️ No test dataset defined in cfg.DATASETS.TEST")

dataset_name = cfg.DATASETS.TEST[0]
print(f"Checking dataset: {dataset_name}")
print("All registered datasets:", DatasetCatalog.list())

dataset = DatasetCatalog.get(dataset_name)
print(f"Dataset size: {len(dataset)}")

if len(dataset) > 0:
    print("First sample keys:", dataset[0].keys())
    print("First image path:", dataset[0]["file_name"])
    print("First annotations:", dataset[0]["annotations"])

# -------------------------------
# 🔮 3. Run predictor on first image
# -------------------------------
predictor = DefaultPredictor(cfg)

img = cv2.imread(dataset[0]["file_name"])
outputs = predictor(img)

instances = outputs["instances"].to("cpu")
print("\n========== PREDICTIONS ==========")
print("Num predictions:", len(instances))
if len(instances) > 0:
    print("Pred classes:", instances.pred_classes.numpy())
    print("Pred scores:", instances.scores.numpy())
    print("Pred boxes:", instances.pred_boxes.tensor.numpy())
else:
    print("⚠️ No predictions returned!")
print("=================================\n")

# -------------------------------
# 🎨 4. Visualization
# -------------------------------
meta = MetadataCatalog.get(dataset_name)
v = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.7)
out = v.draw_instance_predictions(instances)
os.makedirs("debug_vis", exist_ok=True)
cv2.imwrite("debug_vis/pred_debug.jpg", out.get_image()[:, :, ::-1])
print("✅ Saved visualization at debug_vis/pred_debug.jpg")
