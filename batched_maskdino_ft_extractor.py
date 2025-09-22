# ------------------------------------------------------------------------
# MaskDINO Batched Feature Extractor
# Extracts global (mask_features) and object-wise (decoder queries) features
# ------------------------------------------------------------------------

import os
import torch
import cv2
from glob import glob
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor
from os.path import join as path_join
from maskdino import add_maskdino_config

setup_logger()


# ---- Config Setup ----
def setup_cfg(cfg_path, weights_path, device="cuda"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device

    # Force input size 1024x1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.freeze()
    return cfg


# ---- Feature Extractor ----
class FeatureExtractor:
    def __init__(self, cfg_path, weights_path, device="cuda"):
        self.cfg = setup_cfg(cfg_path, weights_path, device)
        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.eval()
        self.device = device

        self.global_feats = None
        self.object_feats = None
        self._register_hooks()

    def _register_hooks(self):
        model = self.predictor.model

        def hook_mask_features(module, input, output):
            self.global_feats = output.detach().cpu()  # [B, C, H/4, W/4]

        def hook_decoder(module, input, output):
            hs, references = output
            self.object_feats = hs[-1].detach().cpu()  # [B, num_queries, d_model]

        model.sem_seg_head.pixel_decoder.mask_features.register_forward_hook(hook_mask_features)
        model.sem_seg_head.predictor.decoder.register_forward_hook(hook_decoder)

    @torch.no_grad()
    def extract_batch(self, imgs):
        """
        imgs: list of numpy arrays [H,W,3] in BGR, already resized
        Returns: global_feats [B,C,H,W], object_feats [B,num_queries,d_model]
        """
        batched_inputs = []
        for img in imgs:
            tensor = torch.as_tensor(img.transpose(2, 0, 1)).to(self.device)  # [3,H,W]
            batched_inputs.append({"image": tensor})

        _ = self.predictor.model(batched_inputs)  # forward pass with list of dicts
        return self.global_feats, self.object_feats


# ---- Main ----
if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = "configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
    weights_path = "/home/jclyons/endovis/challenge_2025/MaskDINO/outputs/endoscapes_6/model_best.pth"
    image_folder = "data/endoscapes/frames"
    output_folder = path_join(this_dir, 'endoscapes2023_features')
    global_ft_folder = path_join(output_folder, 'global_features')
    object_ft_folder = path_join(output_folder, 'object_features')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(global_ft_folder, exist_ok=True)
    os.makedirs(object_ft_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(cfg_path, weights_path, device=device)

    img_list = glob(path_join(image_folder, "*.jpg"))
    batch_size = 8  # adjust depending on GPU memory

    with tqdm(total=len(img_list), desc="Extracting features", unit="frame") as pbar:
        for i in range(0, len(img_list), batch_size):
            batch_paths = img_list[i:i+batch_size]
            batch_imgs = [
                cv2.resize(read_image(p, format="BGR"), (1024, 1024)) for p in batch_paths
            ]

            global_f, object_f = extractor.extract_batch(batch_imgs)  # [B,...]

            for j, img_path in enumerate(batch_paths):
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                if global_f is not None:
                    torch.save(global_f[j], f"{global_ft_folder}/{img_name}_global.pt")

                if object_f is not None:
                    torch.save(object_f[j], f"{object_ft_folder}/{img_name}_object.pt")

                pbar.update(1)
