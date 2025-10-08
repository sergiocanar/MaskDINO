import os
import torch
from tqdm import tqdm
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import ImageList, Boxes
from detectron2.data import build_detection_train_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom




from tqdm import tqdm
from utils import inverse_sigmoid
from maskdino.utils import box_ops
from maskdino import add_maskdino_config
from train_net import register_surgical_dataset
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

def get_raw_preds(model, image_list, h, w):
    
    with torch.no_grad():
        #1. Get the features from the backbone and pixel decoder
        features = model.backbone(image_list.tensor)
        target = None # I dont have masks at this point. The idea is to use available bboxes
        #Extract the mask_features with the multiscale features. That is x
        mask_features, _, x = model.sem_seg_head.pixel_decoder.forward_features(features, target) 

        B = mask_features.shape[0]

        #1.5 Create per level binary masks for the FPN features (empty)
        level_masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]

        #2. Convert my GT boxes to normnalized cxcywh and project to query embeddings
        gt_instances = sample["instances"].to(model.device)
        gt_boxes = gt_instances.gt_boxes.tensor  # [N_objects, 4] in xyxy
        refpoint_embed = box_ops.box_xyxy_to_cxcywh(gt_boxes)  # [N_objects, 4] in cxcywh
        refpoint_embed = refpoint_embed / torch.as_tensor([w, h, w, h], device=model.device)  # Normalize to [0, 1]

        #2.5 Apply unsigmoid to pass this as logit space                
        eps = 1e-4
        norm_boxes = refpoint_embed.clamp(eps, 1-eps)

        Q_gt = norm_boxes.shape[0] # N_objects
        refpoint_embed = inverse_sigmoid(norm_boxes).unsqueeze(0) #1,Qgt,4 #Hasta aqui voy bien 07/10 22h
        
        # 3) Build tgt (query features). Use learned query_feat, then slice/pad to match num_queries and our injected boxes
        predictor_head = model.sem_seg_head.predictor  # MaskDINODecoder
        hidden_dim = predictor_head.hidden_dim
        num_queries = predictor_head.num_queries

        # content features
        if getattr(predictor_head, "learn_tgt", False):
            # learned content embeddings
            base_tgt = predictor_head.query_feat.weight[None].repeat(B, 1, 1)  # [1, num_queries, D]
        else:
            #B,nq,D
            base_tgt = torch.zeros((B, num_queries, hidden_dim), device=model.device) #Built as zero so decoder can change this in the following steps

        #Aqui tengo que llenar los refpoints para que cuadren con el tamaño de los queries #TODO probar con bboxes ruidosos en vez de con zeros
        # Now align refpoints/tgt length with num_queries:
        if Q_gt >= num_queries:
            refpoint_embed_in = refpoint_embed[:, :num_queries, :]          # [1, num_queries, 4]
            tgt_in = base_tgt[:, :num_queries, :]                           # [1, num_queries, D]
        else:
            # pad refpoints with learned anchors (or zeros) to reach num_queries
            if hasattr(predictor_head, "query_embed") and predictor_head.query_embed is not None:
                # if available (two-stage==False path), take the remaining anchors from learned query_embed
                pad_ref = predictor_head.query_embed.weight[None, :num_queries - Q_gt, :].to(refpoint_embed.dtype).to(refpoint_embed.device)
            else:
                pad_ref = torch.zeros((B, num_queries - Q_gt, 4), device=refpoint_embed.device, dtype=refpoint_embed.dtype)
            refpoint_embed_in = torch.cat([refpoint_embed, pad_ref], dim=1)  # [1, num_queries, 4]
            tgt_in = base_tgt  # already [1, num_queries, D]

        # 4) Flatten multi-scale features & masks exactly like decoder.forward()
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        # decoder expects smallest->largest in the flattened sequence
        for i in range(len(x)):
            idx = len(x) - 1 - i  # reverse
            src_l = predictor_head.input_proj[idx](x[idx]).flatten(2).transpose(1, 2)  # [B, H*W, C]
            src_flatten.append(src_l)
            mask_flatten.append(level_masks[i].flatten(1))  # note they use masks in opposite order
            spatial_shapes.append(x[idx].shape[-2:])

        src_flatten = torch.cat(src_flatten, dim=1)                  # [B, sum(HW), C]
        mask_flatten = torch.cat(mask_flatten, dim=1)                # [B, sum(HW)]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # valid ratios (no padding -> ones)
        valid_ratios = torch.ones((B, len(x), 2), device=src_flatten.device)

        # 5) (Optional) get an initial prediction before decoding if initial_pred=True
        predictions_class = []
        predictions_mask = []
        if getattr(predictor_head, "initial_pred", False):
            out_cls0, out_mask0 = predictor_head.forward_prediction_heads(tgt_in.transpose(0, 1), mask_features, pred_mask=True)
            predictions_class.append(out_cls0)
            predictions_mask.append(out_mask0)

        # 6) Run the decoder with your refpoints
        hs, references = predictor_head.decoder(
            tgt=tgt_in.transpose(0, 1),                 # [num_queries, B, D]
            memory=src_flatten.transpose(0, 1),         # [sum(HW), B, C]
            memory_key_padding_mask=mask_flatten,       # [B, sum(HW)]
            pos=None,
            refpoints_unsigmoid=refpoint_embed_in.transpose(0, 1),  # [num_queries, B, 4]
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None
        )
        
        #6.5 I need bboxes predictions for MaskDINO instance segmentation post-processing
        # breakpoint()
        out_boxes = predictor_head.pred_box(references, hs, refpoint_embed_in.sigmoid())

        # 7) Heads after each decoder layer (keep last)
        for i, output in enumerate(hs):
            out_cls, out_mask = predictor_head.forward_prediction_heads(output.transpose(0, 1), mask_features, pred_mask=True)
            predictions_class.append(out_cls)
            predictions_mask.append(out_mask)

        # last layer outputs
        pred_logits = predictions_class[-1]     # [B, num_queries, num_classes]
        pred_masks  = predictions_mask[-1]      # [B, num_queries, H/4, W/4]
        pred_bboxes = out_boxes[-1]
        # upsample to original size
        pred_masks_up = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False).sigmoid()
        
        return pred_masks_up, pred_logits, pred_bboxes



#paths
config_path = "configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
weights_path = "weights/best_endoscapes_cutmargins.pth"

#Setup the model based on the config and weights. In this case Im using a Swin backbone trained with MaskDINO on Endoscapes cutmargins
cfg = get_cfg()
add_deeplab_config(cfg) # for the pixel decoder
add_maskdino_config(cfg) # for the transformer predictor
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.DEVICE = "cuda:0"
# I cannot disable test_intance as it is used in the mask head to determine the number of queries
# cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False  # Skip standard inference
cfg.freeze()

#Register the surgical dataset as in TAPIS
register_surgical_dataset(cfg)
dataset_dicts = DatasetCatalog.get("endoscapes_test")

# Build model
predictor = DefaultPredictor(cfg)
model = predictor.model.to(cfg.MODEL.DEVICE)
model.eval()

#Build dataloader based on the config and the registered dataset
dataset_name = "endoscapes_test"
mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=True)
data_loader = build_detection_train_loader(cfg, mapper=mapper)

#Save directory for the predicted masks
save_dir = "mask_predictions_from_gt/"
os.makedirs(save_dir, exist_ok=True)


# Save the masks in COCO format 
final_dict = {
    "images": [],
    "annotations": [],
    "categories": []
}



with tqdm(total=len(dataset_dicts), desc='Processing frame...', unit='frame') as pbar:
    for batch in tqdm(data_loader):
        for sample in batch:
            
            #Get image and metadata
            file_name = os.path.basename(sample["file_name"])
            img_id = sample["image_id"]
            image = sample["image"].to(model.device)  # [3, H, W]
            h, w = image.shape[-2:]

            # Normalize and pad
            image_input = (image - model.pixel_mean) / model.pixel_std
            image_list = ImageList.from_tensors([image_input], model.size_divisibility)
            
            raw_masks_preds, raw_classes_preds, raw_bboxes_preds = get_raw_preds(model=model,
                                                  image_list=image_list,
                                                  h=h,
                                                  w=w
                                                )
            
            processed_results = []
            
            for i, (mask_cls_result, mask_pred_result, mask_box_result) in enumerate(zip(raw_classes_preds, raw_masks_preds, raw_bboxes_preds)):
                
                new_size = mask_pred_result.shape[-2:]
                og_h = sample['height']
                og_w = sample['width']
                input_img_size = (h,w) 
                
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_result, input_img_size, og_h, og_w)
                mask_cls_result = mask_cls_result.to(mask_pred_result)
                
                
                mask_box_result = mask_box_result.to(mask_pred_result)
                height = new_size[0]/h*og_h
                width = new_size[1]/w*og_w
                mask_box_result = model.box_postprocess(mask_box_result, height, width)

                instance_r, top_q_idx = retry_if_cuda_oom(model.instance_inference)(
                        mask_cls_result, mask_pred_result, mask_box_result
                    )
                
                breakpoint()
                
                            
            #TODO Me está haciendo falta las preds de bboxes para el postproc final. Adaptar a COCO y sale
            #TODO entrenar maskdino en este json. Preguntar por SAM
            
            