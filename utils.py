import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycocotools.mask as mask_util
from os.path import join as path_join
from detectron2.structures import BoxMode
from pycocotools import mask as maskUtils


def decode_rle_to_mask(rle):
    """Decode RLE to a binary numpy mask of shape (h, w)."""
    return maskUtils.decode(rle).astype(np.uint8)

def encode_mask_to_rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')  # make JSON-serializable
    return rle


def remove_duplicates_n_features(instances):
    """
    This function removes duplicate instances based on their bounding boxes, keeping the instance with the highest score.

    Parameters:
    instances : list of dicts
        Each dictionary represents an instance with keys including 'bbox' (bounding box) and 'score'.

    Returns:
    list of dicts
        A list of unique instances, where each bounding box appears only once, keeping the instance with the highest score.
    """

    # Initialize a dictionary to store unique instances keyed by their bounding boxes
    no_dups = {}

    # Iterate through each instance
    for inst in instances:
        # Only consider instances with a positive score
        if inst["score"] > 0:
            # Convert the bounding box to a tuple to use as a dictionary key
            bbox_key = tuple(inst["bbox"])

            # If the bounding box is not already in the dictionary, add the instance
            if bbox_key not in no_dups:
                no_dups[bbox_key] = inst
            else:
                # If the bounding box is already in the dictionary, keep the instance with the higher score
                if inst["score"] > no_dups[bbox_key]["score"]:
                    no_dups[bbox_key] = inst

    # Return the list of unique instances
    return list(no_dups.values())


def instances_to_coco_json(instances, img_id, ann_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    """
    
    num_instance = len(instances)
    if num_instance == 0:
        return []

    # ---- Always move to CPU-friendly containers first
    boxes = []
    scores = []
    classes = []
    embeds = None
    dec_outs = None
    score_dists = None

    # ---- Initialize flags up-front so they're always defined
    has_mask = instances.has("pred_masks")
    has_embd = False
    has_obj_queries = False
    has_score_dist = False

    # ---- Core fields (boxes / scores / classes)
    boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()
    scores = instances.scores_dist.detach().cpu().tolist()
    classes = instances.pred_classes.detach().cpu().tolist()

    # ---- Optional: mask embeddings
    if instances.has("mask_embd"):
        embeds = instances.mask_embd.detach().cpu().tolist()
        has_embd = True

    # ---- Optional: decoder outputs (consistent key names)
    #   If your Instances stores it as 'decoder_out', use that key.
    #   If you actually store 'object_queries', change both lines accordingly.
    if instances.has("object_queries"):
        dec_outs = instances.object_queries.detach().cpu().tolist()
        has_obj_queries = True

    # ---- Optional: per-class score distributions
    #   Use the correct key: 'score_dist' (not 'scores')
    if instances.has("scores_dist"):
        score_dists = instances.scores_dist.detach().cpu().tolist()
        has_score_dist = True

    # ---- Masks (RLE encode if present)
    if has_mask:
        rles = [
            mask_util.encode(np.array(m[:, :, None], order="F", dtype="uint8"))[0]
            for m in instances.pred_masks.detach().cpu().numpy()
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
            
    # breakpoint()
    # ---- Build results
    results = []
    for k in range(num_instance):
        result = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": classes[k] + 1,
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_embd:
            result["mask_embd"] = embeds[k]
        if has_obj_queries:
            result["decoder_out"] = dec_outs[k]
        if has_score_dist:
            result["score_dist"] = score_dists[k]
        
        ann_id+=1
        
        results.append(result)
    
    results = remove_duplicates_n_features(results)
    
    return results

def trivial_batch_collator(batch):
    """
    A simple batch collator that does no collation.
    It simply returns the list of data as-is.

    Args:
        batch (list): A list of data samples, where each sample is typically a dictionary.

    Returns:
        list: The same list of data samples.
    """
    return batch


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_black_frames(src_dir: str, frame_lt: list):
    
    for frame in frame_lt:
        
        new_frame = ''
        
        if ".jpg" not  in frame:
            new_frame = f'{frame}.jpg'
        else:
            new_frame = frame
        
        path = path_join(src_dir, new_frame)
                
        try:
            os.remove(path=path)
            print(f'Removed path: {path}')
        except FileNotFoundError as e:
            print(f'The file to remove does not exist or was already removed: {e}')
            
def join_coco_jsons(json_path_lt: list, output_path: str):
    assert len(json_path_lt) >= 2, "Need at least 2 JSONs to merge."

    final_dict = {"images": [], "annotations": [], "categories": []}
    image_ids = set()
    annot_ids = set()
    categories_set = {}

    for json_path in json_path_lt:
        data = load_json(json_path)
        # --- Merge categories (avoid duplicates) ---
        for cat in data.get("categories", []):
            if cat["id"] not in categories_set:
                categories_set[cat["id"]] = cat

        # --- Merge images without duplicates ---
        for img in data.get("images", []):
            if img["id"] not in image_ids:
                image_ids.add(img["id"])
                final_dict["images"].append(img)

        # --- Merge annotations without duplicates ---
        for ann in data.get("annotations", []):
            if ann["id"] not in annot_ids:
                annot_ids.add(ann["id"])
                final_dict["annotations"].append(ann)

    final_dict["categories"] = list(categories_set.values())
    save_json(final_dict, output_path)
    
    print(f'Joint .json was saved to: {output_path}')
            

def create_symlink(frame_name: str, src_path: str, dst_dir: str):
    """
    Create a symbolic link for a single frame.

    Args:
        frame_name (str): file name (e.g., "120_69225.jpg")
        src_path (str): absolute or relative path to the source file
        dst_dir (str): directory where the symlink should be placed
    """
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, frame_name)

    try:
        if not os.path.exists(dst_path):
            os.symlink(os.path.abspath(src_path), dst_path)
            print(f"Linked: {dst_path}")
        else:
            print(f"Already exists: {dst_path}")
    except OSError as e:
        print(f"Error linking {frame_name}: {e}")

def load_json(coco_json_path: str):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data_dict: dict, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
        
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def dummy_plot(img: torch.Tensor, img2: np.ndarray= None, boxes=None, i: int =0, color='lime', linewidth=2, output_path: str = None):
    """
    Plots an image with Detectron2 Boxes drawn on top.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W)
        img2 (np.ndarray): Image array of shape (H,W,C)
        boxes (detectron2.structures.Boxes or torch.Tensor): Bounding boxes in (x1, y1, x2, y2)
        i (int): Index for saving the figure
        color (str): Box color
        linewidth (int): Box edge width
    """
    # --- Convert image to NumPy ---
    
    if isinstance(img, torch.Tensor):    
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        
    elif isinstance(img, np.ndarray):
        shape = img.shape
        
        if shape[0] == 3:
            img = np.transpose(img, (1,2,0))
        else:
            pass
    else:
        raise TypeError('Image should be an NumPy array or PyTorch Tensor...')
    
    final_path = path_join(output_path, f'debug_comparison_{i}.png')
    
    if img2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=(15,15))
        ax[0].imshow(img)
        ax[0].axis('off')
        
        ax[1].imshow(img2)
        ax[1].axis('off')
        
        plt.savefig(final_path)
        plt.close(fig)
        
    else:        
        # --- Create plot ---
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img)
        ax.axis('off')

        # --- Convert Detectron2 Boxes to tensor if needed ---
        
        if boxes != None:    
            if hasattr(boxes, "tensor"):  # detectron2.structures.Boxes
                boxes = boxes.tensor

            # --- Draw each box ---
            boxes = boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)

        # --- Save and close ---
        plt.savefig(f'debug_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def dummy_plot_three(img: np.ndarray, img_bin: np.ndarray, img_bin2: np.ndarray, crop_coords: tuple,
                     img_crop: np.ndarray, i: int = 0, output_path: str = None):
    """
    Plots:
      1. Original image with crop rectangle
      2. Binary mask
      3. Cropped image
    """
    create_directory_if_not_exists(output_path)
    final_path = path_join(output_path, f'debug_comparison_{i}.png')

    fig, ax = plt.subplots(1, 4, figsize=(18, 6))

    # --- 1. Original image with rectangle ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[0].imshow(img_rgb)
    ax[0].set_title("Original + Detected Crop")
    ax[0].axis('off')

    if crop_coords is not None:
        left, right, bottom, top = crop_coords
        rect = patches.Rectangle(
            (bottom, left),                # (x, y)
            top - bottom,                  # width
            right - left,                  # height
            linewidth=3,
            edgecolor='lime',
            facecolor='none'
        )
        ax[0].add_patch(rect)

    # --- 2. Binary mask ---
    ax[1].imshow(img_bin, cmap='gray')
    ax[1].set_title("Binary Mask")
    ax[1].axis('off')

    ax[3].imshow(img_bin2, cmap='gray')
    ax[3].set_title("Binary Mask")
    ax[3].axis('off')

    # --- 3. Cropped image ---
    if img_crop is not None:
        ax[2].imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        ax[2].set_title("Cropped Region")
        ax[2].axis('off')
    else:
        ax[2].set_title("No crop detected")
        ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(final_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)