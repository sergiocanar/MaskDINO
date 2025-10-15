import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def dummy_plot(img: torch.Tensor, boxes=None, i: int =0, color='lime', linewidth=2):
    """
    Plots an image with Detectron2 Boxes drawn on top.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W)
        boxes (detectron2.structures.Boxes or torch.Tensor): Bounding boxes in (x1, y1, x2, y2)
        i (int): Index for saving the figure
        color (str): Box color
        linewidth (int): Box edge width
    """
    # --- Convert image to NumPy ---
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

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
