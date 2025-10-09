import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
