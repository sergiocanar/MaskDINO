import json
import torch

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

