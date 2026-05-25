import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_voc_loaders
from model import DeepLabV3Plus

def evaluate(model_path='weights/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    _, valloader = get_voc_loaders(batch_size=4, num_workers=2)
    
    model = DeepLabV3Plus(num_classes=21).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]

    intersection = torch.zeros(21).to(device)
    union = torch.zeros(21).to(device)
    pixel_acc = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            valid_mask = targets != 255
            pixel_acc += (preds[valid_mask] == targets[valid_mask]).sum().item()
            total_pixels += valid_mask.sum().item()
            
            for cls in range(21):
                pred_mask = (preds == cls)
                target_mask = (targets == cls)
                intersection[cls] += (pred_mask & target_mask).sum().float()
                union[cls] += (pred_mask | target_mask).sum().float()

    iou = intersection / (union + 1e-10)
    miou = iou.mean().item()
    pa = pixel_acc / total_pixels

    print(f'Pixel Accuracy: {pa*100:.2f}%')
    print(f'Mean IoU: {miou*100:.2f}%')
    print('\nPer-class IoU:')
    for i, name in enumerate(class_names):
        print(f'{name:15s}: {iou[i].item()*100:.2f}%')

    return miou

if __name__ == '__main__':
    evaluate()
