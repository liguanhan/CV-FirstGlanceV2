import torch
import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def visualize_prediction(image, pred, target=None, save_path='prediction.png'):
    image = denormalize(image).permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    
    if target is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(image)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        ax2.imshow(target.cpu().numpy(), cmap='tab20')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3.imshow(pred.cpu().numpy(), cmap='tab20')
        ax3.set_title('Prediction')
        ax3.axis('off')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        ax2.imshow(pred.cpu().numpy(), cmap='tab20')
        ax2.set_title('Prediction')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

