import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_voc_loaders
from model import DeepLabV3Plus

os.makedirs('logs', exist_ok=True)
os.makedirs('weights', exist_ok=True)
os.makedirs('samples', exist_ok=True)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    trainloader, valloader = get_voc_loaders(batch_size=4, num_workers=2)
    
    model = DeepLabV3Plus(num_classes=21).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9, weight_decay=5e-4)

    best_miou = 0
    num_epochs = 100
    
    train_loss_list = []
    val_loss_list = []
    miou_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss/(batch_idx+1)
        train_loss_list.append(avg_train_loss)
        print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}')
        
        model.eval()
        val_loss = 0
        intersection = torch.zeros(21).to(device)
        union = torch.zeros(21).to(device)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                for cls in range(21):
                    pred_mask = (preds == cls)
                    target_mask = (targets == cls)
                    intersection[cls] += (pred_mask & target_mask).sum().float()
                    union[cls] += (pred_mask | target_mask).sum().float()

        avg_val_loss = val_loss/(batch_idx+1)
        iou = intersection / (union + 1e-10)
        miou = iou.mean().item()
        
        val_loss_list.append(avg_val_loss)
        miou_list.append(miou)
        
        print(f'Val | Loss: {avg_val_loss:.4f} | mIoU: {miou:.4f}')

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'weights/best_model.pth')
            print(f'Saved best model with mIoU: {best_miou:.4f}')

    print(f'Best mIoU: {best_miou:.4f}')
    
    epochs = list(range(1, num_epochs+1))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_loss_list, label='Val Loss', linewidth=2, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve ({num_epochs} Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, miou_list, label='mIoU', color='green', linewidth=2, marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU (%)')
    plt.title(f'mIoU Curve (Best: {best_miou:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logs/training_curve.png', dpi=300)
    print("训练曲线已保存")

if __name__ == '__main__':
    train()