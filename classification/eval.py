import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_cifar10_loaders
from model import ResNet18

def evaluate(model_path='resnet18_cifar10_best.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    _, testloader, classes = get_cifar10_loaders(batch_size=100, num_workers=2)
    
    net = ResNet18().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print(f'Overall Accuracy: {100.*correct/total:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_targets, all_preds, target_names=classes, digits=4))
    
    print('\nConfusion Matrix:')
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print('\nPer-class Accuracy:')
    for i, cls in enumerate(classes):
        print(f'{cls:10s}: {class_acc[i]*100:.2f}%')

    return 100. * correct / total

if __name__ == '__main__':
    evaluate()
