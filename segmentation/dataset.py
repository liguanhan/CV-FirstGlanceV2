import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

class VOCSegmentation(Dataset):
    def __init__(self, root, year='2012', image_set='train', transform=None, target_transform=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        
        voc_root = os.path.join(self.root, f'VOCdevkit/VOC{year}')
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = np.array(target, dtype=np.int64)
        target[target == 255] = 0
        target = torch.from_numpy(target)
        
        return img, target

    def __len__(self):
        return len(self.images)

def get_voc_loaders(batch_size=8, num_workers=4, data_root='./data'):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    trainset = VOCSegmentation(
        root=data_root, year='2012', image_set='train', 
        transform=transform_train, target_transform=target_transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valset = VOCSegmentation(
        root=data_root, year='2012', image_set='val', 
        transform=transform_test, target_transform=target_transform)
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader

