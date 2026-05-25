import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.blocks = nn.ModuleList()
        
        self.blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for rate in rates:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.blocks.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = []
        
        for block in self.blocks[:-1]:
            features.append(block(x))
        
        pool = self.blocks[-1](x)
        features.append(F.interpolate(pool, size=size, mode='bilinear', align_corners=False))
        
        x = torch.cat(features, dim=1)
        x = self.conv1x1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_channels=256):
        super(Decoder, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1x1(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.last_conv(x)
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet101'):
        super(DeepLabV3Plus, self).__init__()
        
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        else:
            resnet = models.resnet101(pretrained=False)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.decoder = Decoder(num_classes, low_level_channels=256)

    def forward(self, x):
        size = x.shape[-2:]
        
        x = self.layer0(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        return x
