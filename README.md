# CV-FirstGlanceV2: 计算机视觉第一个项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## 项目简介

CV-FirstGlanceV2是一个简单、初等的计算机视觉完整实践项目。项目完成了计算机视觉两大核心任务：**图像分类**和**图像分割**，使用PyTorch框架实现。

> - 本项目的中文译名我将其命名为“第一瞥”。从此刻起，我的计算机有了“视觉”的能力，这个项目就是它对这个世界的“第一瞥”。现在读者看到的是这个项目的V2版本，V1版本在开发过程中“胎死腹中”了，到最后项目变得很混乱，我无心再调了，索性直接制作并发布第二版。
> - 本项目的一个重要背景是2025~2026学年春季学期，我选修了李东喜老师负责、多位老师共同授课的《大数据思维与技术》选修课。这个项目就是我最后提交的课程作业。



## 项目特性

- **经典模型实现**：ResNet + DeepLabV3+
- **标准数据集**：CIFAR-10 + PASCAL VOC
- **新手友好**：完整教程



## 项目结构

```
cv_firstglancev2/
├── classification/          # 图像分类 (CIFAR-10 + ResNet)
├── segmentation/           # 图像分割 (VOC 2012 + DeepLabV3+)
├── utils/                  # 通用工具函数
├── PROJECT_OVERVIEW.md     # 项目详细说明
├── WHY_THESE_MODELS.md     # 模型选择论证报告
├── EXPERIMENT_REPORT.md    # 完整实验报告
├── requirements.txt        # 依赖清单
└── README.md               # 本文件
```



## 快速开始

### 环境配置

```bash
# 1. 创建conda环境
conda create -n cv_firstglance python=3.10
conda activate cv_firstglance

# 2. 安装PyTorch (GPU版本)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 图像分类

```bash
cd classification
python train.py    # 开始训练
python eval.py     # 评估模型
```

### 图像分割

```bash
cd segmentation
python train.py    # 开始训练
python eval.py     # 评估模型
```



## 预期性能

| 任务     | 模型       | 数据集   | 指标      | 预期值 |
| -------- | ---------- | -------- | --------- | ------ |
| 图像分类 | ResNet18   | CIFAR-10 | Top-1 Acc | 95.45% |
| 图像分割 | DeepLabV3+ | VOC 2012 | mIoU      | 67.64  |

图像分割使用 ImageNet 预训练权重可达到 78-82% mIoU，本项目为从零开始训练，未使用预训练权重。



## 详细文档

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**：项目完整介绍、环境配置、使用教程
- **[WHY_THESE_MODELS.md](WHY_THESE_MODELS.md)**：模型选择的论证报告
- **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)**：完整实验结果与分析



## 展望

1.不久的将来增添“目标检测”模块。



## 许可证

MIT License - 可自由用于学习和研究。



## 贡献

目前本项目由本人独立开发完成。欢迎提交Issue和Pull Request！
