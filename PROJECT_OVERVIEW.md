# PROJECT OVERVIEW: CV-FirstGlanceV2

## 项目整体介绍与背景

### 项目简介
CV-FirstGlanceV2是一个计算机视觉入门项目，完整实现了计算机视觉两大核心任务：**图像分类**、**图像分割**。本项目采用PyTorch框架，使用学术界标准数据集和经典模型架构。

### 项目背景
深度学习和计算机视觉是人工智能领域最热门的方向，但对于初学者而言，现有的开源项目要么过于复杂难以理解，要么过于简化缺乏完整性。

CV-FirstGlanceV2旨在解决这一痛点：
- **完整性**：覆盖CV核心任务的完整pipeline
- **教学性**：代码结构清晰，便于学习和理解
- **可复现性**：所有代码可直接运行，结果可复现

### 项目目标
1. 提供完整的计算机视觉深度学习实践框架
2. 帮助初学者理解分类、分割的本质区别
3. 展示ResNet、DeepLabV3+经典模型的实现细节
4. 建立从数据处理→模型构建→训练→评估的完整工程思维

---

## 项目结构说明

```
cv_firstglancev2/
├── classification/          # 图像分类模块
│   ├── train.py            # 训练脚本
│   ├── model.py            # ResNet模型定义
│   ├── dataset.py          # CIFAR-10数据集加载
│   └── eval.py             # 模型评估脚本
├── segmentation/           # 图像分割模块
│   ├── train.py            # 训练脚本
│   ├── model.py            # DeepLabV3+模型定义
│   ├── dataset.py          # VOC 2012数据集加载
│   └── eval.py             # 模型评估脚本
├── utils/                  # 通用工具模块
│   └── common.py           # 工具函数集合
├── PROJECT_OVERVIEW.md     # 本文件，项目概述
├── WHY_THESE_MODELS.md     # 模型选择论证报告
├── EXPERIMENT_REPORT.md    # 实验报告
├── requirements.txt        # Python依赖清单
└── README.md               # 项目入口说明
```

### 模块设计原则

1. **模块化设计**：每个任务独立成模块，便于单独学习和修改
2. **统一接口**：各模块采用相似的代码结构，降低学习成本
3. **低耦合**：模块间无强依赖，可独立运行
4. **可扩展**：预留接口，便于添加新模型和新功能

---

## 环境配置步骤

### 系统要求
- **操作系统**：Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python版本**：3.8 ~ 3.11
- **GPU**：NVIDIA GPU (推荐8GB+显存)，支持CUDA 11.7+
- **CPU**：4核以上，推荐8核
- **内存**：16GB+，推荐32GB

### 步骤1：创建Conda环境（推荐）

```bash
# 创建虚拟环境
conda create -n cv_firstglance python=3.10
conda activate cv_firstglance
```

### 步骤2：安装PyTorch

```bash
# GPU版本（推荐）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU版本（仅用于测试）
# conda install pytorch torchvision cpuonly -c pytorch
```

### 步骤3：安装其他依赖

```bash
pip install -r requirements.txt
```

### 步骤4：验证安装

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

预期输出：
```
PyTorch version: 2.1.0
CUDA available: True
```

---

## 快速开始指南

### 图像分类任务

#### 1. 开始训练
```bash
cd classification
python train.py
```

训练过程输出示例：
```
Using device: cuda
Files already downloaded and verified
Files already downloaded and verified
Epoch: 1/200 | Loss: 1.876 | Acc: 30.125% (15062/50000)
Epoch: 2/200 | Loss: 1.234 | Acc: 55.432% (27716/50000)
...
```

#### 2. 评估模型
```bash
python eval.py
```

#### 3. 预期结果
- 训练200 epoch后，测试集准确率可达95%+
- 模型权重保存为 `resnet18_cifar10_best.pth`

### 图像分割任务

#### 1. 数据集准备
```bash
# 自动下载VOC 2012数据集（首次运行自动下载）
# 数据集将保存至 ./data/VOCdevkit/
```

#### 2. 开始训练
```bash
cd segmentation
python train.py
```

#### 3. 评估模型
```bash
python eval.py
```

#### 4. 预期结果
- 训练100 epoch后，mIoU可达67.64%
- 模型权重保存为 `best_model.pth`

---

## 各模块功能说明

### 1. Classification 模块

#### dataset.py
- **功能**：CIFAR-10数据集加载与预处理
- **核心特性**：
  - 自动下载数据集
  - 训练集数据增强（RandomCrop + RandomHorizontalFlip）
  - 标准化处理
  - DataLoader封装

#### model.py
- **功能**：ResNet系列模型实现
- **核心特性**：
  - BasicBlock和Bottleneck两种残差块
  - ResNet18/34/50/101/152完整实现
  - 适配CIFAR-10的32×32输入

#### train.py
- **功能**：分类任务训练循环
- **核心特性**：
  - SGD优化器 + 余弦退火学习率
  - 自动GPU/CPU检测
  - 每5 epoch验证一次
  - 自动保存最佳模型

#### eval.py
- **功能**：分类模型评估
- **核心特性**：
  - Top-1/Top-5准确率计算
  - 分类报告生成
  - 混淆矩阵输出
  - 单类别准确率统计

### 2. Segmentation 模块

#### dataset.py
- **功能**：PASCAL VOC 2012分割数据集加载
- **核心特性**：
  - VOC标准格式解析
  - 21类语义分割标注处理
  - 忽略边界像素（255）
  - 图像与掩码同步变换

#### model.py
- **功能**：DeepLabV3+模型实现
- **核心特性**：
  - ASPP（空洞空间金字塔池化）模块
  - 编码器-解码器架构
  - ResNet骨干网络（支持50/101）
  - 空洞卷积与多尺度特征融合

#### train.py
- **功能**：分割任务训练循环
- **核心特性**：
  - 交叉熵损失（忽略255标签）
  - SGD优化器 + 步长学习率衰减
  - 支持ImageNet预训练

#### eval.py
- **功能**：分割模型评估
- **核心特性**：
  - 像素准确率计算
  - 每类IoU计算
  - mIoU（平均交并比）计算

### 3. Utils 模块

#### common.py
- **功能**：通用工具函数
- **核心特性**：
  - 随机种子设置（可复现性）
  - 模型检查点保存
  - 训练曲线绘制
  - 参数量统计
  - 结果可视化工具



---

## 常见问题排查

### 1. CUDA out of memory
- **解决方案**：减小batch_size
- 分类：128 → 64 → 32
- 分割：8 → 4 → 2
- 检测：8 → 4 → 2

### 2. 数据集下载失败
- **解决方案**：手动下载数据集并放置到正确路径
- CIFAR-10: `./data/cifar-10-batches-py/`
- VOC: `./data/VOCdevkit/`

### 3. 训练不收敛
- **检查点**：
  - 学习率是否过大
  - 数据归一化是否正确
  - 损失函数是否匹配任务

### 4. 评估指标异常
- **检查点**：
  - 模型权重是否正确加载
  - 评估时是否设置了model.eval()
  - 数据预处理是否与训练一致

---

## 贡献与反馈

本项目欢迎任何形式的贡献：
- 代码优化与bug修复
- 文档完善与翻译
- 新模型与新功能添加
- 使用经验与问题反馈

## 许可证
MIT License - 可自由用于学习和研究目的。
