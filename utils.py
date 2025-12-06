"""
工具函数模块
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import Config


def set_seed(seed=None):
    """
    设置随机种子以确保可重复性
    
    参数:
        seed: 随机种子，如果为None则使用Config.SEED
    """
    seed = seed or Config.SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"随机种子设置为: {seed}")


def print_model_summary(model, input_size=(3, 224, 224)):
    """打印模型摘要"""
    from torchsummary import summary
    
    try:
        # 获取设备字符串：'cuda' 或 'cpu'（不包括设备编号）
        device_str = Config.DEVICE.type if hasattr(Config.DEVICE, 'type') else str(Config.DEVICE).split(':')[0]
        summary(model, input_size=input_size, device=device_str)
    except ImportError:
        print("未安装torchsummary，跳过模型摘要")
        # print(f"模型: {model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    except AssertionError:
        print("torchsummary 设备验证失败，跳过模型摘要")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    计算评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('micro', 'macro', 'weighted')
    
    返回:
        dict: 包含各种指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_sample_images(dataset, num_images=8, class_names=None):
    """绘制样本图像"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        
        # 反标准化图像
        image_np = image.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        axes[i].imshow(image_np)
        
        if class_names:
            title = class_names[label]
        else:
            title = f'Label: {label}'
        
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('样本图像', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(dataset, class_names=None):
    """绘制类别分布"""
    if hasattr(dataset, 'get_class_distribution'):
        distribution = dataset.get_class_distribution()
        classes = list(distribution.keys())
        counts = list(distribution.values())
    else:
        # 从样本中统计
        labels = [label for _, label in dataset.samples]
        unique_labels = np.unique(labels)
        counts = [labels.count(label) for label in unique_labels]
        
        if class_names:
            classes = [class_names[label] for label in unique_labels]
        else:
            classes = [f'Class {label}' for label in unique_labels]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='black')
    
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title('类别分布', fontsize=14)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    # 在柱状图上添加数量标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def hyperparameter_tuning_suggestions():
    """超参数调优建议"""
    print("=" * 60)
    print("模型优化建议:")
    print("=" * 60)
    
    suggestions = [
        "1. 调整学习率: 尝试0.01, 0.001, 0.0001等不同值",
        "2. 调整批次大小: 尝试16, 32, 64等",
        "3. 增加数据增强: 随机裁剪、颜色抖动、旋转等",
        "4. 使用不同的优化器: Adam, SGD, RMSprop",
        "5. 调整网络深度: 增加或减少卷积层",
        "6. 调整Dropout率: 防止过拟合",
        "7. 使用学习率调度器: StepLR, ReduceLROnPlateau",
        "8. 尝试不同的预训练模型: ResNet50, EfficientNet, ViT",
        "9. 使用标签平滑: 提高模型泛化能力",
        "10. 尝试模型集成: 结合多个模型的预测结果",
        "11. 调整权重衰减: 控制L2正则化强度",
        "12. 使用梯度裁剪: 防止梯度爆炸",
        "13. 尝试不同的激活函数: ReLU, LeakyReLU, ELU",
        "14. 使用批量归一化: 加速训练并提高稳定性",
        "15. 调整图像大小: 尝试不同的输入分辨率",
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{suggestion}")
    
    print("=" * 60)


def save_results(results, filename='results.txt'):
    """保存结果到文件"""
    import json
    import os
    
    # 确保目录存在
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    filepath = os.path.join(Config.MODEL_SAVE_PATH, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # 保存配置
        f.write("=" * 60 + "\n")
        f.write("模型配置:\n")
        f.write("=" * 60 + "\n")
        for key in dir(Config):
            if not key.startswith('_') and not callable(getattr(Config, key)) and key.isupper():
                value = getattr(Config, key)
                f.write(f"{key}: {value}\n")
        
        # 保存结果
        f.write("\n" + "=" * 60 + "\n")
        f.write("训练结果:\n")
        f.write("=" * 60 + "\n")
        
        if isinstance(results, dict):
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(str(results))
    
    print(f"结果已保存到: {filepath}")