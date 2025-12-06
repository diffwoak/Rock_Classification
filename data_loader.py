"""
数据加载和预处理模块
"""

import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import shutil

from config import Config


class RockDataset(Dataset):
    """岩石数据集类"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录路径
            transform: 数据增强和预处理变换
            mode: 'train', 'val', 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        self._load_data()
    
    def _load_data(self):
        """加载数据并创建样本列表"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 获取类别
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有样本
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if self._is_image_file(img_name):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"{self.mode}加载了 {len(self.samples)} 张图片，共 {len(self.classes)} 个类别")
    
    def _is_image_file(self, filename):
        """检查文件是否为图像文件"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 返回黑色图像作为占位符
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_distribution(self):
        """获取类别分布"""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        return distribution


def get_transforms(mode='train'):
    """
    获取数据增强和预处理变换
    
    参数:
        mode: 'train', 'val', 'test'
    """
    if mode == 'train' and Config.USE_AUGMENTATION:
        # 更强的数据增强策略，保持可配置性
        p = Config.AUGMENTATION_PARAMS
        transform_list = []

        # 随机缩放裁剪（优先）
        if p.get('use_random_resized_crop', True):
            transform_list.append(transforms.RandomResizedCrop(Config.IMG_SIZE, scale=p.get('random_resized_crop_scale', (0.8, 1.0))))
        else:
            transform_list.append(transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)))

        # 翻转与旋转
        transform_list.append(transforms.RandomHorizontalFlip(p=p.get('random_horizontal_flip', 0.5)))
        transform_list.append(transforms.RandomVerticalFlip(p=p.get('random_vertical_flip', 0.0)))
        transform_list.append(transforms.RandomRotation(degrees=p.get('random_rotation', 15)))

        # 可选的 RandAugment（取决于 torchvision 版本）
        if p.get('use_randaugment', False):
            try:
                transform_list.append(transforms.RandAugment(num_ops=p.get('randaugment_num_ops', 2), magnitude=p.get('randaugment_magnitude', 9)))
            except Exception:
                # 如果当前 torchvision 不支持 RandAugment，就忽略
                pass

        # 颜色抖动（带概率）
        color_jitter_params = p.get('color_jitter', {})
        color_jitter = transforms.ColorJitter(
            brightness=color_jitter_params.get('brightness', 0.0),
            contrast=color_jitter_params.get('contrast', 0.0),
            saturation=color_jitter_params.get('saturation', 0.0),
            hue=color_jitter_params.get('hue', 0.0)
        )
        transform_list.append(transforms.RandomApply([color_jitter], p=p.get('color_jitter_prob', 0.8)))

        # 仿射变换（平移、缩放、旋转由上面部分控制）
        affine_params = p.get('random_affine', {})
        transform_list.append(transforms.RandomAffine(
            degrees=affine_params.get('degrees', 0),
            translate=affine_params.get('translate', (0.0, 0.0)),
            scale=affine_params.get('scale', (1.0, 1.0))
        ))

        # 高斯模糊（带概率）
        if p.get('gaussian_blur_prob', 0.0) > 0:
            transform_list.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=p.get('gaussian_blur_prob', 0.3)))

        # 转为 Tensor
        transform_list.append(transforms.ToTensor())

        # 随机擦除作用于 tensor
        if p.get('random_erasing_prob', 0.0) > 0:
            transform_list.append(transforms.RandomErasing(p=p.get('random_erasing_prob', 0.25), scale=p.get('random_erasing_scale', (0.02, 0.33))))

        # 归一化
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transform_list)
    else:
        # 验证集和测试集只进行预处理
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders():
    """创建数据加载器"""
    # 检查数据集是否已划分
    train_dir = os.path.join(Config.DATA_DIR, 'train')
    val_dir = os.path.join(Config.DATA_DIR, 'valid')
    test_dir = os.path.join(Config.DATA_DIR, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        print("数据集未划分，正在自动划分...")
        split_dataset()
    
    # 创建数据集
    train_dataset = RockDataset(
        train_dir,
        transform=get_transforms('train'),
        mode='train'
    )
    
    val_dataset = RockDataset(
        val_dir,
        transform=get_transforms('val'),
        mode='val'
    )
    
    test_dataset = RockDataset(
        test_dir,
        transform=get_transforms('test'),
        mode='test'
    )
    
    # 更新类别数量配置
    Config.NUM_CLASSES = len(train_dataset.classes)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    print(f"类别数量: {Config.NUM_CLASSES}")
    print(f"类别列表: {train_dataset.classes}")
    
    # 打印类别分布
    train_dist = train_dataset.get_class_distribution()
    print("\n训练集类别分布:")
    for cls, count in train_dist.items():
        print(f"  {cls}: {count} 张")
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分数据集为训练集、验证集和测试集
    
    参数:
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 创建子目录
    for subset in ['train', 'val', 'test']:
        subset_dir = os.path.join(Config.DATA_DIR, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
    
    # 遍历每个类别
    for class_name in os.listdir(Config.DATA_DIR):
        class_dir = os.path.join(Config.DATA_DIR, class_name)
        if not os.path.isdir(class_dir) or class_name in ['train', 'val', 'test']:
            continue
        
        # 获取该类别的所有图片
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        
        if len(images) == 0:
            print(f"警告: 类别 '{class_name}' 中没有图像文件")
            continue
        
        # 划分数据集
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(1 - train_ratio), 
            random_state=Config.SEED
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, 
            test_size=test_ratio/(val_ratio + test_ratio), 
            random_state=Config.SEED
        )
        
        # 复制文件到对应目录
        for img_list, subset in zip([train_imgs, val_imgs, test_imgs], ['train', 'val', 'test']):
            subset_class_dir = os.path.join(Config.DATA_DIR, subset, class_name)
            if not os.path.exists(subset_class_dir):
                os.makedirs(subset_class_dir)
            
            for img_name in img_list:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(subset_class_dir, img_name)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"复制文件 {src} 到 {dst} 失败: {e}")
        
        print(f"类别 '{class_name}': 训练集 {len(train_imgs)} 张, 验证集 {len(val_imgs)} 张, 测试集 {len(test_imgs)} 张")
    
    print("数据集划分完成！")


def mixup_data(x, y, alpha=None, device=None):
    """
    对批次应用 mixup（返回混合后的输入和两组标签及混合系数）

    调用示例（训练循环中）:
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.4)
        outputs = model(inputs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    """
    if alpha is None:
        alpha = Config.AUGMENTATION_PARAMS.get('mixup_alpha', 0.4)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    if device is None:
        device = Config.DEVICE

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam