"""
配置文件 - 存储所有超参数和配置
"""

import torch

class Config:
    """配置类"""
    
    # 数据配置
    DATA_DIR = 'Rock_Data'  # 修改为您的数据路径
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # 训练配置
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4  # L2正则化
    DROPOUT_RATE = 0.5
    
    # 模型配置
    USE_PRETRAINED = True
    MODEL_NAME = 'resnet50'  # 可选: 'resnet18', 'resnet50', 'vgg16', 'efficientnet'
    FREEZE_BACKBONE = False
    NUM_CLASSES = None  # 将在运行时根据数据确定
    # 微调/冻结策略：'none'（不冻结） | 'keywords'（按关键词冻结）
    FINE_TUNE_STRATEGY = 'none'
    # 当 FINE_TUNE_STRATEGY == 'keywords' 时使用，例: ['classifier.6']
    UNFREEZE_KEYWORDS = ['heads']
    
    # 数据增强配置
    USE_AUGMENTATION = True
    AUGMENTATION_PARAMS = {
        # 基本变换
        'random_horizontal_flip': 0.5,
        'random_vertical_flip': 0.0,
        'random_rotation': 15,

        # 随机缩放裁剪
        'use_random_resized_crop': True,
        'random_resized_crop_scale': (0.8, 1.0),

        # 颜色扰动
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.05
        },
        'color_jitter_prob': 0.8,

        # 仿射/透视
        'random_affine': {
            'degrees': 15,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1)
        },

        # 高级增强
        'use_randaugment': True,
        'randaugment_num_ops': 2,
        'randaugment_magnitude': 9,

        'gaussian_blur_prob': 0.3,

        # 随机擦除（对 tensor 生效）
        'random_erasing_prob': 0.25,
        'random_erasing_scale': (0.02, 0.33),

        # mixup/cutmix（训练时在训练循环中使用）
        'use_mixup': True,
        'mixup_alpha': 0.4
    }
    
    
    # 优化配置
    USE_SCHEDULER = True
    SCHEDULER_PARAMS = {
        'factor': 0.5,
        'patience': 5,
        'mode': 'min'
    }
    
    # 余弦退火调度器配置
    COSINE_ANNEALING_PARAMS = {
        'T_max': 10,           # 总周期数（通常等于 NUM_EPOCHS）
        'eta_min': 1e-6,       # 最小学习率（余弦退火的底部）
        'use_warm_restart': True,  # 是否使用 Warm Restart（周期性重启）
        'T_mult': 2,           # Warm Restart 的乘数（每次周期增长 T_mult 倍）
    }
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 20
    
    # 设备配置
    # 支持多个 CUDA 设备编号（列表），默认空表示自动选择单卡或 CPU
    CUDA_DEVICES = []  # e.g. [0,1]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    MODEL_SAVE_PATH = 'checkpoints/'
    LOG_PATH = 'logs/'
    
    # 随机种子
    SEED = 42
    
    @classmethod
    def update_from_dict(cls, config_dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"警告: 配置中不存在键 '{key}'")

        # 如果用户指定了 CUDA_DEVICES（通过命令行或外部更新），根据其设置主设备
        if hasattr(cls, 'CUDA_DEVICES') and cls.CUDA_DEVICES:
            try:
                # 仅在有可用 CUDA 时设置
                if torch.cuda.is_available():
                    # 使用第一个编号作为主设备
                    primary = int(cls.CUDA_DEVICES[0])
                    cls.DEVICE = torch.device(f'cuda:{primary}')
                    # 将当前上下文的默认 CUDA 设备设置为主设备
                    try:
                        torch.cuda.set_device(primary)
                    except Exception:
                        # 在某些环境中设置设备可能失败（权限/环境问题），忽略即可
                        pass
                else:
                    cls.DEVICE = torch.device('cpu')

            except Exception:
                # 如果 CUDA_DEVICES 解析出错，则回退到自动选择
                cls.CUDA_DEVICES = []
                cls.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 60)
        print("当前配置:")
        print("=" * 60)
        for key in dir(cls):
            if not key.startswith('_') and not callable(getattr(cls, key)):
                if key.isupper():  # 只打印大写字母的配置项（约定为常量）
                    value = getattr(cls, key)
                    print(f"{key}: {value}")
        print("=" * 60)
