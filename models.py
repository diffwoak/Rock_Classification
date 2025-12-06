"""
模型定义模块
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import os

from config import Config


class SimpleCNN(nn.Module):
    """基础CNN模型"""
    
    def __init__(self, num_classes=None, dropout_rate=None, use_batchnorm=True):
        super(SimpleCNN, self).__init__()
        
        # 使用配置或参数
        num_classes = num_classes or Config.NUM_CLASSES
        dropout_rate = dropout_rate or Config.DROPOUT_RATE
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算全连接层输入大小
        # 假设输入为IMG_SIZExIMG_SIZE
        fc_input_size = 256 * (Config.IMG_SIZE // 16) * (Config.IMG_SIZE // 16)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 512),
            nn.BatchNorm1d(512) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class ImprovedCNN(nn.Module):
    """改进的CNN模型，带有残差连接"""
    
    def __init__(self, num_classes=None, dropout_rate=None):
        super(ImprovedCNN, self).__init__()
        
        num_classes = num_classes or Config.NUM_CLASSES
        dropout_rate = dropout_rate or Config.DROPOUT_RATE
        
        # 卷积层块
        self.conv1 = self._make_conv_block(3, 32)
        self.conv2 = self._make_conv_block(32, 64)
        self.conv3 = self._make_conv_block(64, 128)
        self.conv4 = self._make_conv_block(128, 256)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(128, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels):
        """创建卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x


def get_pretrained_model(model_name=None, num_classes=None, freeze_backbone=None):
    """
    获取预训练模型进行迁移学习
    
    参数:
        model_name: 模型名称
        num_classes: 分类数量（若为 None 则使用默认值 10）
        freeze_backbone: 是否冻结主干网络
    """
    model_name = model_name or Config.MODEL_NAME
    num_classes = num_classes or Config.NUM_CLASSES or 10  # 若都为 None，使用默认值 10
    freeze_backbone = freeze_backbone if freeze_backbone is not None else Config.FREEZE_BACKBONE
    
    print(f"加载预训练模型: {model_name}，分类数: {num_classes}")

    def _maybe_load(model_ctor, *args, **kwargs):
        # 兼容不同 torchvision 版本的预训练权重加载
        # 优先尝试新接口 weights 参数（torchvision >= 0.13）
        try:
            # 先尝试使用 weights 参数（新版本）
            return model_ctor(weights='DEFAULT')
        except TypeError:
            pass
        
        try:
            # 再尝试使用 pretrained=True（旧版本）
            return model_ctor(pretrained=True)
        except TypeError:
            pass
        
        try:
            # 最后尝试无参数加载（非预训练）
            return model_ctor()
        except Exception as e:
            raise RuntimeError(f"无法加载模型 {model_ctor.__name__}: {e}")

    def _find_and_replace_last_linear(module, out_features):
        """在模型中查找最后一个 nn.Linear，并替换为 out_features 输出的线性层。"""
        last_parent = None
        last_name = None

        def _rec(cur_module, parent, parent_name):
            nonlocal last_parent, last_name
            for name, child in cur_module.named_children():
                _rec(child, cur_module, name)
            if isinstance(cur_module, nn.Linear):
                last_parent = parent
                last_name = parent_name

        _rec(module, None, None)

        if last_parent is None:
            # 未找到 Linear，抛出异常
            raise RuntimeError('未在模型中找到线性层以替换最终分类器')

        # 获取原始线性层输入特征
        orig = None
        if isinstance(last_parent, nn.Sequential):
            idx = int(last_name)
            orig = last_parent[idx]
            in_features = orig.in_features
            last_parent[idx] = nn.Linear(in_features, out_features)
        else:
            orig = getattr(last_parent, last_name)
            in_features = orig.in_features
            setattr(last_parent, last_name, nn.Linear(in_features, out_features))

    # 支持多种常用网络
    name = model_name.lower()

    # 规范化通用模型名到具体实现，避免 getattr 返回 module 而非可调用构造函数
    # 例如 'efficientnet' -> 'efficientnet_b0'
    if name == 'efficientnet':
        name = 'efficientnet_b0'
    elif name == 'mobilenet':
        name = 'mobilenet_v2'
    elif name == 'resnet':
        name = 'resnet50'
    elif name == 'vgg':
        name = 'vgg16'
    elif name == 'inception':
        name = 'inception_v3'

    if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        ctor = getattr(models, name)
        model = _maybe_load(ctor)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif name == 'densenet121':
        model = _maybe_load(models.densenet121)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # densenet 的分类层是 model.classifier
        in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else model.classifier[0].in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif name in ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']:
        # MobileNet 系列
        model = _maybe_load(getattr(models, name))
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # classifier 处理
        try:
            # 常见结构: Sequential([Dropout, Linear])
            if isinstance(model.classifier, nn.Sequential):
                last_idx = len(model.classifier) - 1
                in_features = model.classifier[last_idx].in_features
                model.classifier[last_idx] = nn.Linear(in_features, num_classes)
            else:
                # 直接替换最后的线性
                _find_and_replace_last_linear(model, num_classes)
        except Exception:
            _find_and_replace_last_linear(model, num_classes)

    elif name.startswith('efficientnet'):
        model = _maybe_load(getattr(models, name))
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # efficientnet 的 classifier 通常在 model.classifier
        try:
            if isinstance(model.classifier, nn.Sequential):
                last_idx = len(model.classifier) - 1
                in_features = model.classifier[last_idx].in_features
                model.classifier[last_idx] = nn.Linear(in_features, num_classes)
            else:
                _find_and_replace_last_linear(model, num_classes)
        except Exception:
            _find_and_replace_last_linear(model, num_classes)

    elif name.startswith('convnext') or name.startswith('regnet') or name.startswith('swin'):
        # ConvNeXt / RegNet / Swin 等模型，加载预训练权重并替换分类器
        model = _maybe_load(getattr(models, name))
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        try:
            _find_and_replace_last_linear(model, num_classes)
        except Exception:
            pass

    elif name == 'vgg16':
        model = _maybe_load(models.vgg16)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # vgg16 classifier[6] 是最后一层
        if isinstance(model.classifier, nn.Sequential):
            # 当 classifier 是 Sequential 时，最后一个模块替换
            last_idx = len(model.classifier) - 1
            in_features = model.classifier[last_idx].in_features
            model.classifier[last_idx] = nn.Linear(in_features, num_classes)
        else:
            _find_and_replace_last_linear(model, num_classes)

    elif name == 'inception_v3':
        # Inception-v3 需要特殊处理：有主分类器和辅助分类器(AuxLogits)
        # 注意：aux_logits=False 在训练时返回张量而不是命名元组，避免与损失函数冲突
        model = _maybe_load(models.inception_v3, aux_logits=False)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # 主分类器
        try:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        except Exception:
            try:
                _find_and_replace_last_linear(model, num_classes)
            except Exception:
                pass

        # 辅助分类器（如果存在）
        if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
            try:
                aux_in = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
            except Exception:
                # 有些 torchvision 版本命名不同，尝试查找并替换最后一层
                try:
                    _find_and_replace_last_linear(model.AuxLogits, num_classes)
                except Exception:
                    pass


    elif name.startswith('vit') or name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit']:
        # Vision Transformer (ViT) 支持（尽量使用 torchvision 的实现），避免依赖 timm
        # 也支持从外部 checkpoint 加载权重：在 `Config.PRETRAINED_WEIGHTS_PATH` 中指定路径
        try:
            ctor = getattr(models, name)
            model = _maybe_load(ctor)
        except Exception:
            # 回退：尝试常见的 torchvision 名称 vit_b_16
            try:
                model = _maybe_load(models.vit_b_16)
            except Exception:
                raise RuntimeError(
                    '无法通过 torchvision 加载 ViT。请升级 torchvision 或提供权重文件，或改为安装 timm。'
                )

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # 替换最后的分类头（不同 torchvision 版本结构可能不同）
        try:
            # torchvision 的 ViT 可能将分类头放在 model.heads 或 model.head
            if hasattr(model, 'heads'):
                # heads 可能是 nn.Sequential / nn.Linear
                if isinstance(model.heads, nn.Linear):
                    in_feats = model.heads.in_features
                    model.heads = nn.Linear(in_feats, num_classes)
                elif isinstance(model.heads, nn.Sequential):
                    # 如果 heads 是 Sequential，尝试替换其中最后一个 Linear
                    replaced = False
                    for i in range(len(model.heads) - 1, -1, -1):
                        try:
                            if isinstance(model.heads[i], nn.Linear):
                                in_feats = model.heads[i].in_features
                                model.heads[i] = nn.Linear(in_feats, num_classes)
                                replaced = True
                                break
                        except Exception:
                            continue
                    if not replaced:
                        _find_and_replace_last_linear(model, num_classes)
                else:
                    _find_and_replace_last_linear(model, num_classes)
            elif hasattr(model, 'head'):
                if isinstance(model.head, nn.Linear):
                    in_feats = model.head.in_features
                    model.head = nn.Linear(in_feats, num_classes)
                else:
                    _find_and_replace_last_linear(model, num_classes)
            else:
                _find_and_replace_last_linear(model, num_classes)
        except Exception:
            try:
                _find_and_replace_last_linear(model, num_classes)
            except Exception:
                pass

        # 如果配置中提供了外部权重路径，尝试加载（非严格匹配以兼容不同实现）
        ckpt_path = getattr(Config, 'PRETRAINED_WEIGHTS_PATH', None)
        if ckpt_path:
            try:
                if os.path.exists(ckpt_path):
                    state = torch.load(ckpt_path, map_location='cpu')
                    # 支持包含在 dict['model'] 的常见格式
                    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
                        state = state['model']
                    try:
                        model.load_state_dict(state, strict=False)
                    except Exception:
                        # 有时 checkpoint 是完整的模型，需要尝试直接赋值
                        try:
                            model = state
                        except Exception:
                            pass
            except Exception:
                pass

    else:
        # 尝试直接通过 torchvision.models 加载任意名字的模型，并替换最后的线性层
        if hasattr(models, name):
            try:
                model = _maybe_load(getattr(models, name))
                if freeze_backbone:
                    for param in model.parameters():
                        param.requires_grad = False
                try:
                    _find_and_replace_last_linear(model, num_classes)
                except Exception:
                    pass
            except Exception:
                raise ValueError(f"无法加载模型: {model_name}")
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    return model


def create_model():
    """创建模型"""
    if Config.USE_PRETRAINED:
        model = get_pretrained_model()
    else:
        model = SimpleCNN()
    
    model = model.to(Config.DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model