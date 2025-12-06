"""
模型层冻结/解冻工具函数（放在 rc_utils 避免与项目根 `utils.py` 冲突）

核心函数：
- unfreeze_by_name_keywords: 通过参数名关键字解冻匹配的参数
- count_trainable_params: 统计可训练参数数量
- get_trainable_parameters: 返回可训练参数的迭代器，便于传入 optimizer
- print_model_param_names: 打印所有参数名称，用于调试 UNFREEZE_KEYWORDS

用法示例（推荐方式）：
    from rc_utils.model_utils import unfreeze_by_name_keywords, get_trainable_parameters
    from models import create_model
    
    model = create_model()
    # 使用 Config.FINE_TUNE_STRATEGY = 'keywords' 和 Config.UNFREEZE_KEYWORDS
    # Trainer 会在 __init__ 中自动应用冻结策略
    
    # 或手动应用：
    unfreeze_by_name_keywords(model, keywords=['classifier.6'])
    optimizer = torch.optim.SGD(get_trainable_parameters(model), lr=1e-3, momentum=0.9)
"""

from typing import Iterable, Tuple
import torch
import torch.nn as nn


def unfreeze_by_name_keywords(model: nn.Module, keywords: Iterable[str]):
    """通过关键字匹配参数名来设置 requires_grad=True/False。

    例如：keywords=('classifier.6', 'fc') 会解冻名称中包含这些关键字的参数，其余保持冻结。
    """
    keys = tuple(keywords)
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in keys)


def count_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    """返回 (trainable_count, total_count, percentage).

    返回值均为参数数量（int）和比例（float 0-100）。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def get_trainable_parameters(model: nn.Module) -> Iterable[torch.nn.parameter.Parameter]:
    """返回可训练参数的 generator，便于直接传给 optimizer。"""
    return (p for p in model.parameters() if p.requires_grad)


def print_model_param_names(model: nn.Module, max_lines: int = None):
    """打印模型中所有参数的完整名称（用于确定 keywords）。
    
    这个函数用于调试 UNFREEZE_KEYWORDS 配置——列出模型中所有参数的名称，
    帮助你确定应该解冻哪些层。
    
    参数:
        model: 要检查的模型
        max_lines: 最多打印多少行（None 表示全部）
    
    使用示例：
        from rc_utils.model_utils import print_model_param_names
        from models import create_model
        model = create_model()
        print_model_param_names(model)
        
        # 输出示例（ResNet50）：
        # layer1.0.conv1.weight
        # layer1.0.bn1.weight
        # layer1.0.bn1.bias
        # ...
        # layer4.2.conv3.weight
        # layer4.2.bn3.weight
        # layer4.2.bn3.bias
        # fc.weight
        # fc.bias
        
        # 根据输出，可以设置 Config.UNFREEZE_KEYWORDS = ['layer4', 'fc']
    """
    print("=" * 80)
    print("模型参数名称列表（用于确定 UNFREEZE_KEYWORDS）:")
    print("=" * 80)
    
    param_names = [name for name, _ in model.named_parameters()]
    
    # 打印参数名
    for i, name in enumerate(param_names):
        if max_lines is not None and i >= max_lines:
            print(f"... 还有 {len(param_names) - i} 个参数（已截断）")
            break
        print(f"{i+1:3d}. {name}")
    
    print("=" * 80)
    print(f"总参数数: {len(param_names)}")
    print("\n建议用法:")
    print("1. 查看上面的参数名列表")
    print("2. 找到你想解冻的层的名称前缀（例如 'layer4', 'fc'）")
    print("3. 在 config.py 中设置: Config.UNFREEZE_KEYWORDS = ['layer4', 'fc']")
    print("4. 设置: Config.FINE_TUNE_STRATEGY = 'keywords'")
    print("=" * 80)
