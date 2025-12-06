"""
模型集成模块
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Union

from config import Config
from models import create_model, get_pretrained_model


class EnsembleModel:
    """模型集成类"""
    
    def __init__(self, models: List[nn.Module], device=None, weights=None):
        """
        初始化模型集成
        
        参数:
            models: 模型列表
            device: 设备
            weights: 各模型的权重
        """
        self.models = models
        self.device = device or Config.DEVICE
        self.weights = weights if weights else [1.0] * len(models)
        
        # 确保模型在正确的设备上
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                output = model(x.to(self.device))
                predictions.append(weight * torch.softmax(output, dim=1))
        
        # 加权平均
        ensemble_pred = torch.stack(predictions).sum(dim=0) / sum(self.weights)
        
        return ensemble_pred
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """预测概率"""
        with torch.no_grad():
            proba = self.predict(x)
        return proba.cpu().numpy()
    
    def predict_classes(self, x: torch.Tensor) -> np.ndarray:
        """预测类别"""
        with torch.no_grad():
            proba = self.predict(x)
            _, predicted = proba.max(1)
        return predicted.cpu().numpy()


def create_ensemble(model_configs: List[Dict]) -> EnsembleModel:
    """
    创建模型集成
    
    参数:
        model_configs: 模型配置列表，每个元素为包含以下键的字典:
            - 'type': 模型类型 ('resnet18', 'resnet50', 'vgg16', 'simple_cnn')
            - 'checkpoint': 检查点文件路径
            - 'weight': 权重（可选）
    
    返回:
        EnsembleModel: 集成模型
    """
    models = []
    weights = []
    
    for config in model_configs:
        # 创建模型
        if config['type'] == 'simple_cnn':
            from models import SimpleCNN
            model = SimpleCNN()
        else:
            model = get_pretrained_model(
                model_name=config['type'],
                num_classes=Config.NUM_CLASSES,
                freeze_backbone=False
            )
        
        # 加载权重
        if 'checkpoint' in config and config['checkpoint']:
            checkpoint = torch.load(config['checkpoint'], map_location=Config.DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        models.append(model)
        weights.append(config.get('weight', 1.0))
    
    return EnsembleModel(models, Config.DEVICE, weights)


def majority_voting(predictions: List[np.ndarray]) -> np.ndarray:
    """
    多数投票法
    
    参数:
        predictions: 预测列表，每个元素为形状为(n_samples,)的数组
    
    返回:
        np.ndarray: 集成预测结果
    """
    predictions = np.array(predictions)
    
    # 对每个样本进行多数投票
    ensemble_pred = []
    for i in range(predictions.shape[1]):
        unique, counts = np.unique(predictions[:, i], return_counts=True)
        ensemble_pred.append(unique[np.argmax(counts)])
    
    return np.array(ensemble_pred)


def weighted_average(probabilities: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """
    加权平均法
    
    参数:
        probabilities: 概率列表，每个元素为形状为(n_samples, n_classes)的数组
        weights: 权重列表
    
    返回:
        np.ndarray: 集成概率
    """
    if weights is None:
        weights = [1.0] * len(probabilities)
    
    probabilities = np.array(probabilities)
    weights = np.array(weights).reshape(-1, 1, 1)
    
    # 加权平均
    weighted_probs = probabilities * weights
    ensemble_probs = weighted_probs.sum(axis=0) / weights.sum()
    
    return ensemble_probs