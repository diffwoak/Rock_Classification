"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
import warnings

from config import Config
from rc_utils.model_utils import (
    unfreeze_by_name_keywords,
    get_trainable_parameters,
    count_trainable_params
)

# 配置matplotlib以支持中文字体，并抑制字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'SimSun', 'FangSong']
matplotlib.rcParams['font.family'] = 'sans-serif'
# 让负号正常显示（可选）
matplotlib.rcParams['axes.unicode_minus'] = False


class Trainer:
    """训练器类"""
    
    def __init__(self, model, device=None, num_classes=None):
        self.model = model
        self.device = device or Config.DEVICE
        self.num_classes = num_classes or Config.NUM_CLASSES

        # 根据配置自动应用冻结/微调策略
        strategy = getattr(Config, 'FINE_TUNE_STRATEGY', 'none')
        if strategy == 'keywords':
            keys = getattr(Config, 'UNFREEZE_KEYWORDS', ['fc'])
            # 默认先冻结所有，再按关键词解冻
            for p in self.model.parameters():
                p.requires_grad = False
            unfreeze_by_name_keywords(self.model, keywords=keys)
        # 若 strategy == 'none'，不做任何冻结处理

        # 打印可训练参数信息，便于用户确认
        trainable, total, pct = count_trainable_params(self.model)
        print(f"模型可训练参数: {trainable} / {total} ({pct:.2f}%)")
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        
        # 最佳模型信息
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.best_epoch = 0
        
        # 创建保存目录
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            # 确保输出为 Tensor（支持 InceptionOutputs / tuple / list / numpy）
            outputs = self._unwrap_outputs(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                outputs = self._unwrap_outputs(outputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 避免除以零：如果验证集为空或总样本数为0，返回0.0并保持预测/目标列表为空
        n_batches = len(val_loader) if hasattr(val_loader, '__len__') else 0
        if n_batches > 0:
            epoch_loss = running_loss / n_batches
        else:
            epoch_loss = 0.0

        if total > 0:
            epoch_acc = 100. * correct / total
        else:
            epoch_acc = 0.0
        
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc, all_preds, all_targets
    
    def train(self, train_loader, val_loader, 
              criterion=None, optimizer=None, scheduler=None):
        """完整训练过程"""
        # 默认设置
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        if optimizer is None:
            optimizer = optim.Adam(
                get_trainable_parameters(self.model),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        
        print("开始训练...")
        early_stopping_counter = 0
        
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, criterion)
            
            # 记录学习率
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 学习率调度
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
                
                # 保存模型
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    val_acc=val_acc,
                    filename='best_model.pth'
                )
                
                print(f"保存最佳模型，准确率: {val_acc:.2f}%")
            else:
                early_stopping_counter += 1
            
            # 定期保存检查点
            # if (epoch + 1) % 10 == 0:
            #     self.save_checkpoint(
            #         epoch=epoch,
            #         optimizer=optimizer,
            #         val_acc=val_acc,
            #         filename=f'checkpoint_epoch_{epoch+1}.pth'
            #     )
            
            # 早停
            if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"早停触发，在 {epoch+1} 个epoch后停止训练")
                break
        
        print(f"\n训练完成！最佳验证准确率: {self.best_val_acc:.2f}% (epoch {self.best_epoch+1})")
        
        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

    def _unwrap_outputs(self, outputs):
        """确保模型输出为 Tensor：
        - 优先使用 outputs.logits（如 InceptionOutputs）
        - 若为 tuple/list，取第一个元素
        - 若为 numpy array，转换为 Tensor 并移动到设备
        - 若无法转换，抛出清晰的错误
        """
        # 延迟导入以避免循环依赖问题
        import numpy as _np

        # 处理具有 logits 属性的命名元组/对象
        try:
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                outputs = outputs[0]

            # 若已经是 Tensor，直接返回（并确保在正确设备）
            if torch.is_tensor(outputs):
                # 如果 tensor 在 CPU 而模型/数据在 GPU，移动到 trainer 的 device
                try:
                    if outputs.device.type == 'cpu' and getattr(self, 'device', None) is not None and str(self.device).startswith('cuda'):
                        outputs = outputs.to(self.device)
                except Exception:
                    pass
                return outputs

            # numpy -> tensor
            if isinstance(outputs, _np.ndarray):
                return torch.from_numpy(outputs).to(self.device)

            # 其他可被 torch.tensor 处理的类型（列表等）
            try:
                return torch.tensor(outputs, device=self.device)
            except Exception as e:
                raise RuntimeError(f"无法将模型输出转换为 Tensor（类型: {type(outputs)}）。错误: {e}")
        except Exception as e:
            raise RuntimeError(f"处理模型输出时发生错误: {e}") from e
    
    def evaluate(self, test_loader, class_names=None):
        """评估模型"""
        print("\n评估模型...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = self._unwrap_outputs(outputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算准确率
        correct = sum(np.array(all_preds) == np.array(all_targets))
        total = len(all_targets)
        accuracy = 100. * correct / total
        
        print(f"测试集准确率: {accuracy:.2f}%")
        
        # 生成分类报告
        if class_names:
            print("\n分类报告:")
            print(classification_report(all_targets, all_preds, target_names=class_names))
        
        return accuracy, all_preds, all_targets, all_probs
    
    def save_checkpoint(self, epoch, optimizer, val_acc, filename='checkpoint.pth'):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': {k: getattr(Config, k) for k in dir(Config) 
                      if not k.startswith('_') and not callable(getattr(Config, k)) and k.isupper()}
        }
        
        save_path = os.path.join(Config.MODEL_SAVE_PATH, filename)
        torch.save(checkpoint, save_path)
        print(f"检查点保存到: {save_path}")
    
    def load_checkpoint(self, filename='best_model.pth'):
        """加载检查点"""
        checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, filename)
        
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        
        print(f"加载检查点: {filename}")
        print(f"验证准确率: {self.best_val_acc:.2f}%")
        
        return True
    
    def plot_training_history(self):
        """绘制训练历史"""
        if not self.train_losses:
            print("没有训练历史可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 绘制损失
        epochs = range(1, len(self.train_losses) + 1)
        
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 绘制准确率
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].axhline(y=self.best_val_acc, color='g', linestyle='--', 
                           label=f'Best Val Acc: {self.best_val_acc:.2f}%')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 绘制学习率
        if self.learning_rates:
            axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 绘制准确率差值
        if len(self.train_accs) == len(self.val_accs):
            diff = [train - val for train, val in zip(self.train_accs, self.val_accs)]
            axes[1, 1].plot(epochs, diff, 'purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy Difference (%)')
            axes[1, 1].set_title('Train-Val Accuracy Difference')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'training_history.png'), dpi=300)
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.title('Confusion matrix', fontsize=16)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predictive label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(Config.MODEL_SAVE_PATH, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
