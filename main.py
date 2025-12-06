"""
主程序 - 岩石分类
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from config import Config
from data_loader import create_data_loaders
from models import create_model
from trainer import Trainer, plot_confusion_matrix
from utils import (
    set_seed, 
    print_model_summary, 
    hyperparameter_tuning_suggestions,
    plot_sample_images,
    save_results
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='岩石分类训练')
    
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                        help='数据目录路径')
    parser.add_argument('--model', type=str, default=Config.MODEL_NAME,
                        choices=['resnet18', 'resnet50', 'vgg16', 'efficientnet_b0', 'inception_v3', 'simple_cnn'],
                        help='模型选择')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='学习率')
    parser.add_argument('--no_augmentation', action='store_false', dest='augmentation',
                        help='不使用数据增强')
    parser.add_argument('--freeze', action='store_true',
                        help='冻结预训练模型的主干网络')
    parser.add_argument('--test_only', action='store_true',
                        help='仅测试，不训练')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='加载检查点文件路径')

    # 指定 CUDA 设备编号（可传多个，例如: --cuda_devices 0 1）
    parser.add_argument('--cuda_devices', type=int, nargs='*', default=None,
                        help='指定要使用的 CUDA 设备编号（空或未指定则自动选择）')

    # 可选优化器与学习率调度器
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
                        help='选择优化器')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD/RMSprop 动量')
    parser.add_argument('--optimizer_weight_decay', type=float, default=None,
                        help='覆盖默认的 weight decay（如果提供）')

    parser.add_argument('--scheduler', type=str, default='reducelronplateau',
                        choices=['none', 'steplr', 'exponential', 'cosineanneal', 'reducelronplateau'],
                        help='选择学习率调度器:\n'
                             '  none: 不使用调度器\n'
                             '  steplr: 每 step_size 个 epoch 将学习率乘以 gamma\n'
                             '  exponential: 每个 epoch 将学习率乘以 gamma\n'
                             '  cosineanneal: 余弦退火（单周期，推荐）\n'
                             '  reducelronplateau: 根据验证损失动态调整学习率（默认）')
    parser.add_argument('--step_size', type=int, default=7,
                        help='StepLR 的 step_size')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR/ExponentialLR 的 gamma')
    parser.add_argument('--t_max', type=int, default=50,
                        help='CosineAnnealingLR 的 T_max')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    config_updates = {
        'DATA_DIR': args.data_dir,
        'MODEL_NAME': args.model if args.model != 'simple_cnn' else 'resnet18',
        'USE_PRETRAINED': args.model != 'simple_cnn',
        'NUM_EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.lr,
        'USE_AUGMENTATION': args.augmentation,
        'FREEZE_BACKBONE': args.freeze,
    }
    Config.update_from_dict(config_updates)

    if args.model == 'inception_v3':
        Config.IMG_SIZE = 299
    
    # 如果命令行提供了 cuda_devices，则更新 Config 并设置主设备
    if args.cuda_devices is not None:
        # 允许传入空列表以显式选择 CPU
        if len(args.cuda_devices) == 0:
            Config.CUDA_DEVICES = []
            Config.DEVICE = torch.device('cpu')
            print("显式选择 CPU（未使用 CUDA 设备）")
        else:
            Config.update_from_dict({'CUDA_DEVICES': args.cuda_devices})

    # 打印配置（包含设备设置）
    Config.print_config()

    # 设置随机种子（在设置 DEVICE 之后调用，保证 cuda seed 正确应用）
    set_seed()
    
    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders()
    
    # 创建模型
    print("\n创建模型...")
    if args.model == 'simple_cnn':
        from models import SimpleCNN
        model = SimpleCNN()
    else:
        model = create_model()



    # 将模型移动到主设备
    model = model.to(Config.DEVICE)

    # 如果指定了多个 CUDA 设备且可用，则使用 DataParallel 包装模型
    try:
        if hasattr(Config, 'CUDA_DEVICES') and Config.CUDA_DEVICES and torch.cuda.is_available() and len(Config.CUDA_DEVICES) > 1:
            device_ids = [int(x) for x in Config.CUDA_DEVICES]
            print(f"使用多 GPU 训练，device_ids={device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            if hasattr(Config, 'CUDA_DEVICES') and Config.CUDA_DEVICES:
                print(f"使用单 GPU: cuda:{Config.CUDA_DEVICES[0]}")
    except Exception as e:
        print(f"GPU 配置时发生异常: {e}. 将使用单设备 {Config.DEVICE}")
    
    # 打印模型摘要
    # print_model_summary(model)
    
    # 创建训练器
    trainer = Trainer(model)
    
    # 加载检查点（如果指定）
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    elif args.test_only:
        # 仅测试模式，尝试加载最佳模型
        trainer.load_checkpoint('best_model.pth')
    
    if not args.test_only:
        # 显示优化建议
        # hyperparameter_tuning_suggestions()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        # 选择 weight_decay （允许命令行覆盖）
        weight_decay = Config.WEIGHT_DECAY if args.optimizer_weight_decay is None else args.optimizer_weight_decay

        # 创建优化器
        opt_name = args.optimizer.lower()
        if opt_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), lr=Config.LEARNING_RATE,
                momentum=args.momentum, weight_decay=weight_decay
            )
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LEARNING_RATE, weight_decay=weight_decay
            )
        elif opt_name == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(), lr=Config.LEARNING_RATE, momentum=args.momentum, weight_decay=weight_decay
            )
        elif opt_name == 'adagrad':
            optimizer = optim.Adagrad(
                model.parameters(), lr=Config.LEARNING_RATE, weight_decay=weight_decay
            )
        else:  # 默认 adam
            optimizer = optim.Adam(
                model.parameters(), lr=Config.LEARNING_RATE, weight_decay=weight_decay
            )

        print(f"使用优化器: {optimizer.__class__.__name__}, 初始 lr={Config.LEARNING_RATE}, weight_decay={weight_decay}")
        
        # 学习率调度器（由命令行 --scheduler 控制；传 'none' 可禁用）
        scheduler = None
        if args.scheduler != 'none':
            sched_name = args.scheduler.lower()
            if sched_name == 'steplr':
                scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            elif sched_name == 'exponential':
                scheduler = ExponentialLR(optimizer, gamma=args.gamma)
            elif sched_name == 'cosineanneal':
                # 余弦退火调度器
                cosine_params = Config.COSINE_ANNEALING_PARAMS
                use_warm_restart = cosine_params.get('use_warm_restart', False)
                
                if use_warm_restart:
                    # 使用 Warm Restart 版本（周期性重启学习率）
                    scheduler = CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=cosine_params.get('T_max', args.t_max),
                        T_mult=cosine_params.get('T_mult', 2),
                        eta_min=cosine_params.get('eta_min', 1e-6)
                    )
                else:
                    # 使用标准余弦退火（单周期）
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=args.t_max if args.t_max != 50 else cosine_params.get('T_max', args.t_max),
                        eta_min=cosine_params.get('eta_min', 1e-6)
                    )
            elif sched_name == 'reducelronplateau':
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=Config.SCHEDULER_PARAMS.get('factor', 0.1),
                    patience=Config.SCHEDULER_PARAMS.get('patience', 5),
                )

        if scheduler is not None:
            print(f"使用学习率调度器: {scheduler.__class__.__name__}")
        else:
            print("未使用学习率调度器")
        
        # 训练模型
        trainer.train(train_loader, val_loader, criterion, optimizer, scheduler)
        
        # 绘制训练历史
        trainer.plot_training_history()
    
    # 评估模型
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    # 在测试集上评估
    accuracy, preds, targets, probs = trainer.evaluate(test_loader, class_names)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(targets, preds, class_names)
    
    # 保存结果
    results = {
        'test_accuracy': f"{accuracy:.2f}%",
        'best_val_accuracy': f"{trainer.best_val_acc:.2f}%",
        'model': args.model,
        'num_classes': Config.NUM_CLASSES,
        'num_epochs_trained': len(trainer.train_losses),
        'batch_size': Config.BATCH_SIZE,
        'learning_rate': Config.LEARNING_RATE,
    }
    
    save_results(results)
    
    # 检查是否达到目标准确率
    if accuracy >= 80:
        print(f"\n🎉 恭喜！模型准确率达到了 {accuracy:.2f}%，超过了80%的目标！")
    else:
        print(f"\n⚠️  模型准确率为 {accuracy:.2f}%，未达到80%的目标。")
        print("建议尝试以下优化方法:")
        hyperparameter_tuning_suggestions()
    
    return accuracy


if __name__ == "__main__":
    try:
        accuracy = main()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()