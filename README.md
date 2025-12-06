# Rock_Classification

简要说明：这是一个基于 PyTorch 的岩石分类训练与推理工程，包含常用数据增强、多个预训练模型（ResNet / VGG / EfficientNet / Inception / ViT 等）、微调/冻结工具、训练器和集成（ensemble）预测脚本。

快速开始：
- 准备环境：确保已安装合适版本的 `python`、`torch` 和 `torchvision`，以及常用依赖。示例：

```
pip install -r requirements.txt   # 如有 requirements
# 或仅安装核心依赖（根据系统与 CUDA 版本调整）
pip install torch torchvision
```

- 训练示例（使用 GPU）：

```
python main.py --model resnet50 --cuda_devices 0 1 --epochs 30 --batch_size 32 --lr 0.001
```

- 使用 ViT：若系统自带的 `torchvision` 支持 ViT，可直接指定 `--model vit_b_16`；也可通过 `Config.PRETRAINED_WEIGHTS_PATH` 加载外部 checkpoint（例如 `ViT-B-16.pt`）。

- Inception-v3：脚本会自动将 `IMG_SIZE` 设置为 299（Inception 需要 299x299 输入）。

模型集成（ensemble）：
- 项目包含 `ensemble.py`（未经测试），示例采用 softmax 后平均法融合多个模型的输出（确保所有模型的输入尺寸与类别数一致）。

常见路径与配置：
- 配置文件：`config.py`
- 模型定义：`models.py`
- 数据 / 增强：`data_loader.py`
- 训练器：`trainer.py`
- 集成：`ensemble.py`

注意事项：
- 若不想安装 `timm`，工程已尽量使用 `torchvision` 实现 ViT；若需要更多模型变体可以安装 `timm`。
- 训练前请确认数据组织与 `data_loader` 要求一致，且 `Config` 中的路径（比如 `MODEL_SAVE_PATH`）存在或可被创建。

如需更详细使用示例（训练/评估/推理/上传权重），告诉我你希望的场景，我会补充具体命令与示例脚本。
