# Rock_Classification

简要说明：这是一个基于 PyTorch 的岩石分类训练与推理工程，包含常用数据增强、多个预训练模型（ResNet / VGG / EfficientNet / Inception / ViT 等）、微调/冻结工具、训练器和集成（ensemble）预测脚本。

快速开始：
- 准备环境：确保已安装合适版本的 `python`、`torch` 和 `torchvision`，以及常用依赖。示例：

```
pip install -r requirements.txt
```

- 训练示例（使用 GPU）：

```
python main.py --model resnet50 --cuda_devices 0 1 --epochs 50 --batch_size 32 --lr 0.001
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


