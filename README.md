# Label Toolkit - YOLO 标注工具整合集

通用图像标注工具，支持 ISAT-SAM 标注格式到 YOLO 分割格式的转换、数据集划分、模型训练和 ONNX 推理。

## 特性

- **ISAT → YOLO 转换**：支持 ISAT JSON 格式到 YOLO 分割标签的批量转换
- **多项目支持**：通过配置文件管理不同项目的类别和数据路径
- **数据集划分**：自动将数据划分为训练集和验证集
- **YOLO 训练**：集成 YOLOv5-seg 训练流程
- **ONNX 导出**：将训练好的模型导出为 ONNX 格式
- **推理测试**：支持图片和视频的实例分割推理

## 目录结构

```
label_toolkit/
├── configs/
│   └── projects/          # 项目配置文件目录
│       └── default.yaml   # 默认项目配置示例
│   └── train_config.yaml  # 训练参数配置
├── core/                  # 核心模块
│   ├── __init__.py
│   ├── config_manager.py  # 配置管理
│   ├── isat_converter.py  # ISAT → YOLO 转换
│   ├── dataset_splitter.py# 数据集划分
│   ├── validator.py       # YOLO 标签验证
│   └── yolo_inference.py  # ONNX 推理封装
├── scripts/              # 可执行脚本
│   ├── export_labels.py  # 标注转换入口
│   ├── split_dataset.py  # 数据集划分入口
│   ├── train.py          # 训练入口
│   ├── export_onnx.py     # ONNX 导出入口
│   └── test.py            # 推理测试入口
├── yolov5/               # YOLOv5 仓库（运行 setup_env.bat 后自动克隆）
├── venv/                 # Python 虚拟环境
├── dataset.yaml         # YOLO 数据集配置（自动生成）
└── setup_env.bat        # 环境初始化脚本
```

## 快速开始

### 1. 初始化环境

```batch
# 双击运行或命令行执行
setup_env.bat
```

### 2. 配置项目

编辑 `configs/projects/default.yaml`：

```yaml
project_name: "my_project"
classes:
  - name: "black_bold_box"
    id: 0
data_root: "./train_data"
origin_dir: "./origin_pics"
isat_yaml: "./origin_pics/isat.yaml"
train_val_ratio: 10
```

### 3. 转换标注

```bash
# 激活虚拟环境
call venv\Scripts\activate.bat

# 转换 ISAT 标注到 YOLO 格式
python scripts/export_labels.py

# 仅检查模式（不写入文件）
python scripts/export_labels.py --check
```

### 4. 划分数据集

```bash
python scripts/split_dataset.py
```

### 5. 训练模型

```bash
python scripts/train.py

# 强制从头开始（忽略断点）
python scripts/train.py --fresh
```

### 6. 导出 ONNX

```bash
python scripts/export_onnx.py
```

### 7. 测试推理

```bash
python scripts/test.py test.jpg

# 指定模型和阈值
python scripts/test.py test.jpg --weights best.onnx --conf 0.4
```

## 多项目切换

为不同项目创建独立的配置文件：

```
configs/projects/
├── default.yaml      # 默认配置
├── project_a.yaml    # 项目 A 配置
└── project_b.yaml    # 项目 B 配置
```

运行时通过 `--project` 参数指定：

```bash
python scripts/export_labels.py --project project_a
python scripts/split_dataset.py --project project_a
```

## 项目配置说明

### project.yaml

| 字段 | 说明 |
|------|------|
| `project_name` | 项目名称 |
| `classes` | 类别列表，每个类别包含 `name` 和 `id` |
| `data_root` | 训练数据根目录 |
| `origin_dir` | 原始标注文件目录 |
| `isat_yaml` | ISAT 类别配置文件路径 |
| `train_val_ratio` | train:val 划分比例 |
| `random_seed` | 随机种子（保证可复现） |

### train_config.yaml

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `weights` | 预训练权重路径 | yolov5/yolov5s-seg.pt |
| `epochs` | 最大训练轮数 | 100 |
| `batch_size` | 批大小 | 8 |
| `imgsz` | 输入图片尺寸 | 640 |
| `device` | 训练设备 | "0" |
| `optimizer` | 优化器 | SGD |
| `patience` | EarlyStopping 轮数 | 50 |

## 依赖

- Python 3.8+
- PyTorch (通过 yolov5)
- ONNX Runtime
- NumPy, Pillow, OpenCV
- PyYAML

## 许可

MIT License
