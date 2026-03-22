# Label Toolkit - YOLO 标注工具整合集

通用图像标注工具，支持 ISAT-SAM 标注格式到 YOLO 分割格式的转换、数据集划分、模型训练和 ONNX 推理。

---

## 目录

1. [功能特性](#功能特性)
2. [环境准备](#环境准备)
3. [目录结构](#目录结构)
4. [快速开始](#快速开始)
5. [完整使用指南](#完整使用指南)
   - [步骤 1：初始化环境](#步骤-1初始化环境)
   - [步骤 2：创建项目配置](#步骤-2创建项目配置)
   - [步骤 3：放置标注文件](#步骤-3放置标注文件)
   - [步骤 4：转换标注格式](#步骤-4转换标注格式)
   - [步骤 5：划分数据集](#步骤-5划分数据集)
   - [步骤 6：训练模型](#步骤-6训练模型)
   - [步骤 7：导出 ONNX](#步骤-7导出-onnx)
   - [步骤 8：测试推理](#步骤-8测试推理)
6. [多项目管理](#多项目管理)
7. [配置参数详解](#配置参数详解)
8. [常见问题](#常见问题)

---

## 功能特性

- **ISAT-SAM 标注工具**：集成 ISAT-SAM 交互式图像分割标注工具
- **MP4 视频帧提取**：支持从视频中提取帧图片用于标注
- **ISAT → YOLO 转换**：支持 ISAT JSON 格式到 YOLO 分割标签的批量转换
- **多项目支持**：通过配置文件管理不同项目的类别和数据路径，互不干扰
- **数据集划分**：自动将数据划分为训练集和验证集，支持自定义比例
- **YOLO 训练**：集成 YOLOv5-seg 训练流程，支持断点续训
- **ONNX 导出**：将训练好的模型导出为 ONNX 格式，支持模型简化
- **推理测试**：支持图片的实例分割推理，可视化输出

---

## 环境准备

### 系统要求

- Windows 10/11 或 Linux
- Python 3.8 或更高版本
- 推荐 NVIDIA GPU（训练加速，CPU 也可运行）

### 所需软件

1. **Python 3.8+** - [下载地址](https://www.python.org/downloads/)
2. **Git** - [下载地址](https://git-scm.com/download/win)
3. **NVIDIA GPU + CUDA**（可选）- 用于加速训练

---

## 目录结构

```
label_toolkit/
├── configs/
│   └── projects/                # 项目配置文件目录
│       └── *.yaml               # 项目配置示例
│   └── train_config.yaml        # 训练参数配置
├── core/                        # 核心模块（无需修改）
│   ├── __init__.py
│   ├── config_manager.py        # 配置管理
│   ├── isat_converter.py        # ISAT → YOLO 转换
│   ├── dataset_splitter.py      # 数据集划分
│   ├── validator.py             # YOLO 标签验证
│   └── yolo_inference.py        # ONNX 推理封装
├── scripts/                     # 可执行脚本
│   ├── extract_frames.py        # MP4 视频帧提取
│   ├── batch_extract_frames.py  # 批量视频帧提取
│   ├── setup_isat_sam.py       # ISAT-SAM 环境安装
│   ├── start_isat_sam.py       # ISAT-SAM Backend 启动
│   ├── export_labels.py         # 标注转换入口
│   ├── split_dataset.py         # 数据集划分入口
│   ├── train.py                 # 训练入口
│   ├── export_onnx.py           # ONNX 导出入口
│   └── test.py                  # 推理测试入口
├── yolov5/                      # YOLOv5 仓库（运行 setup_env.bat 后自动克隆）
├── venv/                        # Python 虚拟环境（自动创建）
├── dataset.yaml                 # YOLO 数据集配置（自动生成）
└── setup_env.bat                # 环境初始化脚本
```

---

## 快速开始

如果您已经熟悉 YOLO 训练流程，按以下顺序执行：

```batch
# 1. 初始化环境
setup_env.bat

# 2. 配置项目（编辑 configs/projects/your_project.yaml）

# 3. 提取视频帧（如果有 MP4 视频）
python scripts/extract_frames.py video.mp4 --interval 1

# 4. 安装 ISAT-SAM（用于标注）
python scripts/setup_isat_sam.py
python scripts/start_isat_sam.py  # 保持运行

# 5. 使用 ISAT GUI 进行标注（下载 ISAT from GitHub releases）

# 6. 转换标注
call venv\Scripts\activate.bat
python scripts/export_labels.py

# 5. 划分数据集
python scripts/split_dataset.py

# 6. 训练模型
python scripts/train.py

# 7. 导出 ONNX
python scripts/export_onnx.py

# 8. 测试推理
python scripts/test.py test.jpg
```

---

## ISAT-SAM 标注工具使用

### 安装 ISAT-SAM

**方式一：自动安装（推荐）**
```batch
python scripts/setup_isat_sam.py
```

**方式二：手动安装**
```batch
# 创建环境
conda create -n isat_sam python=3.10
conda activate isat_sam

# 安装 isat-sam
pip install isat-sam
```

### 启动 ISAT-SAM

**方式一：使用 Python 脚本（推荐）**
```batch
python scripts/start_isat_sam.py
```

**方式二：使用批处理脚本**
```batch
scripts\start_isat_sam.bat
```

**方式三：命令行直接启动**
```batch
conda activate isat_sam
isat-sam
```

### 标注流程

1. **安装 ISAT-SAM**：`python scripts/setup_isat_sam.py`
2. **启动 ISAT-SAM**：`python scripts/start_isat_sam.py`
3. **导入图片**：在 ISAT-SAM 界面中选择 `origin_pics/` 目录
4. **进行标注**：使用 SAM 辅助绘制多边形
5. **保存标注**：ISAT 自动保存 JSON 文件到 `origin_pics/`
6. **导出 YOLO**：运行 `python scripts/export_labels.py --project your_project`

---

## 完整使用指南

### 步骤 1：初始化环境

双击运行 `setup_env.bat` 或在命令行执行：

```batch
setup_env.bat
```

此脚本会自动完成以下操作：
- 检查 Python 版本
- 创建 Python 虚拟环境 `venv`
- 安装所需依赖（PyYAML, Pillow, NumPy, OpenCV, ONNX Runtime 等）
- 克隆 YOLOv5 官方仓库到 `yolov5/` 目录
- 创建必要的工作目录

初始化完成后，命令行会显示可用空间。

---

### 步骤 2：创建项目配置

每个项目需要独立的配置文件，定义类别、数据路径等参数。

**创建配置文件：**

1. 打开 `configs/projects/` 目录
2. 复制 `default.yaml` 并重命名为您的项目名称，例如 `my_project.yaml`
3. 编辑新配置文件

**配置示例 - 单类别项目：**

```yaml
project_name: "my_project"
classes:
  - name: "black_bold_box"
    id: 0
data_root: "./train_data"
origin_dir: "./origin_pics"
isat_yaml: "./origin_pics/isat.yaml"
train_val_ratio: 10
random_seed: 42
```

**配置示例 - 多类别项目：**

```yaml
project_name: "multi_class_project"
classes:
  - name: "person"
    id: 0
  - name: "car"
    id: 1
  - name: "tree"
    id: 2
data_root: "./train_data"
origin_dir: "./origin_pics"
isat_yaml: "./origin_pics/isat.yaml"
train_val_ratio: 10
random_seed: 42
```

**配置参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `project_name` | 项目名称 | `"my_project"` |
| `classes` | 类别列表 | `[{name: "box", id: 0}]` |
| `classes[].name` | 类别名称（必须与 ISAT 标注一致） | `"black_bold_box"` |
| `classes[].id` | 类别 ID（从 0 开始） | `0` |
| `data_root` | 训练数据输出目录 | `"./train_data"` |
| `origin_dir` | ISAT 原始标注文件目录 | `"./origin_pics"` |
| `isat_yaml` | ISAT 类别配置文件 | `"./origin_pics/isat.yaml"` |
| `train_val_ratio` | 训练集:验证集比例 | `10`（表示 10:1） |
| `random_seed` | 随机种子（保证可复现） | `42` |

---

### 步骤 3：放置标注文件

将 ISAT-SAM 生成的标注文件放入配置的原始目录（默认 `origin_pics/`）。

**目录结构示例：**

```
origin_pics/
├── isat.yaml                    # ISAT 类别配置文件（必须）
├── image1.jpg                  # 原始图片
├── image1.json                 # ISAT 标注文件
├── image2.png
├── image2.json
├── image3.jpg
└── image3.json
```

**isat.yaml 格式：**

```yaml
label:
  - name: "black_bold_box"      # 类别名称（必须与 project.yaml 中一致）
    id: 0
  - name: "other_object"
    id: 1
```

**ISAT JSON 格式说明：**

ISAT JSON 文件包含 `info`（图片信息）和 `objects`（标注对象）两部分：

```json
{
  "info": {
    "name": "image1.jpg",       # 图片文件名
    "width": 1920,              # 图片宽度
    "height": 1080,             # 图片高度
    "description": "ISAT"       # 格式标识
  },
  "objects": [
    {
      "category": "black_bold_box",  # 类别名称
      "segmentation": [[x1,y1], [x2,y2], [x3,y3], ...],  # 多边形顶点坐标（像素）
      "iscrowd": false
    }
  ]
}
```

---

### 步骤 4：转换标注格式

将 ISAT JSON 格式转换为 YOLO 分割标签格式。

**基本用法：**

```batch
# 激活虚拟环境
call venv\Scripts\activate.bat

# 使用默认配置转换
python scripts/export_labels.py
```

**指定项目配置：**

```batch
python scripts/export_labels.py --project my_project
```

**仅检查模式（不写入文件）：**

```batch
python scripts/export_labels.py --check
```

**保留 crowd 对象（默认跳过）：**

```batch
python scripts/export_labels.py --include-crowd
```

**转换输出：**

转换成功后，`train_data/` 目录会包含：

```
train_data/
├── image1.jpg                  # 图片
├── image1.txt                  # YOLO 标签
├── image2.jpg
├── image2.txt
├── image3.jpg
└── image3.txt
```

**YOLO 标签格式：**

```
# 每行一个目标：<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
# 坐标已归一化到 [0, 1]
0 0.123 0.234 0.456 0.234 0.456 0.567
```

---

### 步骤 5：划分数据集

将标注好的数据划分为训练集和验证集。

**基本用法：**

```batch
python scripts/split_dataset.py
```

**指定项目配置：**

```batch
python scripts/split_dataset.py --project my_project
```

**划分结果：**

划分后，`train_data/` 目录结构变为：

```
train_data/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── image3.jpg
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── val/
│       └── image3.txt
└── (原图片和标签会被复制到对应子目录)
```

**同时会自动生成/更新 `dataset.yaml`：**

```yaml
# YOLOv5 数据集配置
path: E:/path/to/label_toolkit/train_data

train: images/train
val: images/val

nc: 1

names:
  0: black_bold_box
```

**划分比例说明：**

- `train_val_ratio: 10` 表示每 11 张图片中，1 张分到验证集，10 张分到训练集
- 实际比例约为 91% 训练集，9% 验证集
- 使用固定随机种子 `42` 保证划分结果可复现

---

### 步骤 6：训练模型

使用 YOLOv5-seg 进行模型训练。

**基本用法：**

```batch
python scripts/train.py
```

**从头开始训练（忽略之前的断点）：**

```batch
python scripts/train.py --fresh
```

**训练过程说明：**

1. 脚本自动检测 `yolov5/runs/train-seg/exp*/weights/last.pt` 是否存在
2. 如果存在断点，自动以 `--resume` 模式继续训练
3. 如果不存在，从头开始训练

**训练输出：**

训练过程中，日志和权重保存在 `yolov5/runs/train-seg/exp*/` 目录：

```
yolov5/runs/train-seg/exp/
├── weights/
│   ├── best.pt          # 最佳权重（验证集 mAP 最高）
│   └── last.pt          # 最新权重
├── results.csv          # 训练指标日志
├── results.jpg          # 训练曲线图
└── args.yaml            # 训练参数
```

**训练参数配置：**

编辑 `configs/train_config.yaml` 调整训练参数：

```yaml
weights: yolov5/yolov5s-seg.pt   # 预训练权重
epochs: 100                       # 训练轮数
batch_size: 8                     # 批大小（显存不足时减小）
imgsz: 640                        # 输入图片尺寸（必须为 32 的倍数）
device: "0"                       # GPU 设备（"0" 或 "cpu"）
optimizer: SGD                     # 优化器（SGD/Adam/AdamW）
patience: 50                      # 早停轮数
workers: 2                        # 数据加载线程数
```

**断点续训：**

训练意外中断后，重新运行 `python scripts/train.py` 会自动从 `last.pt` 恢复。

---

### 步骤 7：导出 ONNX

将训练好的 `.pt` 权重导出为 ONNX 格式，便于部署。

**基本用法：**

```batch
python scripts/export_onnx.py
```

**指定权重文件：**

```batch
python scripts/export_onnx.py --weights yolov5/runs/train-seg/exp/weights/best.pt
```

**自定义参数：**

```batch
python scripts/export_onnx.py --imgsz 640 --opset 17
```

**ONNX 输出：**

导出成功后，会在权重同目录生成 `best.onnx`：

```
yolov5/runs/train-seg/exp/weights/
├── best.pt
└── best.onnx         # 导出的 ONNX 模型
```

同时 `label_toolkit/` 根目录也会复制一份 `best.onnx`。

---

### 步骤 8：测试推理

使用导出的 ONNX 模型进行图片推理测试。

**基本用法：**

```batch
python scripts/test.py test.jpg
```

**指定模型和阈值：**

```batch
python scripts/test.py test.jpg --weights best.onnx --conf 0.4 --iou 0.5
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image` | 测试图片路径（必需） | - |
| `--weights` | ONNX 模型路径 | `best.onnx` |
| `--conf` | 置信度阈值 | `0.25` |
| `--iou` | NMS IoU 阈值 | `0.45` |
| `--imgsz` | 推理图片尺寸 | `640` |
| `--output` | 输出图片路径 | `output_test_validation.jpg` |

**推理输出：**

- 终端显示检测到的目标数量和类别
- 生成带标注的可视化图片 `output_test_validation.jpg`
- 自动使用系统默认程序打开结果图片

---

## 多项目管理

### 创建新项目

1. **复制配置文件：**

```batch
cd configs/projects
copy default.yaml new_project.yaml
```

2. **编辑新配置文件：**

```yaml
project_name: "new_project"
classes:
  - name: "class_a"
    id: 0
  - name: "class_b"
    id: 1
data_root: "./new_project_data"
origin_dir: "./new_project_origin"
isat_yaml: "./new_project_origin/isat.yaml"
train_val_ratio: 10
random_seed: 42
```

3. **创建对应目录：**

```batch
mkdir new_project_origin
mkdir new_project_data
```

### 切换项目

运行时通过 `--project` 参数指定项目：

```batch
# 转换新项目标注
python scripts/export_labels.py --project new_project

# 划分新项目数据集
python scripts/split_dataset.py --project new_project

# 测试新项目模型
python scripts/test.py test.jpg --project new_project
```

### 项目隔离

每个项目使用独立的：
- 数据目录
- 类别配置
- 训练输出（自动保存到不同 exp 目录）

---

## 配置参数详解

### project.yaml 完整参数

```yaml
# 项目名称（用于标识）
project_name: "example_project"

# 类别定义列表
classes:
  - name: "category_name"     # 类别名称（必须与 ISAT 标注中的 category 一致）
    id: 0                     # YOLO 类别 ID（从 0 开始）

# 数据目录（相对于项目根目录或绝对路径）
data_root: "./train_data"    # YOLO 训练数据输出目录
origin_dir: "./origin_pics"   # ISAT 原始标注文件目录

# ISAT 配置文件路径
isat_yaml: "./origin_pics/isat.yaml"

# 数据集划分
train_val_ratio: 10          # train:val = 10:1
random_seed: 42              # 随机种子，保证划分可复现
```

### train_config.yaml 完整参数

```yaml
# 模型预训练权重
weights: yolov5/yolov5s-seg.pt

# 训练参数
epochs: 100                  # 最大训练轮数
batch_size: 8                # 批大小（显存不足时改为 4 或 2）
imgsz: 640                   # 输入图片尺寸（必须为 32 的倍数）

# 设备
device: "0"                  # GPU 设备："0"=GPU 0，"cpu"=CPU，可选 "0,1" 多卡

# 优化器
optimizer: SGD               # SGD（收敛稳定）或 Adam/AdamW（收敛较快）

# 收敛控制
patience: 50                 # 连续 N 轮验证集 mAP 无提升则停止训练
                             # 小数据集建议增大到 80-100

# 保存策略
save_period: -1             # 每 N 轮保存一次 checkpoint，-1=仅保存 best/last

# 数据加载
workers: 2                   # Windows 下建议 ≤2，避免 DataLoader 卡死

# 输出目录
project: runs/train-seg       # 训练结果保存根目录
name: exp                     # 实验名称，重复时自动递增为 exp2, exp3...
```

---

## 常见问题

### Q1: 运行脚时报 `ModuleNotFoundError`

**原因：** 未激活虚拟环境

**解决：**
```batch
call venv\Scripts\activate.bat
```

### Q2: 训练时报 `CUDA out of memory`

**原因：** 批大小过大，显存不足

**解决：** 减小 `configs/train_config.yaml` 中的 `batch_size`：
```yaml
batch_size: 4   # 或 2
```

### Q3: 转换标注时提示 `类别不在 isat.yaml 中`

**原因：** ISAT JSON 中的 `category` 与 `isat.yaml` 中的 `name` 不一致

**解决：** 确保两边类别名称完全一致（区分大小写）

### Q4: 验证集为空或训练集太少

**原因：** 数据量小 + 划分比例不当

**解决：** 调整 `train_val_ratio`：
```yaml
train_val_ratio: 3   # 改为 3:1，75% 训练集
```

### Q5: ONNX 导出失败

**原因：** 可能缺少 `onnxslim`

**解决：**
```batch
pip install onnx onnxslim
```

### Q6: 推理结果为空

**原因：** 置信度阈值过高或图片中确实没有目标

**解决：** 降低置信度阈值：
```batch
python scripts/test.py test.jpg --conf 0.15
```

### Q7: 训练中断后无法恢复

**原因：** `last.pt` 损坏或路径错误

**解决：** 使用 `--fresh` 从头开始：
```batch
python scripts/train.py --fresh
```

---

## 依赖列表

- Python 3.8+
- PyTorch 2.0+（通过 yolov5 自动安装）
- ONNX Runtime
- PyYAML
- Pillow
- NumPy
- OpenCV (cv2)
- onnxsim（模型简化）

---

## 许可

MIT License
