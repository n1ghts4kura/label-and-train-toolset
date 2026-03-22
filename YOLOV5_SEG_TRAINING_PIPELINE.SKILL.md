# YOLOv5-seg 训练完整链路

> 本技能描述从零初始化到视频流推理验证的完整工作流。

---

## 阶段一：环境初始化

```bash
# 1. 克隆 YOLOv5 仓库
git clone https://github.com/ultralytics/yolov5.git

# 2. 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 3. 安装依赖
pip install -r yolov5/requirements.txt
pip install onnx onnxruntime onnxscript

# 4. 安装项目依赖
pip install opencv-python numpy pillow pyyaml
```

---

## 阶段二：配置项目

### 1. 创建项目配置文件

在 `configs/projects/<项目名>.yaml` 中定义：

```yaml
project_name: "your_project"
classes:
  - name: "class_a"
    id: 0
    description: "类别A"
  - name: "class_b"
    id: 1
    description: "类别B"
data_root: "./train_data"
origin_dir: "./origin_pics"
train_val_ratio: 9
random_seed: 42
```

### 2. 配置 ISAT 类别文件

在 `origin_pics/isat.yaml` 中定义 ISAT 标注工具的类别（包含颜色）：

```yaml
label:
  - color: '#fff198'
    name: class_a
  - color: '#8b4513'
    name: class_b
```

---

## 阶段三：数据采集

### MP4 视频抽帧

```bash
# 方法1：逐帧抽帧
python scripts/mp4_to_images.py <mp4路径> --output <输出目录>

# 方法2：按间隔抽帧
python scripts/video2jpg.py <mp4路径> --fps 30 --output <输出目录>
```

抽帧后的图片放入 `origin_pics/` 目录。

---

## 阶段四：ISAT-SAM 标注

### 启动标注工具

```bash
# Windows
python -m ISAT

# 或指定项目
python -m ISAT --project rmyc_sim_v1
```

### 标注流程

1. 启动后选择项目配置文件（`configs/projects/<项目名>.yaml`）
2. 加载 `origin_pics/` 中的图片
3. 使用 SAM 自动分割 + 手动修正
4. 每个目标标注 `polygon`（多边形）并选择类别
5. 保存标注（自动生成 JSON 文件到 `origin_pics/`）

### 导出标注

标注完成后运行：

```bash
python scripts/export_labels.py
```

这会将 `origin_pics/*.json` 转换为 YOLO 格式的 `*.txt` 标签文件。

---

## 阶段五：数据集划分

```bash
python scripts/split_dataset.py
```

- 按 `train_val_ratio`（默认 9:1）划分训练集和验证集
- 输出到 `train_data/images/train`、`train_data/images/val`
- 同时生成 `dataset.yaml`（供 YOLOv5 使用）

---

## 阶段六：模型训练

### 启动训练

```bash
python scripts/train.py
```

或手动指定参数：

```bash
python yolov5/seg/train.py --data dataset.yaml --weights yolov5m-seg.pt --epochs 100 --imgsz 640 --batch 8 --project runs/train-seg --name exp
```

### 训练输出

- 权重文件：`yolov5/runs/train-seg/<exp>/weights/best.pt`
- 指标：Box mAP50、Mask mAP50

---

## 阶段七：导出 ONNX 模型

```bash
python scripts/export_onnx.py
```

或手动指定：

```bash
python yolov5/seg/export.py --weights yolov5/runs/train-seg/<exp>/weights/best.pt --data dataset.yaml --imgsz 640
```

导出后的 ONNX 文件位于 `yolov5/runs/train-seg/<exp>/weights/best.onnx`，可复制到项目根目录：

```bash
cp yolov5/runs/train-seg/<exp>/weights/best.onnx best.onnx
```

---

## 阶段八：推理测试

### 图片测试

```bash
python scripts/test.py <图片路径> --weights best.onnx --conf 0.3 --iou 0.4
```

输出：`output_test_validation.jpg`

### 视频流实时推理

```bash
python scripts/test_video.py <mp4路径> --weights best.onnx
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `best.onnx` | ONNX 模型路径 |
| `--conf` | `0.3` | 置信度阈值 |
| `--iou` | `0.4` | NMS IoU 阈值 |
| `--no-mask` | `False` | 关闭 mask 计算以提升帧率 |
| `--imgsz` | `640` | 推理尺寸（必须与导出时一致） |

**控制键：**
- `q` — 退出程序
- `s` — 保存当前帧截图
- `空格` — 暂停/继续

---

## 完整命令流程（示例）

```bash
# 1. 初始化
git clone https://github.com/ultralytics/yolov5.git
python -m venv venv && .\venv\Scripts\activate
pip install -r yolov5/requirements.txt
pip install onnxruntime opencv-python

# 2. 配置项目（在 configs/projects/ 下创建 yaml）

# 3. 采集数据（MP4 → 图片）
python scripts/mp4_to_images.py video.mp4 --output origin_pics/

# 4. ISAT-SAM 标注
python -m ISAT --project your_project

# 5. 导出标签 + 划分数据集
python scripts/export_labels.py
python scripts/split_dataset.py

# 6. 训练
python scripts/train.py

# 7. 导出 ONNX
python scripts/export_onnx.py
cp yolov5/runs/train-seg/exp3/weights/best.onnx best.onnx

# 8. 测试
python scripts/test.py train_data/images/val/img.jpg --weights best.onnx
python scripts/test_video.py video.mp4 --weights best.onnx --no-mask
```

---

## 关键文件说明

| 文件路径 | 作用 |
|---------|------|
| `configs/projects/<项目>.yaml` | 项目配置（类别、数据路径、划分比例） |
| `dataset.yaml` | YOLO 数据集配置（自动生成） |
| `origin_pics/isat.yaml` | ISAT 标注工具的类别配置 |
| `scripts/export_labels.py` | ISAT JSON → YOLO txt 格式转换 |
| `scripts/split_dataset.py` | 训练集/验证集划分 |
| `scripts/train.py` | 训练入口 |
| `scripts/export_onnx.py` | ONNX 导出 |
| `scripts/test.py` | 单张图片推理测试 |
| `scripts/test_video.py` | MP4 视频流实时推理 |
| `core/yolo_inference.py` | ONNX 推理核心模块 |
| `core/config_manager.py` | 配置管理（加载/切换项目） |
| `core/isat_converter.py` | ISAT 格式转换器 |
