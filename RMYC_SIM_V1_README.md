# RMYC Simulator v1 - 配置指南

## 项目信息

- **模型**: YOLOv5-seg（实例分割）
- **类别数量**: 5
- **类别列表**:
  1. `small_wall` - 小障碍物墙体 (ID: 0)
  2. `big_column` - 大障碍物柱子 (ID: 1)
  3. `robot` - 机器人 (ID: 2)
  4. `red_armor` - 红色装甲板 (ID: 3)
  5. `blue_armor` - 蓝色装甲板 (ID: 4)

---

## 当前状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| 项目配置 | ✅ 完成 | `configs/projects/rmyc_sim_v1.yaml` |
| 环境初始化 | ❓ 待确认 | 需要先运行 `setup_env.bat` |
| 放置标注数据 | ❓ 待完成 | 需要把 ISAT 数据放入 `origin_pics/` |
| 格式转换 | ⏳ 待运行 | `python scripts/export_labels.py --project rmyc_sim_v1` |
| 数据划分 | ⏳ 待运行 | `python scripts/split_dataset.py --project rmyc_sim_v1` |
| 模型训练 | ⏳ 待运行 | `python scripts/train.py` |
| ONNX 导出 | ⏳ 待运行 | `python scripts/export_onnx.py` |
| 推理测试 | ⏳ 待运行 | `python scripts/test.py <图片>` |

---

## 使用流程

### 1. 初始化环境（如果还没做过）

```batch
setup_env.bat
```

### 2. 放置标注数据到 `origin_pics/`

目录结构需要如下：

```
origin_pics/
├── isat.yaml              # 已准备好（勿删除）
├── image1.jpg             # 原始图片
├── image1.json            # ISAT 标注
├── image2.jpg
├── image2.json
└── ...
```

**ISAT JSON 格式示例** (`image1.json`):

```json
{
  "info": {
    "name": "image1.jpg",
    "width": 1920,
    "height": 1080,
    "description": "ISAT"
  },
  "objects": [
    {
      "category": "small_wall",
      "segmentation": [[x1,y1], [x2,y2], [x3,y3], ...],
      "iscrowd": false
    },
    {
      "category": "robot",
      "segmentation": [[x1,y1], [x2,y2], [x3,y3], ...],
      "iscrowd": false
    }
  ]
}
```

### 3. 转换标注格式

```batch
call venv\Scripts\activate.bat
python scripts/export_labels.py --project rmyc_sim_v1
```

### 4. 划分数据集

```batch
python scripts/split_dataset.py --project rmyc_sim_v1
```

### 5. 训练模型

```batch
python scripts/train.py
```

### 6. 导出 ONNX

```batch
python scripts/export_onnx.py
```

### 7. 测试推理

```batch
python scripts/test.py test.jpg --conf 0.25 --iou 0.45
```

---

## 训练参数

编辑 `configs/train_config.yaml` 调整参数：

```yaml
epochs: 100           # 训练轮数
batch_size: 8         # 批大小（显存不足时减小）
imgsz: 640            # 输入尺寸
device: "0"           # GPU 设备
optimizer: SGD        # 优化器
patience: 50          # 早停轮数
```

---

## 注意事项

1. **类别名称必须一致**: ISAT JSON 中的 `category` 必须与 `isat.yaml` 和 `rmyc_sim_v1.yaml` 中的名称完全一致
2. **图片格式**: 支持 jpg、png、bmp 等常见格式
3. **标注工具**: 使用 ISAT-SAM 进行标注，导出 JSON 格式
4. **训练中断**: 使用 `python scripts/train.py` 可自动从断点恢复
