"""
yolo_inference.py
────────────────────────────────────────────────────────────────
YOLOv5-seg ONNX 推理封装模块

提供简洁的推理接口，支持实例分割检测。

使用方式:
    from core.yolo_inference import YOLOInference

    infer = YOLOInference("best.onnx", class_names=["black_bold_box"])
    results = infer.detect("test.jpg", conf_thres=0.25, iou_thres=0.45)

    for box, mask, score, cls_id in zip(*results):
        print(f"Class: {cls_id}, Score: {score:.2f}, Box: {box}")
"""

import cv2
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


class YOLOInference:
    """
    YOLOv5-seg ONNX 推理封装类
    """

    def __init__(
        self,
        weights: str,
        class_names: Optional[list[str]] = None,
        colors: Optional[list[tuple[int, int, int]]] = None,
        provider: str = "CPUExecutionProvider"
    ):
        """
        初始化推理器

        参数:
            weights: ONNX 模型路径
            class_names: 类别名称列表
            colors: 每个类别的 BGR 颜色
            provider: ONNX Runtime provider
        """
        self.weights_path = Path(weights)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {weights}")

        self.class_names = class_names or ["unknown"]
        self.colors = colors or [(255, 102, 0)]

        # 加载 ONNX 模型
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [provider]
        if provider == "CPUExecutionProvider":
            providers = ["CPUExecutionProvider"]
        else:
            providers = [provider, "CPUExecutionProvider"]

        self.sess = ort.InferenceSession(
            str(self.weights_path),
            sess_options=sess_options,
            providers=providers
        )

        self.inp_name = self.sess.get_inputs()[0].name
        out_info = self.sess.get_outputs()
        self.out_names = [o.name for o in out_info]

        # 推断类别数（自动检测）
        self.nc = 1

    def _letterbox(self, img_rgb: np.ndarray, new_shape: int = 640):
        """等比缩放并用灰色(114)填充到 new_shape×new_shape。"""
        h, w   = img_rgb.shape[:2]
        ratio  = new_shape / max(h, w)
        new_h  = int(round(h * ratio))
        new_w  = int(round(w * ratio))
        pad_h  = new_shape - new_h
        pad_w  = new_shape - new_w
        pt, pl = pad_h // 2, pad_w // 2

        resized = np.array(
            Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR)
        )
        canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
        canvas[pt:pt + new_h, pl:pl + new_w] = resized
        return canvas, ratio, (pt, pl)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    def _xywh2xyxy(self, boxes: np.ndarray) -> np.ndarray:
        y = np.empty_like(boxes)
        y[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        y[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        y[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        y[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return y

    def _box_iou(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
        iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)

    def _nms(self, boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> list:
        order = scores.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            if rest.size == 0:
                break
            ious = self._box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[rest])[0]
            mask = ious < iou_thres
            order = rest[mask]
        return keep

    def detect(
        self,
        image,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        with_mask: bool = True
    ) -> tuple:
        """
        对单张图片进行目标检测和实例分割。

        参数:
            image: 图片路径（str）或 cv2/numpy 数组（BGR 或 RGB）
            conf_thres: 置信度阈值
            iou_thres: NMS IoU 阈值
            imgsz: 推理尺寸
            with_mask: 是否计算分割掩码（关闭可大幅提升推理速度）

        返回:
            (boxes, masks, scores, class_ids)
            boxes: Nx4 np.ndarray，检测框坐标 [x1, y1, x2, y2]
            masks: NxHxW np.ndarray，二值化分割掩码列表
            scores: N np.ndarray，置信度分数
            class_ids: N np.ndarray，类别 ID
        """
        # 预处理：支持文件路径或 numpy 数组
        if isinstance(image, np.ndarray):
            # cv2.VideoCapture 返回 BGR，需转 RGB
            orig_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image
        else:
            # 文件路径
            orig_pil = Image.open(image).convert("RGB")
            orig_arr = np.array(orig_pil)
        orig_h, orig_w = orig_arr.shape[:2]

        lb_arr, ratio, (pt, pl) = self._letterbox(orig_arr, new_shape=imgsz)
        inp = lb_arr.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]

        # 推理
        outputs = self.sess.run(None, {self.inp_name: inp})

        pred = outputs[0][0]

        # 自动推断 nc
        if pred.shape[1] > 37:  # 4 + 1 + nc + 32
            self.nc = pred.shape[1] - 4 - 1 - 32
        if self.nc < 1:
            self.nc = 1

        obj_scores = pred[:, 4]
        cls_probs  = pred[:, 5:5 + self.nc]
        cls_ids    = cls_probs.argmax(axis=1)
        confs      = obj_scores * cls_probs[np.arange(len(pred)), cls_ids]

        # 置信度筛选
        keep_mask = confs >= conf_thres
        pred_f    = pred[keep_mask]
        confs_f   = confs[keep_mask]
        cls_f     = cls_ids[keep_mask]

        if pred_f.shape[0] == 0:
            return np.array([]), [], np.array([]), np.array([])

        # NMS
        boxes_xyxy = self._xywh2xyxy(pred_f[:, :4])
        keep_idx   = self._nms(boxes_xyxy, confs_f, iou_thres=iou_thres)

        boxes_xyxy = boxes_xyxy[keep_idx]
        confs_f    = confs_f[keep_idx]
        cls_f      = cls_f[keep_idx]
        coef_f     = pred_f[keep_idx, 5 + self.nc:]

        # 还原坐标到原图
        boxes_xyxy[:, [0, 2]] -= pl
        boxes_xyxy[:, [1, 3]] -= pt
        boxes_xyxy /= ratio
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h)

        # 处理 Mask（可关闭以提升速度）
        masks = []
        if with_mask and len(outputs) > 1:
            proto = outputs[1][0]
            mh, mw = proto.shape[1], proto.shape[2]
            proto_flat = proto.reshape(32, -1)
            logits = coef_f @ proto_flat
            probs = self._sigmoid(logits).reshape(-1, mh, mw)

            h_lb = int(round(orig_h * ratio))
            w_lb = int(round(orig_w * ratio))

            for m in probs:
                m_pil = Image.fromarray((m * 255).astype(np.uint8)).resize(
                    (imgsz, imgsz), Image.BILINEAR
                )
                m_arr = np.array(m_pil, dtype=np.float32) / 255.0
                m_crop = m_arr[pt:pt + h_lb, pl:pl + w_lb]
                m_full = np.array(
                    Image.fromarray((m_crop * 255).astype(np.uint8)).resize(
                        (orig_w, orig_h), Image.BILINEAR
                    ), dtype=np.float32
                ) / 255.0
                masks.append(m_full > 0.5)

        return boxes_xyxy, masks, confs_f, cls_f

    def detect_and_draw(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 640
    ) -> np.ndarray:
        """
        检测并绘制结果到图片上。

        返回绘制后的图片（np.ndarray）
        """
        boxes, masks, scores, cls_ids = self.detect(
            image_path, conf_thres, iou_thres, imgsz
        )

        orig_pil = Image.open(image_path).convert("RGB")
        result_rgb = np.array(orig_pil)

        # 绘制半透明 mask
        if len(masks) > 0:
            overlay = result_rgb.astype(np.float32)
            for idx, binary in enumerate(masks):
                color = self.colors[int(cls_ids[idx]) % len(self.colors)]
                for c, cv in enumerate(color):
                    overlay[:, :, c] = np.where(
                        binary,
                        overlay[:, :, c] * 0.55 + cv * 0.45,
                        overlay[:, :, c]
                    )
            result_rgb = overlay.clip(0, 255).astype(np.uint8)

        # 绘制边框 + 标签
        result_pil = Image.fromarray(result_rgb)
        draw = ImageDraw.Draw(result_pil)

        for idx in range(len(boxes)):
            x1, y1, x2, y2 = boxes[idx].tolist()
            conf   = float(scores[idx])
            cls_id = int(cls_ids[idx])
            color  = self.colors[cls_id % len(self.colors)]
            name   = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            label  = f"{name} {conf:.2f}"

            # 描边加粗边框
            for dd in range(3):
                draw.rectangle([x1 - dd, y1 - dd, x2 + dd, y2 + dd], outline=color)
            # 标签背景
            tw, th = 8 * len(label), 14
            by1 = max(y1 - th - 2, 0)
            draw.rectangle([x1, by1, x1 + tw, by1 + th], fill=color)
            draw.text((x1, by1), label, fill=(255, 255, 255))

        result_rgb = np.array(result_pil)

        # 保存
        if output_path:
            Image.fromarray(result_rgb).save(output_path, quality=95)

        return result_rgb
