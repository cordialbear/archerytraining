from __future__ import annotations

"""
scoreDectect.py
---------------

本模块提供 `ScoreDetector` 类，用于：
1. 使用 YOLOv8 模型检测原始图像中所有靶纸位置，并裁剪出单独的靶纸图像；
2. 使用 arrowV1-model 模型检测每张靶纸上的所有箭支坐标与环数。

注意：
- 具体的模型权重文件请在初始化 `ScoreDetector` 时通过路径传入；
- 代码只依赖 ultralytics 官方 YOLO 接口，适合作为其它模块的基础能力调用；
- 默认假设 arrowV1 模型的类别名为 `"0"~"10"` 或 `"X"`（大小写均可）。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class ArrowResult:
    """单支箭的检测结果。"""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 在靶纸局部坐标系下
    center: Tuple[int, int]  # (cx, cy)
    score: int  # 0–10
    is_x: bool
    conf: float


@dataclass
class TargetResult:
    """单张靶纸的检测与计分结果。"""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 在原始图像坐标系下
    crop: np.ndarray  # 裁剪出的靶纸图像，BGR
    arrows: List[ArrowResult]

    @property
    def total_score(self) -> int:
        return int(sum(a.score for a in self.arrows))

    @property
    def x_count(self) -> int:
        return int(sum(1 for a in self.arrows if a.is_x))


class ScoreDetector:
    """靶纸与箭支检测/计分核心类。

    使用示例::

        detector = ScoreDetector(
            target_model_path="weights/target-yolov8.pt",
            arrow_model_path="weights/arrowV1-model.pt",
        )
        image = cv2.imread("test.jpg")
        results = detector.detect(image)

        for idx, t in enumerate(results):
            print(f"靶纸 {idx}: 总分={t.total_score}, X 环数={t.x_count}, 箭数={len(t.arrows)}")
    """

    def __init__(
        self,
        target_model_path: str,
        arrow_model_path: str,
        device: str | None = None,
        target_conf_thres: float = 0.4,
        arrow_conf_thres: float = 0.4,
    ) -> None:
        """
        Args:
            target_model_path: 检测靶纸的 YOLOv8 模型路径。
            arrow_model_path: 检测箭支与得分的 arrowV1 模型路径。
            device: 推理设备，例如 "cpu" / "cuda:0"；None 时交由 YOLO 自动选择。
            target_conf_thres: 靶纸检测最小置信度。
            arrow_conf_thres: 箭支检测最小置信度。
        """

        self.target_model = YOLO(target_model_path)
        self.arrow_model = YOLO(arrow_model_path)

        if device is not None:
            self.target_model.to(device)
            self.arrow_model.to(device)

        self.target_conf_thres = float(target_conf_thres)
        self.arrow_conf_thres = float(arrow_conf_thres)

    # ---------- 外部主接口 ----------
    def detect(self, image: np.ndarray) -> List[TargetResult]:
        """对一张原始图像执行完整的「靶纸检测 + 裁剪 + 箭支计分」流程。

        Args:
            image: 原始 BGR 图像（OpenCV 读取的格式）。

        Returns:
            每张靶纸的检测与计分结果列表。
        """

        if image is None or image.size == 0:
            raise ValueError("输入图像为空，无法进行检测。")

        targets = self._detect_targets(image)

        results: List[TargetResult] = []
        for bbox, crop in targets:
            arrows = self._detect_arrows_on_target(crop)
            results.append(TargetResult(bbox=bbox, crop=crop, arrows=arrows))

        return results

    def detect_with_scores(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, List[int]]]:
        """便捷接口：返回每张靶纸的裁剪图像及该靶纸上所有箭支分数列表。

        Returns:
            列表中的每个元素为 `(crop, scores)`：
            - `crop`: 靶纸裁剪图 (BGR)
            - `scores`: 该靶纸所有箭支的环数列表（按检测顺序，不保证排序）
        """

        targets = self.detect(image)
        return [(t.crop, [a.score for a in t.arrows]) for t in targets]

    # ---------- 靶纸检测 ----------
    def _detect_targets(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """检测图像中所有靶纸并裁剪。

        默认对所有检测到的 box 视为靶纸，如需按类别过滤，可在此处增加过滤逻辑。
        """

        h, w = image.shape[:2]
        results = self.target_model(image, verbose=False)

        targets: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                if conf < self.target_conf_thres:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1_i = max(int(x1), 0)
                y1_i = max(int(y1), 0)
                x2_i = min(int(x2), w - 1)
                y2_i = min(int(y2), h - 1)

                if x2_i <= x1_i or y2_i <= y1_i:
                    continue

                crop = image[y1_i:y2_i, x1_i:x2_i].copy()
                targets.append(((x1_i, y1_i, x2_i, y2_i), crop))

        return targets

    # ---------- 箭支与得分检测 ----------
    def _detect_arrows_on_target(self, target_image: np.ndarray) -> List[ArrowResult]:
        """在单张靶纸图像上检测所有箭支与得分。"""

        h, w = target_image.shape[:2]
        results = self.arrow_model(target_image, verbose=False)

        arrows: List[ArrowResult] = []

        # 获取类别名列表，用于转为文字标签
        names: Dict[int, str] = getattr(self.arrow_model, "names", {})

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                if conf < self.arrow_conf_thres:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1_i = max(int(x1), 0)
                y1_i = max(int(y1), 0)
                x2_i = min(int(x2), w - 1)
                y2_i = min(int(y2), h - 1)

                if x2_i <= x1_i or y2_i <= y1_i:
                    continue

                cx = (x1_i + x2_i) // 2
                cy = (y1_i + y2_i) // 2

                # 解析类别 -> 分数
                cls_idx: int | None = None
                if box.cls is not None:
                    cls_idx = int(box.cls[0].item())

                label = names.get(cls_idx, "") if cls_idx is not None else ""
                score, is_x = self._parse_score_from_label(label)

                arrows.append(
                    ArrowResult(
                        bbox=(x1_i, y1_i, x2_i, y2_i),
                        center=(cx, cy),
                        score=score,
                        is_x=is_x,
                        conf=conf,
                    )
                )

        return arrows

    @staticmethod
    def _parse_score_from_label(label: str) -> Tuple[int, bool]:
        """将模型类别名解析为 (环数, 是否X环)。

        约定：
        - "X" / "x" 表示 X 环，计 10 分，is_x=True；
        - "0"~"10" 直接转为整数；
        - 其他情况视为 0 分。
        """

        if not label:
            return 0, False

        label = label.strip()

        if label.lower() == "x":
            return 10, True

        try:
            v = int(label)
            v = max(0, min(v, 10))
            return v, False
        except ValueError:
            return 0, False


__all__ = ["ScoreDetector", "TargetResult", "ArrowResult"]
