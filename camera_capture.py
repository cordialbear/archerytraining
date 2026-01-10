import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


class CameraCapture:
    """统一处理 USB 摄像头、视频文件与树莓派 Picamera2 的取流。"""

    def __init__(
        self,
        source: str = "usb0",
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ) -> None:
        self.source = source
        self.resolution = resolution
        self.fps = fps

        self._source_type: Optional[str] = None
        self._cap = None  # type: ignore

    def configure(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
    ) -> None:
        """更新分辨率 / 帧率配置（需在 startCapture 前调用）。"""
        if resolution:
            self.resolution = resolution
        if fps:
            self.fps = fps

    def startCapture(self) -> None:
        """根据 source 打开不同的取流方式。"""
        if isinstance(self.source, str) and self.source.startswith("picamera"):
            self._source_type = "picamera"
            self._init_picamera()
            return

        # USB 摄像头（usb0、usb1…）或视频文件
        if isinstance(self.source, str) and self.source.startswith("usb"):
            cam_idx = int(self.source.replace("usb", ""))
            self._cap = cv2.VideoCapture(cam_idx)
            self._source_type = "usb"
        elif isinstance(self.source, str) and os.path.isfile(self.source):
            self._cap = cv2.VideoCapture(self.source)
            self._source_type = "video"
        else:
            raise ValueError(f"不支持的 source: {self.source}")

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("无法打开摄像头/视频流")

        # 设置分辨率与帧率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def _init_picamera(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # 在非树莓派环境下给出清晰提示
            raise ImportError("需要安装 picamera2 才能使用树莓派摄像头") from exc

        picam = Picamera2()
        width, height = self.resolution
        config = picam.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        picam.configure(config)
        picam.start()
        self._cap = picam

    def getFrame(self) -> Optional[np.ndarray]:
        """返回一帧 BGR 图像，读取失败时返回 None。"""
        if self._source_type == "picamera":
            frame = self._cap.capture_array()
            # Picamera2 默认输出 RGB，这里转换为 BGR 以兼容 OpenCV/YOLO
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not self._cap:
            raise RuntimeError("请先调用 startCapture()")

        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self._source_type == "picamera" and self._cap:
            self._cap.stop()
        elif self._cap and hasattr(self._cap, "release"):
            self._cap.release()
        self._cap = None
        self._source_type = None


class PostDetection:
    """从 CameraCapture 拉流并做人体姿态检测。"""

    def __init__(
        self,
        capture: CameraCapture,
        model_path: str = "models/yolov8n-pose_ncnn",
        conf: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        self.capture = capture
        self.model = YOLO(model_path)
        self.conf = conf

    def infer_once(self) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """获取一帧并返回 (原始帧, keypoints 列表)。"""
        frame = self.capture.getFrame()
        if frame is None:
            return None

        results = self.model(frame, verbose=False, conf=self.conf)
        keypoints: List[np.ndarray] = []
        for r in results:
            if r.keypoints is not None:
                keypoints.append(r.keypoints.xy.cpu().numpy())
        return frame, keypoints

    def draw_poses(
        self, frame: np.ndarray, keypoints: List[np.ndarray]
    ) -> np.ndarray:
        """将关键点绘制到图像上，便于可视化。"""
        draw_img = frame.copy()
        for kp_set in keypoints:
            for kp in kp_set:
                for x, y in kp:
                    cv2.circle(draw_img, (int(x), int(y)), 2, (0, 255, 255), -1)
        return draw_img

