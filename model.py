"""
Model - 数据模型层
负责管理应用的所有数据状态
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import yaml

from camera_capture import CameraCapture, PostDetection
from scoreDectect import ScoreDetector, TargetResult


@dataclass
class TrainingMetrics:
    """训练评估指标数据"""
    current_shot_id: str = "--"
    hit_score: str = "--"
    draw_stability: str = "--"
    release_stability: str = "--"
    posture_score: str = "--"
    overall_suggestion: str = "--"


@dataclass
class ScoringData:
    """计分数据"""
    arrow_count: int = 6
    target_ids: List[int] = field(default_factory=list)
    target_scores: Dict[int, List[str]] = field(default_factory=dict)  # 靶位号 -> 每支箭的环数列表


@dataclass
class Settings:
    """设置数据"""
    camera_source: str = "picamera"
    resolution: str = "640x480"
    pose_detection_enabled: bool = False
    pose_model_path: str = "models/yolov8n-pose_ncnn"
    target_model_path: str = "models/target-yolov8_ncnn"
    arrow_model_path: str = "models/arrowV1-model_ncnn"


class ArcheryModel:
    """射箭训练应用的数据模型"""
    
    CONFIG_FILE = "conf.yaml"
    
    def __init__(self, config_file: str = CONFIG_FILE):
        # 配置文件路径
        self.config_file = config_file
        
        # 摄像头相关
        self.camera: Optional[CameraCapture] = None
        self.pose_detector: Optional[PostDetection] = None
        self.score_detector: Optional[ScoreDetector] = None
        
        # 训练数据
        self.training_metrics = TrainingMetrics()
        
        # 计分数据
        self.scoring_data = ScoringData()
        
        # 检测结果
        self.detected_target_results: List[TargetResult] = []
        
        # 设置（从配置文件加载）
        self.settings = self._load_settings()
        
        # 图像缓存（用于防止被垃圾回收）
        self._image_cache: List = []
    
    def _load_settings(self) -> Settings:
        """从 YAML 配置文件加载设置"""
        if not os.path.exists(self.config_file):
            # 如果配置文件不存在，使用默认设置并创建配置文件
            default_settings = Settings()
            self._save_settings(default_settings)
            return default_settings
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 解析配置
            camera_config = config.get("camera", {})
            detection_config = config.get("detection", {})
            
            settings = Settings(
                camera_source=camera_config.get("source", "picamera"),
                resolution=camera_config.get("resolution", "640x480"),
                pose_detection_enabled=detection_config.get("pose_detection_enabled", False),
                pose_model_path=detection_config.get("pose_model_path", "models/yolov8n-pose_ncnn"),
                target_model_path=detection_config.get("target_model_path", "models/target-yolov8_ncnn"),
                arrow_model_path=detection_config.get("arrow_model_path", "models/arrowV1-model_ncnn"),
            )
            
            return settings
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认设置")
            return Settings()
    
    def _save_settings(self, settings: Optional[Settings] = None):
        """保存设置到 YAML 配置文件"""
        if settings is None:
            settings = self.settings
        
        try:
            # 解析分辨率
            resolution_str = settings.resolution
            try:
                w, h = map(int, resolution_str.split("x"))
            except:
                w, h = 640, 480
            
            config = {
                "camera": {
                    "source": settings.camera_source,
                    "resolution": settings.resolution,
                    "fps": 30,
                },
                "detection": {
                    "pose_detection_enabled": settings.pose_detection_enabled,
                    "pose_model_path": settings.pose_model_path,
                    "target_model_path": settings.target_model_path,
                    "arrow_model_path": settings.arrow_model_path,
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def save_config(self):
        """保存当前设置到配置文件"""
        self._save_settings()

    def initialize_camera(self, source: str, resolution: tuple, fps: int) -> bool:
        """初始化摄像头"""
        try:
            self.camera = CameraCapture(source=source, resolution=resolution, fps=fps)
            self.camera.startCapture()
            return True
        except Exception:
            self.camera = None
            return False
    
    def initialize_pose_detector(self) -> bool:
        """初始化姿态检测器"""
        if self.camera is None:
            return False
        try:
            self.pose_detector = PostDetection(self.camera, model_path=self.settings.pose_model_path)
            return True
        except Exception:
            self.pose_detector = None
            return False
    
    def initialize_score_detector(self) -> bool:
        """初始化计分检测器"""
        if self.score_detector is not None:
            return True
        try:
            self.score_detector = ScoreDetector(
                target_model_path=self.settings.target_model_path,
                arrow_model_path=self.settings.arrow_model_path,
            )
            return True
        except Exception:
            self.score_detector = None
            return False
    
    def update_scoring_targets(self, target_results: List[TargetResult]):
        """更新检测到的靶位数据"""
        self.detected_target_results = target_results
        self.scoring_data.target_ids = list(range(1, len(target_results) + 1))
        # 初始化每个靶位的分数列表
        for target_id in self.scoring_data.target_ids:
            if target_id not in self.scoring_data.target_scores:
                self.scoring_data.target_scores[target_id] = ["--"] * self.scoring_data.arrow_count
    
    def reset_scoring_scores(self):
        """重置所有靶位的分数"""
        for target_id in self.scoring_data.target_ids:
            self.scoring_data.target_scores[target_id] = ["--"] * self.scoring_data.arrow_count
    
    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.release()
        self.camera = None
        self.pose_detector = None
        self.score_detector = None
