"""
Controller - 控制器层
负责处理用户交互和协调 Model 与 View
"""
from typing import Optional
import cv2

from model import ArcheryModel
from view import ArcheryView
from scoreDectect import TargetResult


class ArcheryController:
    """射箭训练应用的控制器"""
    
    def __init__(self, model: ArcheryModel, view: ArcheryView):
        self.model = model
        self.view = view
        
        # 从模型加载设置到视图
        self.view.load_settings_from_model(self.model.settings)
        
        # 绑定事件
        self._bind_events()
        
        # 初始化摄像头
        self._initialize_camera()
        
        # 启动训练页面图像更新循环
        self._start_camera_update_loop()
    
    def _bind_events(self):
        """绑定UI事件"""
        # 箭数变化事件
        self.view.bind_arrow_count_change(self._on_arrow_count_changed)
        
        # 按钮事件
        self.view.bind_detect_button(self.on_detect_targets)
        self.view.bind_next_round_button(self.on_next_round)
        self.view.bind_save_settings_button(self.on_save_settings)
    
    def _initialize_camera(self):
        """初始化摄像头"""
        resolution_str = self.model.settings.resolution
        try:
            w, h = map(int, resolution_str.split("x"))
            resolution = (w, h)
        except:
            resolution = (640, 480)
        
        success = self.model.initialize_camera(
            source=self.model.settings.camera_source,
            resolution=resolution,
            fps=30
        )
        
        if not success:
            self.view.show_train_error("摄像头初始化失败")
    
    def _start_camera_update_loop(self):
        """启动摄像头更新循环"""
        self._update_camera_frame()
    
    def _update_camera_frame(self):
        """更新摄像头画面"""
        if not self.model.camera:
            self.view.root.after(30, self._update_camera_frame)
            return
        
        frame = self.model.camera.getFrame()
        if frame is not None:
            # 如果启用姿态检测（从view获取最新设置）
            pose_enabled = self.view.get_settings().get("pose_detection", False)
            if pose_enabled:
                if self.model.pose_detector is None:
                    self.model.initialize_pose_detector()
                
                if self.model.pose_detector is not None:
                    result = self.model.pose_detector.infer_once()
                    if result is not None:
                        det_frame, keypoints = result
                        frame = self.model.pose_detector.draw_poses(det_frame, keypoints)
            
            self.view.update_train_image(frame)
        
        self.view.root.after(30, self._update_camera_frame)
    
    def _on_arrow_count_changed(self):
        """箭数变化回调"""
        arrow_count = self.view.get_arrow_count()
        self.model.scoring_data.arrow_count = arrow_count
        self._render_score_boxes()
    
    def on_detect_targets(self):
        """检测箭靶按钮点击"""
        if not self.model.camera:
            self.view.show_scoring_error("未初始化摄像头")
            return
        
        if not self.model.initialize_score_detector():
            self.view.show_scoring_error("加载计分模型失败")
            return
        
        frame = self.model.camera.getFrame()
        if frame is None:
            self.view.show_scoring_error("无法获取摄像头画面")
            return
        
        try:
            results = self.model.score_detector.detect(frame)
        except Exception as exc:
            self.view.show_scoring_error(f"计分检测失败：{exc}")
            return
        
        if not results:
            self.view.show_scoring_error("未检测到靶纸")
            self.model.detected_target_results = []
            self.model.scoring_data.target_ids = []
            self._render_score_boxes()
            return
        
        # 更新模型数据
        self.model.update_scoring_targets(results)
        
        # 更新视图
        self.view.display_target_images(
            results,
            on_canvas_resize=lambda: self.view.display_target_images(self.model.detected_target_results)
        )
        self._render_score_boxes()
    
    def on_next_round(self):
        """下一轮按钮点击"""
        self.model.reset_scoring_scores()
        self._render_score_boxes()
    
    def _render_score_boxes(self):
        """渲染计分记录框"""
        arrow_count = self.model.scoring_data.arrow_count
        target_ids = self.model.scoring_data.target_ids
        
        self.view.render_score_boxes(
            target_ids,
            arrow_count,
            on_canvas_resize=lambda: self.view.render_score_boxes(
                self.model.scoring_data.target_ids,
                self.model.scoring_data.arrow_count
            )
        )
    
    def on_save_settings(self):
        """保存设置按钮点击"""
        settings = self.view.get_settings()
        
        # 检查设置是否改变
        camera_changed = (
            self.model.settings.camera_source != settings["camera_source"] or
            self.model.settings.resolution != settings["resolution"]
        )
        
        # 更新模型设置
        self.model.settings.camera_source = settings["camera_source"]
        self.model.settings.resolution = settings["resolution"]
        self.model.settings.pose_detection_enabled = settings["pose_detection"]
        
        # 保存到配置文件
        self.model.save_config()
        
        # 如果摄像头源或分辨率改变，重新初始化摄像头
        if camera_changed:
            self._initialize_camera()
        
        print(f"[设置] 配置已保存: 摄像头源={settings['camera_source']}, "
              f"分辨率={settings['resolution']}, "
              f"姿态检测开启={settings['pose_detection']}")
    
    def cleanup(self):
        """清理资源"""
        self.model.cleanup()
