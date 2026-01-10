"""
ScoreDetector 测试用例

测试 ScoreDetector 类的核心功能：
1. 初始化检测器
2. 检测图像中的靶纸
3. 检测靶纸上的箭支和得分
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoreDectect import ScoreDetector, TargetResult, ArrowResult


def load_config() -> dict:
    """从 conf.yaml 加载配置。"""
    config_path = project_root / "conf.yaml"
    if not config_path.exists():
        pytest.skip(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def detector():
    """创建 ScoreDetector 实例，使用 conf.yaml 中的模型路径。"""
    config = load_config()
    detection_config = config.get("detection", {})
    
    target_model_path = detection_config.get("target_model_path")
    arrow_model_path = detection_config.get("arrow_model_path")
    
    if not target_model_path or not arrow_model_path:
        pytest.skip("conf.yaml 中缺少模型路径配置")
    
    # 检查模型文件是否存在
    target_path = project_root / target_model_path
    arrow_path = project_root / arrow_model_path
    
    if not target_path.exists():
        pytest.skip(f"靶纸模型文件不存在: {target_path}")
    if not arrow_path.exists():
        pytest.skip(f"箭支模型文件不存在: {arrow_path}")
    
    return ScoreDetector(
        target_model_path=str(target_path),
        arrow_model_path=str(arrow_path),
        device="cpu",  # 测试使用 CPU
        target_conf_thres=0.3,
        arrow_conf_thres=0.3,
    )


@pytest.fixture
def test_image():
    """加载测试图像 arrow_target.jpg。"""
    image_path = Path(__file__).parent / "arrow_target.jpg"
    
    if not image_path.exists():
        pytest.skip(f"测试图像不存在: {image_path}。请将测试图像放置在此路径。")
    
    image = cv2.imread(str(image_path))
    if image is None:
        pytest.fail(f"无法读取测试图像: {image_path}")
    
    return image


class TestScoreDetector:
    """ScoreDetector 测试类。"""
    
    def test_detector_initialization(self, detector):
        """测试检测器初始化。"""
        assert detector is not None
        assert detector.target_model is not None
        assert detector.arrow_model is not None
        assert detector.target_conf_thres > 0
        assert detector.arrow_conf_thres > 0
    
    def test_detect_empty_image(self, detector):
        """测试空图像输入应抛出异常。"""
        # 测试 None 输入
        with pytest.raises(ValueError, match="输入图像为空"):
            detector.detect(None)
        
        # 测试空数组输入
        empty_array = np.array([])
        with pytest.raises(ValueError, match="输入图像为空"):
            detector.detect(empty_array)
    
    def test_detect_with_test_image(self, detector, test_image):
        """测试使用 arrow_target.jpg 进行检测。"""
        results = detector.detect(test_image)
        
        # 验证返回结果类型
        assert isinstance(results, list)
        
        # 如果有检测结果，验证结果结构
        if len(results) > 0:
            for result in results:
                assert isinstance(result, TargetResult)
                assert result.bbox is not None
                assert len(result.bbox) == 4
                assert result.crop is not None
                assert isinstance(result.crop, type(test_image))  # numpy.ndarray
                assert isinstance(result.arrows, list)
                
                # 验证靶纸裁剪图像尺寸
                assert result.crop.shape[0] > 0
                assert result.crop.shape[1] > 0
                assert result.crop.shape[2] == 3  # BGR 三通道
                
                # 验证箭支结果
                for arrow in result.arrows:
                    assert isinstance(arrow, ArrowResult)
                    assert len(arrow.bbox) == 4
                    assert len(arrow.center) == 2
                    assert 0 <= arrow.score <= 10
                    assert isinstance(arrow.is_x, bool)
                    assert 0.0 <= arrow.conf <= 1.0
                
                # 验证总分计算
                calculated_total = sum(a.score for a in result.arrows)
                assert result.total_score == calculated_total
                
                # 验证 X 环计数
                calculated_x_count = sum(1 for a in result.arrows if a.is_x)
                assert result.x_count == calculated_x_count
    
    def test_detect_with_scores(self, detector, test_image):
        """测试 detect_with_scores 便捷接口。"""
        results = detector.detect_with_scores(test_image)
        
        assert isinstance(results, list)
        
        for crop, scores in results:
            assert crop is not None
            assert isinstance(crop, type(test_image))  # numpy.ndarray
            assert isinstance(scores, list)
            
            # 验证分数范围
            for score in scores:
                assert isinstance(score, int)
                assert 0 <= score <= 10
    
    def test_parse_score_from_label(self):
        """测试分数解析静态方法。"""
        # 测试 X 环
        score, is_x = ScoreDetector._parse_score_from_label("X")
        assert score == 10
        assert is_x is True
        
        score, is_x = ScoreDetector._parse_score_from_label("x")
        assert score == 10
        assert is_x is True
        
        # 测试数字分数
        for i in range(11):
            score, is_x = ScoreDetector._parse_score_from_label(str(i))
            assert score == i
            assert is_x is False
        
        # 测试边界情况
        score, is_x = ScoreDetector._parse_score_from_label("")
        assert score == 0
        assert is_x is False
        
        score, is_x = ScoreDetector._parse_score_from_label("invalid")
        assert score == 0
        assert is_x is False
        
        # 测试超出范围的数字
        score, is_x = ScoreDetector._parse_score_from_label("15")
        assert score == 10  # 被限制到最大值 10
        assert is_x is False
        
        score, is_x = ScoreDetector._parse_score_from_label("-5")
        assert score == 0  # 被限制到最小值 0
        assert is_x is False


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])
