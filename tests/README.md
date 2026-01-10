# 测试说明

本目录包含 `ScoreDetector` 的测试用例。

## 测试文件

- `test_scoreDetect.py`: ScoreDetector 类的单元测试

## 测试依赖

测试需要以下文件：

1. **测试图像**: `arrow_target.jpg`
   - 请将包含靶纸和箭支的测试图像放置在此目录下
   - 图像格式：JPG/PNG 等 OpenCV 支持的格式

2. **模型文件**: 根据 `conf.yaml` 配置
   - 靶纸检测模型: `models/target-yolov8_ncnn`
   - 箭支检测模型: `models/arrowV1-model_ncnn`

3. **配置文件**: `conf.yaml` (项目根目录)
   - 包含模型路径配置

## 运行测试

```bash
# 安装测试依赖
pip install pytest pyyaml

# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_scoreDetect.py

# 显示详细输出
pytest tests/test_scoreDetect.py -v

# 运行特定测试方法
pytest tests/test_scoreDetect.py::TestScoreDetector::test_detect_with_test_image -v
```

## 注意事项

- 如果测试图像或模型文件不存在，相关测试会被跳过（skip）
- 测试默认使用 CPU 设备进行推理
- 测试会验证检测结果的类型、结构和数值范围
