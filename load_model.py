from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n-pose.pt")  # load an official model
model = YOLO("models/yolov8n-pose.pt") 

# Export the model
#model.export(format="onnx")
#model.export(format="ncnn")
