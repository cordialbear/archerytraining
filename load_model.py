from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n-pose.pt")  # load an official model
model = YOLO("yolov8n-pose.pt") 

# Export the model
#model.export(format="onnx")
model.export(format="ncnn")
