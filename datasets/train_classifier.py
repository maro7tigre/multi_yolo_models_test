from ultralytics import YOLO

# Load the heavy classification model
model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt')
model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")
    
# Train the model using your custom dataset configuration
model.train(data='data.yaml', epochs=15, imgsz=640)

