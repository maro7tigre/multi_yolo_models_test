from ultralytics import YOLO

# Load the heavy classification model
model = YOLO('yolov8x-cls.pt')

# Train the model using your custom dataset configuration
model.train(data='data.yaml', epochs=50, imgsz=224)
