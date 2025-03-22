from ultralytics import YOLO

# Load the heavy classification model
model = YOLO('yolov8x-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8x-cls.pt')
model = YOLO("yolov8x-cls.yaml").load("yolov8x-cls.pt")

# Train the model using your custom dataset configuration
model.train(data='C:/Users/marou/Documents/Github/YOLO_TEST/dataset', epochs=50, imgsz=224)

