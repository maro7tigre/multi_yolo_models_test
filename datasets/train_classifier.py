from ultralytics import YOLO

# Load the model with pre-trained weights. Options include:
# 'yolov8n.pt' for YOLOv8 Nano or 'yolov8s.pt' for a small model
model = YOLO('yolov8x-cls.pt')

# Commence training
model.train(
    data='images',         # Path to dataset configuration file
    imgsz=640,                # Image size (recommended: 640x640)
    epochs=50,                # Total number of training epochs
)

# Validate performance on the validation set and compute metrics like mAP, precision, and recall
metrics = model.val()

# Optionally, export the model to alternate formats for deployment
model.export(format='onnx')