from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # or s/m/l/x
model.train(data="dataset", epochs=25)
