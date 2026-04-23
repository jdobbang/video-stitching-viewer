from ultralytics import YOLO
model = YOLO("asset/yolo26n.pt")
model.export(format="onnx", imgsz=640)