from ultralytics import YOLO

# Load YOLOv8 model (nano/small) - bisa diganti backbone nanti
model = YOLO("yolov8n.pt")  # ringan untuk M1 Pro

# Train model
model.train(
    data="/Users/test/Documents/Data Mining/Mengantuk-YOLOv8/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    project="/Users/test/Documents/Data Mining/Mengantuk-YOLOv8/models",
    name="train2_aug",
    augment=True
)

