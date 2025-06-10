from ultralytics import YOLO
model = YOLO("best.pt")
results = model("test.mp4")
results[0].show()