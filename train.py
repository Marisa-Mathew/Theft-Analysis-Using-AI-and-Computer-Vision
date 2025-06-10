import ultralytics
from ultralytics import YOLO

def main():
    print(f"Using Ultralytics v{ultralytics.__version__}")
    model = YOLO("yolov8n.yaml")  # or 'yolov8n.pt' if using pretrained model
    model.train(data="conf.yaml", epochs=300)

if __name__ == "__main__":
    main()
