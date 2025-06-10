import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\maris\OneDrive\Desktop\Mask Detector\runs\detect\train5\weights\best.pt")

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Define your class names (make sure these match your training order)
class_names = ["mask"]

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Perform detection
    results = model(frame, imgsz=640, verbose=False)[0]

    # Draw results
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]}: {conf:.2f}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Green for "mask", red for "no_mask"
        color = (0, 255, 0) if class_names[cls_id] == "mask" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the video frame with detections
    cv2.imshow("Mask Detection - Press 'q' to Quit", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
