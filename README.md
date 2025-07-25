

---

# Theft Detection Using AI and Computer Vision

This project uses **YOLO (You Only Look Once)** object detection model for real-time **theft detection** using computer vision. The system is trained on custom datasets with annotated theft-related activities and tested using live or recorded video. It leverages AI to automate surveillance tasks and alert when suspicious behavior or objects (e.g., unauthorized access, weapons, stolen items) are detected.

---

##  Project Structure

```bash
theft-detection-ai/
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── conf.yaml
├── yolov8_custom_training.ipynb
├── detect.py
├── README.md
```

---

##  Requirements

* Python 3.8+
* [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
* OpenCV
* LabelImg (for annotation)
* PyTorch
* NumPy

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

---

## Dataset Creation and Annotation

1. **Collect Data**: Gather video frames/images showing normal and suspicious activity (e.g., trespassing, object lifting, carrying bags in restricted zones).

2. **Annotation**:

   * Use [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/tzutalin/labelImg) to annotate each image.
   * Define custom classes like: `thief`, `weapon`, `bag`, `suspicious_action`.

3. **Export in YOLO Format**:

   * Each image must have a corresponding `.txt` label file with YOLO bounding box format.
   * Organize as:

     ```
     /images/train, /val, /test
     /labels/train, /val, /test
     ```

---

## Dataset Splitting

* You can use Roboflow’s export tool to automatically split the dataset into **train**, **val**, and **test**.
* Alternatively, use this command in Python:

```python
import splitfolders
splitfolders.ratio("images/", output="data/", seed=42, ratio=(.8, .1, .1))
```

Ensure the YAML config file `data.yaml` includes:

```yaml
train: data/images/train
val: data/images/val
test: data/images/test

nc: 3
names: ['thief', 'weapon', 'bag']
```

---

##  Training the Model

Train a YOLOv8 model using Ultralytics:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

* You can choose `yolov8n.pt`, `yolov8s.pt`, etc., depending on your hardware.
* After training, weights are saved in `runs/detect/train/weights/best.pt`.

---

##  Testing and Inference

Run inference on a video or webcam:

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=your_video.mp4
```

Or for real-time webcam detection:

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=0
```

---


---



