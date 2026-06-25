# Object Detection for Telecom Equipment

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A YOLOv8-based object detection pipeline for identifying and classifying faults in telecom equipment from field photographs. The system enables remote fault diagnosis, reducing the need for on-site engineer visits.

## Problem

Telecom field engineers spend significant time travelling to sites to diagnose equipment faults that could be identified remotely from photographs. This system automates fault detection from images captured via a mobile app, enabling faster remote resolution.

## Dataset

- **Real images**: 50 manually annotated field photographs
- **Synthetic images**: 350,000 generated using GANs and Stable Diffusion (see [synthetic-data-generation](https://github.com/Adham5172001/synthetic-data-generation))
- **Classes**: 12 fault types (cable damage, connector corrosion, antenna misalignment, etc.)
- **Augmentation**: Random flips, rotations, colour jitter, weather simulation

## Model Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Time |
|-------|---------|--------------|----------------|
| YOLOv8n (baseline) | 0.847 | 0.612 | 3.2ms |
| YOLOv8m (synthetic data) | 0.923 | 0.718 | 8.1ms |
| YOLOv8l (synthetic + real) | **0.943** | **0.751** | 14.3ms |

Training on synthetic data improved mAP@0.5 by **+9.6%** over the real-data-only baseline.

## Installation

```bash
git clone https://github.com/Adham5172001/object-detection-telecom.git
cd object-detection-telecom
pip install -r requirements.txt

# Train model
python train.py --model yolov8m --epochs 100 --data config/telecom.yaml

# Run inference on an image
python detect.py --image photos/equipment.jpg --conf 0.5

# Start API server
python api/server.py
```

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/detect",
    files={"image": open("equipment.jpg", "rb")},
    data={"confidence": 0.5}
)

detections = response.json()["detections"]
for det in detections:
    print(f"{det['class']}: {det['confidence']:.1%} at {det['bbox']}")
```

## License

MIT License
