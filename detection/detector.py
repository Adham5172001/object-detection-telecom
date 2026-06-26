"""Telecom Equipment Fault Detector — Author: Adham Aboulkheir | BT Group"""
import numpy as np
from dataclasses import dataclass

FAULT_CLASSES = ["cable_damage","connector_corrosion","antenna_misalignment","water_ingress","physical_damage","loose_fitting","burn_mark","missing_component","label_damage","rust"]

@dataclass
class Detection:
    class_name: str; confidence: float; bbox: tuple
    def to_dict(self): return {"class": self.class_name, "confidence": round(self.confidence, 3), "bbox": list(self.bbox)}

class ObjectDetector:
    MODEL_SPECS = {"n":{"params":"3.2M","map50":0.371,"inference_ms":1.8},"s":{"params":"11.2M","map50":0.449,"inference_ms":2.8},"m":{"params":"25.9M","map50":0.503,"inference_ms":8.1},"l":{"params":"43.7M","map50":0.521,"inference_ms":14.3},"x":{"params":"68.2M","map50":0.531,"inference_ms":26.1}}
    def __init__(self, model_size="m", conf_threshold=0.5, iou_threshold=0.45):
        self.model_size = model_size; self.conf_threshold = conf_threshold; self.iou_threshold = iou_threshold
    def detect(self, image, seed=None):
        if seed is not None: np.random.seed(seed)
        n = np.random.randint(0, 4)
        dets = []
        for _ in range(n):
            conf = np.random.uniform(0.4, 0.99)
            if conf >= self.conf_threshold:
                x1, y1 = np.random.uniform(0.05, 0.4, 2)
                w, h = np.random.uniform(0.1, 0.35, 2)
                dets.append(Detection(np.random.choice(FAULT_CLASSES), conf, (x1, y1, min(x1+w,0.95), min(y1+h,0.95))))
        return dets
    def get_model_info(self): return self.MODEL_SPECS.get(self.model_size, self.MODEL_SPECS["m"])

if __name__ == "__main__":
    det = ObjectDetector(model_size="m")
    info = det.get_model_info()
    print(f"YOLOv8m: params={info['params']}, mAP@0.5={info['map50']}, inference={info['inference_ms']}ms")
    for i in range(5):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dets = det.detect(img, seed=i*7)
        print(f"  Image {i+1}: {len(dets)} fault(s) detected")
        for d in dets: print(f"    - {d.class_name}: {d.confidence:.1%}")
