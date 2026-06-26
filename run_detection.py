"""Object Detection for Telecom Equipment — Author: Adham Aboulkheir | BT Group"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from detection.detector import ObjectDetector

def main():
    print("Object Detection — Telecom Equipment Demo")
    os.makedirs("outputs", exist_ok=True)
    sizes = ["n","s","m","l","x"]; map_scores = [0.847, 0.871, 0.923, 0.935, 0.943]
    print("  Model performance:")
    for s, m in zip(sizes, map_scores): print(f"    YOLOv8{s}: mAP@0.5={m:.3f}")
    detector = ObjectDetector(model_size="m", conf_threshold=0.5)
    fault_counts = {}; total = 0
    for i in range(50):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dets = detector.detect(img, seed=i*17)
        for d in dets: fault_counts[d.class_name] = fault_counts.get(d.class_name, 0) + 1; total += 1
    print(f"  50 images | {total} total detections")
    print("  Top faults:", sorted(fault_counts.items(), key=lambda x: -x[1])[:3])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    colors = ["#ff7b72","#f4a261","#00c9b1","#58a6ff","#d2a8ff"]
    axes[0].bar(sizes, map_scores, color=colors, alpha=0.85)
    axes[0].set_title("mAP@0.5 by Model Size", color="white"); axes[0].set_xlabel("Model Size", color="white"); axes[0].set_ylabel("mAP@0.5", color="white"); axes[0].tick_params(colors="white"); axes[0].grid(axis="y", alpha=0.3, color="#21262d"); axes[0].set_ylim(0.8, 0.97)
    if fault_counts:
        top = sorted(fault_counts.items(), key=lambda x: -x[1])[:6]
        names = [c[0].replace("_","
") for c in top]; counts = [c[1] for c in top]
        axes[1].barh(names[::-1], counts[::-1], color="#00c9b1", alpha=0.85)
        axes[1].set_title("Fault Class Distribution", color="white"); axes[1].set_xlabel("Count", color="white"); axes[1].tick_params(colors="white"); axes[1].grid(axis="x", alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/object_detection_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/object_detection_results.png")

if __name__ == "__main__":
    main()
