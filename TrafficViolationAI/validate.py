from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/detect/traffic_train/weights/best.pt"
DATA_YAML = "data.yaml"


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        plots=True
    )

    print("\n--- Model Evaluation ---")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()