from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "data.yaml"
BASE_MODEL = "yolo11n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8


def main():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"Missing {DATA_YAML}"
        )

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        workers=2,
        patience=20,
        project="runs/detect",
        name="traffic_train",
        plots=True,
        seed=42
    )

    print("Training finished.")
    print(
        "Best weights: "
        "runs/detect/traffic_train/weights/best.pt"
    )


if __name__ == "__main__":
    main()