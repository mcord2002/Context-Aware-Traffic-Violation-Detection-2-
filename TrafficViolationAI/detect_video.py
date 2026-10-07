import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "traffic.mp4"
OUTPUT_DIR = Path("outputs")

CONFIDENCE = 0.35
TRACKER = "bytetrack.yaml"

# -----------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained YOLO11 model
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)

    # Open CCTV video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / "detected_video.mp4"

    video_writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # YOLO11 detection + ByteTrack
            result = model.track(
                frame,
                persist=True,
                tracker=TRACKER,
                conf=CONFIDENCE,
                verbose=False
            )[0]

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    track_id = None

                    if box.id is not None:
                        track_id = int(
                            box.id[0].item()
                        )

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    if track_id is not None:
                        label += f" ID:{track_id}"

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # Display frame number
            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            video_writer.write(frame)

            cv2.imshow(
                "YOLO11 Traffic Detection",
                frame
            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
