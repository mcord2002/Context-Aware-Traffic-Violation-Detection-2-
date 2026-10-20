import argparse
import math
from pathlib import Path

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "traffic.mp4"
OUTPUT_DIR = Path("outputs")

CONFIDENCE = 0.35
TRACKER = "bytetrack.yaml"

# -----------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Track traffic objects in a video.")
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--video", type=Path, default=Path(VIDEO_PATH))
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "detected_video.mp4",
        help="Output MP4 path; parent directories are created automatically.",
    )
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument(
        "--no-display", action="store_true",
        help="Save the annotated video without opening a preview window.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("--confidence must be a finite number between 0 and 1")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must use the .mp4 extension")
    if args.output.resolve() in {args.video.resolve(), args.model.resolve()}:
        parser.error("--output must not overwrite the input video or model")
    return args


def main(argv=None):
    args = parse_args(argv)
    for label, path in (("Model", args.model), ("Video", args.video)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    # Keep argument help and input validation available without ML dependencies.
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    video_writer = None
    frame_number = 0

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if any(not math.isfinite(value) or value < 1 for value in (width, height)):
            raise RuntimeError(f"Invalid video dimensions: {width} x {height}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (int(width), int(height)),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {args.output}")

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
                conf=args.confidence,
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

            if not args.no_display:
                cv2.imshow("YOLO11 Traffic Detection", frame)
                # Press Q to stop.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if frame_number == 0:
            raise RuntimeError(f"No readable frames in video: {args.video}")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Saved {frame_number} frames to {args.output}")


if __name__ == "__main__":
    main()
