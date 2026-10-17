import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import Detection

from helmet_detection import (
    detect as detect_helmet
)
from seatbelt_detection import (
    detect as detect_seatbelt
)
from mobile_phone_detection import (
    detect as detect_mobile
)

from red_light_detection import RedLightDetector
from wrong_lane_detection import WrongLaneDetector

from event_manager import EventManager


MODEL_PATH = "models/best.pt"
VIDEO_PATH = "traffic.mp4"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "road_config.json"

OUTPUT_DIR = "outputs"
CONFIDENCE = 0.35


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run train.py and copy best.pt "
            "into models/."
        )

    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    with open(
        CONFIG_PATH,
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    video_writer = cv2.VideoWriter(
        str(
            Path(OUTPUT_DIR)
            / "annotated_video.mp4"
        ),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    manager = EventManager(
        OUTPUT_DIR,
        config["camera_id"],
        config["location"]
    )

    red_detector = RedLightDetector(config)
    lane_detector = WrongLaneDetector(config)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                verbose=False
            )[0]

            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(
                        box.cls[0].item()
                    )

                    name = str(
                        result.names[class_id]
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    coords = tuple(
                        map(
                            int,
                            box.xyxy[0].tolist()
                        )
                    )

                    track_id = (
                        int(box.id[0].item())
                        if box.id is not None
                        else None
                    )

                    detections.append(
                        Detection(
                            name,
                            confidence,
                            coords,
                            track_id
                        )
                    )

            context = {"config": config}

            # Five independent violation modules.
            events = []

            events += detect_helmet(
                frame, detections, context
            )

            events += detect_seatbelt(
                frame, detections, context
            )

            events += detect_mobile(
                frame, detections, context
            )

            events += red_detector.detect(
                frame, detections, context
            )

            events += lane_detector.detect(
                frame, detections, context
            )

            # Draw all detections.
            for det in detections:
                x1, y1, x2, y2 = det.box

                label = (
                    f"{det.class_name} "
                    f"{det.confidence:.2f}"
                )

                if det.track_id is not None:
                    label += f" ID:{det.track_id}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 200, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    2
                )

            # Save event reports and evidence.
            for event in events:
                manager.record(frame, event)

            # Draw stop line and signal ROI.
            start, end = config["stop_line"]

            cv2.line(
                frame,
                tuple(start),
                tuple(end),
                (255, 0, 0),
                2
            )

            rx1, ry1, rx2, ry2 = (
                config["signal_roi"]
            )

            cv2.rectangle(
                frame,
                (rx1, ry1),
                (rx2, ry2),
                (0, 255, 255),
                2
            )

            video_writer.write(frame)

            cv2.imshow(
                "Traffic Violation Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        video_writer.release()
        manager.close()
        cv2.destroyAllWindows()

    print("Finished. Check outputs/.")


if __name__ == "__main__":
    main()
