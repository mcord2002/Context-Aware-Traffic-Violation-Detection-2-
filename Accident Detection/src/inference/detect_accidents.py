from pathlib import Path
import cv2
from src.detection.vehicle_detector import VehicleDetector
from src.detection.tracker import extract_tracks
from src.utils.logger import get_logger

def process_video(video_path, output_path, model_path="yolo11n.pt", confidence=0.35):
    logger = get_logger()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    detector = VehicleDetector(model_path=model_path, confidence=confidence)
    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.track(frame, persist=True)
            annotated = result.plot()
            tracks = extract_tracks(result)
            cv2.putText(annotated, f"Vehicle detections: {len(tracks)} | frame {frame_number}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            writer.write(annotated)
            frame_number += 1
            if frame_number % 100 == 0:
                logger.info("Processed %d frames", frame_number)
    finally:
        cap.release()
        writer.release()
    logger.info("Saved annotated video to %s", output_path)
    logger.warning("This version detects/tracks vehicles only; accident classification is not integrated yet.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/annotated_videos/annotated.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    process_video(args.video, args.output, args.model, args.confidence)
