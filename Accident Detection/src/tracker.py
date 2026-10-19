from ultralytics import YOLO


model = YOLO("yolo11n.pt")

results = model.track(
    source="sample_cctv.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    verbose=False
)

for result in results:
    # Inspect vehicle boxes and persistent track IDs.
    boxes = result.boxes

    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        print(track_ids)