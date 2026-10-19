from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, weights="yolo11n.pt"):
        self.model = YOLO(weights)

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=0.35,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return detections