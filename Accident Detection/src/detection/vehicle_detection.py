from ultralytics import YOLO

class VehicleDetector:
    """YOLO detector configured for common road-vehicle classes in COCO."""
    VEHICLE_CLASSES = (2, 3, 5, 7)  # car, motorcycle, bus, truck

    def __init__(self, model_path="yolo11n.pt", confidence=0.35, classes=VEHICLE_CLASSES):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = list(classes)

    def track(self, frame, persist=True, tracker="bytetrack.yaml"):
        """Detect vehicles and associate them with track IDs across frames."""
        result = self.model.track(
            source=frame,
            persist=persist,
            tracker=tracker,
            conf=self.confidence,
            classes=self.classes,
            verbose=False,
        )[0]
        return result
