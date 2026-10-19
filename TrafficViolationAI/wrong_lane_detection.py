from common import point_in_polygon

VEHICLES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


class WrongLaneDetector:
    def __init__(self, config):
        self.config = config
        self.counts = {}
        self.reported = set()

        self.required = int(
            config.get(
                "wrong_lane_confirm_frames", 8
            )
        )

    def detect(self, frame, detections, context):
        events = []

        if not self.config.get(
            "wrong_lane_enabled", False
        ):
            return events

        polygon = self.config["wrong_lane_polygon"]
        visible = set()

        for det in detections:
            if (
                det.class_name not in VEHICLES
                or det.track_id is None
            ):
                continue

            tid = det.track_id
            visible.add(tid)

            if point_in_polygon(
                det.bottom_center, polygon
            ):
                self.counts[tid] = (
                    self.counts.get(tid, 0) + 1
                )
            else:
                self.counts[tid] = 0

            if (
                self.counts[tid] >= self.required
                and tid not in self.reported
            ):
                self.reported.add(tid)

                events.append({
                    "violation_type":
                        "wrong_lane_violation",
                    "track_id": tid,
                    "confidence": det.confidence,
                    "box": det.box,
                    "status": "candidate",
                    "reason": (
                        "Vehicle stayed in configured "
                        "polygon for consecutive frames"
                    )
                })

        for tid in list(self.counts):
            if tid not in visible:
                self.counts[tid] = 0

        return events