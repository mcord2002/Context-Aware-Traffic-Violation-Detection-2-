from common import crossed_line

VEHICLES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


class RedLightDetector:
    def __init__(self, config):
        self.config = config
        self.previous = {}
        self.signal_history = []
        self.max_history = max(
            3,
            int(config.get(
                "red_signal_confirm_frames", 3
            ))
        )
        self.crossed_tracks = set()

    def detect(self, frame, detections, context):
        events = []

        if not self.config.get(
            "red_light_enabled", True
        ):
            return events

        rx1, ry1, rx2, ry2 = (
            self.config["signal_roi"]
        )

        states = []

        for det in detections:
            cx, cy = det.center

            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                if det.class_name == "red_light":
                    states.append("red")
                elif det.class_name == "green_light":
                    states.append("green")

        state = states[0] if states else "unknown"

        self.signal_history.append(state)
        self.signal_history = (
            self.signal_history[-self.max_history:]
        )

        red_confirmed = (
            len(self.signal_history)
            == self.max_history
            and all(
                s == "red"
                for s in self.signal_history
            )
        )

        start, end = self.config["stop_line"]

        for det in detections:
            if (
                det.class_name not in VEHICLES
                or det.track_id is None
            ):
                continue

            tid = det.track_id
            point = det.bottom_center
            prev = self.previous.get(tid)

            if (
                red_confirmed
                and tid not in self.crossed_tracks
                and crossed_line(
                    prev, point, start, end
                )
            ):
                self.crossed_tracks.add(tid)

                events.append({
                    "violation_type":
                        "red_light_violation",
                    "track_id": tid,
                    "confidence": det.confidence,
                    "box": det.box,
                    "status": "candidate",
                    "reason": (
                        "Red signal detected in "
                        "consecutive frames and "
                        "tracked point crossed stop line"
                    )
                })

            self.previous[tid] = point

        return events