import csv
import uuid
from datetime import datetime
from pathlib import Path

import cv2


class EventManager:
    def __init__(
        self,
        output_dir,
        camera_id,
        location
    ):
        self.output_dir = Path(output_dir)

        self.evidence_dir = (
            self.output_dir / "evidence"
        )
        self.evidence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.camera_id = camera_id
        self.location = location
        self.seen = set()

        self.csv_file = open(
            self.output_dir / "violation_events.csv",
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "violation_id",
                "camera_id",
                "location",
                "timestamp",
                "track_id",
                "violation_type",
                "confidence",
                "status",
                "reason",
                "evidence_path"
            ]
        )

        self.writer.writeheader()

    def record(self, frame, event):
        track_id = event.get("track_id")
        violation_type = event["violation_type"]

        key = (track_id, violation_type)

        if key in self.seen:
            return

        self.seen.add(key)

        violation_id = (
            "VIO-" + uuid.uuid4().hex[:12].upper()
        )

        timestamp = (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        )

        evidence_path = (
            self.evidence_dir
            / f"{violation_id}.jpg"
        )

        x1, y1, x2, y2 = event.get(
            "box",
            (
                0,
                0,
                frame.shape[1],
                frame.shape[0]
            )
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        annotated = frame.copy()

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated,
            f"{violation_type} {violation_id}",
            (max(5, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        saved = cv2.imwrite(
            str(evidence_path),
            annotated
        )

        self.writer.writerow({
            "violation_id": violation_id,
            "camera_id": self.camera_id,
            "location": self.location,
            "timestamp": timestamp,
            "track_id": (
                track_id
                if track_id is not None
                else ""
            ),
            "violation_type": violation_type,
            "confidence": round(
                float(event.get("confidence", 0)),
                4
            ),
            "status": event.get(
                "status", "candidate"
            ),
            "reason": event.get("reason", ""),
            "evidence_path": (
                str(evidence_path) if saved else ""
            )
        })

        self.csv_file.flush()

        print(
            f"[{event.get('status', 'candidate').upper()}] "
            f"{violation_id}: {violation_type}"
        )

    def close(self):
        self.csv_file.close()