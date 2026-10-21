def detect(frame, detections, context):
    events = []

    for det in detections:
        if det.class_name == "cell_phone":
            events.append({
                "violation_type": "mobile_phone_usage",
                "track_id": det.track_id,
                "confidence": det.confidence,
                "box": det.box,
                "status": "candidate",
                "reason": (
                    "cell_phone detected; "
                    "driver association required"
                )
            })

    return events