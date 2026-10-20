def detect(frame, detections, context):
    events = []

    for det in detections:
        if det.class_name == "no_helmet":
            events.append({
                "violation_type": "helmet_violation",
                "track_id": det.track_id,
                "confidence": det.confidence,
                "box": det.box,
                "status": "candidate",
                "reason": "no_helmet class detected"
            })

    return events