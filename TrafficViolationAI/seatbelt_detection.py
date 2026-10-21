def detect(frame, detections, context):
    events = []

    for det in detections:
        if det.class_name == "no_seatbelt":
            events.append({
                "violation_type": "seatbelt_violation",
                "track_id": det.track_id,
                "confidence": det.confidence,
                "box": det.box,
                "status": "candidate",
                "reason": "no_seatbelt class detected"
            })

    return events