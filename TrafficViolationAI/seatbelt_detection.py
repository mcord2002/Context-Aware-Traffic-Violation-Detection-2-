def detect(frame, detections, context):
    events = []

    for det in detections:
        if det.class_name == "no_seatbelt":
            events.append({
              
            })

    return events