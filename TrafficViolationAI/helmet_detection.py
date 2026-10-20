def detect(frame, detections, context):
    events = []

    for det in detections:
        if det.class_name == "no_helmet":
            events.append({
                
            })

    return events