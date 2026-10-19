def extract_tracks(result):
    """Convert an Ultralytics tracking result into simple Python dictionaries."""
    tracks = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return tracks

    coords = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coords)

    for bbox, class_id, confidence, track_id in zip(coords, class_ids, confidences, track_ids):
        class_id = int(class_id)
        tracks.append({
            "bbox": [float(v) for v in bbox],
            "class_id": class_id,
            "class_name": result.names.get(class_id, str(class_id)),
            "confidence": float(confidence),
            "track_id": int(track_id) if track_id is not None else None,
        })
    return tracks
