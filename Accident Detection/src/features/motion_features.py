from math import hypot

def center_xy(bbox):
    """bbox format: [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def pixel_displacement(previous_bbox, current_bbox):
    """Distance between bbox centres in pixels between two processed frames."""
    px, py = center_xy(previous_bbox)
    cx, cy = center_xy(current_bbox)
    return hypot(cx - px, cy - py)

def motion_summary(displacements, history=5):
    """Summarize motion change; this is a feature, NOT proof of an accident."""
    if len(displacements) < history + 1:
        return None
    previous = displacements[-history-1:-1]
    previous_mean = sum(previous) / len(previous)
    latest = displacements[-1]
    return {
        "previous_mean_px": previous_mean,
        "latest_px": latest,
        "change_px": latest - previous_mean,
    }
