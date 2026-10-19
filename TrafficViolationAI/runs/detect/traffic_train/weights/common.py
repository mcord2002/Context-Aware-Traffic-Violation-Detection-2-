from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]
    track_id: Optional[int]

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    @property
    def bottom_center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, y2)


def point_in_polygon(point, polygon):
    import cv2
    import numpy as np

    contour = np.asarray(
        polygon,
        dtype="int32"
    )

    return cv2.pointPolygonTest(
        contour,
        (float(point[0]), float(point[1])),
        False
    ) >= 0


def crossed_line(previous, current, start, end):
    if previous is None:
        return False

    def side(p, a, b):
        return (
            (b[0] - a[0]) * (p[1] - a[1])
            - (b[1] - a[1]) * (p[0] - a[0])
        )

    before = side(previous, start, end)
    after = side(current, start, end)

    if before * after >= 0:
        return False

    px, py = previous
    qx, qy = current
    ax, ay = start
    bx, by = end

    rx, ry = qx - px, qy - py
    sx, sy = bx - ax, by - ay

    denominator = rx * sy - ry * sx

    if abs(denominator) < 1e-9:
        return False

    t = (
        (ax - px) * sy - (ay - py) * sx
    ) / denominator

    u = (
        (ax - px) * ry - (ay - py) * rx
    ) / denominator

    return 0 <= t <= 1 and 0 <= u <= 1