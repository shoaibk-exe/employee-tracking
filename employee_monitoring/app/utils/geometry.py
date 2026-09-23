"""Geometry used by line crossing and desk association."""

from __future__ import annotations


Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def bottom_center(bbox: BBox) -> Point:
    """Feet / body base. A bbox center sits on the torso and falls into the wrong desk."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(bbox: BBox, point: Point) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def line_cross(p1: Point, p2: Point, point: Point) -> float:
    """Signed side of a segment. Sign depends on the direction from p1 to p2."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def line_distance(p1: Point, p2: Point, point: Point) -> float:
    length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    if length <= 1e-6:
        dx = point[0] - p1[0]
        dy = point[1] - p1[1]
        return (dx * dx + dy * dy) ** 0.5
    return abs(line_cross(p1, p2, point)) / length


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray cast. Boundary points count as inside so a person standing on an edge is not dropped."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            slope = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x <= slope:
                inside = not inside
        if _on_segment((xi, yi), (xj, yj), point):
            return True
        j = i
    return inside


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1.5) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > eps * max(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5, 1.0):
        return False
    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    if dot < -eps:
        return False
    length_sq = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return dot <= length_sq + eps
