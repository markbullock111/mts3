from __future__ import annotations

from typing import Iterable, Sequence, Tuple

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Sequence[float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1)
        if cond:
            inside = not inside
    return inside


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def line_side(point: Point, p1: Point, p2: Point) -> float:
    return _cross(p2[0] - p1[0], p2[1] - p1[1], point[0] - p1[0], point[1] - p1[1])


def segment_intersection(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p: Point, q: Point, r: Point) -> bool:
        return min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) < 1e-9 and on_seg(a1, b1, a2):
        return True
    if abs(o2) < 1e-9 and on_seg(a1, b2, a2):
        return True
    if abs(o3) < 1e-9 and on_seg(b1, a1, b2):
        return True
    if abs(o4) < 1e-9 and on_seg(b1, a2, b2):
        return True
    return False


def crossed_entry_line(prev_pt: Point | None, curr_pt: Point, line_p1: Point, line_p2: Point, roi_polygon: Sequence[Sequence[float]]) -> bool:
    if prev_pt is None:
        return False
    if not segment_intersection(prev_pt, curr_pt, line_p1, line_p2):
        return False
    prev_in = point_in_polygon(prev_pt, roi_polygon)
    curr_in = point_in_polygon(curr_pt, roi_polygon)
    return (not prev_in) and curr_in
