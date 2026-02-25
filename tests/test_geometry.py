from __future__ import annotations

from inference.geometry import crossed_entry_line, point_in_polygon


def test_point_in_polygon():
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), poly) is True
    assert point_in_polygon((15, 5), poly) is False


def test_crossed_entry_line_into_roi():
    roi = [[0, 0], [10, 0], [10, 10], [0, 10]]
    line_p1 = (0, 5)
    line_p2 = (10, 5)
    prev_pt = (5, 12)   # outside ROI
    curr_pt = (5, 4)    # inside ROI and crossed line
    assert crossed_entry_line(prev_pt, curr_pt, line_p1, line_p2, roi) is True


def test_no_cross_if_no_segment_intersection():
    roi = [[0, 0], [10, 0], [10, 10], [0, 10]]
    line_p1 = (0, 5)
    line_p2 = (10, 5)
    prev_pt = (2, 12)
    curr_pt = (2, 11)
    assert crossed_entry_line(prev_pt, curr_pt, line_p1, line_p2, roi) is False
