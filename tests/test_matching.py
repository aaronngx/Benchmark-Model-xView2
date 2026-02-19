"""Tests for IoU matching and metrics."""
from disaster_bench.data.polygons import parse_wkt_polygon
from disaster_bench.eval.matching import polygon_iou
from disaster_bench.eval.metrics import macro_f1, f1_per_class, coverage


def test_polygon_iou():
    shape = (100, 100)
    a = parse_wkt_polygon("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))")
    b = parse_wkt_polygon("POLYGON ((30 30, 70 30, 70 70, 30 70, 30 30))")
    assert a is not None and b is not None
    iou = polygon_iou(a, b, shape)
    assert 0 < iou < 1
    assert polygon_iou(a, a, shape) == 1.0


def test_macro_f1():
    y_true = ["no-damage", "no-damage", "destroyed", "minor-damage"]
    y_pred = ["no-damage", "no-damage", "no-damage", "minor-damage"]
    m = macro_f1(y_true, y_pred)
    assert 0 <= m <= 1
    per = f1_per_class(y_true, y_pred)
    assert "no-damage" in per
    assert "destroyed" in per


def test_coverage():
    assert coverage(5, 10) == 0.5
    assert coverage(0, 10) == 0.0
    assert coverage(10, 10) == 1.0
    assert coverage(0, 0) == 1.0
