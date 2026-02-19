"""Tests for WKT parsing and scaling."""
import pytest
from disaster_bench.data.polygons import (
    parse_wkt_polygon,
    scale_polygon,
    polygon_to_bbox,
    scale_factors,
    parse_and_scale_building,
)


def test_parse_wkt():
    wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    poly = parse_wkt_polygon(wkt)
    assert poly is not None
    assert poly.area == 100.0
    bbox = polygon_to_bbox(poly)
    assert bbox == (0, 0, 10, 10)


def test_scale_polygon():
    wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    poly = parse_wkt_polygon(wkt)
    scaled = scale_polygon(poly, 2.0, 2.0)
    assert scaled.area == 400.0
    assert polygon_to_bbox(scaled) == (0, 0, 20, 20)


def test_scale_factors():
    sx, sy = scale_factors(1024, 1024, 1024, 1024)
    assert sx == sy == 1.0
    sx, sy = scale_factors(512, 512, 1024, 1024)
    assert sx == sy == 0.5


def test_parse_and_scale_building():
    wkt = "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))"
    poly, bbox = parse_and_scale_building(wkt, 2.0, 2.0)
    assert poly is not None
    assert bbox == (0, 0, 10, 10)
