"""
Oracle crop generation: use GT polygons to crop pre/post per building (uid).
Also supports cropping from predicted bboxes/masks (for tracks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
    load_label_json,
)
from disaster_bench.data.polygons import (
    parse_and_scale_building,
    scale_factors,
)
from disaster_bench.data.rasterize import mask_to_bbox, rasterize_polygon
from shapely.geometry import Polygon


def make_oracle_crops_for_tile(
    tile_id: str,
    pre_path: str | Path,
    post_path: str | Path,
    label_json_path: str | Path,
    out_dir: str | Path,
    *,
    save_masked: bool = False,
) -> int:
    """
    For one tile: load pre/post images and post label JSON; for each building
    parse WKT (xy), scale to image if needed, rasterize, bbox, crop pre and post;
    save to out_dir/tile_id/uid/pre_bbox.png, post_bbox.png.
    Returns number of buildings (uid) written.
    """
    out_base = Path(out_dir) / tile_id
    label_data = load_label_json(label_json_path)
    buildings = get_buildings_from_label(label_data, use_xy=True)
    json_w, json_h = get_label_canvas_size(label_data)
    if json_w <= 0 or json_h <= 0:
        json_w, json_h = 1024, 1024  # fallback

    try:
        pre_img = load_image(pre_path)
    except Exception:
        pre_img = None
    try:
        post_img = load_image(post_path)
    except Exception:
        post_img = None

    if pre_img is None and post_img is None:
        return 0

    # Use post image shape for scaling if available; else pre; else json size
    if post_img is not None:
        img_h, img_w = post_img.shape[:2]
    elif pre_img is not None:
        img_h, img_w = pre_img.shape[:2]
    else:
        img_w, img_h = json_w, json_h
    sx, sy = scale_factors(img_w, img_h, json_w, json_h)

    count = 0
    for b in buildings:
        uid = b["uid"]
        poly, bbox = parse_and_scale_building(b["wkt"], sx, sy)
        if poly is None or bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # Clamp to image
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue

        uid_dir = out_base / uid
        uid_dir.mkdir(parents=True, exist_ok=True)

        if pre_img is not None:
            crop = pre_img[y1:y2, x1:x2]
            Image.fromarray(crop).save(uid_dir / "pre_bbox.png")
        if post_img is not None:
            crop = post_img[y1:y2, x1:x2]
            Image.fromarray(crop).save(uid_dir / "post_bbox.png")

        if save_masked:
            mask = rasterize_polygon(poly, (img_h, img_w))
            mask_roi = mask[y1:y2, x1:x2]
            if pre_img is not None:
                pre_roi = pre_img[y1:y2, x1:x2].copy()
                rgba = np.dstack([pre_roi, (mask_roi * 255)])
                Image.fromarray(rgba.astype(np.uint8)).save(uid_dir / "pre_masked.png")
            if post_img is not None:
                post_roi = post_img[y1:y2, x1:x2].copy()
                rgba = np.dstack([post_roi, (mask_roi * 255)])
                Image.fromarray(rgba.astype(np.uint8)).save(uid_dir / "post_masked.png")
        count += 1
    return count


def crop_from_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop image by (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()
