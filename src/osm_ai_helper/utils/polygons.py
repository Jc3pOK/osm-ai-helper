from typing import List, Tuple

import numpy as np

from PIL import Image
from shapely import Polygon
from skimage.draw import polygon as draw_polygon
from skimage.measure import find_contours

from osm_ai_helper.utils.coordinates import (
    lat_lon_to_pixel_col_row,
    meters_col_row_to_lat_lon,
    pixel_col_row_to_meters_col_row,
)


def crop_polygon(polygon, image, margin):
    rows, cols = polygon.exterior.coords.xy
    top_local = int(min(rows) - margin)
    left_local = int(min(cols) - margin)
    bottom_local = int(max(rows) + margin)
    right_local = int(max(cols) + margin)

    crop = image.crop((left_local, top_local, right_local, bottom_local))
    return crop


def pixel_polygon_to_lat_lon_polygon(
    pixel_polygon: Polygon, top_pixel, left_pixel, zoom
):
    lon_lat_polygon = []
    rows, cols = pixel_polygon.exterior.coords.xy

    pixel_rows = top_pixel + np.array(rows)
    pixel_cols = left_pixel + np.array(cols)
    for pixel_row, pixel_col in zip(pixel_rows, pixel_cols):
        meters_col, meters_row = pixel_col_row_to_meters_col_row(
            pixel_col, pixel_row, zoom
        )
        lat, lon = meters_col_row_to_lat_lon(meters_col, meters_row)
        lon_lat_polygon.append([lon, lat])
    return lon_lat_polygon


def lat_lon_bboxes_to_pixel_polygons(bboxes: List[dict], zoom) -> List[Polygon]:
    polygons = []
    for bbox in bboxes:
        polygons.append(
            Polygon(
                [
                    lat_lon_to_pixel_col_row(
                        float(bbox["north"]), float(bbox["west"]), zoom
                    ),
                    lat_lon_to_pixel_col_row(
                        float(bbox["north"]), float(bbox["east"]), zoom
                    ),
                    lat_lon_to_pixel_col_row(
                        float(bbox["south"]), float(bbox["east"]), zoom
                    ),
                    lat_lon_to_pixel_col_row(
                        float(bbox["south"]), float(bbox["west"]), zoom
                    ),
                ]
            )
        )
    return polygons


def mask_to_polygons(mask: np.ndarray, simplify: float = 0.5):
    return [
        Polygon(contour).simplify(simplify, preserve_topology=False)
        for contour in find_contours(mask)
    ]


def polygon_evaluation(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    min_area: int = 50,
    min_iou: float = 0.2,
    simplify: float = 2.0,
) -> Tuple[List[Polygon], List[Polygon], List[Polygon]]:
    true_polygons = mask_to_polygons(mask_true)
    pred_polygons = mask_to_polygons(mask_pred, simplify=simplify)

    pred_polygons = [p for p in pred_polygons if p.area > min_area]

    found = []
    false_alarms = []
    for pred_polygon in pred_polygons:
        matched = None
        for n, true_polygon in enumerate(true_polygons):
            intersection = true_polygon.intersection(pred_polygon).area
            union = true_polygon.union(pred_polygon).area
            if (intersection / union) > min_iou:
                found.append(pred_polygon)
                matched = n
                break

        if matched is not None:
            true_polygons.pop(matched)
        else:
            false_alarms.append(pred_polygon)

    # Any polygons left in true_polygons are false negatives a.k.a missed
    return found, false_alarms, true_polygons


def paint_polygon(polygon, image, color):
    xx, yy = polygon.exterior.coords.xy
    rr, cc = draw_polygon(xx, yy)
    image[rr, cc] = color
    return image


def paint_polygon_evaluation(
    image: Image,
    found: List[Polygon],
    false_alamars: List[Polygon],
    missed: List[Polygon],
):
    painted_output = np.zeros((image.size[1], image.size[0], 3))
    for p in found:
        paint_polygon(p, painted_output, (0, 255, 0))
    for p in false_alamars:
        paint_polygon(p, painted_output, (255, 255, 0))
    for p in missed:
        paint_polygon(p, painted_output, (255, 0, 0))

    painted_output = Image.fromarray(painted_output.astype(np.uint8))

    painted_image = Image.blend(
        image.convert("RGB"), painted_output.convert("RGB"), 0.5
    )

    return painted_image
