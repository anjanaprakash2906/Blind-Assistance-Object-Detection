"""
Spatial reasoning utilities: turns a raw bounding box into human-friendly
direction ("left", "center", "right") and proximity ("very close", "nearby",
"far") descriptions, since a blind user needs "where" and "how urgent" far
more than pixel coordinates.
"""

from dataclasses import dataclass


@dataclass
class SpatialInfo:
    direction: str      # "left", "center", "right"
    proximity: str      # "very close", "nearby", "far"
    area_ratio: float    # bbox area / frame area, used for sorting by closeness


def estimate_direction(box_center_x: float, frame_width: int) -> str:
    """Splits the frame into three equal horizontal zones."""
    left_bound = frame_width / 3
    right_bound = 2 * frame_width / 3

    if box_center_x < left_bound:
        return "left"
    elif box_center_x > right_bound:
        return "right"
    return "center"


def estimate_proximity(area_ratio: float, near_ratio: float, medium_ratio: float) -> str:
    """Classifies proximity from how much of the frame the bounding box fills."""
    if area_ratio >= near_ratio:
        return "very close"
    elif area_ratio >= medium_ratio:
        return "nearby"
    return "far"


def get_spatial_info(
    x1: float, y1: float, x2: float, y2: float,
    frame_width: int, frame_height: int,
    near_ratio: float, medium_ratio: float
) -> SpatialInfo:
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    box_area = box_width * box_height
    frame_area = float(frame_width * frame_height)
    area_ratio = box_area / frame_area if frame_area > 0 else 0.0

    center_x = x1 + (box_width / 2)

    direction = estimate_direction(center_x, frame_width)
    proximity = estimate_proximity(area_ratio, near_ratio, medium_ratio)

    return SpatialInfo(direction=direction, proximity=proximity, area_ratio=area_ratio)
