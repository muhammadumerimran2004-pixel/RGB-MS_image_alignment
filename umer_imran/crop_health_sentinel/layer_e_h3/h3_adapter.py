from __future__ import annotations

import h3
from shapely.geometry import Polygon

from crop_health_sentinel.errors import H3AggregationError


def polygon_to_overlapping_cells(geometry, resolution: int) -> tuple[str, ...]:
    try: return tuple(sorted(h3.h3shape_to_cells_experimental(h3.geo_to_h3shape(geometry.__geo_interface__), resolution, contain="overlap")))
    except Exception as exc: raise H3AggregationError(f"Unable to create overlapping H3 cells: {exc}") from exc


def cell_boundary_wgs84(index: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(index)])


def average_cell_area_m2(resolution: int) -> float:
    return float(h3.average_hexagon_area(resolution, unit="m^2"))


def latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
    return h3.latlng_to_cell(lat, lng, resolution)
