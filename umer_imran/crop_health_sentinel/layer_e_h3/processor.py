from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from shapely import area, intersection, polygons
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree

from crop_health_sentinel.errors import H3CapacityError
from crop_health_sentinel.io.raster_artifacts import write_single_band_raster
from crop_health_sentinel.io.structured_artifacts import write_csv, write_json
from crop_health_sentinel.models.types import H3CellResult, LayerEOutput
from crop_health_sentinel.utils.preview import write_preview

from .h3_adapter import average_cell_area_m2, cell_boundary_wgs84, latlng_to_cell, polygon_to_overlapping_cells


@dataclass(frozen=True, slots=True)
class H3CapacityEstimate:
    analytic_pixels: int
    estimated_candidate_cells: int


def preflight_h3_capacity(context, quality) -> H3CapacityEstimate:
    analytic = int(np.count_nonzero(quality.analytic_valid_mask)); cfg = context.config.h3
    metric = transform_geometry(Transformer.from_crs("EPSG:4326", cfg.metric_crs, always_xy=True).transform, context.crop_geometry_wgs84)
    estimate = math.ceil(metric.area / average_cell_area_m2(cfg.resolution) * cfg.candidate_estimate_safety_factor)
    if analytic > cfg.max_analytic_pixels or estimate > cfg.max_candidate_cells:
        raise H3CapacityError(f"H3 workload analytic={analytic}, estimated_candidates={estimate} exceeds configured capacity.")
    return H3CapacityEstimate(analytic, estimate)


def process_h3(context, quality, health, logger=None) -> LayerEOutput:
    preflight = preflight_h3_capacity(context, quality); cfg = context.config.h3
    indexes = polygon_to_overlapping_cells(context.crop_geometry_wgs84, cfg.resolution)
    if len(indexes) > cfg.max_candidate_cells: raise H3CapacityError("Actual H3 candidate count exceeds configured capacity.")
    metric_transformer = Transformer.from_crs(context.grid.crs, cfg.metric_crs, always_xy=True)
    crop_metric = transform_geometry(Transformer.from_crs("EPSG:4326", cfg.metric_crs, always_xy=True).transform, context.crop_geometry_wgs84)
    cells = [intersection(transform_geometry(Transformer.from_crs("EPSG:4326", cfg.metric_crs, always_xy=True).transform, cell_boundary_wgs84(i)), crop_metric) for i in indexes]
    tree = STRtree(cells); sums=np.zeros(len(cells)); squares=np.zeros(len(cells)); weights=np.zeros(len(cells)); mins=np.full(len(cells),np.inf); maxs=np.full(len(cells),-np.inf)
    rows, cols = np.nonzero(quality.analytic_valid_mask & np.isfinite(health.pixel_health_map)); values=health.pixel_health_map[rows,cols]
    for start in range(0, len(rows), cfg.batch_size):
        r,c,v=rows[start:start+cfg.batch_size],cols[start:start+cfg.batch_size],values[start:start+cfg.batch_size]
        x=np.stack([c,c+1,c+1,c,c],axis=1); y=np.stack([r,r,r+1,r+1,r],axis=1)
        xx,yy=metric_transformer.transform((context.grid.transform.a*x+context.grid.transform.b*y+context.grid.transform.c).ravel(),(context.grid.transform.d*x+context.grid.transform.e*y+context.grid.transform.f).ravel())
        pixels=polygons(np.stack([np.asarray(xx).reshape(-1,5),np.asarray(yy).reshape(-1,5)],axis=2)); pairs=tree.query(pixels,predicate="intersects")
        if pairs.size==0: continue
        pi,ci=pairs; overlaps=area(intersection(pixels.take(pi), np.asarray(cells,dtype=object).take(ci))); valid=overlaps>0; pi,ci,overlaps=pi[valid],ci[valid],overlaps[valid]
        total=np.bincount(pi,weights=overlaps,minlength=len(pixels)); share=overlaps/total[pi]; np.add.at(weights,ci,share); np.add.at(sums,ci,share*v[pi]); np.add.at(squares,ci,share*v[pi]*v[pi]); np.minimum.at(mins,ci,v[pi]); np.maximum.at(maxs,ci,v[pi])
    actual, analytic = _center_counts(context, quality, indexes)
    result=[]
    for i,index in enumerate(indexes):
        if weights[i] == 0: continue
        mean=sums[i]/weights[i]; std=math.sqrt(max(squares[i]/weights[i]-mean*mean,0)); result.append(H3CellResult(index,float(mean),float(std),float(mins[i]),float(maxs[i]),actual.get(index,0),analytic.get(index,0),float(weights[i]),bool(weights[i]<cfg.minimum_effective_support)))
    result=tuple(sorted(result,key=lambda cell:cell.h3_index)); summary={"analyticPixelCount":preflight.analytic_pixels,"estimatedCandidateCells":preflight.estimated_candidate_cells,"actualCandidateCells":len(indexes),"emittedCells":len(result)}
    output=LayerEOutput(result,summary); _write_artifacts(context, output); return output


def _center_counts(context, quality, indexes):
    to_wgs=Transformer.from_crs(context.grid.crs,"EPSG:4326",always_xy=True); counts=[{},{}]
    for n,mask in enumerate((context.field_mask,quality.analytic_valid_mask)):
        rows,cols=np.nonzero(mask); x=context.grid.transform.a*(cols+.5)+context.grid.transform.b*(rows+.5)+context.grid.transform.c; y=context.grid.transform.d*(cols+.5)+context.grid.transform.e*(rows+.5)+context.grid.transform.f; lon,lat=to_wgs.transform(x,y)
        for cell in (latlng_to_cell(a,b,context.config.h3.resolution) for a,b in zip(lat,lon)):
            if cell in indexes: counts[n][cell]=counts[n].get(cell,0)+1
    return counts[0],counts[1]


def _write_artifacts(context, output):
    phase=context.output_dir/"phase_05"; prefix=context.metadata.scene_id; cells=output.cells
    rows=[{"h3_index":c.h3_index,"mean_health":c.mean_health,"std_health":c.std_health,"min_health":c.min_health,"max_health":c.max_health,"actual_pixel_count":c.actual_pixel_count,"analytic_pixel_count":c.analytic_pixel_count,"effective_analytic_pixel_count":c.effective_analytic_pixel_count,"low_support":c.low_support,"score_source":"analytic"} for c in cells]
    columns=list(rows[0]) if rows else ["h3_index","mean_health","std_health","min_health","max_health","actual_pixel_count","analytic_pixel_count","effective_analytic_pixel_count","low_support","score_source"]
    write_csv(phase/f"{prefix}_h3_cells.csv",columns,rows); write_json(phase/f"{prefix}_all_polygon_h3_indexes.json",[c.h3_index for c in cells]); write_json(phase/f"{prefix}_layer_e_h3_summary.json",output.summary)
    features=[]
    for c in cells:
        geom=cell_boundary_wgs84(c.h3_index).__geo_interface__; features.append({"type":"Feature","geometry":geom,"properties":{"h3Index":c.h3_index,"healthScore":c.mean_health,"lowSupport":c.low_support}})
    write_json(phase/f"{prefix}_h3_cells.geojson",{"type":"FeatureCollection","features":features}); write_json(phase/f"{prefix}_h3_cells_overlay.json",{"schemaVersion":"1.0","cells":[{"h3Index":c.h3_index,"healthScore":c.mean_health,"lowSupport":c.low_support} for c in cells]})
    score={c.h3_index:c.mean_health for c in cells}; raster=np.full((context.grid.height,context.grid.width),np.nan,np.float32); to_wgs=Transformer.from_crs(context.grid.crs,"EPSG:4326",always_xy=True); r,co=np.nonzero(context.field_mask); lon,lat=to_wgs.transform(context.grid.transform.a*(co+.5)+context.grid.transform.b*(r+.5)+context.grid.transform.c,context.grid.transform.d*(co+.5)+context.grid.transform.e*(r+.5)+context.grid.transform.f)
    for row,col,cell in zip(r,co,(latlng_to_cell(a,b,context.config.h3.resolution) for a,b in zip(lat,lon))): raster[row,col]=score.get(cell,np.nan)
    write_single_band_raster(phase/f"{prefix}_h3_health_map.tif",raster,context.grid,dtype="float32",nodata=-9999.0)
    if context.config.artifacts.write_previews: write_preview(phase/f"{prefix}_h3_health_map.png",raster,"H3 health",context.config.artifacts.preview_dpi)
