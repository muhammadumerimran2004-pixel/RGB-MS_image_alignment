"""Check a GeoTIFF for non-physical pixel values such as 4294967296 (= 2**32) and say exactly where they are.

Read-only by default. Two different things are counted, because they have different consequences:

  * FILL: non-physical pixels where the file itself says there is NO data (alpha band = 0). Harmless when the file
    declares them as nodata; when it does not, every tool that ignores the alpha band (QGIS statistics, GDAL
    resampling, rasterio) treats them as measurements.
  * CONTAMINATION: non-physical pixels where the file says there IS data (alpha > 0). These corrupt statistics and
    are the ones that end up inside an analysis.

Examples are printed with map coordinates so a reader can open the file in QGIS, use Identify Features there and
see the value with their own eyes.

    python scripts/check_orthophoto_overflow.py path/to/odm_orthophoto.tif
    python scripts/check_orthophoto_overflow.py ortho.tif --json report.json --map map.png
    python scripts/check_orthophoto_overflow.py ortho.tif --write-clean ortho_clean.tif    # a COPY, never in place

Exit code: 0 = nothing to worry about, 1 = contamination, or fill that the file does not declare as nodata.

The same facts without this script:
  * GDAL:   gdalinfo -stats ortho.tif       (look for STATISTICS_MAXIMUM = 4294967296 and "NoData Value" missing)
  * QGIS:   Layer Properties > Information (statistics), then Identify Features at a coordinate printed below
  * Python: rasterio.open(p).read(1, out_shape=(1000, 1000)).max()
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

STRIP_ROWS = 512
EXAMPLES = 8


def is_garbage(block: np.ndarray, threshold: float) -> np.ndarray:
    """Non-finite, or beyond +-threshold (reflectance and ordinary DN data never get near it)."""
    with np.errstate(invalid="ignore"):
        return ~np.isfinite(block) | (np.abs(block) > threshold)


def alpha_index(src) -> int | None:
    return next((i for i, ci in enumerate(src.colorinterp, start=1) if ci.name == "alpha"), None)


def data_mask(src, window: Window, alpha: int | None, out_shape=None) -> np.ndarray:
    """Where the file says there is data: its alpha band when it has one (GDAL does not apply a float file's alpha
    band as a mask on its own), otherwise GDAL's own mask."""
    kwargs = {"window": window} if out_shape is None else {"out_shape": out_shape}
    return (src.read(alpha, **kwargs) > 0) if alpha else (src.read_masks(1, **kwargs) > 0)


def scan(path: Path, threshold: float, measurement_bands: int) -> dict:
    with rasterio.open(path) as src:
        bands = list(range(1, min(src.count, measurement_bands) + 1))
        alpha = alpha_index(src)
        info = {
            "file": str(path), "size_px": [src.width, src.height], "bands": src.count, "dtype": src.dtypes[0],
            "crs": str(src.crs), "pixel_size_m": [abs(src.transform.a), abs(src.transform.e)],
            "declared_nodata": src.nodata, "alpha_band": alpha, "band_descriptions": list(src.descriptions),
            "scanned_bands": bands, "threshold": threshold,
        }
        per_band = {b: {"fill": 0, "contamination": 0, "values": {}, "max": -np.inf, "min": np.inf} for b in bands}
        data_pixels = empty_pixels = empty_not_garbage = 0
        contaminated: list[tuple[int, int]] = []
        fill_examples: list[tuple[int, int]] = []
        same_everywhere = True
        for row in range(0, src.height, STRIP_ROWS):
            window = Window(0, row, src.width, min(STRIP_ROWS, src.height - row))
            has_data = data_mask(src, window, alpha)
            data_pixels += int(has_data.sum())
            empty_pixels += int((~has_data).sum())
            first = None
            for b in bands:
                block = src.read(b, window=window).astype(np.float64)
                bad = is_garbage(block, threshold)
                stats = per_band[b]
                stats["fill"] += int((bad & ~has_data).sum())
                stats["contamination"] += int((bad & has_data).sum())
                good = block[has_data & ~bad]
                if good.size:
                    stats["max"], stats["min"] = max(stats["max"], float(good.max())), min(stats["min"], float(good.min()))
                for value in np.unique(block[bad]):
                    key = repr(float(value))
                    stats["values"][key] = stats["values"].get(key, 0) + int((block[bad] == value).sum())
                if first is None:
                    first = bad
                    empty_not_garbage += int((~has_data & ~bad).sum())
                    contaminated.extend((int(y) + row, int(x)) for y, x in np.argwhere(bad & has_data))
                    if len(fill_examples) < 4000:
                        outside = np.argwhere(bad & ~has_data)
                        fill_examples.extend((int(y) + row, int(x)) for y, x in outside[:: max(1, len(outside) // 50)])
                elif not np.array_equal(bad, first):
                    same_everywhere = False
        info.update(
            data_pixels=data_pixels, empty_pixels=empty_pixels, empty_pixels_that_are_not_garbage=empty_not_garbage,
            contaminated_pixels=len(contaminated), fill_pixels=per_band[bands[0]]["fill"],
            contaminated_fraction_of_data=len(contaminated) / data_pixels if data_pixels else 0.0,
            same_pixels_in_every_scanned_band=same_everywhere, per_band={str(b): per_band[b] for b in bands},
        )

        def examples(points):
            chosen = points[:: max(1, len(points) // EXAMPLES)][:EXAMPLES]
            return [{"col": x, "row": y, "x": src.xy(y, x)[0], "y": src.xy(y, x)[1]} for y, x in chosen]

        info["examples_contamination"] = examples(contaminated)
        info["examples_fill"] = examples(fill_examples)
        info["_coords"] = contaminated
        return info


def edge_distances(path: Path, coords: list[tuple[int, int]]) -> dict:
    """Distance (m) of each contaminated pixel from the boundary of the data, from a 10x coarser mask."""
    if not coords:
        return {}
    with rasterio.open(path) as src:
        k = 10
        mask = data_mask(src, None, alpha_index(src), out_shape=(src.height // k, src.width // k))
        dist_px = ndimage.distance_transform_edt(mask)
        metres = abs(src.transform.a) * k
        distances = np.array([dist_px[min(y // k, mask.shape[0] - 1), min(x // k, mask.shape[1] - 1)] * metres
                              for y, x in coords])
    return {"median_m": float(np.median(distances)), "p90_m": float(np.percentile(distances, 90)),
            "max_m": float(distances.max()), "share_within_1m": float((distances <= 1.0).mean()),
            "share_within_2m": float((distances <= 2.0).mean()), "resolution_note": "from a 10x coarser mask, about +-0.3 m"}


def draw_map(path: Path, coords: list[tuple[int, int]], out: Path, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(path) as src:
        k = max(1, max(src.height, src.width) // 900)
        mask = data_mask(src, None, alpha_index(src), out_shape=(src.height // k, src.width // k))
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        canvas = mask.astype(float)
        for y, x in coords:
            canvas[min(y // k, canvas.shape[0] - 1), min(x // k, canvas.shape[1] - 1)] = 2.0
        axes[0].imshow(ndimage.maximum_filter(canvas, size=5), cmap="viridis", vmin=0, vmax=2)
        axes[0].set_title("green = data, purple = no data, yellow = contaminated pixels (enlarged)")
        if coords:
            y0, x0 = coords[len(coords) // 2]
            window = Window(max(0, x0 - 60), max(0, y0 - 60), 120, 120)
            block = src.read(1, window=window).astype(float)
            bad = is_garbage(block, threshold)
            axes[1].imshow(np.where(bad, np.nan, block), cmap="gray")
            ys, xs = np.nonzero(bad)
            axes[1].scatter(xs, ys, s=60, facecolors="none", edgecolors="red")
            axes[1].set_title(f"native pixels around col {x0}, row {y0}: red rings = non-physical values")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


def write_clean(path: Path, out: Path, threshold: float, measurement_bands: int) -> int:
    """Copy in which every non-physical pixel is set to nodata (0, declared) in all measurement bands."""
    if out.resolve() == path.resolve():
        raise SystemExit("Refusing to overwrite the original; choose a different --write-clean path.")
    fixed = 0
    with rasterio.open(path) as src:
        profile = src.profile
        profile.update(nodata=0.0)
        bands = list(range(1, min(src.count, measurement_bands) + 1))
        with rasterio.open(out, "w", **profile) as dst:
            for row in range(0, src.height, STRIP_ROWS):
                window = Window(0, row, src.width, min(STRIP_ROWS, src.height - row))
                data = src.read(window=window)
                bad = np.zeros(data.shape[1:], bool)
                for b in bands:
                    bad |= is_garbage(data[b - 1].astype(np.float64), threshold)
                for b in bands:
                    data[b - 1][bad] = 0.0
                fixed += int(bad.sum())
                dst.write(data, window=window)
    return fixed


def report_text(report: dict, threshold: float) -> list[str]:
    lines = [f"File: {report['file']}",
             f"  {report['size_px'][0]} x {report['size_px'][1]} px, {report['bands']} bands, {report['dtype']}, "
             f"{report['crs']}, pixel {report['pixel_size_m'][0]:.3f} m, descriptions {report['band_descriptions']}",
             f"  declared nodata value: {report['declared_nodata']}; alpha band: {report['alpha_band']}",
             f"  pixels the file says have data (alpha > 0): {report['data_pixels']:,}; empty (alpha = 0): {report['empty_pixels']:,}"]
    for b, stats in report["per_band"].items():
        lines.append(f"  band {b}: non-physical values {stats['values'] or '-'}; in the empty area {stats['fill']:,}; "
                     f"INSIDE the data {stats['contamination']:,}; ordinary values {stats['min']:.4g} .. {stats['max']:.4g}")
    lines.append(f"  the same pixels are affected in every scanned band: {report['same_pixels_in_every_scanned_band']}")
    if report["fill_pixels"]:
        lines.append(f"  -> FILL: {report['fill_pixels']:,} pixels outside the data hold a non-physical value; "
                     f"{report['empty_pixels_that_are_not_garbage']:,} empty pixels hold anything else.")
        if report["declared_nodata"] is None:
            lines.append("     The file declares NO nodata value, so tools that ignore the alpha band read the fill as measurements.")
    lines.append(f"  -> CONTAMINATION inside the data: {report['contaminated_pixels']:,} pixels "
                 f"({100 * report['contaminated_fraction_of_data']:.4f} % of the data)")
    d = report.get("distance_to_data_edge")
    if d:
        lines.append(f"     distance to the edge of the data: median {d['median_m']:.1f} m, max {d['max_m']:.1f} m; "
                     f"{100 * d['share_within_2m']:.0f} % within 2 m ({d['resolution_note']})")
    for label, key in (("fill (outside the data)", "examples_fill"), ("contamination (inside the data)", "examples_contamination")):
        if report[key]:
            lines.append(f"  examples of {label}; open the file in QGIS, Identify Features at:")
            lines += [f"    col {e['col']}, row {e['row']}  ->  x={e['x']:.2f}, y={e['y']:.2f} ({report['crs']})" for e in report[key]]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tif", type=Path)
    parser.add_argument("--threshold", type=float, default=1.0e6, help="absolute value above which a pixel is non-physical (default 1e6)")
    parser.add_argument("--measurement-bands", type=int, default=4, help="leading bands holding measurements (default 4; the 5th is alpha)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--write-clean", type=Path)
    args = parser.parse_args()

    report = scan(args.tif, args.threshold, args.measurement_bands)
    coords = report.pop("_coords")
    report["distance_to_data_edge"] = edge_distances(args.tif, coords)
    print("\n".join(report_text(report, args.threshold)))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.map:
        draw_map(args.tif, coords, args.map, args.threshold)
        print(f"  picture written: {args.map}")
    if args.write_clean:
        fixed = write_clean(args.tif, args.write_clean, args.threshold, args.measurement_bands)
        print(f"  cleaned copy written: {args.write_clean} ({fixed:,} pixels set to nodata)")
    problem = report["contaminated_pixels"] > 0 or (report["fill_pixels"] > 0 and report["declared_nodata"] is None)
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
