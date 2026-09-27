# Manual GCP Alignment Guide

Manual mode aligns the MS orthomosaic to the RGB reference from control
points you supply, instead of automated feature matching. This guide covers
how to provide those points, how they're interpreted, and how the model that
warps between them gets chosen. For the underlying design and QA-gate
rationale, see `ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md` §7.

## Providing control points

Two ways:

1. **Interactively** — run `--mode manual` (or `-Mode manual` in the
   PowerShell runner) with no `--gcp-file`; the CLI prompts for each point.
2. **From a file** — pass `--gcp-file PATH` to skip the prompts entirely.
   Two formats are supported, selected with `--gcp-format`:

### CSV format (`--gcp-format csv`, the default)

```csv
id,rgb_x,rgb_y,ms_x,ms_y,role
P1,1024.5,2048.0,512.0,1024.0,control
P2,3000.0,500.0,1500.0,250.0,control
...
CHK1,2000.0,2000.0,1000.0,1000.0,check
```

- `id` — any label; it appears in rejection messages and the JSON report.
- `rgb_x,rgb_y` and `ms_x,ms_y` — coordinates in the units set by
  `--gcp-source-units` (`pixel` or `map`), applied to **both** point sets.
- `role` — `control` (default if omitted) or `check`. See "Check points"
  below.

### QGIS Georeferencer format (`--gcp-format qgis`)

Export a `.points` file directly from QGIS's Georeferencer plugin: load the
MS raster as the image to georeference, the RGB raster's CRS as the target,
and place points by eye between the two. QGIS writes
`mapX,mapY,sourceX,sourceY,enable[,dX,dY,residual]`:

- `mapX,mapY` are always the **RGB (reference)** point, in the RGB raster's
  map CRS.
- `sourceX,sourceY` are the **MS (target)** point. Set `--gcp-source-units`
  to match how QGIS is set up: `map` if you georeferenced against a CRS, or
  `pixel` if you placed points on the raw MS image. When `pixel`, note that
  QGIS stores the row as a **negative** Y — the CLI negates it back
  automatically, so paste the file as QGIS wrote it.
- `enable == 0` rows (points you disabled in QGIS) are skipped.
- QGIS `.points` files don't have a check-point column — every point loaded
  this way is a control point. Add check points via a CSV instead if you
  want independent verification (see below), or edit the file's header/rows
  to add a `role` column matching the CSV format above.

## Pixel vs. map coordinates, and the centre convention

`--gcp-source-units pixel` means "count pixels from the top-left of the
raster, in the raster's own native resolution" — not the low-resolution
registration grid. `--gcp-source-units map` means "coordinates in the
raster's CRS", exactly what QGIS's coordinate display or a `gdallocationinfo`
query would give you.

For pixel-unit points, `--pixel-convention` controls what an integer pixel
index means:

- `center` (default): pixel index `N` denotes the **centre** of that pixel,
  i.e. the point at `(N + 0.5, N + 0.5)` in the raster's own transform. This
  matches how a human reads "pixel 100" when pointing at its middle, and how
  most image viewers report cursor position.
- `corner`: pixel index `N` denotes the pixel's top-left corner, with no
  offset.

Getting this wrong shifts every point by half a pixel at native resolution —
usually negligible at MS native GSD, but worth setting deliberately rather
than by default. Map-unit points aren't affected by this setting at all;
they're used exactly as given.

## Choosing between RGB and MS pixel grids

Every point pair needs one coordinate in the RGB raster and one in the MS
raster, describing **the same ground location**. Read them off in whichever
tool you're comfortable with (QGIS's pixel/map coordinate display, or by eye
against the two orthophotos side by side) — the CLI does not guess which
raster a point belongs to, so double-check you haven't swapped a pair.

## How many points, and why 8+ for a good Thin Plate Spline fit

The model (translation, similarity, affine, or thin-plate spline) is chosen
automatically from the number of control points and how well a spline
predicts held-out points, unless you force one with `--manual-model`:

| Control points | What happens |
|---|---|
| 1 | Translation only — the only model determinable from one pair. |
| 2 | Similarity (rotation + uniform scale + translation). |
| 3–7 (default `min_tps_points=8`) | Affine (rotation + scale + shear + translation). Not enough points to safely validate a spline's extra flexibility. |
| 8+, well spread, TPS beats affine on leave-one-out by a real margin | Thin plate spline — a smooth, spatially-variable correction on top of the affine baseline. |

TPS is deliberately conservative: with too few points, or points clustered
in one area (the "hull coverage" check), a spline can extrapolate wildly a
short distance from where you actually clicked. If you have 8+ points but
still get affine, the report's `manual_control_points.model_reason` field
says why (below the leave-one-out gain threshold, insufficient hull
coverage, or SciPy unavailable).

**Spread your points across the full extent** — corners, edges, and centre
— rather than clustering them in one region. A tight cluster inflates the
apparent precision locally while leaving everywhere else unconstrained.

## Check points

A **check point** (`role=check` in the CSV) is excluded from fitting and
instead measures the accuracy of the model fit from the *other* points. Use
a handful of check points placed somewhere you didn't already put a control
point — this is the only way to catch a systematic error (e.g. a
consistently mis-read coordinate) that leave-one-out on the same points
can't reveal, because leave-one-out still only ever sees points from the same
set you're trying to validate.

The JSON report's `manual_control_points.checkpoint_rmse_px` field reports
this, and the run is rejected if it exceeds `manual.max_checkpoint_rmse_px`
(default 2.0 registration pixels).

## What ends up in the JSON report

`manual_control_points` in the alignment report includes, per point, its
role, coordinates, fit residual, leave-one-out residual (when computable),
robust outlier flag, plus the selected model, its LOO/check-point RMSE, the
baseline affine's rotation/scale/translation, and (for TPS) the Jacobian
safety-gate result and the lattice-sampling interpolation error. See the
blueprint §7 (P3.7) for the exact schema.

## Example commands

Interactive, from the PowerShell runner:

```powershell
.\scripts\run-alignment.ps1 `
  -RgbPath '.\Stuff\RGB_odm_orthophoto.tif' `
  -MsPath '.\Stuff\odm_orthophoto.tif' `
  -OutputDir '.\Stuff\aligned-manual' `
  -Mode manual
```

From a GCP CSV, pixel units, forcing affine:

```powershell
.\scripts\run-alignment.ps1 `
  -RgbPath '.\Stuff\RGB_odm_orthophoto.tif' `
  -MsPath '.\Stuff\odm_orthophoto.tif' `
  -OutputDir '.\Stuff\aligned-manual' `
  -Mode manual `
  -GcpFile '.\Stuff\gcps.csv' `
  -GcpSourceUnits pixel `
  -ManualModel affine
```

From a QGIS `.points` file with pixel-unit source points:

```powershell
.\scripts\run-alignment.ps1 `
  -RgbPath '.\Stuff\RGB_odm_orthophoto.tif' `
  -MsPath '.\Stuff\odm_orthophoto.tif' `
  -OutputDir '.\Stuff\aligned-manual' `
  -Mode manual `
  -GcpFile '.\Stuff\ms_georef.points' `
  -GcpFormat qgis `
  -GcpSourceUnits pixel
```
