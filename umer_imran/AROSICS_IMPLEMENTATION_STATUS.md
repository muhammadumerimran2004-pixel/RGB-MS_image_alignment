# AROSICS Local Alignment — Delivery Status

Updated: September 18, 2026

This file tracks delivery status only. The design and the full rationale for
every decision below live in **`ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md`** —
read that first; this file just says which phases are done.

## Phase status (per the blueprint's §12 implementation order)

| Phase | Scope | Status |
|---|---|---|
| P0 Hotfixes | AROSICS attribute/CRS/config bugs, manual TPS bugs | Done — 101/101 tests |
| P1 Shared infrastructure | Lattice field sampling, taper fix, written-footprint helper, shared rejection type | Done — 118/118 tests |
| P2 AROSICS local | Config split, staging, COREG_LOCAL, tie-point gates, dual warp engines, holdout/full-grid verification | Done — 157/157 tests |
| P3 Manual rewrite | `control_points.py`, `gcp_io.py`, model selection, LOO/checkpoint/Jacobian QA | Done — 197/197 tests |
| P4 CLI/runner/config migration/docs | This file, CLI flags, PowerShell runner, docs | Done (this pass) |
| P5 Real-data validation | Amir Waraich Farm A dataset, acceptance criteria | **In progress** (updated 2026-09-19): R1, R2, R3, R4, and the grid-identity check are done with definitive conclusions (not open questions). A second farm (Farm B) exposed four bugs, now fixed (journal §5.6); all four real pairs now publish a verified global result. A third dataset (FAI-AAA0106) led to retiring the MS-on-RGB coverage gate (journal §5.7). It also showed that full-band reads exceed an 8 GB machine at ~150+ MP, which is deferred to QA hardware. 226/226 tests. See `ALIGNMENT_V4_DEVELOPMENT_JOURNAL.md` §4/§6. Remaining: R5 and the independent QGIS check — both need a human placing points in QGIS, not another automated run. |

All of P0–P4 pass their full test suite against real installed AROSICS 1.13.2
and real SciPy/rasterio/GDAL behavior (not mocks alone). See the blueprint's
§10 test plan for what each phase's tests actually exercise.

## Current architecture (post-P2/P3, replaces the pre-blueprint design)

The verified global result (ORB/SIFT/LoFTR/phase correlation, or AROSICS' own
COREG as a last-resort feature-free candidate) is estimated **first** and is
always the safe fallback. AROSICS COREG_LOCAL then refines that result and
performs the warp itself (its own DESHIFTER, or a streaming GDAL thin-plate-
spline warp for large rasters), subject to independent holdout and full-grid
verification this pipeline adds on top of AROSICS' own tie-point filtering.
Any rejection at any gate republishes the verified global result with the
reason recorded in the JSON report's `fallback` field — AROSICS local
refinement never runs unchecked and never runs before the fallback it could
need already exists.

This replaced an earlier design (described in previous revisions of this
file) where AROSICS COREG_LOCAL ran first with no independent verification
and no safe global fallback in front of it. That design is gone; do not
resurrect it from git history without also re-adding the P2 gates.

Manual mode is unrelated to AROSICS: it fits a user-supplied control-point
model (translation/similarity/affine/TPS, auto-selected by leave-one-out
RMSE) and warps with the pipeline's own tiled warper, per the blueprint's §7.

## Phase 5, real-data acceptance — done vs. still needed

The blueprint's §9 defines the full Phase 5 plan (dataset, runs R1–R5,
independent QGIS check, acceptance criteria). Status as of 2026-09-19:

**Done, with conclusions recorded in `ALIGNMENT_V4_DEVELOPMENT_JOURNAL.md` §4/§6:**

1. Full run against the RAH-AAA0140 Amir Waraich Farm A dataset with both
   warp engines. **Engine parity does not hold**: DESHIFTER 2.27px vs
   gdal_tps 2.85px p90 on the same tie points — a genuine TPS
   solve/extrapolation difference, not a bug. Treat DESHIFTER as the
   reference engine; gdal_tps as the large-raster fallback only.
2. `--no-arosics-local` global-only baseline (R4), and confirmation that its
   output grid/extent is identical to R1's (`gdalinfo`-equivalent transform
   comparison passes).
3. Root-caused why `COREG_LOCAL` can't clear the tie-point bar on this
   dataset (5 of 197 valid, 12 needed): ruled out search radius, red vs.
   green channel, and window size (128/256/512 all land in the same 2-6
   range). This is a genuine content/spectral decorrelation in the source
   imagery, not a tunable parameter — stop tuning AROSICS local config for
   this dataset.
4. Band-pair selection validated: RGB is 4-band (R/G/B/alpha), MS is 5-band
   (R/G/NIR/RedEdge/alpha) — red-red and green-green are the only valid
   custom pairs; red-edge/NIR are not available from this RGB reference.

**Still needed — requires a human placing points in QGIS, not another automated run:**

1. An independent QGIS check-point measurement (12 points not used anywhere
   in fitting) against the R1 (default automated) and R5 (manual) outputs —
   this will also settle whether the 1.5px p90 verification bar is right.
2. R5: manual GCP run (10-15 QGIS GCPs + 5 check points).
3. Peak RAM and runtime measurement on real ODM orthophoto sizes (wrapper
   script exists, see the journal §8).
4. A visual look at R1's preview/overlay at 1:50 zoom for bulges/tears —
   the blueprint's own acceptance bar (§9.3) allows R1 to close as a
   documented rejection once this confirms the reason code matches what's
   visible.

## Recommended Phase 5 test command

```powershell
.\scripts\run-alignment.ps1 `
  -RgbPath 'C:\Users\DELL\Downloads\RAH-AAA0140Ch Amir Waraich Farm A\25-07-2026\RGB_0726-C981-b2567c\RAH-AAA0140-0526-A008_Seasonal Cotton\odm_orthophoto\odm_orthophoto.tif' `
  -MsPath 'C:\Users\DELL\Downloads\RAH-AAA0140Ch Amir Waraich Farm A\25-07-2026\Multispectral_0826-A006-517aeb\RAH-AAA0140-0526-A008_Seasonal Cotton\odm_orthophoto\odm_orthophoto.tif' `
  -OutputDir '.\Stuff\Aligned_R1' `
  -Mode automated `
  -DetailedLogs
```

AROSICS local refinement is on by default — no `-EnableArosics` flag is
needed for it to run (that flag now only controls the feature-free global
COREG *candidate* and custom `-ArosicsBandPair` selection). Add
`-NoArosicsLocal` for the R4 global-only baseline, and
`-ArosicsWarpEngine gdal_tps` for the R2 engine-parity comparison. Do not add
`-ArosicsBandPair` until the source band order has been confirmed.
