# WP-7 Local-Correlation Real-Data Validation

**Date:** 2026-09-02  
**Dataset:** `FAI-AAA0106-0626-A001` RGB/MS orthophotos  
**Inputs:**

- RGB: `C:\Users\DELL\Downloads\FAI-AAA0106-0626-A001\RGB\odm_orthophoto\odm_orthophoto.tif`
- MS: `C:\Users\DELL\Downloads\FAI-AAA0106-0626-A001\MS\odm_orthophoto\odm_orthophoto.tif`

## Default safety-gate run

Command:

```powershell
python -m drone_alignment <rgb> <ms> --output-dir WP7_LOCAL_CORRELATION_RESULT --mode local-correlation --verbose
```

Result: **fail closed before local correlation.**

All feature candidates failed independent residual QA. The global Green/Green phase candidate measured image-domain verification correlation `0.285`, below the configured minimum `0.300`. No TIFF, preview, report, or staging artifact was published; the output directory is empty.

This is correct behavior: local correlation is not permitted to proceed without a verified global baseline.

## Exploratory gate-exercise run

The isolated configuration in `WP7_LOCAL_CORRELATION_EXPLORATORY.yaml` lowered only `transform.min_phase_verification_correlation` to `0.28`. It does not alter application defaults or constitute a production threshold decision.

Result: **safe global fallback; local field rejected.**

| Item | Observed value |
|---|---:|
| Global method | Green/Green phase correlation |
| Global phase response | `0.3072` |
| Global image verification correlation | `0.285` |
| Requested mode | `local_correlation` |
| Applied mode | `automated` |
| Local rejection | `FIELD_LOO_FAILED` |
| Regularized mesh LOO RMSE | `2.4592 px` |
| TPS LOO RMSE | `2.3295 px` |
| Mesh max gradient | `0.00678` (safe) |
| TPS max gradient | `0.01621` (safe) |
| Retained source-valid ratio | `1.00001` |
| RGB reference-overlap ratio | `0.88707` |
| End-to-end elapsed time | about 6 minutes |

Artifacts:

- `WP7_LOCAL_CORRELATION_EXPLORATORY_RESULT/odm_orthophoto_aligned.tif` (global fallback)
- `WP7_LOCAL_CORRELATION_EXPLORATORY_RESULT/odm_orthophoto_alignment_preview.png`
- `WP7_LOCAL_CORRELATION_EXPLORATORY_RESULT/odm_orthophoto_alignment_report.json`

## Visual review

The generated preview is usable for a coarse review of the global fallback. It does **not** validate the local field because local-field publication was correctly blocked. A QGIS inspection of center, corners, crop-row boundaries, and nodata perimeter is still required after a dataset passes the local gates.

## Decision

- Keep the production global verification threshold at `0.300`.
- Do not promote local-correlation mode for this dataset: both field candidates exceeded the configured maximum `2.0 px` leave-one-out error.
- Acquire/run at least one additional pair with a verified global baseline and demonstrated non-rigid drift before any threshold tuning decision.
