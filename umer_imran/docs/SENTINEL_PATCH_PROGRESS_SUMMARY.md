# Sentinel-2 patch co-registration: progress summary

As of 2026-10-19. Package `sentinel_patch/`, released as v5 (see the repository tag). Detailed design:
`docs/SENTINEL_PATCH_COREGISTRATION_BLUEPRINT.md`. Real-data findings: `docs/SENTINEL_PATCH_P5_REAL_DATA_FINDINGS.md`.

## The goal

After the drone multispectral (MS) orthomosaic is aligned to the RGB reference, automatically cut the exact Sentinel-2
patch for the field's outline from free satellite catalogues and align it to the drone stack, giving two outputs: the
aligned MS-RGB stack (already existed) and the Sentinel patch registered to it.

## Overall progress: about 65-70 % toward deployment readiness

| Area | Weight | Done | Contribution |
|---|---|---|---|
| Building the feature (6 planned phases + a consensus addition) | 55 % | ~95 % | ~52 |
| Validation on real data (accuracy, thresholds, dates) | 30 % | ~30 % | ~9 |
| Release and deployment (Docker, packaging, handover) | 15 % | ~40 % | ~6 |

The weights are my own judgement of effort and risk; the split matters more than the total. **Building is nearly
complete. Validation is the open part, and it needs inputs from outside the code** (real flight dates, ground truth).

## What is built and tested

- **P0 to P4 complete, plus multi-scene consensus:** configuration, footprint extraction, local and online scene
  discovery (Earth Search and Planetary Computer, both free and anonymous; a paid or other provider is one new file),
  cloud masking, 10 m reference synthesis from the drone data, registration with safety gates, AROSICS cross-check,
  QA preview and JSON reports, command-line and PowerShell integration, Docker files.
- **Fail-closed by design:** if a registration cannot be trusted, the patch is published at its original position with
  a reason code. On real data no wrong shift has been accepted.
- **Size and tests:** about 3,600 lines of source in 42 files and 3,150 lines of tests. The full repository suite has
  554 tests (about 230 for this feature): 553 pass, 1 fails (a path-with-spaces test in `drone_alignment`, unrelated
  to this work; it also failed before these changes).
- **Live check:** both catalogues searched and real Sentinel-2 files read, including Planetary Computer's signed access.

## What the first real-data run showed (5 real fields, 1.3 to 7.7 ha)

1. **Fields are small** (131 to 771 Sentinel pixels), smaller than the design assumed, so a single satellite scene is
   usually too ambiguous to pin down a shift. Response: **multi-scene consensus**, which accepts a shift only when at
   least 3 independent acquisitions agree within 10 m. It verified one field that the single-scene check had refused (6
   of 6 scenes agree, shift about 10.5 m west, 4.9 m north).
2. **Monsoon cloud** removes most scenes in a +-15 day window; several fields had a single clear scene.
3. **A data-format issue in the multispectral orthophotos** (non-physical value 4294967296 as fill, no nodata
   declared) was found and made harmless for this feature. Evidence, a script anyone can run, and fixes are in
   `docs/ORTHOPHOTO_FILL_VALUE_EVIDENCE.md`.
4. Two robustness fixes came out of real data (this value; truncated downloads from the satellite server).

## Not done yet / needs help from outside the code

| Item | Why it matters | Needs |
|---|---|---|
| **Real flight dates** | The matching depends on the date; the dates used so far were guesses from folder names. Orthophoto files carry no date; the folder name gives the month only | One raw photo per dataset (EXIF has the exact date), or the full date in the folder name or a `flight.json` |
| **Ground truth** | "Several scenes agree" is not "correct". Accuracy needs a few measured reference points | A few RTK/GCP points, or visible fixed features measured in QGIS |
| **Docker verification** | Image is written but never built or run: the engine does not start on this machine (WSL 2 kernel missing) | One Windows administrator approval, possibly a restart |
| **Threshold freezing** | Cloud cut-offs, date tiers and match thresholds were tuned on synthetic data; real runs suggest they need review | The real dates and ground truth above |
| Smaller items | ODM `images.json` date source (field names unverified), optional hand-off to the crop-health module, band order assumed for other cameras (now `red=1, green=2, nir=3` for ours) | Low priority |

## Honest limits to state

- Everything that has run on real data used guessed dates and had no ground truth, so no accuracy figure should be quoted yet.
- For fields under about 4 ha the feature often returns the patch unshifted (with a reason), by design, rather than a doubtful shift.
- Consensus protects against noise, cloud and crop change, not against a look-alike that repeats in every scene.

## Suggested next steps (in order)

1. Obtain flight dates (or raw photos) for 5-10 real datasets, and a few ground-truth points.
2. Declare nodata on the multispectral orthophotos at the source (one command per file) and agree who fixes the ODM step.
3. Approve the Windows prompt so Docker can be built and tested.
4. Re-run the validation campaign, then freeze thresholds and release.
