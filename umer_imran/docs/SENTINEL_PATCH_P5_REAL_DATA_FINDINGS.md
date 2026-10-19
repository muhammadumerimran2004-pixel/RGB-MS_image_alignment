# Sentinel patch co-registration: first real-data findings (P5, part 1)

Date: 2026-10-19. Status: **partial**. This is a first look on the aligned outputs that were on hand, not the full validation campaign of blueprint 9.3.

## What was tested

Five distinct fields, taken from the `odm_orthophoto_aligned.tif` files in this repository (`DOCUMENTATION RESULTS/Test1-3`, `Results/Run_2_...`, `WP7_...`). Each aligned file has 5 bands: Red, Green, NIR, RedEdge, and an alpha band. Band numbers `red=1, green=2, nir=3` are right (confirmed in QGIS by the user, and by the band descriptions in the files).

Live Earth Search and Planetary Computer data, every scene within +-15 days of a nominal date, each registered with the relaxed overlap gate (`min_overlap_px: 50`).

**Limits of this test (read before quoting any number):**
- The original RGB and MS inputs are not on this machine, so the aligned MS stood in for the RGB when taking the field outline.
- **The flight dates are not known.** Test2, Test3 and Run2 use the date in the delivery folder name (17-07-2026, 17-07-2026, 25-07-2026); Test1 and WP7 have no date at all and used 17 July and 20 June as stand-ins. Wrong dates mean a different crop state, so WP7 and Test1 results are the least trustworthy.
- There are no ground-truth control points, so "correct" cannot be measured. The only evidence available is agreement between independent satellite acquisitions of the same field.

## Findings

1. **Data-format issue found and made harmless: a fill value that is not declared as nodata.** The multispectral orthophotos fill the area outside the field with 4294967296 (2^32) in all four measurement bands; the alpha band marks those pixels as empty, but the GeoTIFF declares no nodata value. In the original file this is exactly the empty area (31.9 M pixels, zero exceptions in either direction). In aligned outputs, interpolation blends that fill into a thin ring along the data boundary (0.03-0.2 % of cells, in all 7 files). `build_reference_band` now ignores cells more than 50 robust sigmas from the median (`reference.outlier_mad_factor`), with tests. Full evidence, a script anyone can run, and the fixes (declare nodata at the source; defensive change in the warper) are in `docs/ORTHOPHOTO_FILL_VALUE_EVIDENCE.md`. The cause inside ODM is not established.
2. **Real fields are small.** 1.3, 1.4, 1.7, 3.6 and 7.7 ha = 131, 140, 173, 364 and 771 Sentinel pixels. The blueprint assumed fields of at least ~4 ha. The default `min_overlap_px: 400` therefore refuses every field up to ~4 ha (`AOI_TOO_SMALL`), which is the designed, fail-closed behaviour.
3. **Monsoon cloud removes most scenes.** In the +-15-day windows, 5 of 7 scenes (Test2/Test3) and about 1 in 3 (Run2, Test1) were over 30 % masked, many at 100 %. Several fields had a single usable date.
4. **A single scene is usually ambiguous for fields this small.** With the gate relaxed to 50 px, no single-scene registration passed on any field (about 40 scene/field comparisons). Reasons: weak or ambiguous correlation peak (peak ratio 1.04-1.13), peak on the search border, implausible shift. The correlation surface of a 22 x 16 pixel patch is typically a long ridge in one direction (aperture problem), not a point. **No wrong shift was accepted anywhere**: the gates held.
5. **But independent scenes agree with each other, even when single scenes are ambiguous.** Wide brute-force search (+-120 m), best shift per scene:
   - Run2 (1.7 ha, 12 clear scenes): 8 of 12 within one pixel of (+10 m E, 0 to -10 m N); the others scatter (up to +40 m N, +50 m E). Correlation 0.41-0.74.
   - Test1 (3.6 ha, 9 scenes, date uncertain): the last three scenes agree at about (0 to +10 m, 0 to -10 m); the earlier six agree on the row (about +20 to +30 m N) but are spread along x (+40 to +90 m E), i.e. a one-direction ambiguity.
   - WP7 (7.7 ha, date unknown): random scatter, correlation 0.17-0.33, no information.
   - Test2/Test3: one usable scene each, nothing to compare.
   The consistent cluster of about 10 m (one pixel) is the size of Sentinel-2's own absolute geolocation error plus normal drone GNSS error, so these fields are probably **not badly misplaced**, and the achievable gain is about one pixel.

## What this means for the feature

- The pipeline is safe: it refuses instead of guessing, on real data as on synthetic data.
- **It will usually refuse for fields of 1-4 ha**, because one scene of one small field does not contain enough structure to pin down a 10 m shift. Users get the exact Sentinel patch, unshifted, with a reason code.
- Sentinel pixel-grid sub-pixel claims (0.03 px accuracy) from the synthetic study do **not** carry over: real error is dominated by crop change, cloud shadows and scene content, not by the estimator.

## Update 2026-10-19 (later): multi-scene consensus built and run on the real fields

Option B below was approved and implemented (`registration/consensus.py`, `registration.consensus` config). Default config, real data, nominal dates as above:

| Field | Result | Why |
|---|---|---|
| Run2 (1.7 ha) | **verified by consensus**, shift (-10.5 m E, +4.9 m N) | 6 of 6 clear scenes agree within a 6 m spread (individual shifts from (-7.6, +4.8) to (-16.6, +5.0), correlation 0.51-0.74) |
| Test1 (3.6 ha) | refused | only 2 of 4 scenes agree (3 needed); its date is a guess |
| Test2, Test3 (1.3, 1.4 ha) | refused | 0 and 1 usable scene (monsoon cloud) |
| WP7 (7.7 ha), A003 (7.3 ha, same farm, June) | refused | 0 scenes voted: every clear scene's best match is weak (r 0.2-0.4) and scattered over +-120 m, so there is nothing for scenes to agree on. Flight dates unknown for both |

So consensus rescues the case it was built for (a small field with several clear scenes) and refuses the rest, with a reason in the report. It does not make cloud-covered windows usable.

**What consensus cannot do.** Agreement between acquisitions protects against noise, cloud and crop change. It does not protect against a static look-alike that appears in every scene. Each vote therefore still has to show a real correlation peak (>= 0.5), away from the search border and within the plausible shift, and consensus is never tried after a texture, distortion (NON_TRANSLATIONAL) or independent-band refusal. Tiles of one acquisition count once (de-duplicated on the datatake id).

**Date question for FAI-AAA0106-0626-A003** (orthophotos only, no raw photos): nothing inside the files carries a date (GeoTIFF tags, XMP and the `.aux.xml` hold none). The folder name `...-0626-...` reads as June 2026 (MMYY; the same code in the other deliveries matches their months), month only. The file modification time (25 Aug 2026) is when ODM produced the orthophoto, an upper bound and not the flight date. The original MS orthophoto already holds the 2^32 fill value (as its declared-nothing empty area), so the fill comes with the input; see `docs/ORTHOPHOTO_FILL_VALUE_EVIDENCE.md` for how the alignment's resampling turns it into a boundary ring and how to fix it.

## Options (as first proposed; B has since been built, see above)

| Option | What it does | Cost / risk |
|---|---|---|
| A. Keep as is | Fail closed; document that small fields get an unshifted patch | None; feature mostly idle for fields under ~4 ha |
| B. **Multi-scene consensus** (recommended) | Register against every clear scene in the window; accept a shift only when at least N scenes agree within a pixel; publish the consensus shift | More downloads and time; relies on Sentinel's multi-temporal geolocation being stable (ESA: < 5 m), which is documented. Run2 above is the kind of case it would turn from "refused" into "10 m shift, 8 scenes agree" |
| C. Lower `min_overlap_px` / peak ratio | Accept weaker single-scene evidence | Accepts the ridge ambiguity seen in Test1: wrong shifts become possible. Not recommended without B |
| D. Pool fields flown together | Add the correlation surfaces of fields from the same flight/session | Needs reliable session metadata; assumes one shared georeference error |

## Still to do (P5)

- Real flight dates for these fields, and the original RGB inputs.
- Ground truth for at least a few fields (RTK/GCP points, or visible fixed features such as road crossings measured in QGIS) to turn "agreement" into accuracy.
- Re-check the 10 % / 30 % cloud cut-offs and the 3/7/15-day tiers against real runs (the cloud numbers above suggest the 15-day window is often the limit in monsoon).
- Test a field larger than 4 ha with a known date.
