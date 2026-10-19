# Non-physical values (4294967296) in the multispectral orthophotos: evidence, check, and fixes

Written 2026-10-19. Purpose: let anyone reproduce the observation themselves and agree on a fix. It is a description of
how the files behave, not a criticism of anyone's work: the data and the alignment engine each do something
reasonable, and the problem appears where they meet.

## 1. What was observed

Multispectral (MS) orthophotos from ODM contain the value **4294967296** (= 2^32, the float32 form of the largest
unsigned 32-bit integer) in their four measurement bands. Ordinary reflectance values in the same bands run from about
-0.05 to 0.3.

Tested on the original, never-aligned file
`FAI-AAA0106-0626-A003/MS/odm_orthophoto/odm_orthophoto.tif` (8626 x 9410 px, 5 bands: Red, Green, NIR, RedEdge, alpha;
float32; 3 cm):

| Fact | Result |
|---|---|
| Pixels equal to 4294967296 | 31,896,850 (39 % of the image), identical in all 4 measurement bands |
| Pixels with the alpha band = 0 ("no data") | 31,896,850: **the very same pixels**. Every 4294967296 pixel has alpha 0 and every alpha-0 pixel holds 4294967296, with no exceptions in either direction |
| Pixels with data (alpha = 255) holding a non-physical value | **0** |
| Declared `nodata` value in the GeoTIFF | **none** |
| GDAL treats the alpha band as a mask | no (`all_valid`): the 5th band is flagged alpha, but GDAL still reports band 1 as all valid |

So in the original file the value is the **fill for the empty area outside the field**. The alpha band says "no
data" there. What is missing is the GeoTIFF's own nodata declaration.

Consequence: any tool that reads the four measurement bands without consulting the 5th band sees 4.29 billion as a
measurement. the statistics GDAL stored for the Red band (the `STATISTICS_MEAN` / `STATISTICS_STDDEV`
entries, written when the file was opened in QGIS/GDAL) are a mean of about 1.7 billion and a standard deviation of
2.1 billion, against true values near 0.07.

**Second observation, in the aligned outputs.** In the aligned MS files the empty area is mostly `0`/masked (in aligned Test2, 154,254 of its
1,016,079 empty pixels still hold large values), and in addition a thin ring along the whole data boundary holds blended values (for example 1,083,456 or 2,070,114, about 1,100
different values, not one). Aligned Test2: 10,427 pixels = 0.07 % of the data, all within 2.4 m of the data edge, again
the same pixels in all four bands (picture: ring along the whole outline). The same ring was found in all 7 aligned
files checked (0.03-0.2 % of 1 m cells). Blended values are what interpolation produces when it mixes real values with
a 4.29 billion neighbour.

## 2. Why the ring appears (mechanism, tested)

In `drone_alignment/alignment/warper.py` the MS bands are resampled with
`reproject(..., src_nodata=src.nodata, dst_nodata=config.nodata_value)` (bilinear by default). For this file
`src.nodata` is `None`, so GDAL has no way to know the fill is not data, and bilinear interpolation blends it into the
boundary pixels. The engine then masks the empty area using the alpha band, which removes the fill itself but not the
blended boundary pixels.

Controlled test (same GDAL call, a 2000 x 2000 px crop of the real file around the field edge, 0.37 px shift):

| Source nodata passed to `reproject` | Blended huge values created along the edge |
|---|---|
| none (as the file is) | 3,802 |
| 4294967296 declared | 0 |

Limits of this test: it reproduces the resampling call, not the engine's complete pipeline (other alignment modes
resample through `cv2.remap`). It shows the mechanism is sufficient and that declaring nodata removes it. It does not
prove every contaminated pixel in every file arose this way.

Hypothesis, unverified: 4294967296 is what the unsigned-32-bit value of -1 (4294967295) becomes in float32, i.e. a
"-1 = empty" marker written through an unsigned type. A web search found **no published report of this exact value in
ODM output** (related discussions concern nodata handling and radiometric calibration in multispectral runs), so the
cause inside the ODM/WebODM version used is not established. Please do not describe it as a known ODM bug.

## 3. How to verify it yourself (any one is enough)

**A. QGIS, 2 minutes, no code.**
1. Add `FAI-AAA0106-0626-A003/MS/odm_orthophoto/odm_orthophoto.tif`.
2. Layer Properties > Information: the band list shows no "NoData value". (Symbology > Min/Max with "Actual (slower)"
   accuracy shows a maximum of 4294967296 for the Red band.)
3. Use *Identify Features* and click anywhere in the empty (black or transparent) area, or enter a coordinate printed
   by the script below: the Red/Green/NIR/RedEdge values read 4.29497e+09 and band 5 reads 0.

**B. GDAL.** `gdalinfo -stats odm_orthophoto.tif`: `STATISTICS_MAXIMUM=4294967296`, and the `NoData Value=` line is missing.

**C. The script in this repository** (read-only, about 1-2 minutes per large file):

```
python scripts/check_orthophoto_overflow.py "path\to\odm_orthophoto.tif" --json report.json --map picture.png
```

It prints the counts above, whether the non-physical pixels lie in the empty area (fill) or inside the data
(contamination), and example coordinates to click in QGIS. Exit code 0 = clean, 1 = a problem was found. Run it on an
aligned output to see the ring (contamination), and on an original to see the fill.

## 4. Fixes, from the source to the consumer

| # | Where | What | Effort / risk |
|---|---|---|---|
| 1 | **The file (immediate, metadata only)** | Declare the fill as nodata on the original MS orthophoto: `gdal_edit.py -a_nodata 4294967296 odm_orthophoto.tif`. No pixel changes. Check with `gdalinfo` and by re-running the script. Re-align afterwards: the boundary blending disappears (test above) | Seconds per file. The nodata value applies to all bands; the alpha band holds 0/255 so it is unaffected |
| 2 | **The ODM / delivery step** | Make the pipeline write the nodata value itself (or write NaN/0 with a declared nodata) so every file arrives correct. Find which ODM or WebODM option or version writes the 2^32 fill, with the team that runs it | Needs the team that owns the ODM runs; a one-off post-processing step is the fallback |
| 3 | **The alignment engine (defensive)** | In `warper.py`, when the source declares no nodata but has an alpha band, derive the fill value from the pixels where alpha = 0 and pass it as `src_nodata`, or mask by alpha before interpolating. Then a file with the quirk aligns cleanly without anyone editing it | Small code change in code owned by someone else: propose it, do not make it silently. Needs a regression test with a synthetic fill |
| 4 | **Consumers (done)** | The Sentinel stage ignores cells more than 50 robust standard deviations from the median (`reference.outlier_mad_factor`), so it is safe with or without 1-3. `scripts/check_orthophoto_overflow.py --write-clean out.tif` writes a cleaned COPY (bad pixels set to a declared nodata of 0) for files already in circulation | Done and tested. Does not fix the files for other consumers (analysis, NDVI statistics) |

Recommended order: 1 now (and on already aligned outputs use `--write-clean` or re-align), 2 for good, 3 as a safety net.

## 5. What is not known

- Which ODM/WebODM option or version produces the 2^32 fill.
- Whether every dataset has it (only one original MS orthophoto was available: the other 7 files were already aligned).
- Whether the already delivered aligned MS products have been used for NDVI or other statistics with the ring included. The ring is thin (0.03-0.2 % of data), so the effect on field averages is large only if a few values of about 1e6 are averaged in. Plot-level means that include boundary pixels would be visibly wrong, so a quick min/max check of delivered products is worthwhile.
