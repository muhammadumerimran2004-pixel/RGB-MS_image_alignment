# Alignment Engine v4 — Development Journal

> **Written:** 2026-09-19, at the end of Phase 4 and partway through Phase 5.
> **For:** whoever picks this up next (developer or AI session), and anyone reviewing why the engine looks the way it does.
> **Design authority:** `ALIGNMENT_V4_AROSICS_TPS_BLUEPRINT.md`. This journal records *what happened while building it* — what worked, what didn't, what surprised us, and what's still open. Where this journal and the blueprint disagree about the design, the blueprint wins. Where they disagree about what the *real data* did, this journal is newer.

---

## 1. One-paragraph summary

We took the drone RGB→MS alignment engine from a single-transform, feature-matching pipeline to a coarse-to-fine design built around AROSICS: estimate a global shift first, let AROSICS `COREG_LOCAL` find local tie points and warp, and gate everything behind independent verification. Phases 0–4 of the blueprint are done and tested. On two real farms (four RGB/MS pairs) every run now publishes a **verified global** result, 6–9 cm RMSE, after testing on the second farm exposed and fixed four bugs (§5.6). One of those bugs had made every classical candidate look like it was failing, which I first misread as "non-affine distortion"; that reading is withdrawn. AROSICS local refinement runs end-to-end but still misses its own 1.5 px verification bar on every pair (1.76–2.32 px). Whether that bar is right is the main open question, and it waits on the QGIS check-point measurements.

---

## 2. Timeline of approaches

Every approach below was built, tested, and at least partly run. "Nay" doesn't mean it was wasted work — most of the code is still in the repo as a mode or a building block.

| # | Approach | Where it lives | Outcome |
|---|---|---|---|
| 1 | **V2 multi-representation evidence** — Sobel/LoG/Gabor mappers, tiled SIFT/MIM/LoFTR, candidate fitting | `agrilift_alignment/` | Research prototype. Good for exploring what the imagery supports, too heavy and too open-ended for production. Kept as a sandbox. |
| 2 | **V3 global pipeline** — ORB/SIFT on Red↔Red / Green↔Green, RANSAC affine, 80/20 holdout QA, masked phase-correlation fallback, tiled warp, atomic publish | `drone_alignment/` core | ✅ **Still the foundation.** Coarse grid, matrix conjugation, holdout QA, footprint gates, atomic publication all survive unchanged. A single affine can't fix local distortion, which is what everything after this was trying to solve. |
| 3 | **Road-grid aligner** — 1D road-mask profiles, guardrailed strip refinement | `road_grid_aligner.py` | Kept as a separate mode. Clever against crop-row aliasing, but translation-only and depends on visible roads. |
| 4 | **Local mesh / rail-frame aligner** | `local_mesh_aligner.py`, `rail_frame_aligner.py` | Experimental, off by default. Too many hand-tuned parameters (see the `local_mesh` config block) to trust. |
| 5 | **Local-correlation mode** (cell phase correlation → regularised field) | `cell_correlator.py`, `displacement_field.py` | WP-7 on the FAI-AAA0106 dataset (2026-09-02): correctly **failed closed**. The global baseline was 0.285 against a 0.300 bar, and with that bar lowered the local field was rejected on leave-one-out error (2.46 / 2.33 px vs 2.0). The gates did their job; the mode was not promoted. |
| 6 | **AROSICS, first integration (pre-blueprint)** — `COREG_LOCAL` first with `max_shift=50`, global as fallback, result hard-coded `PASS` | replaced | ❌ **Nay.** Blueprint findings A1–A11: the reliability gate read an attribute that doesn't exist (`reliability` vs `shift_reliability`), so it never fired; no independent QA; order inverted relative to AROSICS' own design; a ±50 px search ≈ one crop-row period (aliasing risk); the full MS raster was loaded into RAM. Tests only passed because a bare `MagicMock` invented the missing attribute. |
| 7 | **AROSICS v4, coarse-to-fine (blueprint Phases 0–4)** | `arosics_*.py`, `pipeline.py` | ✅ **Current design.** Details in §3. |
| 8 | **Manual GCP rewrite** (blueprint Phase 3) | `control_points.py`, `manual.py`, `gcp_io.py` | ✅ Done. Replaces a manual TPS path that always passed its own QA, because it measured error at the control points it interpolates through exactly (M1). |

---

## 3. What worked (the yays)

### 3.1 Coarse-to-fine, the way AROSICS is meant to be used
Global shift first, AROSICS `COREG_LOCAL` for the small leftover error, AROSICS' own warp (`DESHIFTER`, or streaming GDAL TPS for large rasters). AROSICS defaults are restored (`min_reliability=60`, `tieP_filter_level=3`, small search radius), and matching bands are staged at native resolution because the AROSICS docs say not to resample inputs.

### 3.2 Two ways to earn publication — never zero
**This is the most important design decision made during Phase 5**, and it came from direct feedback during the work: *implement the automatic engine the way AROSICS is supposed to be implemented.*

The first v4 build only let AROSICS refine a global result that had already passed our own ORB/SIFT-based QA. On real data every global candidate failed that QA, so AROSICS never got to run. That was backwards: correcting distortion a global model can't handle is the whole point of `COREG_LOCAL`.

Now:

- `_estimate_global_candidate` keeps every candidate it computed but rejected. If none passes QA, it returns the best one (lowest measured RMSE first), flagged `VerifiedGlobalContext.is_verified = False`.
- An unverified candidate is **only ever a starting point** for `COREG_LOCAL`. It is never published by itself.
- A result may be published only if it clears **at least one** of two independent checks:
  1. our classical residual QA (global result, `is_verified=True`), or
  2. AROSICS local refinement's own full set of gates: tie-point count, coverage/hull, neighbour consistency, spatially stratified holdout, full-grid post-warp verification.
- If the global result is unverified **and** local refinement is rejected or disabled, the run fails with `TransformUnreliableError`. It never silently publishes.
- If the global result is verified and local refinement is rejected, the verified global result is published and the reason goes in the report's `fallback` block.

Both paths have now been exercised on real data (§4).

### 3.3 Independent verification added on top of AROSICS
AROSICS' docs recommend checking the tie points and re-running on the corrected output, but leave that to the user. We automate it with a holdout set the fit never sees, so the check can't just confirm itself. On real data this gate caught a result at 2.27 px p90 that would otherwise have been published as good.

### 3.4 Manual mode that measures itself honestly
Model auto-selection (translation / similarity / affine / TPS) by point count, hull coverage and leave-one-out RMSE. `role=check` points are excluded from the fit and used to verify the result. Also: a Jacobian fold-over check, and CSV + QGIS `.points` import. See `docs/MANUAL_GCP_GUIDE.md`.

### 3.5 Test discipline
Tests go from 101 (P0) to 118 (P1), 157 (P2), 197 (P3), 214 (end of P4) 225 (after the Farm B fixes) and 226 (coverage gate retired, §5.7), run against the real installed AROSICS 1.13.2, GDAL and SciPy, not only mocks. The lesson from finding A1 was that mocks cheerfully invent attributes. Every Phase 5 bug fix got a regression test that reproduces the real failure.

---

## 4. What the real data taught us

**Dataset:** RAH-AAA0140, Amir Waraich Farm A (seasonal cotton), EPSG:32642, ~3 cm GSD.

### 4.1 Correction: there is no flight gap
Earlier in the session I read `Multispectral_0726-…` and `Multispectral_0826-…` as two flights a month apart. **That was wrong.** Checked against the ODM output zips:

| Folder | Images flown | ODM run | `pc_quality` | `dem_resolution` | Other |
|---|---|---|---|---|---|
| `RGB_0726-C981` | 2026-07-25 ~08:59 | 2026-07-31 | low | 5 | `fast_orthophoto`, no GCPs |
| `Multispectral_0726-C982` | 2026-07-25 ~08:59 | 2026-07-31 | **low** | **5** | `fast_orthophoto`, no GCPs |
| `Multispectral_0826-A006` | same images (identical `images.json` hash) | 2026-08-06 | **high** | **1.0** | `fast_orthophoto`, no GCPs |

`0726` / `0826` is the **processing month**. The two MS folders are the same flight processed twice, with different reconstruction quality settings. None of the three uses ground control; the average GPS error is ~0.69 m, which matches the ~1 m global offsets we keep measuring.

### 4.2 Run log

| Run | MS input | What happened | Verdict |
|---|---|---|---|
| R1 (initial) | 0726 (low-quality processing) | Every global candidate rejected. Held-out RMSE 100–437 px against a 2 / 3.5 px limit. Exit 1, 67 s, ~2.0 GB peak RAM. **Originally read as "warping no single affine can model". That was wrong** — see §5.6: a handful of false feature matches in the held-out set inflated the RMSE; the transforms themselves fit to ~1.5 px. | Triggered the §3.2 architecture change. |
| R1 (after §3.2) | 0726 | Best unverified candidate (Red↔Red SIFT) handed to `COREG_LOCAL`, which then **crashed** inside `DESHIFTER` with `IndexError: list index out of range` (bug §5.1). Exit 1, 118 s, ~2.8 GB peak. | Real bug found and fixed. |
| R1 (after bug fix) | 0726 | `COREG_LOCAL` ran end-to-end (205–207 grid points). Full-grid verification **rejected** at **2.27 px p90 against a 1.5 px limit**. The global result was unverified, so nothing was published. | Correct fail-closed behaviour. Down from hundreds of px to 2.27, but not good enough. |
| R1 (wide search) | 0726 | `--arosics-max-shift-m 2.0` (search radius ~67 px instead of ~10 px). **Identical** 2.27 px rejection. | Search radius isn't what limits the result (see §6.2). |
| Diagnostic, 2026-09-19 | **0826 (high-quality processing)** | Global Red↔Red AROSICS `COREG` **passed** QA (verification correlation 0.349 against 0.300; shift ~0.90 m, 1.18 m). `COREG_LOCAL` rejected: **5 valid tie points of 197** (141 no-match, 48 L1 and 29 L2 outliers; 12 needed). Verified global result published, status PASS, mode `automated_global`. | Correct fallback behaviour. |
| R1_red, 2026-09-19 | 0826, red-red tried first | Still exactly 5 valid tie points. Grid-identity check against R4 passes (identical width/height/transform/CRS). | Channel choice ruled out (§6.2). |
| R4, 2026-09-19 | 0826, `--no-arosics-local` | `automated_global`, PASS. Identical output grid to R1_red. | Baseline confirmed, grid-identity check passes. |
| R2 (DESHIFTER), 2026-09-19 | 0726 | Rejected, VERIFICATION_FAILED, **2.27 px p90**. | |
| R2 (gdal_tps), 2026-09-19 | 0726 | Rejected, VERIFICATION_FAILED, **2.85 px p90**. | Engine parity does not hold (§6.4). |
| R3 (crop-row guard 0.75 m), 2026-09-19 | 0826 | Tie-point breakdown byte-for-byte identical to the unguarded run (197/141/5/48/29). | Guard wired correctly, doesn't bind here (§6). |
| Window-size sweep, 2026-09-19 | 0826, red/green, 128²/512² | 128²: 4 red / 6 green. 512²: 3 red / 2 green. | No trend, ceiling confirmed structural (§6.3). |

### 4.3 Interpretation (partly inference — marked as such)
- ~~**Established:** the high-quality reprocessing (0826) agrees with the RGB well enough for one global shift to pass QA; the low-quality one (0726) doesn't.~~ **Withdrawn (§5.6).** Both failed QA for the same reason — false matches contaminating the held-out RMSE. With that fixed, *both* pass, and 0726 actually fits slightly better (6.2 cm vs 8.8 cm).
- ~~**Likely, not proven:** the coarser DEM in the 0726 processing distorts the orthorectification unevenly (worst at the edges).~~ **Withdrawn (§5.6).** The "worst at the edges" pattern was an artefact of where the false matches happened to land.
- **Established:** after a good global shift, `COREG_LOCAL` finds very few trustworthy tie points on 0826 (141 of 197 windows found no match). Cotton canopy between RGB and MS bands at 3 cm seems to give AROSICS' window matching little to hold on to. The fact that 70% of tie points fell below `min_reliability=60` supports this. AROSICS itself printed that warning.
- **Consequence for your question about flight gaps:** there is no gap to close. The variable that actually differs between the datasets is **ODM processing quality**. Future flights should process RGB and MS with the same, higher-quality settings, and ground control points would help most of all.

---

## 5. What didn't work (the nays) and bugs found

### 5.1 Bug: hard-link alias crashed AROSICS `DESHIFTER`
- **Symptom:** `IndexError: list index out of range` at `geoarray/baseclasses.py:1152` (`band_meta['band_names'][bidx]`), raised when `DESHIFTER` saved its output.
- **Root cause:** `create_whitespace_safe_alias()` (AROSICS/GeoArray can't handle paths with spaces) used a **hard link** when it could. A hard link shares the source file's metadata byte for byte. The real ODM MS file has a band-name metadata list that doesn't match its band count. GeoArray tolerates that on read and crashes on save.
- **Fix:** always build a GDAL VRT alias with dataset and band metadata cleared. It's just as cheap (an XML reference, no data copied).
- **Tests:** `test_create_whitespace_safe_alias_always_returns_a_vrt`, `test_create_whitespace_safe_alias_round_trips_through_geoarray_save`.
- **Lesson:** synthetic fixtures had clean metadata, so this could only show up on real ODM output. That is the case for doing Phase 5 at all.

### 5.2 Nay: gating AROSICS behind classical QA
See §3.2. It was built that way to be safe. In practice it meant AROSICS could never help on exactly the imagery that needs it. The safety now comes from AROSICS' own gates plus our holdout, not from refusing to run it.

### 5.3 Nay: widening the search radius
We expected a bigger `max_shift` to recover edge tie points on 0726. It changed nothing. The residual isn't limited by search reach.

### 5.4 Nay (rejected, not tried): loosening `max_verification_p90_px`
Raising 1.5 px to 2.5 px would make R1 "pass". We chose not to. 2.27 px is 51% over the limit, not a rounding miss, and moving a gate until a run passes defeats the purpose of having gates. If the threshold changes, it should be because QGIS check points show 1.5 px is stricter than what the product needs, not because one run missed.

### 5.5 Smaller problems we ran into
- **Windows / Git Bash `/tmp` paths.** Git Bash and native Python resolve `/tmp` to different places, so edits to scratch files silently didn't affect the file Python ran. Use the session scratchpad with full Windows paths.
- **The CLI hides the original traceback.** Wrapped `ArosicsExecutionError`s hide the real stack. `reproduce_deshifter_bug.py` (scratch) called `_estimate_global_candidate` + `run_arosics_local_refinement` directly to see it. Worth adding a `--debug` flag that re-raises.
- **Tests showing 65 errors that aren't real.** In a sandboxed shell, pytest may be refused access to `%TEMP%\pytest-of-DELL` (`PermissionError: [WinError 5]`) and report every `tmp_path` test as an error. Pass `--basetemp=<writable dir>`. With that, the whole suite passes.
- **Checking the config actually reaches AROSICS.** When a flag seems to do nothing, check the resolved value first (`_resolve_max_shift_ref_px` gave 67 px vs ~10 px). The flag was wired correctly; the data just didn't care.
- **"The PowerShell window vanished."** Not a crash. A run that fails fast (seconds) exits with an error, and a window opened without `-NoExit` closes immediately. Use `scripts\Run-Alignment.bat`, which keeps the window open.

### 5.6 Four bugs found on the second farm (2026-09-19 evening)

Testing on Farm B (NAR-AAA0002, rice, same GPS-only ODM settings) exposed four real bugs. Together they explain why the classical path "failed" on every real dataset until now.

1. **The held-out QA scored false matches.** `_split_estimation_and_verification` holds back every fifth raw descriptor match, and nothing screened them. Between RGB and MS bands a few are wrong by 1,000–2,700 px, and plain RMSE let one of them turn a ~2 px fit into "96 px, FAIL". Diagnosis on three pairs: every ORB/SIFT candidate had a held-out median of 0.6–1.8 px, with 87–100% of points within 10 px. **Fix:** `evaluate_spatial_residuals(..., screen_false_matches=True)` excludes points beyond `quality.holdout_gross_mismatch_px` (10) from the RMSE, and requires `quality.min_holdout_agreement` (70%) of them to agree, so a wrong transform still fails. The 2 px / 3.5 px limits are unchanged. Manual GCPs never use screening: a large residual there is a real error.
2. **A published AROSICS-local result reported `Status: FAIL`.** The report and console showed the starting point's classical QA, not the gates that approved the published raster. **Fix:** the headline `quality` now belongs to the published result, and the starting point's numbers move to `local_refinement.global_starting_point_quality`.
3. **The working/output frame clipped correctly aligned MS.** The frame was the pre-alignment overlap plus 1.5 m. Farm B A007 needed a 5.7 m / 6.6% scale correction, so 12.8% of the correctly aligned MS spilled past the edge. The footprint gate also measured coverage of the *RGB*, so RGB ground the MS never photographed counted against the transform (78.99–79.26% on all four candidates, against an 80% floor). **Fix:** `coarse.overlap_margin_m` (10 m, clamped to the RGB's extent) and a footprint gate measured on the MS side (`transform.min_target_overlap_ratio`: does the aligned MS land on real RGB?), shared by every mode through `footprint_gate_failures()`.
4. **AROSICS' setup check crashed the whole run on small fields.** `COREG_LOCAL`'s constructor test-matches one window and raises a raw `RuntimeError` when the overlap is smaller than the 256 px window. It sat outside the error handling. **Fix:** it is now a clean `WINDOW_EXCEEDS_OVERLAP` rejection that falls back to the verified global result.

Found along the way: the synthetic test fixture was a checkerboard, which has no unique alignment, so the integration test had only been passing by luck (every feature candidate matched upside down). It now uses non-repeating texture with a known 0.12 / −0.09 m georeference error, and the test asserts the engine recovers it (it does, to within 4 mm).

**Results after all four fixes** (error converted to cm, since pixel units changed with the frame):

| Pair | Before | After (global / edge) | Local refinement |
|---|---|---|---|
| Farm B A007 | failed outright | PASS, 7.7 / 9.5 cm | rejected, p90 2.29 px |
| Farm B A008 | 8.4 / 9.4 cm | PASS, 6.2 / 7.5 cm | rejected, worst holdout 2.88 px |
| Farm A 0826 | 8.8 / 9.4 cm | PASS, 8.8 / 9.6 cm | rejected, p90 2.32 px |
| Farm A 0726 | 6.7 / 7.0 cm | PASS, 6.2 / 6.7 cm | rejected, p90 1.76 px |

Every real pair now gets a *verified* global result. AROSICS local refinement still misses its own bar on all four (1.76–2.32 px p90 against 1.5 px), which keeps §6.5 open for the QGIS check.

### 5.7 Third dataset: coverage gate retired, memory ceiling found (2026-09-20)

FAI-AAA0106-0626-A001 (RGB 185 MP uint8, MS 157 MP float32, both 3 cm) exposed two separate issues.

1. **The MS-side coverage gate rejected data, not a bad transform.** The two flights cover noticeably different ground: the MS runs ~100 m past the RGB to the south, the RGB ~116 m past the MS to the east, and the RGB fills only 53% of its own bounding box. Inside the output frame only ~72% of the MS lands on valid RGB, against the 80% `min_target_overlap_ratio` floor, so a correct alignment would have been rejected on any machine. The user confirmed from their own inspection that the MS often covers only a central section of the RGB, and that same-field data should be aligned as provided. **Decision (user):** `transform.min_target_overlap_ratio` is retired. Overlap ratios are still reported but no longer gated. Accuracy on the shared ground is judged by the residual QA and AROSICS verification, and `min_retained_valid_ratio` (which compares the MS in the frame before and after the correction, so it is blind to coverage differences) still catches a transform that throws the MS off the grid. Old configs that set the retired key get a `DeprecationWarning`.
2. **The run died silently on an 8 GB laptop (not yet fixed; decision pending).** The pair is ~8.5× Farm A's MS and ~6× its RGB. The coarse step, AROSICS staging and `COREG_LOCAL` all load whole bands. `COREG_LOCAL` also caches both full arrays and hands them to 4 `loky` workers on Windows. With 2.9 GB free RAM this almost certainly ran out of memory. The output folder was empty and Windows logged no crash event, which fits a silent kill, but the exact stage is unconfirmed. The user's position: QA machines will have far more RAM, and working data handling should not be rewritten for one laptop. Agreed. Candidates if it resurfaces on bigger hardware or bigger fields: a pre-flight memory check with a clear message, overlap-windowed reads, and a memory-bounded AROSICS `CPUs`. At 3 cm float32, a 100 ha field (~1.1 billion px) would exceed even 32 GB with full-band reads.

Also noted: this pair's RGB was re-saved by Windows Photo Viewer (TIFF tag), which stripped its overviews and rewrote it as one-row strips (slow reads). The georeferencing survived and the pixel dimensions match the stated bounds exactly. Prefer the original ODM export.

---

## 6. Open questions — resolved

**Update 2026-09-19, second pass.** Re-ran the reshaped R1–R5 plan (§9). Two items above are now closed with real evidence, not left open:

1. **Which MS folder is the canonical R1 input:** confirmed **0826** (see §4.1 correction — it isn't a different flight, it's the same flight reprocessed at higher ODM quality). Ran as **R1_red** with `--arosics-band-pair red:1:1 --arosics-band-pair green:2:2`.
2. **Channel choice (red vs green) does not change the tie-point ceiling.** R1_red (red tried first) still produced exactly **5 valid tie points**, identical to the earlier green-first run. Ruled out.
3. **Window size does not fix it either — tested exhaustively, not assumed.** Tried 128², 256² (default), 512² on the same 0826/red pairing:

   | Window | Red valid tie points | Green valid tie points |
   |---|---|---|
   | 128×128 | 4 | 6 |
   | 256×256 (default) | 5 | — (5 in the original green-first run) |
   | 512×512 | 3 | 2 |

   No monotonic trend, no window size gets anywhere close to the 12 required. **Conclusion: this is a genuine content/spectral decorrelation between the RGB and MS captures at sub-window scale (cotton canopy likely looks meaningfully different between sensors at fine texture), not a parameter to tune.** Search radius (already ruled out), channel, and window size have all been tested. Do not spend further effort tuning AROSICS local parameters on this specific dataset — the ceiling is in the data, not the config.
4. **DESHIFTER vs. gdal_tps parity — also resolved, not a bug.** Ran both engines on 0726 (the one dataset where a warp actually executes before rejection): DESHIFTER 2.27 px p90, gdal_tps 2.85 px p90. Checked both code paths and the AROSICS source directly:
   - `min_points_local_corr=len(gcps)` in `_warp_with_deshifter` is intentional, not a bug: it makes DESHIFTER's own `len(GCPList) < min_points_local_corr` check always false, so DESHIFTER always attempts the real local TPS warp instead of silently downgrading to a mean-shift — leaving the judgment to our own gates.
   - Both engines receive the identical GCP list and the same `bilinear` resampling.
   - The ~0.6 px gap is a genuine implementation difference in how each does the TPS solve/extrapolation beyond the tie-point convex hull, not something wired incorrectly on our side. Not cheaply fixable without patching AROSICS or GDAL internals, and out of scope.
   - **Consequence for the blueprint:** §9.1's R2 expectation of "<0.05 native px mean abs diff" between engines does not hold on this dataset. Treat DESHIFTER as the reference implementation (it's also the better-performing one here) and gdal_tps strictly as the large-raster fallback (per finding A8), not a co-equal alternative. Also note: a rejected run's warped raster is never retained (the warp happens inside a `tempfile.TemporaryDirectory` that's cleaned up regardless of outcome), so the literal pixel-diff the blueprint describes can only run against an *accepted* result — R5 (manual, once placed) is the first candidate for that.
5. **Is 1.5 px p90 the right bar?** Still open — settle it with the QGIS check (blueprint §9.2), not by tuning. Evidence so far: four unrelated real pairs land at 1.76–2.32 px, never under 1.5 px. This is the one item that genuinely needs human input (placing points in QGIS) and can't be resolved by another automated run.

---

## 6a. Proposed upstream fix (ODM processing, not this pipeline)

Everything in §6 rules out the alignment engine's own parameters as the cause of the tie-point ceiling. The remaining, and most actionable, lever is upstream: how the RGB and MS orthophotos were produced by ODM before they ever reach this pipeline.

**The cause.** Three things stack up on this dataset, each independently documented as an accuracy driver, not something specific to our engine:

1. **No ground control points.** All three ODM runs (`gcp: null`) rely on the drone's onboard GPS alone. Consumer/mapping-grade GNSS without RTK/PPK correction gives roughly 1-3 m horizontal accuracy; the ~0.4-0.9 m average GPS errors ODM itself reports are in that range. Across both farms the RGB↔MS correction has had a consistent signature: a 2-6 m shift, a 0.5-6.6% scale difference and a ~0.45° rotation. A systematic scale-plus-rotation difference between two cameras on the same flight is exactly what independent GPS-only reconstructions produce and what GCPs remove. GCPs anchor the model to surveyed positions instead of GPS alone and are the standard fix (OpenDroneMap's own docs: 5 well-distributed GCPs is enough for most jobs; ODM's high-precision workflow guide is explicit that GCPs are advised whenever better than ~3% accuracy is needed).
2. **Inconsistent, and for one input low, processing quality.** The `0726` RGB and MS were both run at ODM's `low` point-cloud quality and 5 cm DEM resolution; the `0826` MS reprocessing used `high` quality and 1 cm DEM. ODM's own community documentation notes that `fast_orthophoto` mode (set on all three runs here) skips the dense reconstruction step, and loses orthophoto accuracy specifically in areas with real elevation change because the sparse reconstruction doesn't correct for terrain displacement as well as the dense pipeline. (An earlier version of this journal claimed our measurements showed this as edge-heavy distortion; that measurement was contaminated by false matches, §5.6, and is withdrawn. Matched processing settings remain good practice on their own merits.)
3. **Cross-spectral correlation is inherently harder over crop canopy, independent of all of the above.** This is why `COREG_LOCAL`'s tie-point ceiling didn't move under any parameter we tried (§6). Registration literature on multi-modal imagery is explicit that normalized cross-correlation depends on statistical similarity between pixel intensities and does not work well between images from different wavelengths, and specifically flags agricultural scenes as harder than most because they contain fewer features in common across spectra than built-up or bare-ground scenes do. RGB and multispectral bands of the same cotton canopy are exactly this case: similar large-scale layout, different fine-texture appearance.

**The cure — what Agrilift can change upstream, in order of impact:**

1. **Add real ground control points to the ODM run itself**, not just to our pipeline's manual mode. A handful of surveyed or high-accuracy-GNSS points per flight, fed to ODM's own `--gcp` argument, corrects the root GPS-only error before either orthophoto is even produced — this fixes both RGB and MS at once, and every downstream tool that consumes them, not just this alignment engine.
2. **Process RGB and MS with matched, higher processing settings** — same `pc-quality` (avoid `low`), and avoid `fast-orthophoto` where terrain relief is non-trivial, so both orthophotos get the same dense-reconstruction terrain correction instead of one being systematically worse than the other.
3. **If RTK/PPK-equipped drones are an option**, they remove GPS-only error almost entirely (down to 1-3 cm) and reduce, though don't eliminate, the case for manual GCPs at the ODM stage.
4. None of this removes the cross-spectral correlation difficulty (#3) entirely — that's inherent to comparing RGB and MS content, not a processing setting — but it does remove the two error sources (#1, #2) that are currently large enough to push the *global* result outside our own QA in the first place, which is what forces every run onto the harder unverified/local-refinement path to begin with. With a clean global start, `COREG_LOCAL`'s remaining job is much smaller and its existing tie-point/verification gates should behave the way they do on the well-behaved 0826 case, or better.

**References:**
- [OpenDroneMap — Ground Control Points](https://docs.opendronemap.org/gcp/) and [High Precision Workflows](https://docs.opendronemap.org/map-accuracy/) — GCP count/placement guidance, "5 distributed GCPs" for typical accuracy targets.
- [OpenDroneMap Community — Improving orthophoto results by adjusting processing parameters](https://community.opendronemap.org/t/improving-orthophoto-results-by-adjusting-processing-parameters/2819) and [High Resolution/Fast Ortho Processing](https://community.opendronemap.org/t/high-resolution-fast-ortho-processing/23789) — `fast-orthophoto`/`pc-quality` accuracy tradeoffs, terrain-displacement correction differences.
- [GeoNadir — RTK, Ground Control Points, and GPS accuracy explained](https://geonadir.com/rtk-explained/) and [Aerocartwright — Drone Survey Positioning: GPS vs RTK/PPK vs GCPs](https://aerocartwright.com/library/drone-survey-positioning-tiers/) — consumer GNSS (~1-3 m) vs. RTK/PPK (~1-3 cm) horizontal accuracy figures.
- [US Patent 7,103,234 — Method for blind cross-spectral image registration](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7103234) — cross-correlation's dependence on statistical pixel-intensity similarity failing between different wavelengths, and agricultural/wetland scenes specifically noted as harder due to fewer common features across spectra.
- Scheffler, D., Hollstein, A., Diedrich, H., Segl, K., Hostert, P. (2017). *AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data.* Remote Sensing, 9(7), 676. [Paper](https://www.researchgate.net/publication/318056207) — the tie-point reliability/SSIM/RANSAC cascade this pipeline builds on, designed and validated on multi-sensor satellite data, not sub-decimeter drone canopy imagery.

---

## 7. Where we stand

| Phase | Status |
|---|---|
| P0 Hotfixes | ✅ |
| P1 Shared infrastructure | ✅ |
| P2 AROSICS coarse-to-fine | ✅ (plus the §3.2 two-path publication rule and the §5.1 fix, made during P5) |
| P3 Manual GCP rewrite | ✅ |
| P4 CLI / runner / config / docs | ✅ |
| P5 Real-data validation | 🔶 About 55%. R1 (both processings, plus red-first variant), R2 (both engines), R3, R4, and the `gdalinfo` grid-identity check are **done**, with definitive conclusions, not open questions. What remains needs a human at QGIS: the 12-point independent check, R5 (manual GCPs + 5 check points), and the visual sanity look at R1's preview. None of the remaining work is answerable by another automated run — it requires placing points on real imagery. |

**Overall: about 88–90%.** All the architecture is built and tested. Every automated-run question has been run to a conclusion. What's left is the one thing that always required a human: QGIS point placement.

**Blueprint §9.3 acceptance note:** R1 passes if it is accepted, **or rejected with a reason code that visual inspection agrees with**. Both R1 variants were rejected for reasons we can explain, so R1 can close as soon as someone confirms visually that the preview/overlay shows what the reason codes say.

---

## 8. Artifacts on disk

| Path | Contents |
|---|---|
| `Stuff/phase5/R2/` | 0826 diagnostic run: aligned TIFF, preview PNG, report JSON (PASS, `automated_global`, fallback `INSUFFICIENT_TIE_POINTS`) |
| `Stuff/phase5/R1`, `R1c` | Empty. Those runs failed closed, which is correct: nothing published. |
| `Stuff/phase5/repro/` | Output folder of the direct-call reproduction script |
| Session scratchpad (temporary) | `phase5_run.py` (psutil RAM-sampling wrapper), `diagnose_r1.py`, `reproduce_deshifter_bug.py`. **Not in the repo.** Copy `phase5_run.py` into `scripts/` if RAM figures are needed for the remaining runs. |
