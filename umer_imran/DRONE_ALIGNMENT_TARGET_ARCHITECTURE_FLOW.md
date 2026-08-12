# Drone Alignment: Proposed Production Data Flow

**Purpose:** Decision diagram for review before the next performance-focused implementation phase.  
**Status:** Implemented and verified on the supplied RGB/MS pair on 2026-08-06.

## Executive Summary

The proposed design moves full-resolution spectral reprojection and warping **after** low-resolution registration and validation. This prevents expensive processing of the complete multispectral orthomosaic when no transform can be verified.

```mermaid
flowchart TD
    A[RGB reference TIFF<br/>MS target TIFF] --> B[Read metadata once]
    B --> C[Identify semantic bands<br/>RGB: Red + Green<br/>MS: Red + Green + alpha]
    C --> D[Read source validity masks once<br/>dataset mask + alpha + nodata + finite values]

    D --> E[Build low-resolution common registration grid<br/>maximum side: 2048–4096 px]
    E --> F[Reproject only registration inputs<br/>RGB Green/Red, MS Green/Red, masks]
    F --> G[Mask-aware preprocessing<br/>valid-only percentile stretch + CLAHE]

    G --> H1[Green-to-Green candidates<br/>ORB → SIFT]
    G --> H2[Red-to-Red candidates<br/>ORB → SIFT]
    H1 --> I[Estimate candidate transform]
    H2 --> I
    I --> J{Feature candidate<br/>passes validation?}

    J -->|No feature candidate| K[Masked phase correlation<br/>on best channel pair]
    K --> L{Phase response and image-domain<br/>verification pass?}
    L -->|No| X[FAIL fast<br/>Write diagnostic report only<br/>Publish no aligned TIFF]
    L -->|Yes| M[Validated low-resolution transform]

    J -->|Yes| M
    M --> N[Low-resolution guardrails]
    N --> N1[Translation / rotation / scale]
    N --> N2[Inlier count, ratio, and spatial coverage]
    N --> N3[Retained valid-footprint ratio]
    N --> N4[RGB/MS valid-overlap ratio]
    N1 --> O{All mandatory<br/>gates pass?}
    N2 --> O
    N3 --> O
    N4 --> O

    O -->|No| X
    O -->|Yes| P[Convert transform to native output coordinates]
    P --> Q[Re-open original full-resolution MS raster]
    Q --> R[Warp only full-resolution MS validity mask]
    R --> S{Native mask retention and<br/>overlap gates pass?}
    S -->|No| X
    S -->|Yes| T[Create run-scoped staging output]

    T --> U[For each spectral band]
    U --> V[Read only required source window/tile]
    V --> W[Warp tile using native transform]
    W --> Y[Write compressed output tile immediately]
    Y --> U

    U -->|All bands written| Z[Write output mask / alpha]
    Z --> AA[Create bounded thumbnail QA preview]
    AA --> AB[Write report with candidate evidence<br/>and final footprint metrics]
    AB --> AC[Atomic publish<br/>TIFF + preview + report]
```

## Current Versus Proposed Cost Boundary

```mermaid
flowchart LR
    subgraph Current[Current corrected implementation]
        C1[Full-resolution common grid] --> C2[Full-resolution registration bands + masks]
        C2 --> C3[Candidate validation]
        C3 --> C4[Full-resolution warp]
    end

    subgraph Proposed[Proposed architecture]
        P1[Low-resolution registration grid] --> P2[Candidate validation]
        P2 -->|Rejected| P3[Stop cheaply]
        P2 -->|Accepted| P4[Full-resolution streamed warp]
    end
```

## Native-Coordinate Transform Rule

The registration matrix estimated in the low-resolution grid must not be copied directly into native-resolution warping.

```text
M_native = S_destination^-1 × M_low_resolution × S_source
```

Where `S_source` and `S_destination` map native pixel coordinates into their corresponding registration-grid coordinates. If both images use the same downsampled common target grid, the conversion simplifies; translation still must be scaled correctly, while rotation, shear, and scale must retain their geometric meaning.

## Mandatory Publication Gates

| Gate | Purpose | Failure result |
|---|---|---|
| Valid mask available | Exclude alpha-invalid/fill pixels | `INSUFFICIENT_VALID_DATA` |
| Registration candidate evidence | Prevent arbitrary transforms | Try next candidate or `FAIL` |
| Translation / rotation / scale | Reject implausible geometry | `TRANSFORM_OUT_OF_BOUNDS` |
| Feature spatial coverage or phase response | Reject weak evidence | `INSUFFICIENT_MATCHES` / `LOW_PHASE_RESPONSE` |
| Low-resolution footprint retention and overlap | Reject obvious clipping cheaply | `FOOTPRINT_RETENTION_FAILED` |
| Native-resolution mask confirmation | Confirm the winning transform at export scale | `FOOTPRINT_RETENTION_FAILED` |
| Output tile completion and mask write | Prevent partial/corrupt data | `OUTPUT_WRITE_FAILED` |
| Atomic publication | Ensure final names represent complete products | No final publication on error |

## Why This Is the Recommended Architecture

- Most registrations can be accepted or rejected from 2048–4096 px inputs rather than full orthomosaics.
- A failed candidate avoids full-resolution spectral reprojection and output work.
- Validity masks are read once and reused rather than repeatedly rebuilt per band.
- The full-resolution stage streams tiles, so peak memory is bounded by tile size rather than total mosaic size.
- A final native-mask check preserves the strict clipping and overlap safeguards established during registration.
- Partial output remains staged until every validation and write step has completed.

## Implementation Status

| Capability | Status |
|---|---|
| Alpha/dataset-mask-aware normalization | Implemented |
| Green/red candidate registration and CLAHE | Implemented |
| Transform, phase-response, and footprint guardrails | Implemented |
| Band-at-a-time staged output | Implemented |
| Low-resolution-only registration grid | Implemented |
| Source mask cache | Implemented |
| Native-coordinate transform conversion | Implemented |
| Tile/window streaming from source raster | Implemented |
| Native-resolution final mask confirmation | Implemented |

## Controlled Verification Result

The optimized pipeline was run against the supplied orthomosaics and published its result to `Stuff/aligned_v3/`.

| Metric | Result |
|---|---|
| Registration method | Masked phase correlation, Green/Green |
| Native residual translation | `(-1.19 px, -0.19 px)` / `(-9.0 cm, -1.4 cm)` |
| Phase response | `0.801` |
| Retained MS valid footprint | `99.9986%` |
| RGB-reference overlap | `92.83%` |
| Output valid-mask coverage | `91.84%` |
| Final TIFF | `502.03 MiB` |
| QA preview | `17.06 MiB` |
| Run time | approximately 3m51s |

The candidate was accepted through phase response and masked image-domain verification. Feature candidates were rejected by held-out residual QA. A QGIS visual review remains a release gate because no independent feature-correspondence residual was available for the selected phase result.
