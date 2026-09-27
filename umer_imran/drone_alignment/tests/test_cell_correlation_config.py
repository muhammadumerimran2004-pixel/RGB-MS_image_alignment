import pytest
from pydantic import ValidationError

from drone_alignment.alignment.displacement_field import LocalMatchSample as FieldLocalMatchSample
from drone_alignment.alignment.local_evidence import (
    CellMatchStatus,
    LocalMatchSample,
    SparseDisplacementResult,
)
from drone_alignment.alignment.local_mesh_aligner import LocalMatchSample as LegacyLocalMatchSample
from drone_alignment.config.schema import AlignmentConfig, AlignmentMode, CellCorrelationConfig


def test_local_correlation_mode_and_default_config_are_available():
    config = AlignmentConfig(alignment_mode=AlignmentMode.LOCAL_CORRELATION)

    assert config.alignment_mode is AlignmentMode.LOCAL_CORRELATION
    assert config.cell_correlation.grid_rows == 7
    assert config.cell_correlation.grid_cols == 7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"cell_halo_px": 12, "max_search_radius_px": 12, "interpolation_margin_px": 2},
            "cell_halo_px",
        ),
        (
            {"grid_rows": 2, "grid_cols": 2, "min_trusted_cells": 5},
            "min_trusted_cells",
        ),
        (
            {"min_peak_sharpness": 2.0, "full_confidence_peak_sharpness": 2.0},
            "full_confidence_peak_sharpness",
        ),
        (
            {"min_holdout_cells": 4, "max_holdout_cells": 3},
            "max_holdout_cells",
        ),
    ],
)
def test_cell_correlation_config_rejects_unsafe_combinations(overrides, message):
    with pytest.raises(ValidationError, match=message):
        CellCorrelationConfig(**overrides)


def test_shared_evidence_contract_has_one_canonical_type():
    """Legacy imports remain valid while generic consumers use the neutral module."""
    assert LegacyLocalMatchSample is LocalMatchSample
    assert FieldLocalMatchSample is LocalMatchSample

    sample = LocalMatchSample(
        row=0,
        col=0,
        center_xy=(10.0, 10.0),
        source_ms_xy=(10.0, 10.0),
        predicted_rgb_xy=(10.0, 10.0),
        residual_dx_dy=(1.0, -1.0),
        displacement_dx_dy=(1.0, -1.0),
        confidence=0.8,
        road_score=None,
        second_road_score=None,
        tree_residual_dx_dy=None,
        status=CellMatchStatus.ACCEPTED,
        channel="green",
        phase_response=0.9,
        peak_sharpness=2.1,
        valid_fraction=0.95,
        phase_shift_dx_dy=(1.0, -1.0),
    )
    result = SparseDisplacementResult(
        samples=(sample,),
        rgb_features=None,
        ms_features=None,
        global_translation_px=(0.0, 0.0),
        trusted_count=1,
        coverage=1.0,
    )

    assert result.accepted == (sample,)
