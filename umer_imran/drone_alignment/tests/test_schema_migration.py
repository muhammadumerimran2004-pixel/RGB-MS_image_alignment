"""Legacy configuration key migration (blueprint §8.3).

These exercise AlignmentConfig/ArosicsConfig's mode="before" validators
directly against plain dicts, the same shape a YAML config file loads into.
"""
import pytest

from drone_alignment.config.schema import AlignmentConfig, ArosicsConfig


# --- ArosicsConfig: pre-split flat keys -> global_candidate / local --------

def test_legacy_arosics_yaml_loads_with_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        cfg = ArosicsConfig(**{
            "window_size": [128, 128],
            "max_shift_px": 40,
            "min_reliability": 50.0,
            "grid_res": 32,
            "min_local_tie_points": 20,
        })
    assert cfg.global_candidate.window_size == (128, 128)
    assert cfg.global_candidate.max_shift_px == 40
    assert cfg.global_candidate.min_reliability == 50.0
    assert cfg.local.window_size == (128, 128)
    assert cfg.local.min_reliability == 50.0
    assert cfg.local.grid_res_px == 32
    assert cfg.local.min_valid_tie_points == 20


def test_legacy_arosics_ignore_errors_is_dropped_with_warning():
    with pytest.warns(DeprecationWarning, match="ignore_errors"):
        cfg = ArosicsConfig(**{"ignore_errors": False})
    assert not hasattr(cfg, "ignore_errors")


def test_legacy_arosics_keys_do_not_override_explicit_nested_config():
    cfg = ArosicsConfig(**{
        "max_shift_px": 40,
        "global_candidate": {"max_shift_px": 90},
    })
    assert cfg.global_candidate.max_shift_px == 90


def test_arosics_config_without_legacy_keys_loads_cleanly_and_without_warning(recwarn):
    ArosicsConfig()
    assert len(recwarn) == 0


# --- AlignmentConfig: transform.manual_tps_* -> manual.* -------------------

def test_legacy_manual_tps_keys_migrate_with_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        cfg = AlignmentConfig(**{
            "transform": {
                "manual_tps_smoothing": 0.0,
                "manual_tps_fallback_weight": 0.4,
                "manual_tps_support_radius_px": 300.0,
            },
        })
    assert cfg.manual.tps_smoothing_m == 0.0
    assert cfg.manual.taper_fallback_weight == 0.4
    assert cfg.manual.taper_support_radius_px == 300.0


def test_legacy_manual_tps_smoothing_nonzero_is_ignored_with_warning():
    with pytest.warns(DeprecationWarning, match="cannot be converted to metres"):
        cfg = AlignmentConfig(**{"transform": {"manual_tps_smoothing": 5.0}})
    assert cfg.manual.tps_smoothing_m == 0.0


@pytest.mark.parametrize("data, match", [
    ({"transform": {"min_reference_overlap_ratio": 0.7}}, "min_reference_overlap_ratio has been retired"),
    ({"transform": {"min_target_overlap_ratio": 0.7}}, "min_target_overlap_ratio has been retired"),
    ({"coarse": {"overlap_margin_px": 50}}, "overlap_margin_m"),
])
def test_retired_footprint_keys_warn_instead_of_being_silently_ignored(data, match):
    with pytest.warns(DeprecationWarning, match=match):
        cfg = AlignmentConfig(**data)
    assert not hasattr(cfg.transform, "min_target_overlap_ratio")
    assert cfg.transform.min_retained_valid_ratio == 0.80
    assert cfg.coarse.overlap_margin_m == 10.0


def test_alignment_config_without_legacy_keys_loads_cleanly_and_without_warning(recwarn):
    AlignmentConfig()
    assert len(recwarn) == 0
