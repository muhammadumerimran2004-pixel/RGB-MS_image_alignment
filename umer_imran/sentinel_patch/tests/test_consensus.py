"""Multi-scene consensus: the decision rules, then the whole pipeline on synthetic scenes."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from sentinel_patch import run_sentinel_patch
from sentinel_patch.config.schema import ConsensusConfig, parse_config
from sentinel_patch.registration.consensus import Vote, decide, largest_cluster

from . import scene_files

CFG = ConsensusConfig()  # 3 scenes, 50 %, 1 px


def vote(name, east, north, gap=0):
    return Vote(name, name, gap, (east, north), (east / 10, -north / 10), 0.6, 1.05, 150)


def verdict(shifts, cfg=CFG, max_shift=25.0):
    return decide([vote(f"s{i}", *s) for i, s in enumerate(shifts)], cfg, 10.0, max_shift)


# -- the decision rules -----------------------------------------------------------------------------

def test_a_tight_majority_wins_and_the_shift_is_their_median():
    result = verdict([(10, 0), (11, -1), (9, 1), (10, 2), (-40, 30), (45, 5)])
    assert result.agreed and result.members == (0, 1, 2, 3) and result.shift_m == pytest.approx((10, 0.5))
    assert result.spread_m < 3 and result.runner_up_size == 1


def test_outliers_do_not_drag_the_shift():
    assert verdict([(10, 0), (10, 0), (10, 0), (60, 60)]).shift_m == (10.0, 0.0)


def test_too_few_scenes_is_not_a_consensus():
    result = verdict([(10, 0), (10, 0)])
    assert result.status == "too_few_votes" and not result.agreed and "at least 3" in result.reason


def test_scattered_votes_do_not_agree():
    result = verdict([(0, 0), (40, 0), (0, 40), (-40, 0), (0, -40)])
    assert result.status == "no_agreement" and len(result.members) == 1


def test_three_agreeing_out_of_many_is_a_coincidence_not_a_consensus():
    shifts = [(10, 0)] * 3 + [(i * 15.0 - 100, 70 - i * 9.0) for i in range(9)]
    result = verdict(shifts)
    assert result.status == "no_agreement" and result.needed == 6  # half of 12 votes


def test_two_equally_supported_shifts_are_ambiguous():
    result = verdict([(10, 0), (10, 1), (10, -1), (-30, 20), (-30, 21), (-31, 20)])
    assert result.status == "ambiguous" and result.runner_up_size == 3


def test_an_agreed_shift_beyond_the_plausible_limit_is_refused():
    assert verdict([(40, 0)] * 4, max_shift=25.0).status == "no_agreement"


def test_agreement_radius_is_in_pixels():
    spread = [(0, 0), (9, 0), (-9, 0)]
    assert verdict(spread).agreed
    assert verdict(spread, ConsensusConfig(agreement_radius_px=0.5)).status == "no_agreement"


def test_cluster_is_deterministic_and_prefers_the_tighter_group_on_ties():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0], [20.2, 0.0]])
    assert sorted(largest_cluster(points, 3.0)) == [2, 3]  # equal size: the tighter pair wins
    assert largest_cluster(points, 3.0).tolist() == largest_cluster(points, 3.0).tolist()


def test_consensus_config_bounds():
    with pytest.raises(Exception):
        ConsensusConfig(min_scenes=1)


# -- end to end ---------------------------------------------------------------------------------------

FLIGHT = date(2026, 10, 10)
ERROR_M = (15.0, -12.0)  # content offset; the truth shift is (-15, +12)


@pytest.fixture(scope="module")
def scene():
    return scene_files.full_scene()


@pytest.fixture
def drone(tmp_path, scene):
    return scene_files.write_drone_rasters(tmp_path / "drone", scene)


def scenes(tmp_path, scene, specs):
    """specs: (folder, day, content_offset, datatake_id)."""
    root = tmp_path / "scenes"
    for name, day, offset, datatake in specs:
        scene_files.write_s2_scene(root / name, scene, content_offset_m=offset, item_id=name,
                                   acquired=f"2026-10-{day:02d}T05:30:00Z", datatake_id=datatake)
    return root


def run(tmp_path, drone, root, **registration):
    # A peak ratio nobody can reach makes every single scene "ambiguous", the situation consensus is for.
    registration = {"arosics_mode": "off", "min_peak_ratio": 1e9, **registration}
    config = parse_config({"local_folder": str(root), "registration": registration})
    ms, rgb = drone
    result = run_sentinel_patch(ms, rgb, tmp_path / "out", flight_date=FLIGHT, ms_band_map=scene_files.MS_BAND_MAP, config=config)
    return result, json.loads(result.report_path.read_text(encoding="utf-8"))


def test_ambiguous_single_scene_is_rescued_by_agreeing_scenes(tmp_path, scene, drone):
    root = scenes(tmp_path, scene, [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12, 13, 14)])
    result, report = run(tmp_path, drone, root)

    assert result.status == "verified" and "SHIFT_FROM_CONSENSUS" in result.warnings
    assert result.shift_m == pytest.approx((-ERROR_M[0], -ERROR_M[1]), abs=1.5)
    registration = report["registration"]
    assert registration["method"] == "consensus" and registration["single_scene"]["reason_code"] == "WEAK_PEAK"
    assert registration["consensus"]["status"] == "agreed" and len(registration["consensus"]["members"]) == 4
    assert registration["shift_px"] is not None and report["status"] == "verified"


def test_a_scene_that_disagrees_is_outvoted(tmp_path, scene, drone):
    specs = [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12, 13)] + [("odd", 14, (-30.0, 20.0), "Todd")]
    result, report = run(tmp_path, drone, scenes(tmp_path, scene, specs))
    assert result.status == "verified" and report["registration"]["consensus"]["spread_m"] < 3
    assert result.shift_m == pytest.approx((-ERROR_M[0], -ERROR_M[1]), abs=1.5)


def test_disagreeing_scenes_leave_the_patch_unshifted_and_say_why(tmp_path, scene, drone):
    offsets = [(15.0, -12.0), (-20.0, 10.0), (5.0, 20.0), (-10.0, -20.0)]
    specs = [(f"d{i}", 11 + i, o, f"T{i}") for i, o in enumerate(offsets)]
    result, report = run(tmp_path, drone, scenes(tmp_path, scene, specs))
    assert result.status == "not_applied" and result.reason_code == "WEAK_PEAK" and result.shift_m == (0.0, 0.0)
    assert report["registration"]["consensus"]["status"] in {"no_agreement", "ambiguous"}
    assert report["registration"]["method"] == "single_scene"


def test_two_scenes_are_not_enough(tmp_path, scene, drone):
    result, report = run(tmp_path, drone, scenes(tmp_path, scene, [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12)]))
    assert result.status == "not_applied" and report["registration"]["consensus"]["status"] == "too_few_votes"


def test_tiles_of_one_acquisition_count_once(tmp_path, scene, drone):
    specs = [("a", 11, ERROR_M, "SAME"), ("b", 11, ERROR_M, "SAME"), ("c", 12, ERROR_M, "T12")]
    result, report = run(tmp_path, drone, scenes(tmp_path, scene, specs))
    scenes_seen = {s["item_id"]: s for s in report["registration"]["consensus"]["scenes"]}
    assert result.status == "not_applied" and report["registration"]["consensus"]["votes"] == 2
    assert "DUPLICATE_ACQUISITION" in {s["reason_code"] for s in scenes_seen.values()}


def test_cloudy_scenes_do_not_vote(tmp_path, scene, drone):
    root = scenes(tmp_path, scene, [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12, 13)])
    scene_files.write_s2_scene(root / "cloudy", scene, content_offset_m=(40.0, 30.0), item_id="cloudy",
                               acquired="2026-10-10T05:30:00Z", datatake_id="Tc",
                               edit_scl=lambda a: a.__setitem__((slice(None), slice(None)), 9))
    result, report = run(tmp_path, drone, root)
    cloudy = next(s for s in report["registration"]["consensus"]["scenes"] if s["item_id"] == "cloudy")
    assert cloudy["vote"] is False and cloudy["reason_code"] == "AOI_TOO_CLOUDY" and result.status == "verified"


def test_consensus_can_be_switched_off(tmp_path, scene, drone):
    root = scenes(tmp_path, scene, [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12, 13)])
    result, report = run(tmp_path, drone, root, consensus={"enabled": False})
    assert result.status == "not_applied" and report["registration"]["consensus"] is None


def test_consensus_never_overrides_a_texture_refusal(tmp_path, scene, drone):
    root = scenes(tmp_path, scene, [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12, 13)])
    result, report = run(tmp_path, drone, root, min_texture_score=1000.0)
    assert (result.status, result.reason_code) == ("not_applied", "LOW_TEXTURE")
    assert report["registration"]["consensus"] is None  # not even tried


def test_a_good_single_scene_does_not_trigger_the_consensus(tmp_path, scene, drone):
    root = scenes(tmp_path, scene, [(f"d{d}", d, ERROR_M, f"T{d}") for d in (11, 12, 13)])
    result, report = run(tmp_path, drone, root, min_peak_ratio=1.15)
    assert result.status == "verified" and report["registration"]["method"] == "single_scene"
    assert report["registration"]["consensus"] is None and "SHIFT_FROM_CONSENSUS" not in result.warnings
