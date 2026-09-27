import pytest

from drone_alignment.alignment.rejections import LocalRefinementRejected
import drone_alignment.pipeline as pipeline


def test_local_correlation_rejected_is_an_alias_for_local_refinement_rejected():
    """pipeline.LocalCorrelationRejected must remain the exact same class object
    as alignment.rejections.LocalRefinementRejected, so existing imports and
    isinstance checks against the old name keep working unchanged."""
    assert pipeline.LocalCorrelationRejected is LocalRefinementRejected


def test_local_refinement_rejected_carries_reason_code_and_details():
    with pytest.raises(LocalRefinementRejected) as excinfo:
        raise LocalRefinementRejected("SOME_REASON", "human readable message", {"count": 3})

    assert excinfo.value.reason_code == "SOME_REASON"
    assert str(excinfo.value) == "human readable message"
    assert excinfo.value.details == {"count": 3}


def test_local_refinement_rejected_details_default_to_empty_dict():
    error = LocalRefinementRejected("CODE", "message")
    assert error.details == {}
