import logging

from drone_alignment.progress import ProgressReporter


def test_progress_reports_bar_percentage_and_stage(caplog):
    reporter = ProgressReporter(logging.getLogger("progress-test"), width=10)
    with caplog.at_level(logging.INFO, logger="progress-test"):
        reporter.update(25, "working")
    assert "[##--------]  25% - working" in caplog.text


def test_progress_never_moves_backwards_or_repeats(caplog):
    reporter = ProgressReporter(logging.getLogger("progress-test"))
    with caplog.at_level(logging.INFO, logger="progress-test"):
        reporter.update(60, "first")
        reporter.update(40, "stale")
        reporter.update(60, "duplicate")
    assert reporter.percent == 60
    assert caplog.text.count("Progress [") == 1
