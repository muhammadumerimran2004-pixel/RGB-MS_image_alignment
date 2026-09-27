"""Shared rejection type for expected local-alignment safety-gate failures.

Raising this (rather than a generic exception) signals that the failure is an
*expected* safety-gate rejection, not an operational or programming error:
the caller should fall back to the already-verified global result and record
the rejection reason, rather than letting the failure propagate and abort the
whole run. Operational/programming errors must still propagate normally so a
broken local path is never silently reported as a successful global run.
"""
from __future__ import annotations


class LocalRefinementRejected(RuntimeError):
    """Expected local safety-gate rejection; global publication remains safe."""

    def __init__(self, reason_code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details or {}
