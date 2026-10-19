"""Last-resort flight-date estimate from orthophoto file and folder names (blueprint 6.9.1).

Pure functions. The estimate is only ever *proposed*: callers confirm it with the user, or accept it
unattended only when it is unambiguous and high confidence and the configuration allows that.
Nothing here reads file-system times; the optional GeoTIFF ``TIFFTAG_DATETIME`` is an upper bound only
(a flight cannot happen after its orthophoto was produced).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

SENTINEL_2A_LAUNCH = date(2015, 6, 23)
MONTHS = {name: number for number, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_MONTH_RE = "(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_START, _END = r"(?<![0-9A-Za-z])", r"(?![0-9A-Za-z])"

CONFIDENCE_RANK = {"medium": 1, "high": 2}
PATTERN_RANK = {"iso": 4, "compact": 3, "month_name": 3, "numeric_year_last": 1}

# (name, regex, confidence); each regex has named groups, interpreted in ``_interpret``.
_PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("iso", re.compile(_START + r"(?P<y>\d{4})[-_.](?P<m>\d{2})[-_.](?P<d>\d{2})" + _END), "high"),
    ("compact", re.compile(_START + r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})" + _END), "high"),
    ("month_name", re.compile(
        _START + r"(?P<d>\d{1,2})[-_. ]?(?P<mn>" + _MONTH_RE + r")[-_. ,]?(?P<y>\d{4}|\d{2})" + _END, re.I), "high"),
    ("month_name", re.compile(
        _START + r"(?P<mn>" + _MONTH_RE + r")[-_. ]?(?P<d>\d{1,2})[-_. ,]+(?P<y>\d{4})" + _END, re.I), "high"),
    ("numeric_year_last", re.compile(
        _START + r"(?P<a>\d{1,2})[-_.](?P<b>\d{1,2})[-_.](?P<y>\d{4})" + _END), "medium"),
)


@dataclass(frozen=True)
class DateCandidate:
    value: date
    matched_text: str
    component: str  # the file or folder name it was found in
    depth: int  # 0 = file name, 1 = parent folder, ...
    pattern: str
    confidence: Literal["high", "medium"]
    ambiguous_readings: tuple[date, ...] = ()  # more than one when day and month cannot be told apart
    sources: tuple[int, ...] = ()  # indices of the paths it was found in (RGB, MS)

    @property
    def ambiguous(self) -> bool:
        return len(self.ambiguous_readings) > 1


@dataclass(frozen=True)
class PathEstimate:
    status: Literal["FOUND", "MULTIPLE_DATES", "NONE"]
    candidates: tuple[DateCandidate, ...] = field(default_factory=tuple)  # best first

    @property
    def best(self) -> DateCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def usable_unattended(self) -> bool:
        best = self.best
        return (self.status == "FOUND" and best is not None and not best.ambiguous and best.confidence == "high")


def _valid(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _interpret(name: str, match: re.Match, date_order: str) -> tuple[date | None, tuple[date, ...]]:
    """The date a match stands for, plus every reading when it is ambiguous."""
    groups = match.groupdict()
    year = int(groups["y"])
    if year < 100:
        year += 2000
    if name in ("iso", "compact"):
        found = _valid(year, int(groups["m"]), int(groups["d"]))
        return found, ((found,) if found else ())
    if name == "month_name":
        found = _valid(year, MONTHS[groups["mn"][:3].lower()], int(groups["d"]))
        return found, ((found,) if found else ())
    a, b = int(groups["a"]), int(groups["b"])
    readings = [r for r in (_valid(year, b, a), _valid(year, a, b)) if r]  # day-first, then month-first
    unique = tuple(dict.fromkeys(readings))
    if not unique:
        return None, ()
    if len(unique) == 1:
        return unique[0], unique
    primary = unique[0] if date_order == "dmy" else unique[1]
    return primary, unique


def _components(path: Path, depth: int) -> list[tuple[str, int]]:
    parts = [(path.stem, 0)]
    for level, parent in enumerate(path.parents, start=1):
        if level > depth or not parent.name:
            break
        parts.append((parent.name, level))
    return parts


def _scan(path: Path, source: int, depth: int, date_order: str) -> list[DateCandidate]:
    found: list[DateCandidate] = []
    for text, level in _components(path, depth):
        taken: list[tuple[int, int]] = []
        for name, pattern, confidence in _PATTERNS:
            for match in pattern.finditer(text):
                if any(match.start() < end and start < match.end() for start, end in taken):
                    continue  # a stronger pattern already claimed these characters
                value, readings = _interpret(name, match, date_order)
                if value is None:
                    continue
                taken.append((match.start(), match.end()))
                found.append(DateCandidate(value, match.group(0), text, level, name, confidence,
                                           readings if len(readings) > 1 else (), (source,)))
    return found


def estimate_flight_date(
    paths: Sequence[Path], *, depth: int = 3, date_order: str = "dmy", upper_bound: date | None = None,
    today: date | None = None,
) -> PathEstimate:
    """Rank the plausible dates named in ``paths`` (RGB orthophoto first, then the MS orthophoto)."""
    today = today or date.today()
    raw: list[DateCandidate] = []
    for source, path in enumerate(paths):
        raw.extend(_scan(Path(path), source, depth, date_order))
    plausible = [c for c in raw if SENTINEL_2A_LAUNCH <= c.value <= today and (upper_bound is None or c.value <= upper_bound)]
    if not plausible:
        return PathEstimate("NONE")

    merged: list[DateCandidate] = []
    for value in dict.fromkeys(c.value for c in plausible):
        group = [c for c in plausible if c.value == value]
        strongest = max(group, key=lambda c: (CONFIDENCE_RANK[c.confidence], PATTERN_RANK[c.pattern], -c.depth))
        sources = tuple(sorted({s for c in group for s in c.sources}))
        confidence = strongest.confidence
        if len(sources) > 1 and confidence == "medium" and not strongest.ambiguous:
            confidence = "high"  # the same date in both orthophoto paths
        merged.append(DateCandidate(value, strongest.matched_text, strongest.component, min(c.depth for c in group),
                                    strongest.pattern, confidence, strongest.ambiguous_readings, sources))
    merged.sort(key=lambda c: (-len(c.sources), c.depth, -CONFIDENCE_RANK[c.confidence], -PATTERN_RANK[c.pattern], c.value))
    return PathEstimate("FOUND" if len(merged) == 1 else "MULTIPLE_DATES", tuple(merged))


def read_tiff_processing_date(path: Path) -> date | None:
    """``TIFFTAG_DATETIME`` of a GeoTIFF (processing time) for use as an upper bound; None when absent."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            text = src.tags().get("TIFFTAG_DATETIME")
        return date(int(text[0:4]), int(text[5:7]), int(text[8:10])) if text else None
    except Exception:  # noqa: BLE001  (an unreadable tag simply means "no bound")
        return None
