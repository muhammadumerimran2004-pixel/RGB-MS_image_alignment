"""Shared fixtures for Sentinel patch tests."""

from __future__ import annotations

import pytest
from affine import Affine

from . import synthetic

S2_SHAPE = (60, 60)  # 600 m x 600 m at 10 m


@pytest.fixture
def s2_transform() -> Affine:
    return synthetic.make_s2_transform()


@pytest.fixture
def s2_shape() -> tuple[int, int]:
    return S2_SHAPE


@pytest.fixture
def scene(s2_transform, s2_shape):
    return synthetic.make_scene(s2_transform, s2_shape)
