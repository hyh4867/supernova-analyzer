"""Tests for observed peak detection."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from supernova_analyzer import PeakResult, detect_observed_peak


def test_detect_observed_peak_returns_expected_result() -> None:
    light_curve = pd.DataFrame(
        {"time": [0.0, 1.0, 2.0], "magnitude": [18.5, 17.2, 17.8]}
    )

    result = detect_observed_peak(light_curve)

    assert result == PeakResult(peak_index=1, peak_time=1.0, peak_magnitude=17.2)
    with pytest.raises(FrozenInstanceError):
        result.peak_time = 2.0  # type: ignore[misc]


def test_detect_observed_peak_uses_time_sorted_position() -> None:
    light_curve = pd.DataFrame(
        {"time": [3.0, 1.0, 2.0], "magnitude": [18.0, 18.5, 17.0]},
        index=[20, 40, 60],
    )

    result = detect_observed_peak(light_curve)

    assert result == PeakResult(peak_index=1, peak_time=2.0, peak_magnitude=17.0)


def test_detect_observed_peak_can_be_first_row() -> None:
    light_curve = pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "magnitude": [16.9, 17.2, 17.8]}
    )

    assert detect_observed_peak(light_curve).peak_index == 0


def test_detect_observed_peak_can_be_last_row() -> None:
    light_curve = pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "magnitude": [18.2, 17.8, 17.1]}
    )

    assert detect_observed_peak(light_curve).peak_index == 2


def test_detect_observed_peak_chooses_earliest_tied_minimum() -> None:
    light_curve = pd.DataFrame(
        {"time": [4.0, 1.0, 3.0], "magnitude": [17.0, 17.0, 18.0]}
    )

    result = detect_observed_peak(light_curve)

    assert result == PeakResult(peak_index=0, peak_time=1.0, peak_magnitude=17.0)


def test_detect_observed_peak_rejects_empty_dataframe() -> None:
    light_curve = pd.DataFrame(columns=["time", "magnitude"])

    with pytest.raises(ValueError, match="empty"):
        detect_observed_peak(light_curve)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_detect_observed_peak_rejects_invalid_magnitude(invalid_value: float) -> None:
    light_curve = pd.DataFrame(
        {"time": [1.0, 2.0], "magnitude": [17.0, invalid_value]}
    )

    with pytest.raises(ValueError, match="magnitude.*finite"):
        detect_observed_peak(light_curve)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_detect_observed_peak_rejects_invalid_time(invalid_value: float) -> None:
    light_curve = pd.DataFrame(
        {"time": [1.0, invalid_value], "magnitude": [17.0, 18.0]}
    )

    with pytest.raises(ValueError, match="time.*finite"):
        detect_observed_peak(light_curve)


def test_detect_observed_peak_does_not_modify_input() -> None:
    light_curve = pd.DataFrame(
        {"time": [2.0, 1.0], "magnitude": [17.0, 18.0]}, index=[5, 3]
    )
    original = light_curve.copy(deep=True)

    detect_observed_peak(light_curve)

    assert_frame_equal(light_curve, original)
