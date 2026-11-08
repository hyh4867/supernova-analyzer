"""Tests for linear Delta m15 calculation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from supernova_analyzer import (
    Dm15Result,
    PeakResult,
    calculate_dm15_linear,
    detect_observed_peak,
)


def make_interpolation_data() -> pd.DataFrame:
    """Create a light curve whose day-15 magnitude requires interpolation."""
    return pd.DataFrame(
        {
            "time": [20.0, 0.0, 10.0],
            "magnitude": [19.0, 17.0, 18.0],
        }
    )


def test_calculate_dm15_linear_interpolates_expected_value() -> None:
    light_curve = make_interpolation_data()
    peak = detect_observed_peak(light_curve)

    result = calculate_dm15_linear(light_curve, peak)

    assert result == Dm15Result(
        peak_time=0.0,
        peak_magnitude=17.0,
        target_time=15.0,
        interpolated_magnitude=18.5,
        delta_m15=1.5,
        method="linear",
    )
    with pytest.raises(FrozenInstanceError):
        result.delta_m15 = 2.0  # type: ignore[misc]


def test_calculate_dm15_linear_rejects_target_outside_range() -> None:
    light_curve = pd.DataFrame(
        {"time": [0.0, 5.0, 10.0], "magnitude": [17.0, 17.5, 18.0]}
    )
    peak = detect_observed_peak(light_curve)

    with pytest.raises(ValueError, match="extrapolation"):
        calculate_dm15_linear(light_curve, peak)


def test_calculate_dm15_linear_rejects_duplicate_time() -> None:
    light_curve = pd.DataFrame(
        {"time": [0.0, 10.0, 10.0, 20.0], "magnitude": [17.0, 18.0, 18.2, 19.0]}
    )
    peak = PeakResult(peak_index=0, peak_time=0.0, peak_magnitude=17.0)

    with pytest.raises(ValueError, match="duplicate time"):
        calculate_dm15_linear(light_curve, peak)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("time", np.nan),
        ("time", np.inf),
        ("magnitude", np.nan),
        ("magnitude", np.inf),
    ],
)
def test_calculate_dm15_linear_rejects_non_finite_values(
    column: str,
    invalid_value: float,
) -> None:
    light_curve = make_interpolation_data()
    light_curve.loc[1, column] = invalid_value
    peak = PeakResult(peak_index=0, peak_time=0.0, peak_magnitude=17.0)

    with pytest.raises(ValueError, match=rf"{column}.*finite"):
        calculate_dm15_linear(light_curve, peak)


def test_calculate_dm15_linear_does_not_modify_input() -> None:
    light_curve = make_interpolation_data()
    original = light_curve.copy(deep=True)
    peak = detect_observed_peak(light_curve)

    calculate_dm15_linear(light_curve, peak)

    assert_frame_equal(light_curve, original)


def test_calculate_dm15_linear_uses_exact_observation() -> None:
    light_curve = pd.DataFrame(
        {"time": [0.0, 15.0, 20.0], "magnitude": [17.0, 18.25, 19.0]}
    )
    peak = detect_observed_peak(light_curve)

    result = calculate_dm15_linear(light_curve, peak)

    assert result.interpolated_magnitude == pytest.approx(18.25)
    assert result.delta_m15 == pytest.approx(1.25)
