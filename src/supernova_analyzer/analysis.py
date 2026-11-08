"""Analysis utilities for supernova light curve data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_PEAK_COLUMNS = ("time", "magnitude")


@dataclass(frozen=True, slots=True)
class PeakResult:
    """Observed peak measured directly from discrete light curve data.

    Attributes:
        peak_index: Zero-based row position in the time-sorted observations.
        peak_time: Time of the observed minimum magnitude.
        peak_magnitude: Minimum apparent magnitude in the observations.
    """

    peak_index: int
    peak_time: float
    peak_magnitude: float


@dataclass(frozen=True, slots=True)
class Dm15Result:
    """Result of the linear 15-day decline baseline from an observed peak."""

    peak_time: float
    peak_magnitude: float
    target_time: float
    interpolated_magnitude: float
    delta_m15: float
    method: str


def _validated_sorted_copy(light_curve: pd.DataFrame) -> pd.DataFrame:
    """Validate analysis columns and return a time-sorted copy."""
    missing_columns = [
        column for column in REQUIRED_PEAK_COLUMNS if column not in light_curve.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Light curve is missing required columns: {missing}.")

    if light_curve.empty:
        raise ValueError("Light curve DataFrame is empty.")

    for column in REQUIRED_PEAK_COLUMNS:
        if not is_numeric_dtype(light_curve[column]):
            raise ValueError(f"Column '{column}' must contain numeric values.")
        if not np.isfinite(light_curve[column].to_numpy()).all():
            raise ValueError(f"Column '{column}' must contain only finite values.")

    return light_curve.sort_values("time", kind="stable").reset_index(drop=True)


def detect_observed_peak(light_curve: pd.DataFrame) -> PeakResult:
    """Detect the observed peak as the earliest minimum apparent magnitude.

    The calculation uses a time-sorted copy, leaving the input DataFrame
    unchanged. This result describes a discrete observed point and is not an
    interpolated or model-fitted peak.

    Args:
        light_curve: DataFrame containing numeric ``time`` and ``magnitude``
            columns with finite values.

    Returns:
        The observed peak and its zero-based position in time-sorted data.

    Raises:
        ValueError: If required columns are missing, the DataFrame is empty, or
            time or magnitude contains non-numeric or non-finite values.
    """
    sorted_light_curve = _validated_sorted_copy(light_curve)
    peak_index = int(sorted_light_curve["magnitude"].argmin())
    peak_row = sorted_light_curve.iloc[peak_index]

    return PeakResult(
        peak_index=peak_index,
        peak_time=float(peak_row["time"]),
        peak_magnitude=float(peak_row["magnitude"]),
    )


def calculate_dm15_linear(
    light_curve: pd.DataFrame,
    peak: PeakResult,
) -> Dm15Result:
    """Calculate the linear 15-day decline baseline from an observed peak.

    The baseline is ``m(t_peak + 15 days) - m(t_peak)``. The target magnitude is
    linearly interpolated from a time-sorted copy. It is not a noise-aware fit,
    and extrapolation is not performed.

    Args:
        light_curve: DataFrame containing numeric, finite ``time`` and
            ``magnitude`` columns.
        peak: Observed peak used as the reference for time and magnitude.

    Returns:
        An immutable result containing the interpolated magnitude and decline.

    Raises:
        ValueError: If the input is invalid, contains duplicate times, or does
            not cover the target time.
    """
    sorted_light_curve = _validated_sorted_copy(light_curve)

    if sorted_light_curve["time"].duplicated().any():
        raise ValueError("Cannot interpolate a light curve with duplicate time values.")

    target_time = peak.peak_time + 15.0
    minimum_time = float(sorted_light_curve["time"].iloc[0])
    maximum_time = float(sorted_light_curve["time"].iloc[-1])
    if target_time < minimum_time or target_time > maximum_time:
        raise ValueError(
            f"Target time {target_time:g} is outside the observed time range "
            f"[{minimum_time:g}, {maximum_time:g}]; extrapolation is not allowed."
        )

    interpolated_magnitude = float(
        np.interp(
            target_time,
            sorted_light_curve["time"].to_numpy(),
            sorted_light_curve["magnitude"].to_numpy(),
        )
    )

    return Dm15Result(
        peak_time=peak.peak_time,
        peak_magnitude=peak.peak_magnitude,
        target_time=target_time,
        interpolated_magnitude=interpolated_magnitude,
        delta_m15=interpolated_magnitude - peak.peak_magnitude,
        method="linear",
    )
