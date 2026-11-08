"""Pure presentation services for the Supernova Analyzer application."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from numbers import Real
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from supernova_analyzer.analysis import Dm15Result, PeakResult
from supernova_analyzer.salt3_diagnostics import Salt3Diagnostics
from supernova_analyzer.salt3_fitting import Salt3FitResult
from supernova_analyzer.standardization import (
    Salt3AmplitudeMagnitudeResult,
    SyntheticRestFrameMagnitudeResult,
    TrippStandardizationResult,
)


def _json_ready(value: Any) -> Any:
    """Recursively convert supported scientific results to JSON primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON output cannot contain non-finite values.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported serialization type: {type(value).__name__}.")


def result_to_json(value: Any) -> str:
    """Serialize a frozen result or primitive mapping as stable JSON text."""
    return json.dumps(
        _json_ready(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def scientific_input_fingerprint(
    file_identity: bytes | str,
    *,
    redshift: float,
    include_model_covariance: bool,
    min_snr: float,
    initial_t0: float | None,
    bounds: Mapping[str, tuple[float, float]] | None,
) -> str:
    """Return a stable identity for every input that can affect a SALT3 fit.

    Tripp nuisance parameters are intentionally absent because changing them
    must not invalidate or rerun the fitted SALT3 model.
    """
    content_hash = hashlib.sha256(
        file_identity if isinstance(file_identity, bytes) else file_identity.encode()
    ).hexdigest()
    normalized_bounds = {
        name: [float(limits[0]), float(limits[1])]
        for name, limits in sorted((bounds or {}).items())
    }
    payload = {
        "content_sha256": content_hash,
        "redshift": validate_gui_number(redshift, "redshift", minimum=0.0),
        "include_model_covariance": bool(include_model_covariance),
        "min_snr": validate_gui_number(
            min_snr,
            "minimum S/N",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "initial_t0": (
            None
            if initial_t0 is None
            else validate_gui_number(initial_t0, "initial t0")
        ),
        "bounds": normalized_bounds,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_gui_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Validate a finite GUI numeric value with optional bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        invalid = number < minimum if minimum_inclusive else number <= minimum
        if invalid:
            qualifier = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {qualifier} {minimum:g}.")
    if maximum is not None:
        invalid = number > maximum if maximum_inclusive else number >= maximum
        if invalid:
            qualifier = "at most" if maximum_inclusive else "less than"
            raise ValueError(f"{name} must be {qualifier} {maximum:g}.")
    return number


def parse_optional_number(value: str, name: str) -> float | None:
    """Parse an optional finite number from a GUI text field."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric.") from error
    return validate_gui_number(number, name)


def parse_bounds(
    values: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[float, float]] | None:
    """Parse optional fitting bounds, requiring complete ordered pairs."""
    bounds: dict[str, tuple[float, float]] = {}
    for name, (lower_text, upper_text) in values.items():
        lower = parse_optional_number(lower_text, f"{name} lower bound")
        upper = parse_optional_number(upper_text, f"{name} upper bound")
        if lower is None and upper is None:
            continue
        if lower is None or upper is None:
            raise ValueError(f"Both bounds are required for parameter '{name}'.")
        if lower >= upper:
            raise ValueError(
                f"Lower bound for parameter '{name}' must be less than its upper bound."
            )
        bounds[name] = (lower, upper)
    return bounds or None


def basic_analysis_summary(
    peak: PeakResult,
    decline: Dm15Result,
    observation_count: int,
) -> dict[str, Any]:
    """Build a JSON-friendly Basic Analysis summary."""
    return {
        "analysis": "Basic light-curve analysis",
        "observed_peak": _json_ready(peak),
        "linear_15_day_decline_baseline": _json_ready(decline),
        "number_of_observations": int(observation_count),
    }


def basic_summary_text(summary: Mapping[str, Any]) -> str:
    """Create a compact human-readable Basic Analysis report."""
    peak = summary["observed_peak"]
    decline = summary["linear_15_day_decline_baseline"]
    return "\n".join(
        (
            "Supernova Analyzer — Basic Analysis",
            f"Observations: {summary['number_of_observations']}",
            f"Observed peak time: {peak['peak_time']:.6g}",
            f"Observed peak magnitude: {peak['peak_magnitude']:.6g}",
            (
                "Linear 15-day decline baseline: "
                f"{decline['delta_m15']:.6g} mag"
            ),
            (
                "Interpolated magnitude at peak + 15 days: "
                f"{decline['interpolated_magnitude']:.6g}"
            ),
            "",
            "The observed peak is the minimum sampled magnitude.",
            "The decline value is a linear-interpolation baseline.",
        )
    )


def scientific_combined_summary(
    fit: Salt3FitResult,
    diagnostics: Salt3Diagnostics,
    amplitude: Salt3AmplitudeMagnitudeResult,
    synthetic: SyntheticRestFrameMagnitudeResult,
    tripp: TrippStandardizationResult,
) -> str:
    """Create a human-readable scientific fit and standardization report."""
    fit_values = (
        f"t0={fit.t0:.8g}, x0={fit.x0:.8e}, x1={fit.x1:.8g}, c={fit.c:.8g}"
    )
    return "\n".join(
        (
            "Supernova Analyzer — Scientific SALT3 Analysis",
            f"Source: {fit.source_name} {fit.source_version}",
            f"Fixed redshift: {fit.redshift:.8g}",
            f"Fitted parameters: {fit_values}",
            (
                f"Fit quality: chi-square={fit.chi2:.8g}, ndof={fit.ndof}, "
                f"reduced chi-square={fit.reduced_chi2:.8g}"
            ),
            (
                f"Observations used: {fit.n_observations_used}/"
                f"{fit.n_observations_total}"
            ),
            (
                f"Residual mean/RMS: {diagnostics.residual_mean:.8g} / "
                f"{diagnostics.residual_rms:.8g}"
            ),
            (
                "Normalized residual mean/RMS: "
                f"{diagnostics.normalized_residual_mean:.8g} / "
                f"{diagnostics.normalized_residual_rms:.8g}"
            ),
            (
                f"Amplitude m_B: {amplitude.apparent_magnitude:.8g}; "
                f"normalization C={amplitude.normalization_constant:.8g}"
            ),
            (
                "Full synthetic rest-frame B magnitude: "
                f"{synthetic.apparent_magnitude:.8g}"
            ),
            (
                f"Illustrative Tripp mu: {tripp.distance_modulus:.8g}; "
                "fit-statistical uncertainty="
                f"{tripp.distance_modulus_error_statistical}"
            ),
            "",
            "This is an educational single-object analysis, not survey cosmology.",
        )
    )


def figure_to_png_bytes(figure: Figure, *, dpi: int = 150) -> bytes:
    """Render a Matplotlib figure to non-empty PNG bytes without closing it."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    payload = buffer.getvalue()
    if not payload:
        raise ValueError("Figure rendering produced an empty PNG.")
    return payload
