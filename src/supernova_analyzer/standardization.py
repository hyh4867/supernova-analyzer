"""SALT3 amplitude magnitude, spectral diagnostics, and standardization."""

import hashlib
import json
import warnings
from dataclasses import dataclass
from numbers import Real

import numpy as np

from supernova_analyzer.salt3_fitting import (
    Salt3FitResult,
    build_fitted_salt3_model,
)

REQUIRED_COVARIANCE_PARAMETERS = ("t0", "x0", "x1", "c")
REST_FRAME_PHASE = 0.0
FINITE_DIFFERENCE_SCALE = np.cbrt(np.finfo(float).eps)


class StandardizationError(RuntimeError):
    """Raised when SALT3 magnitude or Tripp-like standardization fails."""


@dataclass(frozen=True, slots=True)
class Salt3AmplitudeMagnitudeResult:
    """Project-derived, x0-only SALT3 amplitude magnitude."""

    source_name: str
    source_version: str
    magnitude_system: str
    reference_band: str
    normalization_constant: float
    x0: float
    apparent_magnitude: float
    apparent_magnitude_error: float | None
    method: str
    fit_fingerprint: str


@dataclass(frozen=True, slots=True)
class SyntheticRestFrameMagnitudeResult:
    """Full synthetic phase-zero magnitude including fitted x0, x1, and c.

    This is a spectral diagnostic and is not the default magnitude for a
    literature-style Tripp relation. Its uncertainty fixes the t0 derivative
    to exactly zero and finite-differences only x0, x1, and c.
    """

    band_name: str
    magnitude_system: str
    phase: float
    observer_time: float
    apparent_magnitude: float
    apparent_magnitude_error: float | None
    method: str
    source_name: str
    source_version: str
    fit_fingerprint: str


# Backward-compatible name for the former full synthetic magnitude result.
Salt3PeakMagnitudeResult = SyntheticRestFrameMagnitudeResult


@dataclass(frozen=True, slots=True)
class TrippStandardizationResult:
    """Illustrative Tripp-like standardization with fit-statistical uncertainty."""

    apparent_magnitude: float
    apparent_magnitude_error: float | None
    x1: float
    x1_error: float | None
    color: float
    color_error: float | None
    alpha: float
    beta: float
    absolute_magnitude: float
    distance_modulus: float
    distance_modulus_error_statistical: float | None
    covariance_included: bool
    formula: str


def _validated_scalar(value: object, name: str) -> float:
    """Return a finite real scalar or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    return numeric_value


def _validated_label(value: object, name: str) -> str:
    """Return a non-empty string label or raise a validation error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _fit_fingerprint(fit_result: Salt3FitResult) -> str:
    """Create a deterministic SHA-256 fingerprint for fit provenance."""
    payload = {
        "source_name": fit_result.source_name,
        "source_version": fit_result.source_version,
        "redshift": float(fit_result.redshift),
        "t0": float(fit_result.t0),
        "x0": float(fit_result.x0),
        "x1": float(fit_result.x1),
        "c": float(fit_result.c),
        "chi2": float(fit_result.chi2),
        "ndof": int(fit_result.ndof),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parameter_values(fit_result: Salt3FitResult) -> dict[str, float]:
    """Return finite fitted parameters and require positive x0."""
    values = {
        "t0": float(fit_result.t0),
        "x0": float(fit_result.x0),
        "x1": float(fit_result.x1),
        "c": float(fit_result.c),
    }
    if not np.isfinite(tuple(values.values())).all():
        raise StandardizationError(
            "Fitted parameters t0, x0, x1, and c must be finite."
        )
    if values["x0"] <= 0.0:
        raise StandardizationError("Fitted SALT3 parameter x0 must be positive.")
    return values


def _covariance_with_names(
    fit_result: Salt3FitResult,
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    """Return finite covariance and its explicit parameter ordering."""
    if fit_result.covariance is None:
        return None
    names = tuple(str(name) for name in fit_result.varied_parameter_names)
    covariance = np.asarray(fit_result.covariance, dtype=float)
    expected_shape = (len(names), len(names))
    if covariance.shape != expected_shape:
        raise StandardizationError(
            f"Fitted covariance has shape {covariance.shape}; expected "
            f"{expected_shape} from varied_parameter_names."
        )
    if not np.isfinite(covariance).all():
        raise StandardizationError("Fitted covariance must contain finite values.")
    if len(set(names)) != len(names):
        raise StandardizationError("varied_parameter_names must be unique.")
    return covariance, names


def _propagated_uncertainty(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    quantity_name: str,
) -> float:
    """Propagate a full covariance matrix through an aligned Jacobian."""
    variance = float(jacobian @ covariance @ jacobian)
    scale = float(np.abs(jacobian) @ np.abs(covariance) @ np.abs(jacobian))
    tolerance = 1.0e-12 * max(1.0, scale)
    if not np.isfinite(variance):
        raise StandardizationError(f"{quantity_name} variance is not finite.")
    if variance < -tolerance:
        raise StandardizationError(f"{quantity_name} variance is negative.")
    return float(np.sqrt(max(variance, 0.0)))


def _reference_normalization_constant(
    fit_result: Salt3FitResult,
    reference_band: str,
    magnitude_system: str,
) -> float:
    """Derive the x0=1 SALT3 magnitude normalization from model photometry."""
    model = build_fitted_salt3_model(fit_result)
    try:
        model.set(z=0.0, t0=0.0, x0=1.0, x1=0.0, c=0.0)
        normalization = float(
            model.bandmag(reference_band, magnitude_system, time=0.0)
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise StandardizationError(
            f"Unable to derive the SALT3 amplitude normalization in band "
            f"'{reference_band}' with magnitude system '{magnitude_system}'. "
            "The requested convention may be invalid or outside model coverage."
        ) from error
    if not np.isfinite(normalization):
        raise StandardizationError(
            "Derived SALT3 amplitude normalization constant is not finite."
        )
    return normalization


def calculate_salt3_amplitude_magnitude(
    fit_result: Salt3FitResult,
    reference_band: str = "bessellb",
    magnitude_system: str = "ab",
) -> Salt3AmplitudeMagnitudeResult:
    """Calculate an x0-only amplitude magnitude for Tripp standardization.

    The project-derived normalization ``C`` is evaluated reproducibly from the
    same SALT3 source/version with ``z=0, t0=0, x0=1, x1=0, c=0`` through the
    requested reference band and magnitude system. The returned magnitude is
    ``-2.5*log10(x0) + C`` and does not use fitted x1 or c.

    If covariance is available, the analytic x0 derivative and the x0 variance
    in the explicit ``varied_parameter_names`` ordering are used. Off-diagonal
    terms cannot contribute because every other derivative is exactly zero.
    """
    validated_band = _validated_label(reference_band, "reference_band")
    validated_magsys = _validated_label(magnitude_system, "magnitude_system")
    parameters = _parameter_values(fit_result)
    normalization = _reference_normalization_constant(
        fit_result,
        validated_band,
        validated_magsys,
    )
    x0 = parameters["x0"]
    magnitude = -2.5 * np.log10(x0) + normalization
    if not np.isfinite(magnitude):
        raise StandardizationError("SALT3 amplitude magnitude is not finite.")

    uncertainty = None
    covariance_details = _covariance_with_names(fit_result)
    if covariance_details is not None:
        covariance, names = covariance_details
        if "x0" not in names:
            raise StandardizationError(
                "Fitted covariance does not contain parameter 'x0'."
            )
        jacobian = np.zeros(len(names), dtype=float)
        jacobian[names.index("x0")] = -2.5 / (np.log(10.0) * x0)
        uncertainty = _propagated_uncertainty(
            jacobian,
            covariance,
            "SALT3 amplitude magnitude",
        )

    return Salt3AmplitudeMagnitudeResult(
        source_name=fit_result.source_name,
        source_version=fit_result.source_version,
        magnitude_system=validated_magsys,
        reference_band=validated_band,
        normalization_constant=float(normalization),
        x0=x0,
        apparent_magnitude=float(magnitude),
        apparent_magnitude_error=uncertainty,
        method=(
            "x0-only SALT amplitude convention with project-derived model "
            "normalization at z=0, t0=0, x0=1, x1=0, c=0"
        ),
        fit_fingerprint=_fit_fingerprint(fit_result),
    )


def _synthetic_magnitude(
    model: object,
    parameters: dict[str, float],
    band_name: str,
    magnitude_system: str,
) -> float:
    """Evaluate full fitted spectral magnitude at rest-frame phase zero."""
    try:
        model.set(
            z=0.0,
            t0=parameters["t0"],
            x0=parameters["x0"],
            x1=parameters["x1"],
            c=parameters["c"],
        )
        magnitude = float(
            model.bandmag(band_name, magnitude_system, parameters["t0"])
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise StandardizationError(
            f"Unable to calculate full synthetic rest-frame magnitude in band "
            f"'{band_name}' with magnitude system '{magnitude_system}'. The "
            "requested convention may be invalid or outside model coverage."
        ) from error
    if not np.isfinite(magnitude):
        raise StandardizationError(
            "Full synthetic rest-frame magnitude is not finite."
        )
    return magnitude


def _spectral_jacobian(
    function,
    parameters: dict[str, float],
    covariance_names: tuple[str, ...],
) -> np.ndarray:
    """Differentiate x0, x1, and c while fixing the absolute t0 derivative."""
    jacobian = np.zeros(len(covariance_names), dtype=float)
    for name in ("x0", "x1", "c"):
        if name not in covariance_names:
            raise StandardizationError(
                f"Fitted covariance does not contain parameter '{name}'."
            )
        value = parameters[name]
        if name == "x0":
            step = min(
                FINITE_DIFFERENCE_SCALE * max(abs(value), 1.0e-4),
                0.5 * value,
            )
        else:
            step = FINITE_DIFFERENCE_SCALE * max(abs(value), 1.0)
        upper = parameters.copy()
        lower = parameters.copy()
        upper[name] += step
        lower[name] -= step
        derivative = (function(upper) - function(lower)) / (2.0 * step)
        jacobian[covariance_names.index(name)] = derivative
    if not np.isfinite(jacobian).all():
        raise StandardizationError(
            "Full synthetic magnitude finite-difference Jacobian is not finite."
        )
    return jacobian


def calculate_synthetic_restframe_b_magnitude(
    fit_result: Salt3FitResult,
    band_name: str = "bessellb",
    magnitude_system: str = "ab",
) -> SyntheticRestFrameMagnitudeResult:
    """Calculate a full synthetic rest-frame phase-zero spectral magnitude.

    This diagnostic uses fitted x0, x1, and c and is not the default input to
    literature-style Tripp standardization. Full-covariance uncertainty fixes
    the absolute t0 derivative to zero and finite-differences only x0, x1, c.
    """
    validated_band = _validated_label(band_name, "band_name")
    validated_magsys = _validated_label(magnitude_system, "magnitude_system")
    parameters = _parameter_values(fit_result)
    model = build_fitted_salt3_model(fit_result)

    def magnitude_function(values: dict[str, float]) -> float:
        return _synthetic_magnitude(
            model,
            values,
            validated_band,
            validated_magsys,
        )

    magnitude = magnitude_function(parameters)
    uncertainty = None
    covariance_details = _covariance_with_names(fit_result)
    if covariance_details is not None:
        covariance, names = covariance_details
        jacobian = _spectral_jacobian(magnitude_function, parameters, names)
        uncertainty = _propagated_uncertainty(
            jacobian,
            covariance,
            "Full synthetic rest-frame magnitude",
        )

    return SyntheticRestFrameMagnitudeResult(
        band_name=validated_band,
        magnitude_system=validated_magsys,
        phase=REST_FRAME_PHASE,
        observer_time=parameters["t0"],
        apparent_magnitude=magnitude,
        apparent_magnitude_error=uncertainty,
        method=(
            "full synthetic rest-frame spectral diagnostic at z=0 and phase 0; "
            "uses fitted x0, x1, c with full-covariance finite differences"
        ),
        source_name=fit_result.source_name,
        source_version=fit_result.source_version,
        fit_fingerprint=_fit_fingerprint(fit_result),
    )


def calculate_salt3_peak_magnitude(
    fit_result: Salt3FitResult,
    band_name: str = "bessellb",
    magnitude_system: str = "ab",
) -> SyntheticRestFrameMagnitudeResult:
    """Deprecated alias for ``calculate_synthetic_restframe_b_magnitude``."""
    warnings.warn(
        "calculate_salt3_peak_magnitude() is deprecated; use "
        "calculate_synthetic_restframe_b_magnitude() for the full spectral "
        "diagnostic or calculate_salt3_amplitude_magnitude() for Tripp input.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calculate_synthetic_restframe_b_magnitude(
        fit_result,
        band_name=band_name,
        magnitude_system=magnitude_system,
    )


def calculate_tripp_standardization(
    fit_result: Salt3FitResult,
    amplitude_magnitude_result: Salt3AmplitudeMagnitudeResult,
    alpha: float,
    beta: float,
    absolute_magnitude: float,
) -> TrippStandardizationResult:
    """Calculate an illustrative Tripp relation from x0-only amplitude m_B.

    The central relation is ``mu = m_B - M + alpha*x1 - beta*c``. Statistical
    uncertainty uses the analytic Jacobian ``[0, dm_B/dx0, alpha, -beta]``
    aligned to ``varied_parameter_names`` and the full fitted covariance once.
    It excludes nuisance calibration uncertainty and other survey systematics.
    """
    if not isinstance(amplitude_magnitude_result, Salt3AmplitudeMagnitudeResult):
        raise TypeError(
            "amplitude_magnitude_result must be a "
            "Salt3AmplitudeMagnitudeResult; full synthetic magnitudes are "
            "diagnostic-only and cannot be used for Tripp standardization."
        )
    validated_alpha = _validated_scalar(alpha, "alpha")
    validated_beta = _validated_scalar(beta, "beta")
    validated_absolute_magnitude = _validated_scalar(
        absolute_magnitude,
        "absolute_magnitude",
    )
    if (
        fit_result.source_name != amplitude_magnitude_result.source_name
        or fit_result.source_version != amplitude_magnitude_result.source_version
    ):
        raise ValueError(
            "fit_result and amplitude_magnitude_result must use the same "
            "source name and version."
        )
    if amplitude_magnitude_result.fit_fingerprint != _fit_fingerprint(fit_result):
        raise ValueError(
            "amplitude_magnitude_result fingerprint does not match fit_result."
        )
    apparent_magnitude = _validated_scalar(
        amplitude_magnitude_result.apparent_magnitude,
        "amplitude_magnitude_result.apparent_magnitude",
    )
    parameters = _parameter_values(fit_result)
    distance_modulus = (
        apparent_magnitude
        - validated_absolute_magnitude
        + validated_alpha * parameters["x1"]
        - validated_beta * parameters["c"]
    )
    if not np.isfinite(distance_modulus):
        raise StandardizationError("Tripp-like distance modulus is not finite.")

    uncertainty = None
    covariance_details = _covariance_with_names(fit_result)
    if covariance_details is not None:
        covariance, names = covariance_details
        missing = [
            name for name in REQUIRED_COVARIANCE_PARAMETERS if name not in names
        ]
        if missing:
            raise StandardizationError(
                "Fitted covariance is missing required parameters: "
                + ", ".join(missing)
                + "."
            )
        jacobian = np.zeros(len(names), dtype=float)
        jacobian[names.index("x0")] = -2.5 / (
            np.log(10.0) * parameters["x0"]
        )
        jacobian[names.index("x1")] = validated_alpha
        jacobian[names.index("c")] = -validated_beta
        uncertainty = _propagated_uncertainty(
            jacobian,
            covariance,
            "Tripp-like distance modulus",
        )

    return TrippStandardizationResult(
        apparent_magnitude=apparent_magnitude,
        apparent_magnitude_error=(
            amplitude_magnitude_result.apparent_magnitude_error
        ),
        x1=parameters["x1"],
        x1_error=fit_result.x1_error,
        color=parameters["c"],
        color_error=fit_result.c_error,
        alpha=validated_alpha,
        beta=validated_beta,
        absolute_magnitude=validated_absolute_magnitude,
        distance_modulus=float(distance_modulus),
        distance_modulus_error_statistical=uncertainty,
        covariance_included=covariance_details is not None,
        formula="mu = m_B - M + alpha*x1 - beta*c",
    )
