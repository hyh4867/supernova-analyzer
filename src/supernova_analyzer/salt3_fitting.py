"""SALT3 light-curve fitting for validated multi-band photometry."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import sncosmo
from astropy.table import Table

from supernova_analyzer.scientific_io import (
    _validated_scientific_photometry_copy,
)

SALT3_SOURCE_NAME = "salt3"
SALT3_SOURCE_VERSION = "2.0"
VARIED_PARAMETER_NAMES = ("t0", "x0", "x1", "c")


class Salt3FitError(RuntimeError):
    """Raised when SALT3 optimization or fit-result validation fails."""


class Salt3ModelUnavailableError(Salt3FitError):
    """Raised when pinned SALT3 model or required bandpass data are unavailable."""


@dataclass(frozen=True, slots=True)
class Salt3FitResult:
    """Immutable, JSON-friendly summary of a SALT3 light-curve fit."""

    source_name: str
    source_version: str
    sncosmo_version: str
    redshift: float
    t0: float
    t0_error: float | None
    x0: float
    x0_error: float | None
    x1: float
    x1_error: float | None
    c: float
    c_error: float | None
    chi2: float
    ndof: int
    reduced_chi2: float
    ncall: int
    nfit: int
    n_observations_total: int
    n_observations_used: int
    varied_parameter_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    data_mask: tuple[bool, ...]
    model_covariance_included: bool
    fit_message: str


def _validated_real(value: object, name: str, *, positive: bool = False) -> float:
    """Return a finite real value or raise a clear input validation error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    if positive and numeric_value <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return numeric_value


def _build_salt3_model(
    source_name: str = SALT3_SOURCE_NAME,
    source_version: str = SALT3_SOURCE_VERSION,
) -> sncosmo.Model:
    """Construct a pinned SALT3 model with cache-aware error reporting."""
    try:
        source = sncosmo.get_source(source_name, version=source_version)
        return sncosmo.Model(source=source)
    except (OSError, RuntimeError, ValueError) as error:
        raise Salt3ModelUnavailableError(
            f"Unable to load {source_name} version {source_version}. Check your "
            "internet connection and the local Astropy/sncosmo cache."
        ) from error


def _result_value(result: object, name: str) -> Any:
    """Read a required sncosmo result attribute with a fit-specific error."""
    try:
        return getattr(result, name)
    except AttributeError:
        try:
            return result[name]  # type: ignore[index]
        except (KeyError, TypeError) as error:
            raise Salt3FitError(
                f"SALT3 fit result is missing required field '{name}'."
            ) from error


def _fitted_parameter(model: sncosmo.Model, name: str) -> float:
    """Read and validate a parameter from the fitted model."""
    if name not in model.param_names:
        raise Salt3FitError(
            f"Fitted SALT3 model is missing required parameter '{name}'."
        )
    value = float(model.get(name))
    if not np.isfinite(value):
        raise Salt3FitError(f"Fitted SALT3 parameter '{name}' is not finite.")
    return value


def _parameter_error(errors: object, name: str) -> float | None:
    """Extract an optional finite, non-negative parameter uncertainty."""
    if errors is None:
        return None
    try:
        value = errors[name]  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    if value is None:
        return None
    uncertainty = float(value)
    if not np.isfinite(uncertainty) or uncertainty < 0.0:
        raise Salt3FitError(
            f"SALT3 uncertainty for parameter '{name}' must be finite and "
            "non-negative."
        )
    return uncertainty


def _immutable_covariance(
    covariance: object,
) -> tuple[tuple[float, ...], ...] | None:
    """Validate and convert optional covariance to immutable tuples."""
    if covariance is None:
        return None
    array = np.asarray(covariance, dtype=float)
    expected_shape = (len(VARIED_PARAMETER_NAMES), len(VARIED_PARAMETER_NAMES))
    if array.shape != expected_shape:
        raise Salt3FitError(
            f"SALT3 covariance has shape {array.shape}; expected {expected_shape}."
        )
    if not np.isfinite(array).all():
        raise Salt3FitError("SALT3 covariance must contain only finite values.")
    return tuple(tuple(float(value) for value in row) for row in array)


def fit_salt3(
    photometry: Table,
    redshift: float,
    include_model_covariance: bool = True,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    initial_t0: float | None = None,
    min_snr: float = 5.0,
) -> Salt3FitResult:
    """Fit pinned SALT3 parameters to validated multi-band flux photometry.

    Redshift is fixed externally. Exactly ``t0``, ``x0``, ``x1``, and ``c`` are
    varied, in that order. The input Table is validated and copied before it is
    passed to sncosmo.

    Args:
        photometry: Multi-band flux photometry with the scientific input schema.
        redshift: Fixed, externally supplied non-negative redshift.
        include_model_covariance: Pass model covariance to ``sncosmo.fit_lc``.
        bounds: Optional parameter bounds forwarded to ``sncosmo.fit_lc``.
        initial_t0: Optional fixed initial value for the fitted ``t0`` parameter.
        min_snr: Positive S/N threshold used for initial parameter guesses.

    Returns:
        An immutable summary of fitted parameters and fit statistics.

    Raises:
        ValueError: If user inputs or photometry are invalid.
        Salt3ModelUnavailableError: If required model or bandpass data cannot load.
        Salt3FitError: If optimization fails or returns an invalid result.
    """
    validated_redshift = _validated_real(redshift, "redshift")
    if validated_redshift < 0.0:
        raise ValueError("redshift must be greater than or equal to zero.")
    validated_min_snr = _validated_real(min_snr, "min_snr", positive=True)
    validated_initial_t0 = (
        None
        if initial_t0 is None
        else _validated_real(initial_t0, "initial_t0")
    )
    fit_data = _validated_scientific_photometry_copy(photometry)

    model = _build_salt3_model()
    model.set(z=validated_redshift)
    if validated_initial_t0 is not None:
        model.set(t0=validated_initial_t0)

    fit_bounds = None if bounds is None else dict(bounds)
    try:
        result, fitted_model = sncosmo.fit_lc(
            fit_data,
            model,
            list(VARIED_PARAMETER_NAMES),
            bounds=fit_bounds,
            guess_amplitude=True,
            guess_t0=validated_initial_t0 is None,
            guess_z=False,
            minsnr=validated_min_snr,
            modelcov=include_model_covariance,
        )
    except OSError as error:
        raise Salt3ModelUnavailableError(
            "Unable to load SALT3 model or bandpass data during fitting. Check "
            "your internet connection and the local Astropy/sncosmo cache."
        ) from error
    except RuntimeError as error:
        raise Salt3FitError(f"SALT3 optimization failed: {error}") from error

    message = str(_result_value(result, "message"))
    if not bool(_result_value(result, "success")):
        raise Salt3FitError(f"SALT3 fit was unsuccessful: {message}")

    result_parameter_names = tuple(str(name) for name in _result_value(result, "param_names"))
    missing_parameters = [
        name for name in VARIED_PARAMETER_NAMES if name not in result_parameter_names
    ]
    if missing_parameters:
        missing = ", ".join(missing_parameters)
        raise Salt3FitError(f"SALT3 fit result is missing parameters: {missing}.")

    fitted_values = {
        name: _fitted_parameter(fitted_model, name)
        for name in VARIED_PARAMETER_NAMES
    }
    chi2 = float(_result_value(result, "chisq"))
    if not np.isfinite(chi2):
        raise Salt3FitError("SALT3 chi-square is not finite.")
    ndof = int(_result_value(result, "ndof"))
    if ndof <= 0:
        raise Salt3FitError("SALT3 fit must have positive degrees of freedom.")

    result_varied_names = tuple(
        str(name) for name in _result_value(result, "vparam_names")
    )
    if result_varied_names != VARIED_PARAMETER_NAMES:
        raise Salt3FitError(
            "SALT3 fit result varied parameters do not match "
            f"{VARIED_PARAMETER_NAMES}."
        )

    raw_mask = np.asarray(_result_value(result, "data_mask"))
    if raw_mask.ndim != 1 or len(raw_mask) != len(fit_data):
        raise Salt3FitError(
            "SALT3 data_mask length must match the total observation count."
        )
    data_mask = tuple(bool(value) for value in raw_mask)
    errors = getattr(result, "errors", None)
    covariance = _immutable_covariance(getattr(result, "covariance", None))

    return Salt3FitResult(
        source_name=SALT3_SOURCE_NAME,
        source_version=SALT3_SOURCE_VERSION,
        sncosmo_version=str(sncosmo.__version__),
        redshift=validated_redshift,
        t0=fitted_values["t0"],
        t0_error=_parameter_error(errors, "t0"),
        x0=fitted_values["x0"],
        x0_error=_parameter_error(errors, "x0"),
        x1=fitted_values["x1"],
        x1_error=_parameter_error(errors, "x1"),
        c=fitted_values["c"],
        c_error=_parameter_error(errors, "c"),
        chi2=chi2,
        ndof=ndof,
        reduced_chi2=chi2 / ndof,
        ncall=int(_result_value(result, "ncall")),
        nfit=int(_result_value(result, "nfit")),
        n_observations_total=len(fit_data),
        n_observations_used=sum(data_mask),
        varied_parameter_names=result_varied_names,
        covariance=covariance,
        data_mask=data_mask,
        model_covariance_included=bool(include_model_covariance),
        fit_message=message,
    )


def build_fitted_salt3_model(result: Salt3FitResult) -> sncosmo.Model:
    """Rebuild a fitted SALT3 model for future plotting and residual analysis.

    A new model instance is returned; the immutable fit result is not modified.
    """
    model = _build_salt3_model(result.source_name, result.source_version)
    model.set(
        z=result.redshift,
        t0=result.t0,
        x0=result.x0,
        x1=result.x1,
        c=result.c,
    )
    return model
