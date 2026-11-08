"""Dataset-independent Hubble-diagram and residual analysis."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib.figure import Figure
from scipy.linalg import cho_solve

from supernova_analyzer.covariance import (
    CovarianceAlignmentError,
    CovarianceMatrix,
    catalog_order_fingerprint,
    require_positive_definite,
    validate_covariance_alignment,
)
from supernova_analyzer.population_io import (
    DES_SN5YR_CATALOG_RELEASE,
    DesSn5yrCatalog,
)


class HubbleAnalysisError(ValueError):
    """Raised when normalized population data or analysis outputs are invalid."""


def _finite_real(value: object, name: str) -> float:
    """Return a finite real scalar or raise a clear analysis error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HubbleAnalysisError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise HubbleAnalysisError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class NormalizedDistanceCatalog:
    """Immutable common input representation for population distance releases."""

    object_id: tuple[str, ...]
    redshift: tuple[float, ...]
    distance_modulus: tuple[float, ...]
    distance_modulus_error: tuple[float, ...]
    source_row_index: tuple[int, ...]
    catalog_name: str
    catalog_release: str
    redshift_definition: str
    distance_definition: str
    heliocentric_redshift: tuple[float, ...] | None = None
    reference_distance_convention: str = "redshift only"

    def __post_init__(self) -> None:
        """Validate row alignment and every selected population value."""
        for name in (
            "object_id",
            "redshift",
            "distance_modulus",
            "distance_modulus_error",
            "source_row_index",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise HubbleAnalysisError(
                    f"{name} must be an immutable tuple."
                )
        lengths = {
            len(self.object_id),
            len(self.redshift),
            len(self.distance_modulus),
            len(self.distance_modulus_error),
            len(self.source_row_index),
        }
        if self.heliocentric_redshift is not None:
            if not isinstance(self.heliocentric_redshift, tuple):
                raise HubbleAnalysisError(
                    "heliocentric_redshift must be an immutable tuple when present."
                )
            lengths.add(len(self.heliocentric_redshift))
        if len(lengths) != 1:
            raise HubbleAnalysisError(
                "Normalized distance catalog fields must have equal lengths."
            )
        if not self.object_id:
            raise HubbleAnalysisError(
                "Normalized distance catalog must contain at least one row."
            )
        for name, value in (
            ("catalog_name", self.catalog_name),
            ("catalog_release", self.catalog_release),
            ("redshift_definition", self.redshift_definition),
            ("distance_definition", self.distance_definition),
            ("reference_distance_convention", self.reference_distance_convention),
        ):
            if not isinstance(value, str) or not value.strip():
                raise HubbleAnalysisError(f"{name} must be a non-empty string.")
        for position, object_id in enumerate(self.object_id):
            if not isinstance(object_id, str) or not object_id.strip():
                raise HubbleAnalysisError(
                    f"object_id at position {position} must be non-empty."
                )
        if len(set(self.object_id)) != len(self.object_id):
            raise HubbleAnalysisError("object_id values must be unique.")
        for position, source_index in enumerate(self.source_row_index):
            if isinstance(source_index, bool) or not isinstance(
                source_index, Integral
            ):
                raise HubbleAnalysisError(
                    f"source_row_index at position {position} must be an integer."
                )
        if len(set(self.source_row_index)) != len(self.source_row_index):
            raise HubbleAnalysisError(
                "source_row_index values must provide unique row correspondence."
            )
        for name, values, positive in (
            ("redshift", self.redshift, True),
            ("distance_modulus", self.distance_modulus, False),
            ("distance_modulus_error", self.distance_modulus_error, True),
        ):
            array = np.asarray(values, dtype=float)
            if not np.isfinite(array).all():
                raise HubbleAnalysisError(
                    f"{name} must contain only finite values."
                )
            if positive and np.any(array <= 0.0):
                raise HubbleAnalysisError(
                    f"{name} must contain values greater than zero."
                )
        if self.heliocentric_redshift is not None:
            heliocentric = np.asarray(self.heliocentric_redshift, dtype=float)
            if not np.isfinite(heliocentric).all() or np.any(heliocentric <= 0.0):
                raise HubbleAnalysisError(
                    "heliocentric_redshift must contain finite values greater "
                    "than zero."
                )


@dataclass(frozen=True, slots=True)
class HubbleDiagramResult:
    """Immutable Hubble-diagram result with a fitted nuisance intercept."""

    object_id: tuple[str, ...]
    source_row_index: tuple[int, ...]
    redshift: tuple[float, ...]
    observed_distance_modulus: tuple[float, ...]
    distance_modulus_error: tuple[float, ...]
    provisional_reference_distance_modulus: tuple[float, ...]
    fitted_reference_distance_modulus: tuple[float, ...]
    fitted_intercept: float
    fitted_intercept_error: float
    omega_m: float
    reference_h0: float
    catalog_name: str
    catalog_release: str
    redshift_definition: str
    number_of_supernovae: int
    reference_distance_convention: str = "redshift only"


@dataclass(frozen=True, slots=True)
class HubbleResidualResult:
    """Immutable diagonal-statistical residual summary."""

    object_id: tuple[str, ...]
    source_row_index: tuple[int, ...]
    redshift: tuple[float, ...]
    residual: tuple[float, ...]
    distance_modulus_error: tuple[float, ...]
    fitted_intercept: float
    fitted_intercept_error: float
    weighted_residual_mean: float
    residual_mean: float
    residual_rms: float
    weighted_residual_rms: float
    chi2_diagonal: float
    ndof: int
    reduced_chi2_diagonal: float
    omega_m: float
    reference_h0: float
    catalog_name: str
    catalog_release: str
    covariance_mode: str
    number_of_supernovae: int


@dataclass(frozen=True, slots=True)
class HubbleCovarianceResidualResult:
    """Immutable full-covariance nuisance-intercept residual result."""

    object_id: tuple[str, ...]
    source_row_index: tuple[int, ...]
    redshift: tuple[float, ...]
    residual: tuple[float, ...]
    distance_modulus_error: tuple[float, ...]
    fitted_reference_distance_modulus: tuple[float, ...]
    fitted_intercept: float
    fitted_intercept_error: float
    generalized_residual_mean: float
    residual_mean: float
    residual_rms: float
    chi2: float
    ndof: int
    reduced_chi2: float
    omega_m: float
    reference_h0: float
    catalog_name: str
    catalog_release: str
    covariance_mode: str
    catalog_order_fingerprint: str
    reference_distance_convention: str
    number_of_supernovae: int


@dataclass(frozen=True, slots=True)
class HubbleCovarianceComparison:
    """JSON-friendly comparison of diagonal and full-covariance analyses."""

    diagonal_fitted_intercept: float
    diagonal_fitted_intercept_error: float
    covariance_fitted_intercept: float
    covariance_fitted_intercept_error: float
    diagonal_chi2: float
    diagonal_ndof: int
    covariance_chi2: float
    covariance_ndof: int
    maximum_residual_difference: float
    diagonal_residual_rms: float
    covariance_residual_rms: float
    covariance_mode: str
    catalog_release: str
    catalog_order_fingerprint: str


def normalize_des_sn5yr_distances(
    catalog: DesSn5yrCatalog,
) -> NormalizedDistanceCatalog:
    """Convert the legacy DES-SN5YR catalog using zHD without substitutions.

    Row ordering and original source-row correspondence are preserved. No
    quality cut is applied and the source catalog is not modified.
    """
    frame = catalog.to_pandas()
    required = (
        "object_id",
        "z_hel",
        "z_hd",
        "distance_modulus",
        "distance_modulus_error",
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise HubbleAnalysisError(
            "Legacy DES-SN5YR adapter is missing normalized fields: "
            + ", ".join(missing)
            + "."
        )
    source_indices = catalog.source_row_indices
    if len(source_indices) != len(frame):
        raise HubbleAnalysisError(
            "Legacy DES-SN5YR source-row correspondence is unavailable or "
            "does not match the catalog row count."
        )
    return NormalizedDistanceCatalog(
        object_id=tuple(str(value) for value in frame["object_id"]),
        redshift=tuple(float(value) for value in frame["z_hd"]),
        distance_modulus=tuple(
            float(value) for value in frame["distance_modulus"]
        ),
        distance_modulus_error=tuple(
            float(value) for value in frame["distance_modulus_error"]
        ),
        source_row_index=tuple(int(value) for value in source_indices),
        catalog_name="Legacy DES-SN5YR HD + metadata catalog",
        catalog_release=DES_SN5YR_CATALOG_RELEASE,
        redshift_definition="zHD",
        distance_definition=(
            "Released DES-SN5YR distance modulus MU with MUERR_FINAL"
        ),
        heliocentric_redshift=tuple(float(value) for value in frame["z_hel"]),
        reference_distance_convention=(
            "zHD cosmology redshift with zHEL frame correction"
        ),
    )


def _validated_cosmology(omega_m: object, reference_h0: object) -> tuple[float, float]:
    """Validate fixed reference-cosmology inputs."""
    validated_omega_m = _finite_real(omega_m, "omega_m")
    if not 0.0 < validated_omega_m < 1.0:
        raise HubbleAnalysisError("omega_m must be between zero and one.")
    validated_h0 = _finite_real(reference_h0, "reference_h0")
    if validated_h0 <= 0.0:
        raise HubbleAnalysisError("reference_h0 must be greater than zero.")
    return validated_omega_m, validated_h0


def calculate_reference_distance_modulus(
    redshift: tuple[float, ...],
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
    heliocentric_redshift: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Calculate provisional FlatLambdaCDM distance moduli.

    ``reference_h0`` supplies only a provisional normalization. Supernova
    distances retain an absolute-magnitude/H0 degeneracy, so this function does
    not measure H0. ``omega_m`` is fixed rather than inferred.
    """
    validated_omega_m, validated_h0 = _validated_cosmology(
        omega_m,
        reference_h0,
    )
    values = np.asarray(redshift, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise HubbleAnalysisError("redshift must be a non-empty one-dimensional tuple.")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise HubbleAnalysisError(
            "redshift must contain finite values greater than zero."
        )
    cosmology = FlatLambdaCDM(H0=validated_h0, Om0=validated_omega_m)
    if heliocentric_redshift is None:
        result = np.asarray(cosmology.distmod(values).value, dtype=float)
    else:
        heliocentric = np.asarray(heliocentric_redshift, dtype=float)
        if heliocentric.ndim != 1 or len(heliocentric) != len(values):
            raise HubbleAnalysisError(
                "heliocentric_redshift must match the redshift array length."
            )
        if not np.isfinite(heliocentric).all() or np.any(heliocentric <= 0.0):
            raise HubbleAnalysisError(
                "heliocentric_redshift must contain finite values greater "
                "than zero."
            )
        luminosity_distance = (
            np.asarray(cosmology.luminosity_distance(values).value, dtype=float)
            * (1.0 + heliocentric)
            / (1.0 + values)
        )
        result = 5.0 * np.log10(luminosity_distance) + 25.0
    if not np.isfinite(result).all():
        raise HubbleAnalysisError(
            "Reference distance modulus contains non-finite values."
        )
    return tuple(float(value) for value in result)


def _fit_intercept(
    observed: np.ndarray,
    reference: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[float, float]:
    """Fit the diagonal weighted additive nuisance intercept."""
    if len(observed) < 2:
        raise HubbleAnalysisError(
            "At least two supernovae are required for intercept fitting."
        )
    weights = 1.0 / uncertainty**2
    raw_residual = observed - reference
    intercept = float(np.sum(weights * raw_residual) / np.sum(weights))
    intercept_error = float(1.0 / np.sqrt(np.sum(weights)))
    if not np.isfinite((intercept, intercept_error)).all():
        raise HubbleAnalysisError(
            "Fitted intercept and uncertainty must be finite."
        )
    return intercept, intercept_error


def fit_additive_intercept_diagonal(
    catalog: NormalizedDistanceCatalog,
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
) -> tuple[float, float]:
    """Fit a diagonal-statistical normalization offset, not H0 or absolute M."""
    reference = np.asarray(
        calculate_reference_distance_modulus(
            catalog.redshift,
            omega_m=omega_m,
            reference_h0=reference_h0,
            heliocentric_redshift=catalog.heliocentric_redshift,
        )
    )
    return _fit_intercept(
        np.asarray(catalog.distance_modulus),
        reference,
        np.asarray(catalog.distance_modulus_error),
    )


def calculate_hubble_diagram(
    catalog: NormalizedDistanceCatalog,
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
) -> HubbleDiagramResult:
    """Calculate a fixed-shape Hubble diagram with one fitted offset."""
    validated_omega_m, validated_h0 = _validated_cosmology(
        omega_m,
        reference_h0,
    )
    reference = np.asarray(
        calculate_reference_distance_modulus(
            catalog.redshift,
            omega_m=validated_omega_m,
            reference_h0=validated_h0,
            heliocentric_redshift=catalog.heliocentric_redshift,
        )
    )
    observed = np.asarray(catalog.distance_modulus)
    uncertainty = np.asarray(catalog.distance_modulus_error)
    intercept, intercept_error = _fit_intercept(
        observed,
        reference,
        uncertainty,
    )
    fitted_reference = reference + intercept
    if not np.isfinite(fitted_reference).all():
        raise HubbleAnalysisError(
            "Fitted reference distance modulus must be finite."
        )
    return HubbleDiagramResult(
        object_id=catalog.object_id,
        source_row_index=catalog.source_row_index,
        redshift=catalog.redshift,
        observed_distance_modulus=catalog.distance_modulus,
        distance_modulus_error=catalog.distance_modulus_error,
        provisional_reference_distance_modulus=tuple(
            float(value) for value in reference
        ),
        fitted_reference_distance_modulus=tuple(
            float(value) for value in fitted_reference
        ),
        fitted_intercept=intercept,
        fitted_intercept_error=intercept_error,
        omega_m=validated_omega_m,
        reference_h0=validated_h0,
        catalog_name=catalog.catalog_name,
        catalog_release=catalog.catalog_release,
        redshift_definition=catalog.redshift_definition,
        number_of_supernovae=len(catalog.object_id),
        reference_distance_convention=catalog.reference_distance_convention,
    )


def calculate_hubble_residuals_diagonal(
    diagram: HubbleDiagramResult,
) -> HubbleResidualResult:
    """Calculate diagonal residual statistics after fitting one intercept.

    The weighted residual RMS is
    ``sqrt(sum(w_i * residual_i**2) / sum(w_i))`` with
    ``w_i = 1 / sigma_i**2``. This diagonal statistic does not reproduce an
    official survey likelihood based on a full covariance matrix.

    The weighted residual mean is approximately zero by construction after the
    weighted additive intercept is fitted. That normal-equation consequence is
    not independent evidence of fit quality; reduced chi-square and residual
    structure remain the relevant diagnostics.
    """
    residual = np.asarray(diagram.observed_distance_modulus) - np.asarray(
        diagram.fitted_reference_distance_modulus
    )
    uncertainty = np.asarray(diagram.distance_modulus_error)
    weights = 1.0 / uncertainty**2
    weighted_mean = float(np.sum(weights * residual) / np.sum(weights))
    residual_mean = float(np.mean(residual))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    weighted_rms = float(
        np.sqrt(np.sum(weights * residual**2) / np.sum(weights))
    )
    chi2 = float(np.sum((residual / uncertainty) ** 2))
    ndof = diagram.number_of_supernovae - 1
    if ndof <= 0:
        raise HubbleAnalysisError(
            "At least two supernovae are required for residual statistics."
        )
    reduced_chi2 = chi2 / ndof
    outputs = (
        *residual,
        weighted_mean,
        residual_mean,
        residual_rms,
        weighted_rms,
        chi2,
        reduced_chi2,
    )
    if not np.isfinite(outputs).all():
        raise HubbleAnalysisError("Hubble residual outputs must be finite.")
    return HubbleResidualResult(
        object_id=diagram.object_id,
        source_row_index=diagram.source_row_index,
        redshift=diagram.redshift,
        residual=tuple(float(value) for value in residual),
        distance_modulus_error=diagram.distance_modulus_error,
        fitted_intercept=diagram.fitted_intercept,
        fitted_intercept_error=diagram.fitted_intercept_error,
        weighted_residual_mean=weighted_mean,
        residual_mean=residual_mean,
        residual_rms=residual_rms,
        weighted_residual_rms=weighted_rms,
        chi2_diagonal=chi2,
        ndof=ndof,
        reduced_chi2_diagonal=reduced_chi2,
        omega_m=diagram.omega_m,
        reference_h0=diagram.reference_h0,
        catalog_name=diagram.catalog_name,
        catalog_release=diagram.catalog_release,
        covariance_mode="diagonal_statistical",
        number_of_supernovae=diagram.number_of_supernovae,
    )


def _covariance_intercept_from_raw(
    raw: np.ndarray,
    covariance: CovarianceMatrix,
) -> tuple[float, float, tuple[np.ndarray, bool]]:
    """Fit a nuisance intercept using Cholesky solves without inversion."""
    if len(raw) < 2:
        raise HubbleAnalysisError(
            "At least two supernovae are required for intercept fitting."
        )
    factor = require_positive_definite(covariance)
    one = np.ones_like(raw)
    solved_one = cho_solve(factor, one, check_finite=False)
    solved_raw = cho_solve(factor, raw, check_finite=False)
    denominator = float(one @ solved_one)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise HubbleAnalysisError(
            "Full-covariance intercept denominator must be finite and positive."
        )
    intercept = float((one @ solved_raw) / denominator)
    intercept_error = float(np.sqrt(1.0 / denominator))
    if not np.isfinite((intercept, intercept_error)).all():
        raise HubbleAnalysisError(
            "Full-covariance intercept outputs must be finite."
        )
    return intercept, intercept_error, factor


def fit_additive_intercept_covariance(
    catalog: NormalizedDistanceCatalog,
    covariance: CovarianceMatrix,
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
) -> tuple[float, float]:
    """Fit the additive nuisance intercept using a full covariance matrix."""
    validate_covariance_alignment(covariance, catalog)
    reference = np.asarray(
        calculate_reference_distance_modulus(
            catalog.redshift,
            omega_m=omega_m,
            reference_h0=reference_h0,
            heliocentric_redshift=catalog.heliocentric_redshift,
        )
    )
    raw = np.asarray(catalog.distance_modulus) - reference
    intercept, intercept_error, _ = _covariance_intercept_from_raw(
        raw,
        covariance,
    )
    return intercept, intercept_error


def calculate_hubble_residuals_covariance(
    catalog: NormalizedDistanceCatalog,
    covariance: CovarianceMatrix,
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
) -> HubbleCovarianceResidualResult:
    """Calculate residual statistics using the complete covariance likelihood.

    The generalized residual mean is approximately zero by construction after
    fitting the covariance-weighted intercept. It is a normal-equation
    consequence, not independent evidence of fit quality.
    """
    validate_covariance_alignment(covariance, catalog)
    validated_omega_m, validated_h0 = _validated_cosmology(
        omega_m,
        reference_h0,
    )
    reference = np.asarray(
        calculate_reference_distance_modulus(
            catalog.redshift,
            omega_m=validated_omega_m,
            reference_h0=validated_h0,
            heliocentric_redshift=catalog.heliocentric_redshift,
        )
    )
    observed = np.asarray(catalog.distance_modulus)
    raw = observed - reference
    intercept, intercept_error, factor = _covariance_intercept_from_raw(
        raw,
        covariance,
    )
    one = np.ones_like(raw)
    residual = raw - intercept * one
    solved_residual = cho_solve(factor, residual, check_finite=False)
    solved_one = cho_solve(factor, one, check_finite=False)
    chi2 = float(residual @ solved_residual)
    generalized_mean = float(
        (one @ solved_residual) / (one @ solved_one)
    )
    ndof = len(residual) - 1
    reduced_chi2 = chi2 / ndof
    residual_mean = float(np.mean(residual))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    fitted_reference = reference + intercept
    outputs = (
        *residual,
        *fitted_reference,
        chi2,
        generalized_mean,
        reduced_chi2,
        residual_mean,
        residual_rms,
    )
    if not np.isfinite(outputs).all():
        raise HubbleAnalysisError(
            "Full-covariance Hubble residual outputs must be finite."
        )
    return HubbleCovarianceResidualResult(
        object_id=catalog.object_id,
        source_row_index=catalog.source_row_index,
        redshift=catalog.redshift,
        residual=tuple(float(value) for value in residual),
        distance_modulus_error=catalog.distance_modulus_error,
        fitted_reference_distance_modulus=tuple(
            float(value) for value in fitted_reference
        ),
        fitted_intercept=intercept,
        fitted_intercept_error=intercept_error,
        generalized_residual_mean=generalized_mean,
        residual_mean=residual_mean,
        residual_rms=residual_rms,
        chi2=chi2,
        ndof=ndof,
        reduced_chi2=reduced_chi2,
        omega_m=validated_omega_m,
        reference_h0=validated_h0,
        catalog_name=catalog.catalog_name,
        catalog_release=catalog.catalog_release,
        covariance_mode=covariance.covariance_kind,
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        reference_distance_convention=catalog.reference_distance_convention,
        number_of_supernovae=len(catalog.object_id),
    )


def compare_diagonal_and_covariance_results(
    diagonal: HubbleResidualResult,
    covariance: HubbleCovarianceResidualResult,
) -> HubbleCovarianceComparison:
    """Compare residual results under two uncertainty models without ranking."""
    if (
        diagonal.object_id != covariance.object_id
        or diagonal.source_row_index != covariance.source_row_index
        or diagonal.catalog_release != covariance.catalog_release
    ):
        raise CovarianceAlignmentError(
            "Diagonal and covariance results do not describe the same rows."
        )
    differences = np.asarray(covariance.residual) - np.asarray(
        diagonal.residual
    )
    return HubbleCovarianceComparison(
        diagonal_fitted_intercept=diagonal.fitted_intercept,
        diagonal_fitted_intercept_error=diagonal.fitted_intercept_error,
        covariance_fitted_intercept=covariance.fitted_intercept,
        covariance_fitted_intercept_error=covariance.fitted_intercept_error,
        diagonal_chi2=diagonal.chi2_diagonal,
        diagonal_ndof=diagonal.ndof,
        covariance_chi2=covariance.chi2,
        covariance_ndof=covariance.ndof,
        maximum_residual_difference=float(np.max(np.abs(differences))),
        diagonal_residual_rms=diagonal.residual_rms,
        covariance_residual_rms=covariance.residual_rms,
        covariance_mode=covariance.covariance_mode,
        catalog_release=covariance.catalog_release,
        catalog_order_fingerprint=covariance.catalog_order_fingerprint,
    )
def _save_figure(
    figure: Figure,
    output_path: str | PathLike[str] | None,
) -> None:
    """Save an open figure when an output path is supplied."""
    if output_path is None:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)


def calculate_uncertainty_display_limit(
    distance_modulus_error: tuple[float, ...],
    display_error_limit: float | None = None,
) -> float:
    """Return a deterministic display-only error-bar limit.

    When no explicit limit is supplied, the 95th percentile of the complete
    positive uncertainty array is used. This rule affects only rendered
    error-bar lengths; calculations and serialized uncertainties remain
    unchanged.
    """
    uncertainty = np.asarray(distance_modulus_error, dtype=float)
    if (
        uncertainty.ndim != 1
        or len(uncertainty) == 0
        or not np.isfinite(uncertainty).all()
        or np.any(uncertainty <= 0.0)
    ):
        raise HubbleAnalysisError(
            "distance_modulus_error must contain finite positive values."
        )
    if display_error_limit is None:
        return float(np.percentile(uncertainty, 95.0))
    limit = _finite_real(display_error_limit, "display_error_limit")
    if limit <= 0.0:
        raise HubbleAnalysisError(
            "display_error_limit must be greater than zero."
        )
    return limit


def _display_errors(
    distance_modulus_error: tuple[float, ...],
    display_error_limit: float | None,
) -> tuple[np.ndarray, float, int]:
    """Return visually limited errors plus an explicit exceedance count."""
    uncertainty = np.asarray(distance_modulus_error, dtype=float)
    limit = calculate_uncertainty_display_limit(
        distance_modulus_error,
        display_error_limit,
    )
    exceeded = int(np.sum(uncertainty > limit))
    return np.minimum(uncertainty, limit), limit, exceeded


def _annotate_display_limit(
    axis: object,
    limit: float,
    exceeded: int,
) -> None:
    """State the display-only truncation policy directly on a plot."""
    axis.text(
        0.01,
        0.02,
        (
            f"Display-only error limit: {limit:.3g} mag; "
            f"{exceeded} error bars exceed it. "
            "Full uncertainties are retained in calculations and output."
        ),
        transform=axis.transAxes,
        fontsize="small",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.8",
        },
    )


def plot_hubble_diagram(
    result: HubbleDiagramResult,
    output_path: str | PathLike[str] | None = None,
    logarithmic_redshift: bool = True,
    display_error_limit: float | None = None,
    analysis_label: str | None = None,
) -> Figure:
    """Plot all distances with display-limited error bars and a reference curve."""
    redshift = np.asarray(result.redshift)
    order = np.argsort(redshift)
    display_errors, limit, exceeded = _display_errors(
        result.distance_modulus_error,
        display_error_limit,
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.errorbar(
        redshift,
        result.observed_distance_modulus,
        yerr=display_errors,
        fmt="o",
        markersize=3,
        alpha=0.65,
        capsize=1.5,
        label="Observed distance modulus",
    )
    axis.plot(
        redshift[order],
        np.asarray(result.fitted_reference_distance_modulus)[order],
        linewidth=2,
        label="FlatLambdaCDM reference + nuisance offset",
    )
    if logarithmic_redshift:
        axis.set_xscale("log")
    axis.set_xlabel(f"Redshift ({result.redshift_definition})")
    axis.set_ylabel("Distance modulus")
    title = (
        f"{result.catalog_name}\n{result.catalog_release} | "
        f"nuisance offset={result.fitted_intercept:.4f} ± "
        f"{result.fitted_intercept_error:.4f} mag"
    )
    if analysis_label:
        title += f" | {analysis_label}"
    axis.set_title(title)
    axis.grid(visible=True, alpha=0.25)
    axis.legend()
    _annotate_display_limit(axis, limit, exceeded)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure


def plot_hubble_residuals(
    result: HubbleResidualResult | HubbleCovarianceResidualResult,
    output_path: str | PathLike[str] | None = None,
    logarithmic_redshift: bool = True,
    display_error_limit: float | None = None,
) -> Figure:
    """Plot all residuals with display-limited error bars."""
    display_errors, limit, exceeded = _display_errors(
        result.distance_modulus_error,
        display_error_limit,
    )
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.errorbar(
        result.redshift,
        result.residual,
        yerr=display_errors,
        fmt="o",
        markersize=3,
        alpha=0.65,
        capsize=1.5,
        label="Hubble residual",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    if logarithmic_redshift:
        axis.set_xscale("log")
    axis.set_xlabel("Redshift")
    axis.set_ylabel("Residual (mag)")
    axis.set_title(
        f"{result.catalog_name} residuals | "
        f"{result.covariance_mode.replace('_', ' ')}"
    )
    axis.grid(visible=True, alpha=0.25)
    axis.legend()
    _annotate_display_limit(axis, limit, exceeded)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure


def plot_diagonal_vs_covariance_residuals(
    diagonal: HubbleResidualResult,
    covariance: HubbleCovarianceResidualResult,
    output_path: str | PathLike[str] | None = None,
    logarithmic_redshift: bool = True,
) -> Figure:
    """Plot full-covariance residuals and their difference from diagonal results."""
    compare_diagonal_and_covariance_results(diagonal, covariance)
    redshift = np.asarray(covariance.redshift)
    full_residual = np.asarray(covariance.residual)
    difference = full_residual - np.asarray(diagonal.residual)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    axes[0].scatter(
        redshift,
        full_residual,
        s=12,
        alpha=0.65,
        label="Full-covariance residual",
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Residual (mag)")
    axes[0].legend()
    axes[1].scatter(
        redshift,
        difference,
        s=12,
        alpha=0.65,
        label="Full covariance − diagonal",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Difference (mag)")
    axes[1].set_xlabel("Redshift")
    axes[1].legend()
    if logarithmic_redshift:
        axes[1].set_xscale("log")
    for axis in axes:
        axis.grid(visible=True, alpha=0.25)
    figure.suptitle(
        f"{covariance.catalog_name}\nDiagonal versus full-covariance residuals"
    )
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure


def plot_distance_uncertainty_distribution(
    result: HubbleDiagramResult,
    output_path: str | PathLike[str] | None = None,
) -> Figure:
    """Plot the complete distance-uncertainty distribution on a log x-axis.

    The distribution is descriptive. Its extreme tail is retained and is not
    labeled as erroneous data.
    """
    uncertainty = np.asarray(result.distance_modulus_error, dtype=float)
    calculate_uncertainty_display_limit(result.distance_modulus_error)
    minimum = float(np.min(uncertainty))
    maximum = float(np.max(uncertainty))
    if minimum == maximum:
        bins = np.array([minimum * 0.9, maximum * 1.1])
    else:
        bins = np.geomspace(minimum, maximum, 45)
    statistics = (
        ("Median", float(np.median(uncertainty))),
        ("p95", float(np.percentile(uncertainty, 95.0))),
        ("p99", float(np.percentile(uncertainty, 99.0))),
        ("Maximum", maximum),
    )

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.hist(uncertainty, bins=bins, alpha=0.75, label="Released uncertainties")
    for label, value in statistics:
        axis.axvline(value, linestyle="--", linewidth=1.2, label=f"{label}: {value:.3g}")
    axis.set_xscale("log")
    axis.set_xlabel("Released distance-modulus uncertainty (mag)")
    axis.set_ylabel("Number of supernovae")
    axis.set_title(
        f"{result.catalog_name}\nFull uncertainty distribution; "
        "extreme values are retained"
    )
    axis.grid(visible=True, alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure
