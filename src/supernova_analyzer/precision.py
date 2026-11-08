"""Precision-matrix Hubble residual analysis for ordered population catalogs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.linalg import LinAlgError, cho_factor, eigvalsh

from supernova_analyzer.covariance import catalog_order_fingerprint
from supernova_analyzer.hubble_analysis import (
    HubbleAnalysisError,
    NormalizedDistanceCatalog,
    calculate_reference_distance_modulus,
)

SYMMETRY_RTOL = 1.0e-10
SYMMETRY_ATOL = 1.0e-12


class PrecisionMatrixError(ValueError):
    """Raised when precision content, provenance, or alignment is invalid."""


@dataclass(frozen=True, slots=True)
class PrecisionMatrix:
    """Read-only complete inverse covariance with explicit order provenance."""

    values: np.ndarray
    dimension: int
    precision_kind: str
    source_path_name: str
    source_sha256: str
    catalog_order_fingerprint: str
    release_name: str
    composition_description: str
    npz_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Copy, validate, Cholesky-check, and freeze the precision array."""
        matrix = np.array(self.values, dtype=float, copy=True)
        if self.dimension <= 0 or matrix.shape != (
            self.dimension,
            self.dimension,
        ):
            raise PrecisionMatrixError(
                "Precision shape must equal its positive declared dimension."
            )
        if not np.isfinite(matrix).all():
            raise PrecisionMatrixError("Precision values must be finite.")
        difference = float(np.max(np.abs(matrix - matrix.T)))
        scale = float(max(1.0, np.max(np.abs(matrix))))
        if difference > SYMMETRY_ATOL + SYMMETRY_RTOL * scale:
            raise PrecisionMatrixError(
                "Precision matrix must be symmetric within "
                f"rtol={SYMMETRY_RTOL:g} and atol={SYMMETRY_ATOL:g}."
            )
        if np.any(np.diag(matrix) <= 0.0):
            raise PrecisionMatrixError(
                "Precision diagonal must be strictly positive."
            )
        try:
            cho_factor(matrix, lower=True, check_finite=False)
        except LinAlgError as error:
            raise PrecisionMatrixError(
                "Precision matrix is not positive definite; no jitter was added."
            ) from error
        for name in (
            "precision_kind",
            "source_path_name",
            "source_sha256",
            "catalog_order_fingerprint",
            "release_name",
            "composition_description",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PrecisionMatrixError(f"{name} must be a non-empty string.")
        matrix.flags.writeable = False
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "npz_keys", tuple(self.npz_keys))


@dataclass(frozen=True, slots=True)
class PrecisionSummary:
    """JSON-friendly precision summary without the dense matrix."""

    dimension: int
    diagonal_min: float
    diagonal_median: float
    diagonal_max: float
    off_diagonal_min: float
    off_diagonal_max: float
    off_diagonal_rms: float
    symmetry_maximum_absolute_difference: float
    cholesky_success: bool
    minimum_eigenvalue: float | None
    source_sha256: str
    release_name: str
    precision_kind: str
    composition_description: str
    npz_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HubblePrecisionResidualResult:
    """Immutable JSON-friendly precision-weighted residual result."""

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
    profiled_chi2: float
    omega_m: float
    reference_h0: float
    catalog_name: str
    catalog_release: str
    covariance_mode: str
    catalog_order_fingerprint: str
    reference_distance_convention: str
    number_of_supernovae: int


@dataclass(frozen=True, slots=True)
class PrecisionAnalysisComparison:
    """Comparison of two complete precision products without ranking either."""

    first_kind: str
    second_kind: str
    first_intercept: float
    first_intercept_error: float
    second_intercept: float
    second_intercept_error: float
    first_chi2: float
    second_chi2: float
    ndof: int
    first_residual_rms: float
    second_residual_rms: float
    maximum_residual_difference: float
    catalog_order_fingerprint: str
    catalog_release: str


def load_packed_precision_matrix(
    source: str | PathLike[str],
    *,
    precision_kind: str,
    catalog_order_fingerprint: str,
    release_name: str,
    composition_description: str,
) -> PrecisionMatrix:
    """Load an NPZ packed upper-triangular complete precision matrix.

    The archive is always opened with ``allow_pickle=False``. Its stored
    ``allow_pickle`` scalar is retained only as key metadata and never used as
    permission. Precision semantics must be supplied explicitly by the caller.
    """
    path = Path(source)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PrecisionMatrixError(
            f"Unable to read precision file: {error}"
        ) from error
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            required = {"nsn", "cov", "allow_pickle"}
            if set(keys) != required:
                raise PrecisionMatrixError(
                    "Precision NPZ keys must be exactly nsn, cov, and "
                    "allow_pickle."
                )
            nsn = np.array(archive["nsn"], copy=True)
            packed = np.array(archive["cov"], dtype=float, copy=True)
    except (OSError, ValueError) as error:
        if isinstance(error, PrecisionMatrixError):
            raise
        raise PrecisionMatrixError(
            f"Unable to load precision NPZ with allow_pickle=False: {error}"
        ) from error
    if nsn.size != 1:
        raise PrecisionMatrixError("NPZ nsn must contain exactly one value.")
    dimension_float = float(nsn.reshape(-1)[0])
    if (
        not np.isfinite(dimension_float)
        or not dimension_float.is_integer()
        or dimension_float <= 0.0
    ):
        raise PrecisionMatrixError("NPZ nsn must be one positive integer.")
    dimension = int(dimension_float)
    if packed.ndim != 1:
        raise PrecisionMatrixError("NPZ cov must be one-dimensional packed data.")
    expected = dimension * (dimension + 1) // 2
    if len(packed) != expected:
        raise PrecisionMatrixError(
            f"Packed precision length must be {expected} for dimension "
            f"{dimension}; found {len(packed)}."
        )
    if not np.isfinite(packed).all():
        raise PrecisionMatrixError("Packed precision values must be finite.")
    upper = np.triu_indices(dimension)
    matrix = np.zeros((dimension, dimension), dtype=float)
    matrix[upper] = packed
    matrix[(upper[1], upper[0])] = packed
    return PrecisionMatrix(
        values=matrix,
        dimension=dimension,
        precision_kind=precision_kind,
        source_path_name=path.name,
        source_sha256=digest,
        catalog_order_fingerprint=catalog_order_fingerprint,
        release_name=release_name,
        composition_description=composition_description,
        npz_keys=keys,
    )


def validate_precision_alignment(
    precision: PrecisionMatrix,
    catalog: NormalizedDistanceCatalog,
) -> None:
    """Require exact dimension, HD-order fingerprint, and release provenance."""
    if precision.dimension != len(catalog.object_id):
        raise PrecisionMatrixError(
            "Precision dimension does not match the catalog row count."
        )
    expected = catalog_order_fingerprint(catalog)
    if precision.catalog_order_fingerprint != expected:
        raise PrecisionMatrixError(
            "Precision catalog-order fingerprint does not match HD order."
        )
    if precision.release_name != catalog.catalog_release:
        raise PrecisionMatrixError(
            "Precision release provenance does not match the catalog release."
        )


def precision_summary(precision: PrecisionMatrix) -> PrecisionSummary:
    """Summarize a precision product without serializing the dense matrix."""
    matrix = precision.values
    diagonal = np.diag(matrix)
    off_diagonal = np.array(matrix, copy=True)
    np.fill_diagonal(off_diagonal, np.nan)
    count = precision.dimension * (precision.dimension - 1)
    minimum_eigenvalue = (
        float(
            eigvalsh(
                matrix,
                subset_by_index=(0, 0),
                check_finite=False,
            )[0]
        )
        if precision.dimension <= 512
        else None
    )
    return PrecisionSummary(
        dimension=precision.dimension,
        diagonal_min=float(np.min(diagonal)),
        diagonal_median=float(np.median(diagonal)),
        diagonal_max=float(np.max(diagonal)),
        off_diagonal_min=float(np.nanmin(off_diagonal)) if count else 0.0,
        off_diagonal_max=float(np.nanmax(off_diagonal)) if count else 0.0,
        off_diagonal_rms=(
            float(np.sqrt(np.nansum(off_diagonal**2) / count))
            if count
            else 0.0
        ),
        symmetry_maximum_absolute_difference=float(
            np.max(np.abs(matrix - matrix.T))
        ),
        cholesky_success=True,
        minimum_eigenvalue=minimum_eigenvalue,
        source_sha256=precision.source_sha256,
        release_name=precision.release_name,
        precision_kind=precision.precision_kind,
        composition_description=precision.composition_description,
        npz_keys=precision.npz_keys,
    )


def _raw_residual(
    catalog: NormalizedDistanceCatalog,
    omega_m: float,
    reference_h0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference modulus and raw residual under the shared convention."""
    reference = np.asarray(
        calculate_reference_distance_modulus(
            catalog.redshift,
            omega_m=omega_m,
            reference_h0=reference_h0,
            heliocentric_redshift=catalog.heliocentric_redshift,
        )
    )
    return reference, np.asarray(catalog.distance_modulus) - reference


def _precision_intercept_from_raw(
    raw: np.ndarray,
    precision: PrecisionMatrix,
) -> tuple[float, float]:
    """Calculate the analytic nuisance intercept directly with precision P."""
    if len(raw) < 2:
        raise HubbleAnalysisError(
            "At least two supernovae are required for intercept fitting."
        )
    one = np.ones_like(raw)
    denominator = float(one @ precision.values @ one)
    numerator = float(one @ precision.values @ raw)
    if not np.isfinite((denominator, numerator)).all() or denominator <= 0.0:
        raise PrecisionMatrixError(
            "Precision intercept contractions must be finite with positive "
            "normalization."
        )
    return numerator / denominator, float(np.sqrt(1.0 / denominator))


def fit_additive_intercept_precision(
    catalog: NormalizedDistanceCatalog,
    precision: PrecisionMatrix,
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
) -> tuple[float, float]:
    """Fit the additive nuisance intercept directly from a precision matrix."""
    validate_precision_alignment(precision, catalog)
    _, raw = _raw_residual(catalog, omega_m, reference_h0)
    return _precision_intercept_from_raw(raw, precision)


def calculate_profiled_precision_chi2(
    raw: np.ndarray,
    precision: PrecisionMatrix,
) -> float:
    """Return A - B²/C for the intercept-profiled quadratic form only."""
    vector = np.asarray(raw, dtype=float)
    if vector.shape != (precision.dimension,) or not np.isfinite(vector).all():
        raise PrecisionMatrixError(
            "Raw residual must be a finite vector matching precision dimension."
        )
    one = np.ones_like(vector)
    a_value = float(vector @ precision.values @ vector)
    b_value = float(one @ precision.values @ vector)
    c_value = float(one @ precision.values @ one)
    result = a_value - b_value**2 / c_value
    if not np.isfinite(result):
        raise PrecisionMatrixError("Profiled precision chi-square must be finite.")
    return result


def calculate_hubble_residuals_precision(
    catalog: NormalizedDistanceCatalog,
    precision: PrecisionMatrix,
    omega_m: float = 0.3,
    reference_h0: float = 70.0,
) -> HubblePrecisionResidualResult:
    """Calculate precision-weighted residuals without forming an inverse."""
    validate_precision_alignment(precision, catalog)
    reference, raw = _raw_residual(catalog, omega_m, reference_h0)
    intercept, intercept_error = _precision_intercept_from_raw(raw, precision)
    one = np.ones_like(raw)
    residual = raw - intercept * one
    denominator = float(one @ precision.values @ one)
    generalized_mean = float(
        (one @ precision.values @ residual) / denominator
    )
    chi2 = float(residual @ precision.values @ residual)
    profiled_chi2 = calculate_profiled_precision_chi2(raw, precision)
    ndof = len(raw) - 1
    reduced_chi2 = chi2 / ndof
    residual_mean = float(np.mean(residual))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    fitted_reference = reference + intercept
    outputs = (
        *residual,
        *fitted_reference,
        intercept,
        intercept_error,
        generalized_mean,
        chi2,
        profiled_chi2,
        reduced_chi2,
        residual_mean,
        residual_rms,
    )
    if not np.isfinite(outputs).all():
        raise HubbleAnalysisError(
            "Precision-based Hubble residual outputs must be finite."
        )
    return HubblePrecisionResidualResult(
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
        profiled_chi2=profiled_chi2,
        omega_m=float(omega_m),
        reference_h0=float(reference_h0),
        catalog_name=catalog.catalog_name,
        catalog_release=catalog.catalog_release,
        covariance_mode=precision.precision_kind,
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        reference_distance_convention=catalog.reference_distance_convention,
        number_of_supernovae=len(catalog.object_id),
    )


def compare_precision_results(
    first: HubblePrecisionResidualResult,
    second: HubblePrecisionResidualResult,
) -> PrecisionAnalysisComparison:
    """Compare two complete released precision analyses without ranking them."""
    if (
        first.object_id != second.object_id
        or first.source_row_index != second.source_row_index
        or first.catalog_order_fingerprint != second.catalog_order_fingerprint
    ):
        raise PrecisionMatrixError(
            "Precision results must describe the same HD-ordered catalog."
        )
    difference = np.asarray(first.residual) - np.asarray(second.residual)
    return PrecisionAnalysisComparison(
        first_kind=first.covariance_mode,
        second_kind=second.covariance_mode,
        first_intercept=first.fitted_intercept,
        first_intercept_error=first.fitted_intercept_error,
        second_intercept=second.fitted_intercept,
        second_intercept_error=second.fitted_intercept_error,
        first_chi2=first.chi2,
        second_chi2=second.chi2,
        ndof=first.ndof,
        first_residual_rms=first.residual_rms,
        second_residual_rms=second.residual_rms,
        maximum_residual_difference=float(np.max(np.abs(difference))),
        catalog_order_fingerprint=first.catalog_order_fingerprint,
        catalog_release=first.catalog_release,
    )


def plot_precision_residual_comparison(
    first: HubblePrecisionResidualResult,
    second: HubblePrecisionResidualResult,
    output_path: str | PathLike[str] | None = None,
) -> Figure:
    """Plot two precision residual products and their difference."""
    compare_precision_results(first, second)
    redshift = np.asarray(first.redshift)
    first_residual = np.asarray(first.residual)
    second_residual = np.asarray(second.residual)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    axes[0].scatter(
        redshift,
        first_residual,
        s=11,
        alpha=0.6,
        label=first.covariance_mode.replace("_", " "),
    )
    axes[0].scatter(
        redshift,
        second_residual,
        s=11,
        alpha=0.6,
        label=second.covariance_mode.replace("_", " "),
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Residual (mag)")
    axes[0].legend()
    axes[1].scatter(
        redshift,
        second_residual - first_residual,
        s=11,
        alpha=0.65,
        label="STAT+SYS − STATONLY residual",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Redshift (zHD)")
    axes[1].set_ylabel("Difference (mag)")
    axes[1].legend()
    for axis in axes:
        axis.grid(visible=True, alpha=0.25)
    figure.suptitle(
        f"{first.catalog_name}\nComplete precision-product comparison"
    )
    figure.tight_layout()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination)
    return figure
