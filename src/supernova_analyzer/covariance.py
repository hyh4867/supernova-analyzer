"""Validated covariance loading and composition for population distances."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
from scipy.linalg import LinAlgError, cho_factor

CovarianceSource = str | PathLike[str] | TextIO | BinaryIO
SYMMETRY_RTOL = 1.0e-10
SYMMETRY_ATOL = 1.0e-12


class CovarianceError(ValueError):
    """Raised when covariance content or metadata are invalid."""


class CovarianceAlignmentError(CovarianceError):
    """Raised when covariance provenance does not match catalog ordering."""


class CovarianceNotPositiveDefiniteError(CovarianceError):
    """Raised when a covariance matrix cannot be Cholesky-factorized."""


@dataclass(frozen=True, slots=True)
class CovarianceMatrix:
    """Read-only covariance array with explicit release and composition metadata."""

    values: np.ndarray
    dimension: int
    covariance_kind: str
    source_path_name: str
    source_sha256: str
    catalog_order_fingerprint: str
    release_name: str
    composition_description: str
    symmetry_rtol: float = SYMMETRY_RTOL
    symmetry_atol: float = SYMMETRY_ATOL
    parent_source_sha256: str | None = None
    original_dimension: int | None = None
    retained_source_row_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Copy and validate the matrix, then make its storage read-only."""
        matrix = np.array(self.values, dtype=float, copy=True)
        if matrix.shape != (self.dimension, self.dimension):
            raise CovarianceError(
                "Covariance shape must equal (dimension, dimension)."
            )
        if self.dimension <= 0:
            raise CovarianceError("Covariance dimension must be positive.")
        if not np.isfinite(matrix).all():
            raise CovarianceError("Covariance values must be finite.")
        rtol = float(self.symmetry_rtol)
        atol = float(self.symmetry_atol)
        if not np.isfinite((rtol, atol)).all() or rtol < 0.0 or atol < 0.0:
            raise CovarianceError(
                "Covariance symmetry tolerances must be finite and non-negative."
            )
        difference = float(np.max(np.abs(matrix - matrix.T)))
        scale = float(max(1.0, np.max(np.abs(matrix))))
        if difference > atol + rtol * scale:
            raise CovarianceError(
                "Covariance matrix must be symmetric within "
                f"rtol={rtol:g} and atol={atol:g}."
            )
        for name in (
            "covariance_kind",
            "source_path_name",
            "source_sha256",
            "catalog_order_fingerprint",
            "release_name",
            "composition_description",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CovarianceError(f"{name} must be a non-empty string.")
        matrix.flags.writeable = False
        object.__setattr__(self, "values", matrix)
        if self.original_dimension is not None:
            if self.original_dimension < self.dimension:
                raise CovarianceError(
                    "original_dimension cannot be smaller than dimension."
                )
        if self.retained_source_row_indices is not None:
            retained = tuple(int(value) for value in self.retained_source_row_indices)
            if len(retained) != self.dimension or len(set(retained)) != len(retained):
                raise CovarianceError(
                    "Retained source-row indices must be unique and match dimension."
                )
            object.__setattr__(self, "retained_source_row_indices", retained)


@dataclass(frozen=True, slots=True)
class CovarianceSummary:
    """JSON-friendly numerical summary without the full covariance matrix."""

    dimension: int
    diagonal_min: float
    diagonal_median: float
    diagonal_max: float
    off_diagonal_min: float
    off_diagonal_max: float
    off_diagonal_rms: float
    symmetry_maximum_absolute_difference: float
    cholesky_success: bool
    condition_estimate: float | None
    source_sha256: str
    covariance_kind: str
    composition_description: str
    symmetry_rtol: float
    symmetry_atol: float
    parent_source_sha256: str | None
    original_dimension: int | None


def catalog_order_fingerprint(catalog: object) -> str:
    """Return a stable SHA-256 fingerprint for exact normalized catalog order."""
    payload = {
        "object_id": list(catalog.object_id),
        "source_row_index": list(catalog.source_row_index),
        "catalog_name": catalog.catalog_name,
        "catalog_release": catalog.catalog_release,
        "redshift_definition": catalog.redshift_definition,
        "number_of_objects": len(catalog.object_id),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_bytes(source: CovarianceSource) -> tuple[bytes, str]:
    """Read bytes while restoring a seekable stream position when possible."""
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        try:
            return path.read_bytes(), path.name
        except OSError as error:
            raise CovarianceError(f"Unable to read covariance file: {error}") from error
    position: int | None = None
    try:
        position = source.tell()
    except (AttributeError, OSError):
        pass
    try:
        if position is not None:
            source.seek(0)
        content = source.read()
    except (AttributeError, OSError, TypeError) as error:
        raise CovarianceError(f"Unable to read covariance stream: {error}") from error
    finally:
        if position is not None:
            source.seek(position)
    name = Path(str(getattr(source, "name", "<stream>"))).name
    if isinstance(content, str):
        return content.encode("utf-8"), name
    return bytes(content), name


def load_flattened_covariance(
    source: CovarianceSource,
    *,
    covariance_kind: str,
    catalog_order_fingerprint: str,
    release_name: str,
    composition_description: str,
    symmetry_rtol: float = SYMMETRY_RTOL,
    symmetry_atol: float = SYMMETRY_ATOL,
) -> CovarianceMatrix:
    """Load one dimension token followed by exactly N*N row-major values.

    The caller must declare covariance semantics explicitly. No meaning is
    inferred from the filename.
    """
    raw, source_name = _source_bytes(source)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CovarianceError(f"Unable to decode covariance content: {error}") from error
    tokens = text.split()
    if not tokens:
        raise CovarianceError("Covariance input is empty.")
    try:
        dimension_float = float(tokens[0])
    except ValueError as error:
        raise CovarianceError(
            "The first covariance token must be an integer dimension."
        ) from error
    if not np.isfinite(dimension_float) or not dimension_float.is_integer():
        raise CovarianceError(
            "The first covariance token must be an integer dimension."
        )
    dimension = int(dimension_float)
    if dimension <= 0:
        raise CovarianceError("Covariance dimension must be positive.")
    expected = dimension * dimension
    actual = len(tokens) - 1
    if actual != expected:
        qualifier = "truncated" if actual < expected else "oversized"
        raise CovarianceError(
            f"Covariance data are {qualifier}: expected {expected} values "
            f"after dimension {dimension}, found {actual}."
        )
    try:
        values = np.asarray(tokens[1:], dtype=float)
    except ValueError as error:
        raise CovarianceError("Covariance values must be numeric.") from error
    if not np.isfinite(values).all():
        raise CovarianceError("Covariance values must be finite.")
    return CovarianceMatrix(
        values=values.reshape((dimension, dimension), order="C"),
        dimension=dimension,
        covariance_kind=covariance_kind,
        source_path_name=source_name,
        source_sha256=digest,
        catalog_order_fingerprint=catalog_order_fingerprint,
        release_name=release_name,
        composition_description=composition_description,
        symmetry_rtol=symmetry_rtol,
        symmetry_atol=symmetry_atol,
    )


def validate_covariance_alignment(
    covariance: CovarianceMatrix,
    catalog: object,
) -> None:
    """Require dimension, exact catalog-order fingerprint, and release match."""
    if covariance.dimension != len(catalog.object_id):
        raise CovarianceAlignmentError(
            "Covariance dimension does not match the catalog row count."
        )
    expected = catalog_order_fingerprint(catalog)
    if covariance.catalog_order_fingerprint != expected:
        raise CovarianceAlignmentError(
            "Covariance catalog-order fingerprint does not match the catalog."
        )
    if covariance.release_name != catalog.catalog_release:
        raise CovarianceAlignmentError(
            "Covariance release provenance does not match the catalog release."
        )


def compose_covariance_with_statistical_diagonal(
    component: CovarianceMatrix,
    statistical_uncertainty: tuple[float, ...],
    *,
    covariance_kind: str,
    composition_description: str,
    component_is_released_total: bool = False,
) -> CovarianceMatrix:
    """Return a component plus a statistical variance diagonal exactly once."""
    uncertainty = np.asarray(statistical_uncertainty, dtype=float)
    if uncertainty.shape != (component.dimension,):
        raise CovarianceAlignmentError(
            "Statistical uncertainty length must match covariance dimension."
        )
    if not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0.0):
        raise CovarianceError(
            "Statistical uncertainties must be finite and greater than zero."
        )
    total = np.array(component.values, copy=True)
    if not component_is_released_total:
        total[np.diag_indices(component.dimension)] += uncertainty**2
    if np.any(np.diag(total) <= 0.0):
        raise CovarianceError(
            "Composed total covariance must have a strictly positive diagonal."
        )
    return CovarianceMatrix(
        values=total,
        dimension=component.dimension,
        covariance_kind=covariance_kind,
        source_path_name=component.source_path_name,
        source_sha256=component.source_sha256,
        catalog_order_fingerprint=component.catalog_order_fingerprint,
        release_name=component.release_name,
        composition_description=composition_description,
        symmetry_rtol=component.symmetry_rtol,
        symmetry_atol=component.symmetry_atol,
        parent_source_sha256=component.parent_source_sha256,
        original_dimension=component.original_dimension,
        retained_source_row_indices=component.retained_source_row_indices,
    )


def compose_legacy_des_total_covariance(
    component: CovarianceMatrix,
    catalog: object,
    *,
    component_convention: str,
) -> CovarianceMatrix:
    """Compose the official tutorial covariance treatment for legacy DES.

    ``component_convention="official_tutorial_component"`` adds
    ``MUERR_FINAL**2`` exactly once. ``"verified_released_total"`` preserves
    the supplied matrix without adding a diagonal.
    """
    validate_covariance_alignment(component, catalog)
    if "Legacy DES-SN5YR" not in catalog.catalog_release:
        raise CovarianceAlignmentError(
            "Legacy DES covariance composition requires legacy DES provenance."
        )
    if component_convention == "official_tutorial_component":
        add_diagonal = True
        kind = "legacy_des_stat_plus_systematic_full_covariance"
        description = (
            "Official legacy DES tutorial composition: supplied covariance "
            "component + diag(MUERR_FINAL**2), added exactly once."
        )
    elif component_convention == "verified_released_total":
        add_diagonal = False
        kind = "legacy_des_verified_released_total_full_covariance"
        description = (
            "Caller-verified released total covariance; MUERR_FINAL diagonal "
            "was not added."
        )
    else:
        raise CovarianceError(
            "component_convention must be 'official_tutorial_component' or "
            "'verified_released_total'."
        )
    return compose_covariance_with_statistical_diagonal(
        component,
        catalog.distance_modulus_error,
        covariance_kind=kind,
        composition_description=description,
        component_is_released_total=not add_diagonal,
    )


def covariance_summary(covariance: CovarianceMatrix) -> CovarianceSummary:
    """Summarize covariance structure and test Cholesky factorization.

    An exact condition number is intentionally omitted because it is expensive
    for release-scale dense matrices.
    """
    matrix = covariance.values
    diagonal = np.diag(matrix)
    off_diagonal = np.array(matrix, copy=True)
    np.fill_diagonal(off_diagonal, np.nan)
    count = covariance.dimension * (covariance.dimension - 1)
    off_rms = (
        float(np.sqrt(np.nansum(off_diagonal**2) / count))
        if count
        else 0.0
    )
    try:
        cho_factor(matrix, lower=True, check_finite=False)
        cholesky_success = True
    except LinAlgError:
        cholesky_success = False
    return CovarianceSummary(
        dimension=covariance.dimension,
        diagonal_min=float(np.min(diagonal)),
        diagonal_median=float(np.median(diagonal)),
        diagonal_max=float(np.max(diagonal)),
        off_diagonal_min=(
            float(np.nanmin(off_diagonal)) if count else 0.0
        ),
        off_diagonal_max=(
            float(np.nanmax(off_diagonal)) if count else 0.0
        ),
        off_diagonal_rms=off_rms,
        symmetry_maximum_absolute_difference=float(
            np.max(np.abs(matrix - matrix.T))
        ),
        cholesky_success=cholesky_success,
        condition_estimate=None,
        source_sha256=covariance.source_sha256,
        covariance_kind=covariance.covariance_kind,
        composition_description=covariance.composition_description,
        symmetry_rtol=covariance.symmetry_rtol,
        symmetry_atol=covariance.symmetry_atol,
        parent_source_sha256=covariance.parent_source_sha256,
        original_dimension=covariance.original_dimension,
    )


def require_positive_definite(
    covariance: CovarianceMatrix,
) -> tuple[np.ndarray, bool]:
    """Return a Cholesky factor or raise without adding numerical jitter."""
    if np.any(np.diag(covariance.values) <= 0.0):
        raise CovarianceNotPositiveDefiniteError(
            "Total covariance diagonal must be strictly positive."
        )
    try:
        return cho_factor(covariance.values, lower=True, check_finite=False)
    except LinAlgError as error:
        raise CovarianceNotPositiveDefiniteError(
            "Covariance is not positive definite; no jitter was added."
        ) from error
