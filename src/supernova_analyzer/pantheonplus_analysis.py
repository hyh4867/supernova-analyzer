"""Pantheon+-specific covariance validation and descriptive comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from supernova_analyzer.covariance import (
    CovarianceMatrix,
    catalog_order_fingerprint,
    load_flattened_covariance,
    require_positive_definite,
    validate_covariance_alignment,
)
from supernova_analyzer.hubble_analysis import HubbleCovarianceResidualResult
from supernova_analyzer.pantheonplus_io import (
    PantheonPlusCatalog,
    PantheonPlusHubbleFlowSubset,
    normalize_pantheonplus_distances,
)

PANTHEONPLUS_SYMMETRY_RTOL = 1.0e-7
PANTHEONPLUS_SYMMETRY_ATOL = 1.0e-12


@dataclass(frozen=True, slots=True)
class PantheonPlusDiagonalComparison:
    """Descriptive covariance-diagonal comparison without semantic inference."""

    covariance_kind: str
    dimension: int
    mu_error_diagonal_rms_difference: float
    mu_error_diagonal_maximum_absolute_difference: float
    mu_error_diagonal_maximum_relative_difference: float
    mb_error_diagonal_rms_difference: float | None
    mb_error_diagonal_maximum_absolute_difference: float | None
    mb_error_diagonal_maximum_relative_difference: float | None


@dataclass(frozen=True, slots=True)
class PantheonPlusCovarianceComparison:
    """Immutable comparison of two separate released covariance products."""

    first_kind: str
    second_kind: str
    first_intercept: float
    first_intercept_error: float
    second_intercept: float
    second_intercept_error: float
    first_chi2: float
    second_chi2: float
    first_reduced_chi2: float
    second_reduced_chi2: float
    ndof: int
    first_residual_rms: float
    second_residual_rms: float
    maximum_residual_difference: float
    catalog_order_fingerprint: str
    catalog_release: str


def load_pantheonplus_covariance(
    source: str | PathLike[str],
    catalog: PantheonPlusCatalog,
    *,
    covariance_kind: str,
    composition_description: str,
) -> CovarianceMatrix:
    """Load one complete Pantheon+ covariance in sequential row-major order.

    The official release README defines the first token as dimension N and the
    following N*N values as a matrix read sequentially. Product semantics are
    supplied explicitly and never inferred from the filename. No diagonal is
    added and the matrix is not combined with another product.
    """
    normalized = normalize_pantheonplus_distances(catalog)
    covariance = load_flattened_covariance(
        source,
        covariance_kind=covariance_kind,
        catalog_order_fingerprint=catalog_order_fingerprint(normalized),
        release_name=normalized.catalog_release,
        composition_description=composition_description,
        symmetry_rtol=PANTHEONPLUS_SYMMETRY_RTOL,
        symmetry_atol=PANTHEONPLUS_SYMMETRY_ATOL,
    )
    validate_covariance_alignment(covariance, normalized)
    require_positive_definite(covariance)
    return covariance


def subset_pantheonplus_covariance(
    covariance: CovarianceMatrix,
    selection: PantheonPlusHubbleFlowSubset,
) -> CovarianceMatrix:
    """Extract the exact Hubble-flow principal submatrix with provenance."""
    validate_covariance_alignment(covariance, selection.full_catalog)
    position_by_source_row = {
        source_row: position
        for position, source_row in enumerate(selection.full_catalog.source_row_index)
    }
    try:
        positions = tuple(
            position_by_source_row[index]
            for index in selection.retained_source_row_indices
        )
    except KeyError as error:
        raise ValueError(
            "Pantheon+ retained source-row index is absent from the full catalog."
        ) from error
    values = covariance.values[np.ix_(positions, positions)]
    subset = CovarianceMatrix(
        values=values,
        dimension=selection.hubble_flow_row_count,
        covariance_kind=covariance.covariance_kind,
        source_path_name=covariance.source_path_name,
        source_sha256=covariance.source_sha256,
        catalog_order_fingerprint=selection.subset_catalog_fingerprint,
        release_name=covariance.release_name,
        composition_description=(
            "Principal submatrix of the released Pantheon+SH0ES covariance "
            "for IS_CALIBRATOR == 0 source rows; no reordering, "
            "deduplication, or diagonal addition was applied."
        ),
        symmetry_rtol=covariance.symmetry_rtol,
        symmetry_atol=covariance.symmetry_atol,
        parent_source_sha256=covariance.source_sha256,
        original_dimension=covariance.dimension,
        retained_source_row_indices=selection.retained_source_row_indices,
    )
    validate_covariance_alignment(subset, selection.normalized_catalog)
    require_positive_definite(subset)
    return subset


def _relative_maximum(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return a finite maximum relative difference with zero-safe scaling."""
    scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected) / scale))


def compare_pantheonplus_covariance_diagonal(
    covariance: CovarianceMatrix,
    catalog: PantheonPlusCatalog,
) -> PantheonPlusDiagonalComparison:
    """Compare released covariance diagonal with plotting-only error columns.

    This diagnostic does not assert equality and does not alter covariance
    semantics or values.
    """
    normalized = normalize_pantheonplus_distances(catalog)
    validate_covariance_alignment(covariance, normalized)
    frame = catalog.to_pandas()
    diagonal = np.diag(covariance.values)
    mu_variance = np.asarray(frame["MU_SH0ES_ERR_DIAG"], dtype=float) ** 2
    mu_difference = diagonal - mu_variance
    mb_rms: float | None = None
    mb_maximum: float | None = None
    mb_relative: float | None = None
    if "m_b_corr_err_DIAG" in frame:
        mb_variance = np.asarray(frame["m_b_corr_err_DIAG"], dtype=float) ** 2
        mb_difference = diagonal - mb_variance
        if np.isfinite(mb_variance).all():
            mb_rms = float(np.sqrt(np.mean(mb_difference**2)))
            mb_maximum = float(np.max(np.abs(mb_difference)))
            mb_relative = _relative_maximum(diagonal, mb_variance)
    return PantheonPlusDiagonalComparison(
        covariance_kind=covariance.covariance_kind,
        dimension=covariance.dimension,
        mu_error_diagonal_rms_difference=float(
            np.sqrt(np.mean(mu_difference**2))
        ),
        mu_error_diagonal_maximum_absolute_difference=float(
            np.max(np.abs(mu_difference))
        ),
        mu_error_diagonal_maximum_relative_difference=_relative_maximum(
            diagonal,
            mu_variance,
        ),
        mb_error_diagonal_rms_difference=mb_rms,
        mb_error_diagonal_maximum_absolute_difference=mb_maximum,
        mb_error_diagonal_maximum_relative_difference=mb_relative,
    )


def compare_pantheonplus_covariance_results(
    first: HubbleCovarianceResidualResult,
    second: HubbleCovarianceResidualResult,
) -> PantheonPlusCovarianceComparison:
    """Compare separate covariance products without ranking physical quality."""
    if (
        first.object_id != second.object_id
        or first.source_row_index != second.source_row_index
        or first.catalog_order_fingerprint != second.catalog_order_fingerprint
    ):
        raise ValueError(
            "Pantheon+ covariance results must describe the same released rows."
        )
    difference = np.asarray(first.residual) - np.asarray(second.residual)
    return PantheonPlusCovarianceComparison(
        first_kind=first.covariance_mode,
        second_kind=second.covariance_mode,
        first_intercept=first.fitted_intercept,
        first_intercept_error=first.fitted_intercept_error,
        second_intercept=second.fitted_intercept,
        second_intercept_error=second.fitted_intercept_error,
        first_chi2=first.chi2,
        second_chi2=second.chi2,
        first_reduced_chi2=first.reduced_chi2,
        second_reduced_chi2=second.reduced_chi2,
        ndof=first.ndof,
        first_residual_rms=first.residual_rms,
        second_residual_rms=second.residual_rms,
        maximum_residual_difference=float(np.max(np.abs(difference))),
        catalog_order_fingerprint=first.catalog_order_fingerprint,
        catalog_release=first.catalog_release,
    )


def plot_pantheonplus_covariance_comparison(
    first: HubbleCovarianceResidualResult,
    second: HubbleCovarianceResidualResult,
    output_path: str | PathLike[str] | None = None,
) -> Figure:
    """Plot residuals under two separate complete covariance products."""
    compare_pantheonplus_covariance_results(first, second)
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
        s=10,
        alpha=0.55,
        label=first.covariance_mode.replace("_", " "),
    )
    axes[0].scatter(
        redshift,
        second_residual,
        s=10,
        alpha=0.55,
        label=second.covariance_mode.replace("_", " "),
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Residual (mag)")
    axes[0].legend()
    axes[1].scatter(
        redshift,
        second_residual - first_residual,
        s=10,
        alpha=0.6,
        label="STAT+SYS − STATONLY residual",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Redshift (zHD)")
    axes[1].set_ylabel("Difference (mag)")
    axes[1].legend()
    for axis in axes:
        axis.grid(visible=True, alpha=0.25)
    figure.suptitle("Pantheon+ separate covariance-product comparison")
    figure.tight_layout()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination)
    return figure
