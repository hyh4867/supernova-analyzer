"""Descriptive comparison of independently fitted catalog residuals.

This module deliberately has no knowledge of release file formats or matrix
representations.  It consumes validated residual results produced elsewhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class CrossCatalogError(ValueError):
    """Raised when residual results cannot be compared descriptively."""


class ResidualResult(Protocol):
    """Structural interface shared by validated precision/covariance results."""

    object_id: tuple[str, ...]
    source_row_index: tuple[int, ...]
    redshift: tuple[float, ...]
    residual: tuple[float, ...]
    fitted_intercept: float
    fitted_intercept_error: float
    chi2: float
    ndof: int
    reduced_chi2: float
    catalog_name: str
    catalog_release: str
    covariance_mode: str
    catalog_order_fingerprint: str
    number_of_supernovae: int


@dataclass(frozen=True, slots=True)
class CatalogResidualSummary:
    """Catalog-specific statistics after its own nuisance-intercept fit."""

    catalog_name: str
    analysis_mode: str
    number_of_fitted_rows: int
    minimum_redshift: float
    median_redshift: float
    maximum_redshift: float
    fitted_nuisance_intercept: float
    intercept_uncertainty: float
    catalog_specific_chi2: float
    ndof: int
    catalog_specific_reduced_chi2: float
    arithmetic_residual_mean: float
    residual_median: float
    residual_rms: float
    residual_mad_scatter: float
    catalog_release: str
    catalog_order_fingerprint: str


@dataclass(frozen=True, slots=True)
class RedshiftBinSummary:
    """One descriptive percentile summary; its interval is not uncertainty."""

    lower_edge: float
    upper_edge: float
    representative_redshift: float | None
    row_count: int
    median_residual: float | None
    residual_percentile_16: float | None
    residual_percentile_84: float | None


@dataclass(frozen=True, slots=True)
class CommonBinComparison:
    """Aligned bin summaries and an optional descriptive median difference."""

    lower_edge: float
    upper_edge: float
    first_catalog: RedshiftBinSummary
    second_catalog: RedshiftBinSummary
    descriptive_median_difference: float | None
    minimum_count_required: int


@dataclass(frozen=True, slots=True)
class CrossCatalogComparison:
    """Immutable comparison with full-catalog and overlap-only descriptions."""

    first_catalog: CatalogResidualSummary
    second_catalog: CatalogResidualSummary
    common_redshift_minimum: float
    common_redshift_maximum: float
    first_catalog_rows_in_common_range: int
    second_catalog_rows_in_common_range: int
    bin_edges: tuple[float, ...]
    bins: tuple[CommonBinComparison, ...]
    independent_nuisance_intercepts: bool = True
    object_cross_matching_performed: bool = False
    cross_catalog_covariance_constructed: bool = False
    comparison_kind: str = "descriptive only"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping without any dense uncertainty matrix."""
        return asdict(self)


def _arrays(result: ResidualResult) -> tuple[np.ndarray, np.ndarray]:
    redshift = np.asarray(result.redshift, dtype=float)
    residual = np.asarray(result.residual, dtype=float)
    count = result.number_of_supernovae
    if redshift.ndim != 1 or residual.shape != redshift.shape or len(redshift) != count:
        raise CrossCatalogError("Residual result row fields must have equal lengths.")
    if len(result.object_id) != count or len(result.source_row_index) != count:
        raise CrossCatalogError("Residual result must preserve every fitted row.")
    if count == 0 or not np.isfinite(redshift).all() or not np.isfinite(residual).all():
        raise CrossCatalogError("Residual results must contain finite fitted rows.")
    return redshift, residual


def summarize_catalog(result: ResidualResult) -> CatalogResidualSummary:
    """Summarize a validated result without refitting or reordering it."""
    redshift, residual = _arrays(result)
    scalars = (
        result.fitted_intercept,
        result.fitted_intercept_error,
        result.chi2,
        result.reduced_chi2,
    )
    if not np.isfinite(scalars).all() or result.ndof != len(redshift) - 1:
        raise CrossCatalogError("Residual fit statistics are invalid or misaligned.")
    median = float(np.median(residual))
    return CatalogResidualSummary(
        catalog_name=result.catalog_name,
        analysis_mode=result.covariance_mode,
        number_of_fitted_rows=len(redshift),
        minimum_redshift=float(np.min(redshift)),
        median_redshift=float(np.median(redshift)),
        maximum_redshift=float(np.max(redshift)),
        fitted_nuisance_intercept=float(result.fitted_intercept),
        intercept_uncertainty=float(result.fitted_intercept_error),
        catalog_specific_chi2=float(result.chi2),
        ndof=result.ndof,
        catalog_specific_reduced_chi2=float(result.reduced_chi2),
        arithmetic_residual_mean=float(np.mean(residual)),
        residual_median=median,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_mad_scatter=float(1.4826 * np.median(np.abs(residual - median))),
        catalog_release=result.catalog_release,
        catalog_order_fingerprint=result.catalog_order_fingerprint,
    )


def common_redshift_range(
    first: ResidualResult, second: ResidualResult
) -> tuple[float, float]:
    """Return the closed overlap interval or refuse disjoint catalogs."""
    first_z, _ = _arrays(first)
    second_z, _ = _arrays(second)
    lower = float(max(np.min(first_z), np.min(second_z)))
    upper = float(min(np.max(first_z), np.max(second_z)))
    if not lower < upper:
        raise CrossCatalogError("Catalog redshift ranges do not have a non-zero overlap.")
    return lower, upper


def _validated_edges(
    lower: float, upper: float, bins: int | Sequence[float]
) -> tuple[float, ...]:
    if isinstance(bins, bool):
        raise CrossCatalogError("bins must be an integer or edge sequence.")
    if isinstance(bins, Integral):
        if bins < 1:
            raise CrossCatalogError("bins must be positive.")
        edges = np.linspace(lower, upper, int(bins) + 1)
    else:
        edges = np.asarray(tuple(bins), dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise CrossCatalogError("At least two bin edges are required.")
        if not np.isclose(edges[0], lower) or not np.isclose(edges[-1], upper):
            raise CrossCatalogError("Explicit bin edges must span the exact overlap range.")
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0.0):
        raise CrossCatalogError("Bin edges must be finite and strictly increasing.")
    return tuple(float(value) for value in edges)


def _bin(result: ResidualResult, edges: tuple[float, ...]) -> tuple[RedshiftBinSummary, ...]:
    redshift, residual = _arrays(result)
    summaries = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (redshift >= lower) & (redshift < upper)
        if index == len(edges) - 2:
            mask |= redshift == upper
        z_values = redshift[mask]
        values = residual[mask]
        summaries.append(RedshiftBinSummary(
            lower_edge=lower,
            upper_edge=upper,
            representative_redshift=float(np.median(z_values)) if len(values) else None,
            row_count=int(len(values)),
            median_residual=float(np.median(values)) if len(values) else None,
            residual_percentile_16=float(np.percentile(values, 16)) if len(values) else None,
            residual_percentile_84=float(np.percentile(values, 84)) if len(values) else None,
        ))
    return tuple(summaries)


def compare_catalog_residuals(
    first: ResidualResult,
    second: ResidualResult,
    *,
    bins: int | Sequence[float] = 10,
    minimum_count_per_bin: int = 5,
) -> CrossCatalogComparison:
    """Compare independently fitted residual sets descriptively only."""
    if (
        isinstance(minimum_count_per_bin, bool)
        or not isinstance(minimum_count_per_bin, Integral)
        or minimum_count_per_bin < 1
    ):
        raise CrossCatalogError("minimum_count_per_bin must be a positive integer.")
    minimum_count_per_bin = int(minimum_count_per_bin)
    lower, upper = common_redshift_range(first, second)
    edges = _validated_edges(lower, upper, bins)
    first_bins = _bin(first, edges)
    second_bins = _bin(second, edges)
    paired = []
    for first_bin, second_bin in zip(first_bins, second_bins, strict=True):
        difference = None
        if (
            first_bin.row_count >= minimum_count_per_bin
            and second_bin.row_count >= minimum_count_per_bin
        ):
            assert first_bin.median_residual is not None
            assert second_bin.median_residual is not None
            difference = first_bin.median_residual - second_bin.median_residual
        paired.append(CommonBinComparison(
            lower_edge=first_bin.lower_edge,
            upper_edge=first_bin.upper_edge,
            first_catalog=first_bin,
            second_catalog=second_bin,
            descriptive_median_difference=difference,
            minimum_count_required=minimum_count_per_bin,
        ))
    first_z, _ = _arrays(first)
    second_z, _ = _arrays(second)
    return CrossCatalogComparison(
        first_catalog=summarize_catalog(first),
        second_catalog=summarize_catalog(second),
        common_redshift_minimum=lower,
        common_redshift_maximum=upper,
        first_catalog_rows_in_common_range=int(np.count_nonzero((first_z >= lower) & (first_z <= upper))),
        second_catalog_rows_in_common_range=int(np.count_nonzero((second_z >= lower) & (second_z <= upper))),
        bin_edges=edges,
        bins=tuple(paired),
    )


def maximum_residual_difference(first: ResidualResult, shifted: ResidualResult) -> float:
    """Check H0 invariance while requiring exact fitted-row/order preservation."""
    if (
        first.object_id != shifted.object_id
        or first.source_row_index != shifted.source_row_index
        or first.catalog_order_fingerprint != shifted.catalog_order_fingerprint
    ):
        raise CrossCatalogError("H0-invariance results must describe identical ordered rows.")
    _, first_residual = _arrays(first)
    _, shifted_residual = _arrays(shifted)
    return float(np.max(np.abs(first_residual - shifted_residual)))


def _save(figure: Figure, output_path: str | PathLike[str] | None) -> Figure:
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def plot_cross_catalog_redshift_distribution(first: ResidualResult, second: ResidualResult, output_path: str | PathLike[str] | None = None) -> Figure:
    lower, upper = common_redshift_range(first, second)
    first_z, _ = _arrays(first)
    second_z, _ = _arrays(second)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(first_z, bins=40, histtype="step", linewidth=1.8, label=first.catalog_name)
    axis.hist(second_z, bins=40, histtype="step", linewidth=1.8, label=second.catalog_name)
    axis.axvspan(lower, upper, color="grey", alpha=0.12, label="Common redshift range")
    axis.set(xlabel="Redshift", ylabel="Row count", title="Catalog redshift distributions")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(figure, output_path)


def plot_cross_catalog_hubble_residuals(first: ResidualResult, second: ResidualResult, output_path: str | PathLike[str] | None = None) -> Figure:
    first_z, first_r = _arrays(first)
    second_z, second_r = _arrays(second)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.scatter(first_z, first_r, s=8, alpha=0.35, label=f"{first.catalog_name} rows")
    axis.scatter(second_z, second_r, s=8, alpha=0.25, label=f"{second.catalog_name} rows")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xscale="log", xlabel="Redshift", ylabel="Fitted Hubble residual (mag)", title="Independently fitted catalog residuals")
    axis.text(0.01, 0.02, "Points are raw visualization; intercept fits use each catalog's released matrix.", transform=axis.transAxes, fontsize=8)
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(figure, output_path)


def plot_cross_catalog_binned_residuals(comparison: CrossCatalogComparison, output_path: str | PathLike[str] | None = None) -> Figure:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for key, label, color in (("first_catalog", comparison.first_catalog.catalog_name, "C0"), ("second_catalog", comparison.second_catalog.catalog_name, "C1")):
        valid = [
            getattr(item, key)
            for item in comparison.bins
            if getattr(item, key).row_count >= item.minimum_count_required
        ]
        z = np.asarray([item.representative_redshift for item in valid], dtype=float)
        median = np.asarray([item.median_residual for item in valid], dtype=float)
        lower = median - np.asarray([item.residual_percentile_16 for item in valid], dtype=float)
        upper = np.asarray([item.residual_percentile_84 for item in valid], dtype=float) - median
        axis.errorbar(z, median, yerr=(lower, upper), fmt="o-", capsize=3, color=color, label=label)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xscale="log", xlabel="Median redshift in bin", ylabel="Median residual (mag)", title="Descriptive binned residuals (16th–84th percentiles)")
    axis.text(0.01, 0.02, "Percentile ranges are descriptive, not covariance-aware uncertainties.", transform=axis.transAxes, fontsize=8)
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(figure, output_path)


def plot_cross_catalog_residual_distribution(first: ResidualResult, second: ResidualResult, output_path: str | PathLike[str] | None = None) -> Figure:
    _, first_r = _arrays(first)
    _, second_r = _arrays(second)
    combined = np.concatenate((first_r, second_r))
    edges = np.histogram_bin_edges(combined, bins="auto")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(first_r, bins=edges, density=True, histtype="step", linewidth=1.8, label=first.catalog_name)
    axis.hist(second_r, bins=edges, density=True, histtype="step", linewidth=1.8, label=second.catalog_name)
    axis.set(xlabel="Fitted Hubble residual (mag)", ylabel="Density", title="Residual distributions (descriptive only)")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(figure, output_path)
