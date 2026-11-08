"""Independent regression guards for descriptive cross-catalog analysis."""

from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import numpy as np
import pytest

from supernova_analyzer import (
    CovarianceMatrix,
    NormalizedDistanceCatalog,
    PrecisionMatrix,
    calculate_hubble_residuals_covariance,
    calculate_hubble_residuals_precision,
    catalog_order_fingerprint,
    compare_catalog_residuals,
    maximum_residual_difference,
    summarize_catalog,
)


@dataclass(frozen=True)
class ResidualFixture:
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


def residual_fixture(
    name: str,
    *,
    redshift: tuple[float, ...] = (0.01, 0.25, 0.5, 0.75, 1.0),
    residual: tuple[float, ...] = (-0.4, -0.1, 0.0, 0.2, 0.5),
    object_ids: tuple[str, ...] | None = None,
    intercept: float = 0.0,
) -> ResidualFixture:
    count = len(redshift)
    return ResidualFixture(
        object_id=object_ids or tuple(f"{name}-{index}" for index in range(count)),
        source_row_index=tuple(range(10, 10 + count)),
        redshift=redshift,
        residual=residual,
        fitted_intercept=intercept,
        fitted_intercept_error=0.03,
        chi2=float(count - 1),
        ndof=count - 1,
        reduced_chi2=1.0,
        catalog_name=name,
        catalog_release=f"Synthetic {name}",
        covariance_mode=f"independent {name} uncertainty",
        catalog_order_fingerprint=f"fingerprint-{name}",
        number_of_supernovae=count,
    )


def test_summary_preserves_full_catalog_row_count_and_input_order() -> None:
    first = residual_fixture("DES")
    before = (first.object_id, first.source_row_index, first.redshift, first.residual)

    summary = summarize_catalog(first)
    compare_catalog_residuals(first, residual_fixture("Pantheon"), bins=3)

    assert summary.number_of_fitted_rows == first.number_of_supernovae
    assert before == (first.object_id, first.source_row_index, first.redshift, first.residual)


def test_overlap_rows_are_conserved_and_each_belongs_to_exactly_one_bin() -> None:
    first = residual_fixture("DES")
    second = residual_fixture(
        "Pantheon",
        redshift=(0.2, 0.25, 0.5, 0.75, 0.9),
    )
    comparison = compare_catalog_residuals(first, second, bins=(0.2, 0.5, 0.9))

    assert comparison.first_catalog_rows_in_common_range == 3
    assert comparison.second_catalog_rows_in_common_range == 5
    assert sum(item.first_catalog.row_count for item in comparison.bins) == 3
    assert sum(item.second_catalog.row_count for item in comparison.bins) == 5
    # The internal edge (0.5) is assigned to the upper bin, not both bins.
    assert [item.second_catalog.row_count for item in comparison.bins] == [2, 3]


def test_final_upper_edge_is_included_once_and_bin_count_is_configurable() -> None:
    comparison = compare_catalog_residuals(
        residual_fixture("DES"), residual_fixture("Pantheon"), bins=4
    )

    assert len(comparison.bins) == 4
    assert len(comparison.bin_edges) == 5
    assert comparison.bins[-1].first_catalog.row_count == 1
    assert comparison.bins[-1].first_catalog.representative_redshift == 1.0
    assert sum(item.first_catalog.row_count for item in comparison.bins) == 5


def test_minimum_count_controls_difference_suppression_and_sign() -> None:
    first = residual_fixture("DES", residual=(0.2, 0.2, 0.2, 0.2, 0.2))
    second = residual_fixture("Pantheon", residual=(0.1, 0.1, 0.1, 0.1, 0.1))

    retained = compare_catalog_residuals(
        first, second, bins=1, minimum_count_per_bin=5
    )
    suppressed = compare_catalog_residuals(
        first, second, bins=1, minimum_count_per_bin=6
    )

    assert retained.bins[0].descriptive_median_difference == pytest.approx(0.1)
    assert retained.bins[0].minimum_count_required == 5
    assert suppressed.bins[0].descriptive_median_difference is None
    assert suppressed.bins[0].minimum_count_required == 6


def test_mad_scatter_uses_scaled_median_absolute_deviation() -> None:
    values = (-3.0, -1.0, 0.0, 1.0, 9.0)
    summary = summarize_catalog(residual_fixture("DES", residual=values))

    expected = 1.4826 * np.median(np.abs(np.asarray(values) - np.median(values)))
    assert summary.residual_mad_scatter == pytest.approx(expected)


def test_comparison_is_deeply_immutable_and_json_serializable() -> None:
    comparison = compare_catalog_residuals(
        residual_fixture("DES"), residual_fixture("Pantheon"), bins=2
    )

    with pytest.raises(FrozenInstanceError):
        comparison.bins[0].lower_edge = -1.0  # type: ignore[misc]
    encoded = json.dumps(comparison.to_dict(), allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["first_catalog"]["number_of_fitted_rows"] == 5
    assert "object_id" not in encoded


def test_ids_are_not_intersected_and_distinct_intercepts_are_preserved() -> None:
    first = residual_fixture(
        "DES", object_ids=("same", "d1", "d2", "d3", "d4"), intercept=1.25
    )
    second = residual_fixture(
        "Pantheon",
        object_ids=("same", "p1", "p2", "p3", "p4"),
        intercept=-0.75,
    )
    comparison = compare_catalog_residuals(first, second, bins=1)

    assert comparison.first_catalog.number_of_fitted_rows == 5
    assert comparison.second_catalog.number_of_fitted_rows == 5
    assert comparison.object_cross_matching_performed is False
    assert comparison.independent_nuisance_intercepts is True
    assert comparison.first_catalog.fitted_nuisance_intercept == 1.25
    assert comparison.second_catalog.fitted_nuisance_intercept == -0.75


def _catalog(name: str, offset: float) -> NormalizedDistanceCatalog:
    redshift = (0.02, 0.08, 0.2, 0.45, 0.8)
    # Deliberately non-constant perturbations leave non-trivial final residuals.
    distance = tuple(
        base + offset + perturbation
        for base, perturbation in zip(
            (34.7, 37.8, 40.0, 42.0, 43.5),
            (-0.12, 0.03, 0.18, -0.04, 0.09),
            strict=True,
        )
    )
    return NormalizedDistanceCatalog(
        object_id=tuple(f"{name}-{index}" for index in range(5)),
        redshift=redshift,
        heliocentric_redshift=redshift,
        distance_modulus=distance,
        distance_modulus_error=(0.1, 0.12, 0.11, 0.13, 0.14),
        source_row_index=(4, 1, 8, 2, 7),
        catalog_name=name,
        catalog_release=f"Synthetic {name}",
        redshift_definition="synthetic zHD",
        distance_definition="synthetic distance modulus",
    )


def _precision(catalog: NormalizedDistanceCatalog) -> PrecisionMatrix:
    values = np.asarray(
        [
            [120.0, 8.0, 0.0, 0.0, 0.0],
            [8.0, 80.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 100.0, 4.0, 0.0],
            [0.0, 0.0, 4.0, 70.0, 3.0],
            [0.0, 0.0, 0.0, 3.0, 55.0],
        ]
    )
    return PrecisionMatrix(
        values=values,
        dimension=5,
        precision_kind="synthetic complete precision",
        source_path_name="synthetic-precision",
        source_sha256="synthetic-precision-sha256",
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description="Synthetic positive-definite precision.",
    )


def _covariance(catalog: NormalizedDistanceCatalog) -> CovarianceMatrix:
    values = np.asarray(
        [
            [0.020, 0.003, 0.0, 0.0, 0.0],
            [0.003, 0.030, 0.004, 0.0, 0.0],
            [0.0, 0.004, 0.025, 0.002, 0.0],
            [0.0, 0.0, 0.002, 0.035, 0.003],
            [0.0, 0.0, 0.0, 0.003, 0.040],
        ]
    )
    return CovarianceMatrix(
        values=values,
        dimension=5,
        covariance_kind="synthetic complete covariance",
        source_path_name="synthetic-covariance",
        source_sha256="synthetic-covariance-sha256",
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description="Synthetic positive-definite covariance.",
    )


def test_arithmetic_mean_differs_from_generalized_weighted_mean() -> None:
    catalog = _catalog("DES", 0.0)
    fitted = calculate_hubble_residuals_precision(catalog, _precision(catalog))

    assert fitted.generalized_residual_mean == pytest.approx(0.0, abs=1e-12)
    assert abs(fitted.residual_mean) > 1e-4
    assert summarize_catalog(fitted).arithmetic_residual_mean == pytest.approx(
        fitted.residual_mean
    )


def test_both_pipelines_and_binned_difference_are_h0_invariant() -> None:
    des_catalog = _catalog("DES", 0.0)
    pantheon_catalog = _catalog("Pantheon", 0.25)
    des = calculate_hubble_residuals_precision(des_catalog, _precision(des_catalog))
    pantheon = calculate_hubble_residuals_covariance(
        pantheon_catalog, _covariance(pantheon_catalog)
    )
    shifted_des = calculate_hubble_residuals_precision(
        des_catalog, _precision(des_catalog), reference_h0=83.0
    )
    shifted_pantheon = calculate_hubble_residuals_covariance(
        pantheon_catalog, _covariance(pantheon_catalog), reference_h0=83.0
    )

    assert maximum_residual_difference(des, shifted_des) < 1e-12
    assert maximum_residual_difference(pantheon, shifted_pantheon) < 1e-12
    baseline = compare_catalog_residuals(des, pantheon, bins=2, minimum_count_per_bin=1)
    shifted = compare_catalog_residuals(
        shifted_des, shifted_pantheon, bins=baseline.bin_edges, minimum_count_per_bin=1
    )
    assert [item.descriptive_median_difference for item in shifted.bins] == pytest.approx(
        [item.descriptive_median_difference for item in baseline.bins], abs=1e-12
    )


def test_cross_catalog_layer_has_no_out_of_scope_scientific_calls() -> None:
    """AST guard allows plotting/numpy while excluding inferential machinery."""
    module_path = Path(__file__).parents[1] / "src/supernova_analyzer/cross_catalog.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    forbidden_imports = {
        "supernova_analyzer.covariance",
        "supernova_analyzer.precision",
        "supernova_analyzer.hubble_analysis",
        "scipy.stats",
    }
    forbidden_calls = {
        "calculate_hubble_residuals_covariance",
        "fit_additive_intercept_covariance",
        "fit_additive_intercept_precision",
        "load_pantheonplus_covariance",
        "ks_2samp",
        "curve_fit",
        "minimize",
    }
    assert imported_modules.isdisjoint(forbidden_imports)
    assert called_names.isdisjoint(forbidden_calls)

    comparison = compare_catalog_residuals(
        residual_fixture("DES"), residual_fixture("Pantheon"), bins=2
    )
    assert comparison.cross_catalog_covariance_constructed is False
    assert comparison.object_cross_matching_performed is False
    assert comparison.comparison_kind == "descriptive only"
    assert not hasattr(comparison, "h0")
    assert not hasattr(comparison, "omega_m")
    assert not hasattr(comparison, "w")
    assert not hasattr(comparison, "significance")
