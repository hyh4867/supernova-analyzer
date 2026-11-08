"""Offline and opt-in tests for descriptive cross-catalog comparison."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from supernova_analyzer.cross_catalog import (
    CrossCatalogError,
    common_redshift_range,
    compare_catalog_residuals,
    maximum_residual_difference,
    summarize_catalog,
)
from supernova_analyzer import (
    calculate_hubble_residuals_covariance,
    calculate_hubble_residuals_precision,
    catalog_order_fingerprint,
    load_des_dovekie_hd,
    load_packed_precision_matrix,
    load_pantheonplus_catalog,
    load_pantheonplus_covariance,
    normalize_des_dovekie_distances,
    select_pantheonplus_hubble_flow_subset,
    subset_pantheonplus_covariance,
)


@dataclass(frozen=True)
class SyntheticResult:
    object_id: tuple[str, ...]
    source_row_index: tuple[int, ...]
    redshift: tuple[float, ...]
    residual: tuple[float, ...]
    fitted_intercept: float = 0.1
    fitted_intercept_error: float = 0.02
    chi2: float = 4.0
    ndof: int = 3
    reduced_chi2: float = 4 / 3
    catalog_name: str = "First"
    catalog_release: str = "Synthetic release"
    covariance_mode: str = "complete precision"
    catalog_order_fingerprint: str = "abc"
    number_of_supernovae: int = 4


def result(name: str = "First", shift: float = 0.0) -> SyntheticResult:
    return SyntheticResult(
        object_id=tuple(f"{name}-{index}" for index in range(4)),
        source_row_index=(7, 3, 9, 1),
        redshift=(0.01, 0.1, 0.5, 1.0),
        residual=tuple(value + shift for value in (-0.2, 0.0, 0.1, 0.3)),
        catalog_name=name,
        catalog_order_fingerprint=f"fingerprint-{name}",
    )


def test_catalog_summary_statistics_and_robust_scatter() -> None:
    summary = summarize_catalog(result())
    assert summary.number_of_fitted_rows == 4
    assert summary.minimum_redshift == 0.01
    assert summary.median_redshift == 0.3
    assert summary.maximum_redshift == 1.0
    assert summary.arithmetic_residual_mean == pytest.approx(0.05)
    assert summary.residual_median == pytest.approx(0.05)
    assert summary.residual_rms == pytest.approx(np.sqrt(0.14 / 4))
    assert summary.residual_mad_scatter == pytest.approx(0.22239)


def test_common_range_bins_percentiles_and_exact_row_preservation() -> None:
    first = result()
    second = replace(
        result("Second", 0.05),
        redshift=(0.05, 0.1, 0.6, 1.2),
    )
    original = (first.object_id, first.source_row_index, first.redshift, first.residual)
    comparison = compare_catalog_residuals(
        first, second, bins=(0.05, 0.2, 1.0), minimum_count_per_bin=1
    )
    assert common_redshift_range(first, second) == (0.05, 1.0)
    assert comparison.first_catalog.number_of_fitted_rows == 4
    assert comparison.second_catalog.number_of_fitted_rows == 4
    assert comparison.first_catalog_rows_in_common_range == 3
    assert comparison.second_catalog_rows_in_common_range == 3
    assert comparison.bins[0].first_catalog.row_count == 1
    assert comparison.bins[0].first_catalog.residual_percentile_16 == 0.0
    assert comparison.bins[0].descriptive_median_difference == pytest.approx(0.05)
    assert original == (first.object_id, first.source_row_index, first.redshift, first.residual)


def test_empty_and_underpopulated_bins_are_explicit() -> None:
    comparison = compare_catalog_residuals(
        result(), result("Second"), bins=(0.01, 0.05, 0.08, 1.0), minimum_count_per_bin=2
    )
    empty = comparison.bins[1]
    assert empty.first_catalog.row_count == 0
    assert empty.first_catalog.median_residual is None
    assert empty.descriptive_median_difference is None
    assert comparison.bins[0].descriptive_median_difference is None


def test_final_common_bin_upper_edge_is_included_exactly_once() -> None:
    first = result()
    second = result("Second")
    comparison = compare_catalog_residuals(
        first,
        second,
        bins=(0.01, 0.5, 1.0),
        minimum_count_per_bin=1,
    )
    assert comparison.bins[0].first_catalog.row_count == 2
    assert comparison.bins[1].first_catalog.row_count == 2
    assert sum(
        item.first_catalog.row_count for item in comparison.bins
    ) == first.number_of_supernovae


def test_results_are_immutable_and_json_safe() -> None:
    comparison = compare_catalog_residuals(result(), result("Second"), bins=2)
    with pytest.raises(FrozenInstanceError):
        comparison.common_redshift_minimum = 0.0  # type: ignore[misc]
    payload = comparison.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False))["comparison_kind"] == "descriptive only"
    assert payload["object_cross_matching_performed"] is False
    assert "object_id" not in json.dumps(payload)


def test_nonoverlap_is_refused() -> None:
    second = replace(result("Second"), redshift=(2.0, 3.0, 4.0, 5.0))
    with pytest.raises(CrossCatalogError, match="overlap"):
        compare_catalog_residuals(result(), second)


def test_independent_intercepts_and_h0_invariance() -> None:
    first = result()
    second = replace(result("Second"), fitted_intercept=-0.7)
    comparison = compare_catalog_residuals(first, second)
    assert comparison.independent_nuisance_intercepts is True
    assert comparison.first_catalog.fitted_nuisance_intercept == 0.1
    assert comparison.second_catalog.fitted_nuisance_intercept == -0.7
    shifted = replace(first, fitted_intercept=1.0)
    assert maximum_residual_difference(first, shifted) == 0.0
    shifted_comparison = compare_catalog_residuals(shifted, second)
    assert [item.descriptive_median_difference for item in comparison.bins] == [
        item.descriptive_median_difference for item in shifted_comparison.bins
    ]
    altered = replace(shifted, residual=(-0.2, 0.0, 0.1, 0.3000000000001))
    assert maximum_residual_difference(first, altered) < 2e-13


def test_order_preservation_is_required_for_invariance_check() -> None:
    first = result()
    reordered = replace(first, object_id=first.object_id[::-1])
    with pytest.raises(CrossCatalogError, match="ordered rows"):
        maximum_residual_difference(first, reordered)


def test_ids_are_never_used_for_cross_matching() -> None:
    first = result()
    second = replace(result("Second"), object_id=first.object_id)
    comparison = compare_catalog_residuals(first, second)
    assert comparison.object_cross_matching_performed is False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_REAL_ENVIRONMENT = (
    "DES_DOVEKIE_HD_PATH",
    "DES_DOVEKIE_STAT_SYS_PATH",
    "PANTHEONPLUS_DATA_PATH",
    "PANTHEONPLUS_STAT_SYS_PATH",
)


@pytest.mark.integration
@pytest.mark.skipif(
    not all(os.getenv(name) for name in _REAL_ENVIRONMENT),
    reason="Set all DES-Dovekie and Pantheon+ STAT+SYS source paths.",
)
def test_real_cross_catalog_comparison() -> None:
    paths = {name: Path(os.environ[name]) for name in _REAL_ENVIRONMENT}
    hashes_before = {name: _sha256(path) for name, path in paths.items()}
    des_catalog = normalize_des_dovekie_distances(
        load_des_dovekie_hd(paths["DES_DOVEKIE_HD_PATH"])
    )
    des_precision = load_packed_precision_matrix(
        paths["DES_DOVEKIE_STAT_SYS_PATH"],
        precision_kind="dovekie_stat_plus_systematic_complete_precision",
        catalog_order_fingerprint=catalog_order_fingerprint(des_catalog),
        release_name=des_catalog.catalog_release,
        composition_description="Complete released DES-Dovekie STAT+SYS precision; unchanged.",
    )
    pantheon_release = load_pantheonplus_catalog(paths["PANTHEONPLUS_DATA_PATH"])
    selection = select_pantheonplus_hubble_flow_subset(pantheon_release)
    full_covariance = load_pantheonplus_covariance(
        paths["PANTHEONPLUS_STAT_SYS_PATH"],
        pantheon_release,
        covariance_kind="pantheonplus_stat_plus_systematic_complete_covariance",
        composition_description="Complete released Pantheon+ STAT+SYS covariance; unchanged.",
    )
    covariance = subset_pantheonplus_covariance(full_covariance, selection)
    des = calculate_hubble_residuals_precision(des_catalog, des_precision)
    pantheon = calculate_hubble_residuals_covariance(
        selection.normalized_catalog, covariance
    )
    comparison = compare_catalog_residuals(des, pantheon)
    shifted_des = calculate_hubble_residuals_precision(
        des_catalog, des_precision, reference_h0=80.0
    )
    shifted_pantheon = calculate_hubble_residuals_covariance(
        selection.normalized_catalog, covariance, reference_h0=80.0
    )
    assert des.number_of_supernovae == 1820
    assert pantheon.number_of_supernovae == 1624
    assert des.catalog_order_fingerprint == catalog_order_fingerprint(des_catalog)
    assert pantheon.catalog_order_fingerprint == selection.subset_catalog_fingerprint
    assert comparison.common_redshift_minimum < comparison.common_redshift_maximum
    for summary in (summarize_catalog(des), summarize_catalog(pantheon)):
        numeric = [
            value
            for value in asdict(summary).values()
            if isinstance(value, (int, float))
        ]
        assert np.isfinite(numeric).all()
    assert maximum_residual_difference(des, shifted_des) < 1e-12
    assert maximum_residual_difference(pantheon, shifted_pantheon) < 1e-12
    shifted_comparison = compare_catalog_residuals(
        shifted_des,
        shifted_pantheon,
        bins=comparison.bin_edges,
    )
    for baseline, shifted in zip(
        comparison.bins, shifted_comparison.bins, strict=True
    ):
        if baseline.descriptive_median_difference is None:
            assert shifted.descriptive_median_difference is None
        else:
            assert shifted.descriptive_median_difference == pytest.approx(
                baseline.descriptive_median_difference,
                abs=1e-12,
            )
    assert hashes_before == {name: _sha256(path) for name, path in paths.items()}
