"""Run the DES-Dovekie versus Pantheon+ descriptive residual comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from supernova_analyzer import (
    calculate_hubble_residuals_covariance,
    calculate_hubble_residuals_precision,
    catalog_order_fingerprint,
    compare_catalog_residuals,
    load_des_dovekie_hd,
    load_packed_precision_matrix,
    load_pantheonplus_catalog,
    load_pantheonplus_covariance,
    maximum_residual_difference,
    normalize_des_dovekie_distances,
    plot_cross_catalog_binned_residuals,
    plot_cross_catalog_hubble_residuals,
    plot_cross_catalog_redshift_distribution,
    plot_cross_catalog_residual_distribution,
    select_pantheonplus_hubble_flow_subset,
    subset_pantheonplus_covariance,
)


def _path(value: str | None, environment_name: str) -> Path:
    candidate = value or os.getenv(environment_name)
    if not candidate:
        raise ValueError(f"Provide the path or set {environment_name}.")
    return Path(candidate)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--des-hd")
    parser.add_argument("--des-stat-sys")
    parser.add_argument("--pantheon-data")
    parser.add_argument("--pantheon-stat-sys")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-count", type=int, default=5)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--reference-h0", type=float, default=70.0)
    parser.add_argument("--invariance-reference-h0", type=float, default=80.0)
    parser.add_argument("--output-directory", type=Path, default=Path("outputs"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arguments = _arguments()
    des_hd_path = _path(arguments.des_hd, "DES_DOVEKIE_HD_PATH")
    des_matrix_path = _path(arguments.des_stat_sys, "DES_DOVEKIE_STAT_SYS_PATH")
    pantheon_data_path = _path(arguments.pantheon_data, "PANTHEONPLUS_DATA_PATH")
    pantheon_matrix_path = _path(arguments.pantheon_stat_sys, "PANTHEONPLUS_STAT_SYS_PATH")

    des_catalog = normalize_des_dovekie_distances(load_des_dovekie_hd(des_hd_path))
    des_precision = load_packed_precision_matrix(
        des_matrix_path,
        precision_kind="dovekie_stat_plus_systematic_complete_precision",
        catalog_order_fingerprint=catalog_order_fingerprint(des_catalog),
        release_name=des_catalog.catalog_release,
        composition_description="Complete released DES-Dovekie STAT+SYS precision; unchanged.",
    )
    pantheon_release = load_pantheonplus_catalog(pantheon_data_path)
    pantheon_selection = select_pantheonplus_hubble_flow_subset(pantheon_release)
    pantheon_full_covariance = load_pantheonplus_covariance(
        pantheon_matrix_path,
        pantheon_release,
        covariance_kind="pantheonplus_stat_plus_systematic_complete_covariance",
        composition_description="Complete released Pantheon+ STAT+SYS covariance; unchanged.",
    )
    pantheon_covariance = subset_pantheonplus_covariance(
        pantheon_full_covariance, pantheon_selection
    )

    analysis_options = dict(omega_m=arguments.omega_m, reference_h0=arguments.reference_h0)
    des_result = calculate_hubble_residuals_precision(des_catalog, des_precision, **analysis_options)
    pantheon_result = calculate_hubble_residuals_covariance(
        pantheon_selection.normalized_catalog, pantheon_covariance, **analysis_options
    )
    comparison = compare_catalog_residuals(
        des_result,
        pantheon_result,
        bins=arguments.bins,
        minimum_count_per_bin=arguments.minimum_count,
    )
    des_shifted = calculate_hubble_residuals_precision(
        des_catalog, des_precision, omega_m=arguments.omega_m,
        reference_h0=arguments.invariance_reference_h0,
    )
    pantheon_shifted = calculate_hubble_residuals_covariance(
        pantheon_selection.normalized_catalog, pantheon_covariance,
        omega_m=arguments.omega_m, reference_h0=arguments.invariance_reference_h0,
    )
    des_invariance = maximum_residual_difference(des_result, des_shifted)
    pantheon_invariance = maximum_residual_difference(pantheon_result, pantheon_shifted)
    shifted_comparison = compare_catalog_residuals(
        des_shifted, pantheon_shifted, bins=comparison.bin_edges,
        minimum_count_per_bin=arguments.minimum_count,
    )
    bin_differences = [
        abs(a.descriptive_median_difference - b.descriptive_median_difference)
        for a, b in zip(comparison.bins, shifted_comparison.bins, strict=True)
        if a.descriptive_median_difference is not None
        and b.descriptive_median_difference is not None
    ]
    binned_invariance = max(bin_differences, default=0.0)

    provenance = {
        "des_dovekie": {
            "catalog_release": des_catalog.catalog_release,
            "data_source_path_name": des_hd_path.name,
            "data_source_sha256": _sha256(des_hd_path),
            "matrix_source_path_name": des_matrix_path.name,
            "matrix_source_sha256": des_precision.source_sha256,
            "catalog_order_fingerprint": des_result.catalog_order_fingerprint,
            "analysis_mode": des_result.covariance_mode,
            "fitted_row_count": des_result.number_of_supernovae,
        },
        "pantheonplus": {
            "catalog_release": pantheon_release.catalog_release,
            "data_source_path_name": pantheon_data_path.name,
            "data_source_sha256": _sha256(pantheon_data_path),
            "matrix_source_path_name": pantheon_matrix_path.name,
            "matrix_source_sha256": pantheon_full_covariance.source_sha256,
            "full_catalog_order_fingerprint": pantheon_selection.full_catalog_fingerprint,
            "subset_catalog_order_fingerprint": pantheon_result.catalog_order_fingerprint,
            "full_release_row_count": pantheon_selection.full_release_row_count,
            "retained_source_row_indices": list(pantheon_selection.retained_source_row_indices),
            "analysis_mode": pantheon_result.covariance_mode,
            "fitted_row_count": pantheon_result.number_of_supernovae,
        },
    }
    limitations = [
        "Descriptive residual comparison only; no formal survey-consistency significance.",
        "Each catalog has its own fitted additive nuisance intercept; neither intercept nor their difference is an H0 measurement.",
        "No cross-catalog covariance or joint likelihood was constructed.",
        "No object-level cross-matching was attempted.",
        "No cosmological parameter inference was performed.",
        "Binned 16th–84th percentiles are descriptive distributions, not covariance-aware parameter uncertainties.",
    ]
    payload = {
        "provenance": provenance,
        "reference_cosmology": {"omega_m": arguments.omega_m, "reference_h0": arguments.reference_h0},
        "comparison": comparison.to_dict(),
        "h0_invariance": {
            "alternate_reference_h0": arguments.invariance_reference_h0,
            "des_maximum_absolute_residual_difference": des_invariance,
            "pantheonplus_maximum_absolute_residual_difference": pantheon_invariance,
            "maximum_absolute_binned_difference_change": binned_invariance,
        },
        "scientific_limitations": limitations,
    }
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross_catalog_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "cross_catalog_binned_residuals.json").write_text(
        json.dumps({
            "bin_edges": comparison.bin_edges,
            "minimum_count_per_bin": arguments.minimum_count,
            "difference_definition": "DES-Dovekie median residual minus Pantheon+ median residual; descriptive only",
            "bins": [asdict(item) for item in comparison.bins],
        }, indent=2, allow_nan=False), encoding="utf-8"
    )
    figures = (
        plot_cross_catalog_redshift_distribution(des_result, pantheon_result, output / "cross_catalog_redshift_distribution.png"),
        plot_cross_catalog_hubble_residuals(des_result, pantheon_result, output / "cross_catalog_hubble_residuals.png"),
        plot_cross_catalog_binned_residuals(comparison, output / "cross_catalog_binned_residuals.png"),
        plot_cross_catalog_residual_distribution(des_result, pantheon_result, output / "cross_catalog_residual_distribution.png"),
    )
    for figure in figures:
        plt.close(figure)

    print(json.dumps(payload, indent=2, allow_nan=False))
    print("Scientific limitations:")
    for limitation in limitations:
        print(f"- {limitation}")


if __name__ == "__main__":
    main()
