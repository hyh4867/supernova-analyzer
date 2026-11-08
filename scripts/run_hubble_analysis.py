"""Run diagonal-statistical Hubble analysis for the legacy DES-SN5YR catalog."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from supernova_analyzer import (
    calculate_hubble_diagram,
    calculate_hubble_residuals_diagonal,
    calculate_uncertainty_display_limit,
    load_des_sn5yr_catalog,
    normalize_des_sn5yr_distances,
    plot_hubble_diagram,
    plot_hubble_residuals,
    plot_distance_uncertainty_distribution,
)


def _arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the legacy DES-SN5YR HD + metadata catalog with a "
            "diagonal-statistical reference Hubble diagram."
        )
    )
    parser.add_argument(
        "catalog_path",
        type=Path,
        help="Path to the official DES-SN5YR_HD+MetaData.csv file.",
    )
    parser.add_argument(
        "--omega-m",
        type=float,
        default=0.3,
        help="Fixed reference matter density (default: 0.3).",
    )
    parser.add_argument(
        "--reference-h0",
        type=float,
        default=70.0,
        help="Provisional reference normalization in km/s/Mpc (default: 70).",
    )
    return parser.parse_args()


def main() -> None:
    """Load the legacy adapter, analyze distances, and write result artifacts."""
    arguments = _arguments()
    legacy = load_des_sn5yr_catalog(arguments.catalog_path)
    catalog = normalize_des_sn5yr_distances(legacy)
    diagram = calculate_hubble_diagram(
        catalog,
        omega_m=arguments.omega_m,
        reference_h0=arguments.reference_h0,
    )
    residuals = calculate_hubble_residuals_diagonal(diagram)

    output_directory = Path("outputs")
    output_directory.mkdir(parents=True, exist_ok=True)
    diagram_png = output_directory / "legacy_des_hubble_diagram.png"
    residual_png = output_directory / "legacy_des_hubble_residuals.png"
    uncertainty_png = (
        output_directory / "legacy_des_distance_uncertainties.png"
    )
    diagram_json = output_directory / "legacy_des_hubble_diagram.json"
    residual_json = output_directory / "legacy_des_hubble_residuals.json"

    display_limit = calculate_uncertainty_display_limit(
        diagram.distance_modulus_error
    )
    uncertainty = np.asarray(diagram.distance_modulus_error)
    display_exceeded = int(np.sum(uncertainty > display_limit))
    diagram_figure = plot_hubble_diagram(
        diagram,
        output_path=diagram_png,
        display_error_limit=display_limit,
    )
    residual_figure = plot_hubble_residuals(
        residuals,
        output_path=residual_png,
        display_error_limit=display_limit,
    )
    uncertainty_figure = plot_distance_uncertainty_distribution(
        diagram,
        output_path=uncertainty_png,
    )
    plt.close(diagram_figure)
    plt.close(residual_figure)
    plt.close(uncertainty_figure)
    diagram_json.write_text(
        json.dumps(asdict(diagram), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    residual_json.write_text(
        json.dumps(asdict(residuals), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Catalog: {catalog.catalog_name}")
    print(f"Release: {catalog.catalog_release}")
    print(f"Rows: {diagram.number_of_supernovae}")
    print(
        f"Redshift range ({catalog.redshift_definition}): "
        f"{min(catalog.redshift):.6f} to {max(catalog.redshift):.6f}"
    )
    print(
        f"Fitted nuisance intercept: {diagram.fitted_intercept:.6f} ± "
        f"{diagram.fitted_intercept_error:.6f} mag"
    )
    print(
        f"Residual mean / RMS: {residuals.residual_mean:.6f} / "
        f"{residuals.residual_rms:.6f} mag"
    )
    print(
        f"Weighted residual mean / RMS: "
        f"{residuals.weighted_residual_mean:.6e} / "
        f"{residuals.weighted_residual_rms:.6f} mag"
    )
    print(
        "The weighted residual mean is approximately zero by construction "
        "after fitting the weighted nuisance intercept; reduced chi-square "
        "and residual structure are the fit diagnostics."
    )
    print(
        f"Diagonal chi-square / ndof: {residuals.chi2_diagonal:.3f} / "
        f"{residuals.ndof}"
    )
    percentiles = np.percentile(uncertainty, (50.0, 90.0, 95.0, 99.0))
    print("Distance-modulus uncertainty summary (mag):")
    print(f"  minimum: {np.min(uncertainty):.6f}")
    print(f"  median: {percentiles[0]:.6f}")
    print(f"  p90: {percentiles[1]:.6f}")
    print(f"  p95: {percentiles[2]:.6f}")
    print(f"  p99: {percentiles[3]:.6f}")
    print(f"  maximum: {np.max(uncertainty):.6f}")
    for threshold in (0.5, 1.0, 10.0, 100.0):
        print(
            f"  count above {threshold:g} mag: "
            f"{int(np.sum(uncertainty > threshold))}"
        )
    print(f"Display-only error limit (p95): {display_limit:.6f} mag")
    print(f"Error bars exceeding display range: {display_exceeded}")
    print(
        "All original uncertainties remain in calculations and JSON output."
    )
    print(
        "Warning: this analysis uses diagonal statistical uncertainties only; "
        "it does not include an official survey covariance matrix."
    )


if __name__ == "__main__":
    main()
