"""Run legacy DES Hubble residual analysis with an explicit full covariance."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh

from supernova_analyzer import (
    calculate_hubble_diagram,
    calculate_hubble_residuals_covariance,
    calculate_hubble_residuals_diagonal,
    catalog_order_fingerprint,
    compare_diagonal_and_covariance_results,
    compose_legacy_des_total_covariance,
    covariance_summary,
    load_des_sn5yr_catalog,
    load_flattened_covariance,
    normalize_des_sn5yr_distances,
    plot_diagonal_vs_covariance_residuals,
    plot_hubble_diagram,
    plot_hubble_residuals,
)


def _arguments() -> argparse.Namespace:
    """Parse explicit catalog and covariance paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare diagonal and full-covariance Hubble residual analyses "
            "for the legacy DES-SN5YR release."
        )
    )
    parser.add_argument("catalog_path", type=Path)
    parser.add_argument("stat_sys_path", type=Path)
    parser.add_argument("--statonly", type=Path)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--reference-h0", type=float, default=70.0)
    return parser.parse_args()


def _load_component(path: Path, catalog: object, kind: str):
    """Load a release component with explicit provenance and semantics."""
    return load_flattened_covariance(
        path,
        covariance_kind=kind,
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description=(
            "Released covariance component; semantics are declared by the "
            "caller and not inferred from the filename."
        ),
    )


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without modifying the source file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run the explicit legacy tutorial composition and write artifacts."""
    arguments = _arguments()
    legacy = load_des_sn5yr_catalog(arguments.catalog_path)
    catalog = normalize_des_sn5yr_distances(legacy)
    fingerprint = catalog_order_fingerprint(catalog)
    component = _load_component(
        arguments.stat_sys_path,
        catalog,
        "legacy_des_stat_plus_systematic_component",
    )
    total = compose_legacy_des_total_covariance(
        component,
        catalog,
        component_convention="official_tutorial_component",
    )
    component_summary = covariance_summary(component)
    total_summary = covariance_summary(total)

    diagonal_diagram = calculate_hubble_diagram(
        catalog,
        omega_m=arguments.omega_m,
        reference_h0=arguments.reference_h0,
    )
    diagonal = calculate_hubble_residuals_diagonal(diagonal_diagram)
    full = calculate_hubble_residuals_covariance(
        catalog,
        total,
        omega_m=arguments.omega_m,
        reference_h0=arguments.reference_h0,
    )
    comparison = compare_diagonal_and_covariance_results(diagonal, full)
    full_diagram = replace(
        diagonal_diagram,
        fitted_reference_distance_modulus=full.fitted_reference_distance_modulus,
        fitted_intercept=full.fitted_intercept,
        fitted_intercept_error=full.fitted_intercept_error,
    )

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    diagram_path = output / "legacy_des_fullcov_hubble_diagram.png"
    residual_path = output / "legacy_des_fullcov_hubble_residuals.png"
    comparison_path = output / "legacy_des_diagonal_vs_fullcov.png"
    summary_path = output / "legacy_des_covariance_summary.json"
    result_path = output / "legacy_des_fullcov_result.json"
    comparison_json = output / "legacy_des_diagonal_fullcov_comparison.json"

    figures = (
        plot_hubble_diagram(
            full_diagram,
            diagram_path,
            analysis_label="full covariance likelihood",
        ),
        plot_hubble_residuals(full, residual_path),
        plot_diagonal_vs_covariance_residuals(
            diagonal,
            full,
            comparison_path,
        ),
    )
    for figure in figures:
        plt.close(figure)
    summary_payload = {
        "component": asdict(component_summary),
        "composed_total": asdict(total_summary),
        "catalog_order_fingerprint": fingerprint,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(asdict(full), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    comparison_json.write_text(
        json.dumps(asdict(comparison), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Catalog: {catalog.catalog_name}")
    print(f"Release: {catalog.catalog_release}")
    print(f"Rows: {len(catalog.object_id)}")
    print(
        "Reference-distance convention: "
        f"{catalog.reference_distance_convention}"
    )
    print(f"Catalog order fingerprint: {fingerprint}")
    print(f"Catalog source SHA-256: {_sha256(arguments.catalog_path)}")
    print(f"STAT+SYS source SHA-256: {component.source_sha256}")
    print(f"Composition: {total.composition_description}")
    print(f"Component summary: {asdict(component_summary)}")
    print(f"Composed total summary: {asdict(total_summary)}")
    if arguments.statonly is not None:
        statonly = _load_component(
            arguments.statonly,
            catalog,
            "legacy_des_statistical_component",
        )
        statonly_summary = covariance_summary(statonly)
        difference = component.values - statonly.values
        minimum_eigenvalue = float(
            eigvalsh(
                difference,
                subset_by_index=(0, 0),
                check_finite=False,
            )[0]
        )
        released_variance = np.asarray(catalog.distance_modulus_error) ** 2
        print(f"STATONLY source SHA-256: {statonly.source_sha256}")
        print(f"STATONLY summary: {asdict(statonly_summary)}")
        print(
            "STAT+SYS - STATONLY minimum eigenvalue: "
            f"{minimum_eigenvalue:.6e}"
        )
        for label, matrix in (("STATONLY", statonly), ("STAT+SYS", component)):
            component_diagonal = np.diag(matrix.values)
            print(
                f"{label} diagonal versus MUERR_FINAL^2 RMS: "
                f"{np.sqrt(np.mean((component_diagonal - released_variance) ** 2)):.6e}"
            )
    print(
        "Diagonal intercept / uncertainty: "
        f"{diagonal_diagram.fitted_intercept:.6f} / "
        f"{diagonal_diagram.fitted_intercept_error:.6f} mag"
    )
    print(
        "Full-covariance intercept / uncertainty: "
        f"{full.fitted_intercept:.6f} / "
        f"{full.fitted_intercept_error:.6f} mag"
    )
    print(f"Diagonal chi-square / ndof: {diagonal.chi2_diagonal:.3f} / {diagonal.ndof}")
    print(f"Full chi-square / ndof: {full.chi2:.3f} / {full.ndof}")
    print(
        "Maximum residual difference: "
        f"{comparison.maximum_residual_difference:.6e} mag"
    )
    print(
        "Warning: this uses a released covariance treatment for residual "
        "analysis; it is not cosmological parameter inference."
    )


if __name__ == "__main__":
    main()
