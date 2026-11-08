"""Run DES-Dovekie Hubble analysis using released precision matrices."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from supernova_analyzer import (
    calculate_hubble_diagram,
    calculate_hubble_residuals_precision,
    catalog_order_fingerprint,
    compare_precision_results,
    join_dovekie_metadata_to_hd_order,
    load_des_dovekie_hd,
    load_des_dovekie_metadata,
    load_packed_precision_matrix,
    normalize_des_dovekie_distances,
    plot_hubble_diagram,
    plot_hubble_residuals,
    plot_precision_residual_comparison,
    precision_summary,
)


def _arguments() -> argparse.Namespace:
    """Parse explicit current-release input paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DES-Dovekie distances with complete released precision "
            "matrices and one profiled nuisance intercept."
        )
    )
    parser.add_argument("hd_path", type=Path)
    parser.add_argument("stat_sys_path", type=Path)
    parser.add_argument("--statonly", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--reference-h0", type=float, default=70.0)
    return parser.parse_args()


def _load_precision(
    path: Path,
    catalog: object,
    kind: str,
):
    """Load one complete product with explicit semantics and HD fingerprint."""
    return load_packed_precision_matrix(
        path,
        precision_kind=kind,
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description=(
            "Complete released DES-Dovekie inverse covariance; no matrix "
            "combination or MUERR diagonal addition was applied."
        ),
    )


def main() -> None:
    """Run current-release precision analyses and write ignored artifacts."""
    arguments = _arguments()
    hd = load_des_dovekie_hd(arguments.hd_path)
    catalog = normalize_des_dovekie_distances(hd)
    fingerprint = catalog_order_fingerprint(catalog)
    statsys = _load_precision(
        arguments.stat_sys_path,
        catalog,
        "dovekie_stat_plus_systematic_complete_precision",
    )
    statsys_result = calculate_hubble_residuals_precision(
        catalog,
        statsys,
        omega_m=arguments.omega_m,
        reference_h0=arguments.reference_h0,
    )
    statsys_summary = precision_summary(statsys)

    metadata_count: int | None = None
    order_mismatch_count: int | None = None
    if arguments.metadata is not None:
        metadata = load_des_dovekie_metadata(arguments.metadata)
        joined = join_dovekie_metadata_to_hd_order(hd, metadata)
        metadata_count = metadata.row_count
        order_mismatch_count = joined.order_mismatch_count

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    statsys_diagram_path = output / "dovekie_statsys_hubble_diagram.png"
    statsys_residual_path = output / "dovekie_statsys_hubble_residuals.png"
    statsys_result_path = output / "dovekie_statsys_result.json"
    summary_path = output / "dovekie_precision_summary.json"
    base_diagram = calculate_hubble_diagram(
        catalog,
        omega_m=arguments.omega_m,
        reference_h0=arguments.reference_h0,
    )
    statsys_diagram = replace(
        base_diagram,
        fitted_reference_distance_modulus=(
            statsys_result.fitted_reference_distance_modulus
        ),
        fitted_intercept=statsys_result.fitted_intercept,
        fitted_intercept_error=statsys_result.fitted_intercept_error,
    )
    figures = [
        plot_hubble_diagram(
            statsys_diagram,
            statsys_diagram_path,
            analysis_label="STAT+SYS complete precision",
        ),
        plot_hubble_residuals(statsys_result, statsys_residual_path),
    ]
    statsys_result_path.write_text(
        json.dumps(asdict(statsys_result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_payload: dict[str, object] = {
        "catalog_order_fingerprint": fingerprint,
        "stat_plus_systematic": asdict(statsys_summary),
    }

    statonly_result = None
    statonly_matrix = None
    statonly_summary = None
    comparison = None
    if arguments.statonly is not None:
        statonly_matrix = _load_precision(
            arguments.statonly,
            catalog,
            "dovekie_statonly_complete_precision",
        )
        statonly_result = calculate_hubble_residuals_precision(
            catalog,
            statonly_matrix,
            omega_m=arguments.omega_m,
            reference_h0=arguments.reference_h0,
        )
        statonly_summary = precision_summary(statonly_matrix)
        comparison = compare_precision_results(
            statonly_result,
            statsys_result,
        )
        statonly_diagram = replace(
            base_diagram,
            fitted_reference_distance_modulus=(
                statonly_result.fitted_reference_distance_modulus
            ),
            fitted_intercept=statonly_result.fitted_intercept,
            fitted_intercept_error=statonly_result.fitted_intercept_error,
        )
        figures.extend(
            (
                plot_hubble_diagram(
                    statonly_diagram,
                    output / "dovekie_statonly_hubble_diagram.png",
                    analysis_label="STATONLY complete precision",
                ),
                plot_hubble_residuals(
                    statonly_result,
                    output / "dovekie_statonly_hubble_residuals.png",
                ),
                plot_precision_residual_comparison(
                    statonly_result,
                    statsys_result,
                    output / "dovekie_statonly_statsys_comparison.png",
                ),
            )
        )
        (output / "dovekie_statonly_result.json").write_text(
            json.dumps(asdict(statonly_result), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (output / "dovekie_precision_comparison.json").write_text(
            json.dumps(asdict(comparison), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        summary_payload["statonly"] = asdict(statonly_summary)
    for figure in figures:
        plt.close(figure)
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    statsys_h0 = calculate_hubble_residuals_precision(
        catalog,
        statsys,
        omega_m=arguments.omega_m,
        reference_h0=80.0,
    )
    h0_difference = float(
        np.max(
            np.abs(
                np.asarray(statsys_result.residual)
                - np.asarray(statsys_h0.residual)
            )
        )
    )
    print(f"Catalog: {catalog.catalog_name}")
    print(f"Release: {catalog.catalog_release}")
    print(f"HD rows: {hd.row_count}")
    if metadata_count is not None:
        print(f"Metadata rows: {metadata_count}")
        print("CID sets identical: true")
        print(f"Metadata/HD order mismatch count: {order_mismatch_count}")
    print(f"Catalog order fingerprint: {fingerprint}")
    print(
        f"STAT+SYS precision: dimension={statsys.dimension}, "
        f"SHA-256={statsys.source_sha256}"
    )
    print(f"STAT+SYS summary: {asdict(statsys_summary)}")
    print(
        "STAT+SYS intercept / uncertainty: "
        f"{statsys_result.fitted_intercept:.6f} / "
        f"{statsys_result.fitted_intercept_error:.6f} mag"
    )
    print(
        f"STAT+SYS chi-square / ndof: {statsys_result.chi2:.3f} / "
        f"{statsys_result.ndof}"
    )
    if statonly_result is not None and comparison is not None:
        assert statonly_matrix is not None
        assert statonly_summary is not None
        print(
            f"STATONLY precision: dimension={statonly_matrix.dimension}, "
            f"SHA-256={statonly_matrix.source_sha256}"
        )
        print(f"STATONLY summary: {asdict(statonly_summary)}")
        print(
            "STATONLY intercept / uncertainty: "
            f"{statonly_result.fitted_intercept:.6f} / "
            f"{statonly_result.fitted_intercept_error:.6f} mag"
        )
        print(
            f"STATONLY chi-square / ndof: {statonly_result.chi2:.3f} / "
            f"{statonly_result.ndof}"
        )
        print(
            "Maximum STATONLY/STAT+SYS residual difference: "
            f"{comparison.maximum_residual_difference:.6e} mag"
        )
        print(
            "The larger STAT+SYS intercept uncertainty reflects reduced "
            "effective information from correlated systematics and does not "
            "mean the physical model is worse."
        )
    print(
        "Reference-H0 invariance maximum residual difference: "
        f"{h0_difference:.3e}"
    )
    print(
        "STATONLY and STAT+SYS are separate complete precision products; "
        "neither was inverted, combined, or augmented with MUERR."
    )
    print(
        "No H0 or dark-energy parameter is inferred by this residual analysis."
    )


if __name__ == "__main__":
    main()
