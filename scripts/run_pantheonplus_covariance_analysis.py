"""Run Pantheon+SH0ES residual analysis with released covariance products."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from supernova_analyzer import (
    calculate_hubble_diagram,
    calculate_hubble_residuals_covariance,
    compare_pantheonplus_covariance_diagonal,
    compare_pantheonplus_covariance_results,
    covariance_summary,
    load_pantheonplus_catalog,
    load_pantheonplus_covariance,
    plot_hubble_diagram,
    plot_hubble_residuals,
    plot_pantheonplus_covariance_comparison,
    select_pantheonplus_hubble_flow_subset,
    subset_pantheonplus_covariance,
)


def _arguments() -> argparse.Namespace:
    """Parse explicit user-supplied Pantheon+ release paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Pantheon+SH0ES rows with separate complete released "
            "covariance products and one nuisance intercept."
        )
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("stat_sys_path", type=Path)
    parser.add_argument("--statonly", type=Path)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--reference-h0", type=float, default=70.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Hash a source file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_covariance(path: Path, catalog: object, kind: str):
    """Load one explicitly labeled complete release product."""
    return load_pantheonplus_covariance(
        path,
        catalog,
        covariance_kind=kind,
        composition_description=(
            "Complete released Pantheon+ covariance product; no covariance "
            "combination or plotting-error diagonal addition was applied."
        ),
    )


def _result_payload(result: object, selection: object, covariance: object) -> dict:
    """Return JSON result plus compact full/subset provenance."""
    return {
        "analysis_label": selection.analysis_label,
        "full_release_row_count": selection.full_release_row_count,
        "calibrator_row_count": selection.calibrator_row_count,
        "hubble_flow_row_count": selection.hubble_flow_row_count,
        "retained_source_row_indices": list(
            selection.retained_source_row_indices
        ),
        "full_catalog_fingerprint": selection.full_catalog_fingerprint,
        "subset_catalog_fingerprint": selection.subset_catalog_fingerprint,
        "covariance_parent_source_sha256": covariance.parent_source_sha256,
        "subset_covariance_dimension": covariance.dimension,
        "result": asdict(result),
    }


def main() -> None:
    """Run covariance analyses and write ignored plots and JSON summaries."""
    arguments = _arguments()
    catalog = load_pantheonplus_catalog(arguments.data_path)
    selection = select_pantheonplus_hubble_flow_subset(catalog)
    normalized = selection.normalized_catalog
    fingerprint = selection.full_catalog_fingerprint
    statsys_full = _load_covariance(
        arguments.stat_sys_path,
        catalog,
        "pantheonplus_stat_plus_systematic_complete_covariance",
    )
    statsys = subset_pantheonplus_covariance(statsys_full, selection)
    statsys_result = calculate_hubble_residuals_covariance(
        normalized,
        statsys,
        omega_m=arguments.omega_m,
        reference_h0=arguments.reference_h0,
    )
    statsys_full_summary = covariance_summary(statsys_full)
    statsys_summary = covariance_summary(statsys)
    statsys_diagonal = compare_pantheonplus_covariance_diagonal(
        statsys_full,
        catalog,
    )
    base_diagram = calculate_hubble_diagram(
        normalized,
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

    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_hubble_diagram(
            statsys_diagram,
            output / "pantheonplus_statsys_hubble_diagram.png",
            analysis_label="STAT+SYS complete covariance",
        ),
        plot_hubble_residuals(
            statsys_result,
            output / "pantheonplus_statsys_hubble_residuals.png",
        ),
    ]
    (output / "pantheonplus_statsys_result.json").write_text(
        json.dumps(
            _result_payload(statsys_result, selection, statsys),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    summary_payload: dict[str, object] = {
        "catalog_order_fingerprint": fingerprint,
        "subset_catalog_fingerprint": selection.subset_catalog_fingerprint,
        "full_release_row_count": selection.full_release_row_count,
        "calibrator_row_count": selection.calibrator_row_count,
        "hubble_flow_row_count": selection.hubble_flow_row_count,
        "data_source_sha256": _sha256(arguments.data_path),
        "stat_plus_systematic_full_release": asdict(statsys_full_summary),
        "stat_plus_systematic_hubble_flow_subset": asdict(statsys_summary),
        "stat_plus_systematic_diagonal_diagnostic": asdict(statsys_diagonal),
    }

    statonly_result = None
    statonly_covariance = None
    statonly_full_covariance = None
    statonly_summary = None
    statonly_full_summary = None
    statonly_diagonal = None
    comparison = None
    if arguments.statonly is not None:
        statonly_full_covariance = _load_covariance(
            arguments.statonly,
            catalog,
            "pantheonplus_statonly_complete_covariance",
        )
        statonly_covariance = subset_pantheonplus_covariance(
            statonly_full_covariance,
            selection,
        )
        statonly_result = calculate_hubble_residuals_covariance(
            normalized,
            statonly_covariance,
            omega_m=arguments.omega_m,
            reference_h0=arguments.reference_h0,
        )
        statonly_full_summary = covariance_summary(statonly_full_covariance)
        statonly_summary = covariance_summary(statonly_covariance)
        statonly_diagonal = compare_pantheonplus_covariance_diagonal(
            statonly_full_covariance,
            catalog,
        )
        comparison = compare_pantheonplus_covariance_results(
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
                    output / "pantheonplus_statonly_hubble_diagram.png",
                    analysis_label="STATONLY complete covariance",
                ),
                plot_hubble_residuals(
                    statonly_result,
                    output / "pantheonplus_statonly_hubble_residuals.png",
                ),
                plot_pantheonplus_covariance_comparison(
                    statonly_result,
                    statsys_result,
                    output / "pantheonplus_statonly_statsys_comparison.png",
                ),
            )
        )
        (output / "pantheonplus_statonly_result.json").write_text(
            json.dumps(
                _result_payload(statonly_result, selection, statonly_covariance),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        (output / "pantheonplus_covariance_comparison.json").write_text(
            json.dumps(
                {
                    "analysis_label": selection.analysis_label,
                    "full_release_row_count": selection.full_release_row_count,
                    "calibrator_row_count": selection.calibrator_row_count,
                    "hubble_flow_row_count": selection.hubble_flow_row_count,
                    "full_catalog_fingerprint": (
                        selection.full_catalog_fingerprint
                    ),
                    "subset_catalog_fingerprint": (
                        selection.subset_catalog_fingerprint
                    ),
                    "comparison": asdict(comparison),
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        summary_payload["statonly_full_release"] = asdict(
            statonly_full_summary
        )
        summary_payload["statonly_hubble_flow_subset"] = asdict(
            statonly_summary
        )
        summary_payload["statonly_diagonal_diagnostic"] = asdict(
            statonly_diagonal
        )
    for figure in figures:
        plt.close(figure)
    (output / "pantheonplus_covariance_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "pantheonplus_duplicate_cid_summary.json").write_text(
        json.dumps(
            asdict(catalog.duplicate_cid_summary),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    shifted = calculate_hubble_residuals_covariance(
        normalized,
        statsys,
        omega_m=arguments.omega_m,
        reference_h0=80.0,
    )
    h0_difference = float(
        np.max(
            np.abs(
                np.asarray(statsys_result.residual)
                - np.asarray(shifted.residual)
            )
        )
    )
    duplicate = catalog.duplicate_cid_summary
    print(f"Catalog: {catalog.catalog_name} ({catalog.data_product})")
    print(f"Full release rows: {catalog.row_count}")
    print(f"Unique physical CID values: {duplicate.number_of_unique_cids}")
    print(f"Repeated-CID rows: {duplicate.repeated_cid_row_count}")
    print(f"Unique CID+IDSURVEY rows: {catalog.row_count}")
    print(f"Calibrator rows: {catalog.calibrator_row_count}")
    print(f"Non-calibrator rows: {catalog.non_calibrator_row_count}")
    print(
        "Scientific analysis: Pantheon+ Hubble-flow subset (SN-only), "
        f"rows={selection.hubble_flow_row_count}"
    )
    print(f"Catalog order fingerprint: {fingerprint}")
    print(f"Data source SHA-256: {_sha256(arguments.data_path)}")
    print(
        f"STAT+SYS parent covariance: dimension={statsys_full.dimension}, "
        f"SHA-256={statsys_full.source_sha256}"
    )
    print(f"STAT+SYS full-release summary: {asdict(statsys_full_summary)}")
    print(f"STAT+SYS subset summary: {asdict(statsys_summary)}")
    print(f"STAT+SYS diagonal diagnostics: {asdict(statsys_diagonal)}")
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
        assert statonly_covariance is not None
        assert statonly_full_covariance is not None
        assert statonly_summary is not None
        assert statonly_full_summary is not None
        assert statonly_diagonal is not None
        print(
            "STATONLY Hubble-flow covariance: "
            f"dimension={statonly_covariance.dimension}, "
            f"parent SHA-256={statonly_covariance.parent_source_sha256}"
        )
        print(f"STATONLY full-release summary: {asdict(statonly_full_summary)}")
        print(f"STATONLY summary: {asdict(statonly_summary)}")
        print(f"STATONLY diagonal diagnostics: {asdict(statonly_diagonal)}")
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
        "Reference-H0 invariance maximum residual difference: "
        f"{h0_difference:.3e}"
    )
    print(
        "STATONLY and STAT+SYS are separate complete covariance products; "
        "plotting-only diagonal errors were not added to either matrix."
    )
    print(
        "No H0, Omega_m, or dark-energy parameter is inferred by this "
        "residual analysis."
    )


if __name__ == "__main__":
    main()
