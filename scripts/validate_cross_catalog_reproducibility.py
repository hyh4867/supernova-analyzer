"""Validate read-only DES-Dovekie/Pantheon+ cross-catalog reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

ENVIRONMENT_PATHS = (
    "DES_DOVEKIE_HD_PATH",
    "DES_DOVEKIE_STAT_SYS_PATH",
    "PANTHEONPLUS_DATA_PATH",
    "PANTHEONPLUS_STAT_SYS_PATH",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-count", type=int, default=5)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--reference-h0", type=float, default=70.0)
    parser.add_argument("--alternate-reference-h0", type=float, default=80.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optionally write a compact JSON manifest (never a dense matrix).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_from_environment() -> dict[str, Path] | None:
    missing = [name for name in ENVIRONMENT_PATHS if not os.getenv(name)]
    if missing:
        print(
            "SKIPPED: real-data validation requires environment variables: "
            + ", ".join(missing)
        )
        return None
    paths = {name: Path(os.environ[name]).expanduser() for name in ENVIRONMENT_PATHS}
    absent = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if absent:
        print("SKIPPED: configured official source files are unavailable: " + ", ".join(absent))
        return None
    return paths


def _binned_invariance(baseline: Any, shifted: Any) -> float:
    changes = []
    for first, second in zip(baseline.bins, shifted.bins, strict=True):
        first_value = first.descriptive_median_difference
        second_value = second.descriptive_median_difference
        if first_value is None or second_value is None:
            if first_value is not second_value:
                raise RuntimeError("H0 shift changed binned minimum-count eligibility.")
            continue
        changes.append(abs(first_value - second_value))
    return max(changes, default=0.0)


def _validate(arguments: argparse.Namespace, paths: dict[str, Path]) -> dict[str, object]:
    # Delay scientific imports until after the offline environment check so a
    # missing real-data environment is a fast, dependency-light clean skip.
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
        select_pantheonplus_hubble_flow_subset,
        subset_pantheonplus_covariance,
    )

    hashes_before = {name: _sha256(path) for name, path in paths.items()}

    des_catalog = normalize_des_dovekie_distances(
        load_des_dovekie_hd(paths["DES_DOVEKIE_HD_PATH"])
    )
    des_precision = load_packed_precision_matrix(
        paths["DES_DOVEKIE_STAT_SYS_PATH"],
        precision_kind="dovekie_stat_plus_systematic_complete_precision",
        catalog_order_fingerprint=catalog_order_fingerprint(des_catalog),
        release_name=des_catalog.catalog_release,
        composition_description=(
            "Complete released DES-Dovekie STAT+SYS precision; unchanged."
        ),
    )
    pantheon_release = load_pantheonplus_catalog(paths["PANTHEONPLUS_DATA_PATH"])
    pantheon_selection = select_pantheonplus_hubble_flow_subset(pantheon_release)
    pantheon_full_covariance = load_pantheonplus_covariance(
        paths["PANTHEONPLUS_STAT_SYS_PATH"],
        pantheon_release,
        covariance_kind="pantheonplus_stat_plus_systematic_complete_covariance",
        composition_description=(
            "Complete released Pantheon+ STAT+SYS covariance; unchanged."
        ),
    )
    pantheon_covariance = subset_pantheonplus_covariance(
        pantheon_full_covariance, pantheon_selection
    )

    options = {"omega_m": arguments.omega_m, "reference_h0": arguments.reference_h0}
    des = calculate_hubble_residuals_precision(des_catalog, des_precision, **options)
    pantheon = calculate_hubble_residuals_covariance(
        pantheon_selection.normalized_catalog, pantheon_covariance, **options
    )
    comparison = compare_catalog_residuals(
        des,
        pantheon,
        bins=arguments.bins,
        minimum_count_per_bin=arguments.minimum_count,
    )

    shifted_des = calculate_hubble_residuals_precision(
        des_catalog,
        des_precision,
        omega_m=arguments.omega_m,
        reference_h0=arguments.alternate_reference_h0,
    )
    shifted_pantheon = calculate_hubble_residuals_covariance(
        pantheon_selection.normalized_catalog,
        pantheon_covariance,
        omega_m=arguments.omega_m,
        reference_h0=arguments.alternate_reference_h0,
    )
    shifted_comparison = compare_catalog_residuals(
        shifted_des,
        shifted_pantheon,
        bins=comparison.bin_edges,
        minimum_count_per_bin=arguments.minimum_count,
    )

    hashes_after = {name: _sha256(path) for name, path in paths.items()}
    if hashes_before != hashes_after:
        changed = [name for name in ENVIRONMENT_PATHS if hashes_before[name] != hashes_after[name]]
        raise RuntimeError("Official source SHA-256 changed during validation: " + ", ".join(changed))

    return {
        "validation_kind": "software reproducibility; descriptive comparison only",
        "official_sources_unchanged": True,
        "des_dovekie": {
            "fitted_row_count": des.number_of_supernovae,
            "source_file_name": paths["DES_DOVEKIE_HD_PATH"].name,
            "source_sha256": hashes_before["DES_DOVEKIE_HD_PATH"],
            "precision_file_name": paths["DES_DOVEKIE_STAT_SYS_PATH"].name,
            "precision_sha256": hashes_before["DES_DOVEKIE_STAT_SYS_PATH"],
            "catalog_order_fingerprint": des.catalog_order_fingerprint,
        },
        "pantheonplus": {
            "full_release_row_count": pantheon_selection.full_release_row_count,
            "calibrator_count": pantheon_selection.calibrator_row_count,
            "hubble_flow_row_count": pantheon_selection.hubble_flow_row_count,
            "source_file_name": paths["PANTHEONPLUS_DATA_PATH"].name,
            "source_sha256": hashes_before["PANTHEONPLUS_DATA_PATH"],
            "covariance_file_name": paths["PANTHEONPLUS_STAT_SYS_PATH"].name,
            "covariance_sha256": hashes_before["PANTHEONPLUS_STAT_SYS_PATH"],
            "full_catalog_order_fingerprint": (
                pantheon_selection.full_catalog_fingerprint
            ),
            "subset_catalog_order_fingerprint": (
                pantheon_selection.subset_catalog_fingerprint
            ),
        },
        "cross_catalog": {
            "common_redshift_interval": [
                comparison.common_redshift_minimum,
                comparison.common_redshift_maximum,
            ],
            "overlap_counts": {
                "des_dovekie": comparison.first_catalog_rows_in_common_range,
                "pantheonplus": comparison.second_catalog_rows_in_common_range,
            },
            "binning": {
                "bin_count": len(comparison.bins),
                "minimum_count_per_bin": arguments.minimum_count,
                "bin_edges": list(comparison.bin_edges),
            },
            "catalog_summaries": {
                "des_dovekie": asdict(comparison.first_catalog),
                "pantheonplus": asdict(comparison.second_catalog),
            },
            "binned_summaries": [asdict(item) for item in comparison.bins],
            "difference_definition": (
                "DES-Dovekie median residual minus Pantheon+ median residual; "
                "descriptive only"
            ),
        },
        "h0_invariance": {
            "baseline_reference_h0": arguments.reference_h0,
            "alternate_reference_h0": arguments.alternate_reference_h0,
            "des_maximum_absolute_final_residual_change": (
                maximum_residual_difference(des, shifted_des)
            ),
            "pantheonplus_maximum_absolute_final_residual_change": (
                maximum_residual_difference(pantheon, shifted_pantheon)
            ),
            "maximum_absolute_binned_difference_change": _binned_invariance(
                comparison, shifted_comparison
            ),
        },
        "scientific_scope": {
            "independent_nuisance_intercepts": True,
            "cross_catalog_covariance_constructed": False,
            "object_cross_matching_performed": False,
            "cosmological_parameter_inference_performed": False,
            "tension_significance_test_performed": False,
        },
    }


def main() -> int:
    arguments = _arguments()
    paths = _paths_from_environment()
    if paths is None:
        return 0
    payload = _validate(arguments, paths)
    encoded = json.dumps(payload, indent=2, allow_nan=False, sort_keys=True)
    print(encoded)
    if arguments.manifest is not None:
        arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
        arguments.manifest.write_text(encoded + "\n", encoding="utf-8")
        print(f"Wrote JSON-safe manifest: {arguments.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
