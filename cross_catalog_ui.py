"""Presentation helpers for the Streamlit cross-catalog explorer.

This module only orchestrates validated public analysis APIs and converts their
immutable results into UI-friendly tables and JSON.  It contains no fitting,
selection, covariance, precision, or binning implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

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

PATH_ENVIRONMENT_NAMES = (
    "DES_DOVEKIE_HD_PATH",
    "DES_DOVEKIE_STAT_SYS_PATH",
    "PANTHEONPLUS_DATA_PATH",
    "PANTHEONPLUS_STAT_SYS_PATH",
)


@dataclass(frozen=True, slots=True)
class CatalogAvailability:
    """File availability for one catalog's required data products."""

    available: bool
    missing: tuple[str, ...]


def catalog_availability(paths: Mapping[str, str]) -> dict[str, CatalogAvailability]:
    """Describe required-file availability without opening official data."""

    def missing(*names: str) -> tuple[str, ...]:
        return tuple(
            name
            for name in names
            if not paths.get(name) or not Path(paths[name]).is_file()
        )

    des_missing = missing("DES_DOVEKIE_HD_PATH", "DES_DOVEKIE_STAT_SYS_PATH")
    pantheon_missing = missing(
        "PANTHEONPLUS_DATA_PATH", "PANTHEONPLUS_STAT_SYS_PATH"
    )
    return {
        "des": CatalogAvailability(not des_missing, des_missing),
        "pantheon": CatalogAvailability(not pantheon_missing, pantheon_missing),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validated_results(paths: Mapping[str, str]) -> dict[str, Any]:
    """Load official products and run only the existing validated analyses."""
    des_data_path = Path(paths["DES_DOVEKIE_HD_PATH"])
    des_matrix_path = Path(paths["DES_DOVEKIE_STAT_SYS_PATH"])
    pantheon_data_path = Path(paths["PANTHEONPLUS_DATA_PATH"])
    pantheon_matrix_path = Path(paths["PANTHEONPLUS_STAT_SYS_PATH"])

    des_catalog = normalize_des_dovekie_distances(
        load_des_dovekie_hd(des_data_path)
    )
    des_precision = load_packed_precision_matrix(
        des_matrix_path,
        precision_kind="dovekie_stat_plus_systematic_complete_precision",
        catalog_order_fingerprint=catalog_order_fingerprint(des_catalog),
        release_name=des_catalog.catalog_release,
        composition_description=(
            "Complete released DES-Dovekie STAT+SYS precision; unchanged."
        ),
    )
    pantheon_release = load_pantheonplus_catalog(pantheon_data_path)
    pantheon_selection = select_pantheonplus_hubble_flow_subset(pantheon_release)
    pantheon_full_covariance = load_pantheonplus_covariance(
        pantheon_matrix_path,
        pantheon_release,
        covariance_kind=(
            "pantheonplus_stat_plus_systematic_complete_covariance"
        ),
        composition_description=(
            "Complete released Pantheon+ STAT+SYS covariance; unchanged."
        ),
    )
    pantheon_covariance = subset_pantheonplus_covariance(
        pantheon_full_covariance, pantheon_selection
    )

    options = {"omega_m": 0.3, "reference_h0": 70.0}
    des_result = calculate_hubble_residuals_precision(
        des_catalog, des_precision, **options
    )
    pantheon_result = calculate_hubble_residuals_covariance(
        pantheon_selection.normalized_catalog, pantheon_covariance, **options
    )
    provenance = {
        "des_dovekie": {
            "catalog_release": des_catalog.catalog_release,
            "data_source_path_name": des_data_path.name,
            "data_source_sha256": _sha256(des_data_path),
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
            "full_catalog_order_fingerprint": (
                pantheon_selection.full_catalog_fingerprint
            ),
            "subset_catalog_order_fingerprint": (
                pantheon_result.catalog_order_fingerprint
            ),
            "analysis_mode": pantheon_result.covariance_mode,
            "fitted_row_count": pantheon_result.number_of_supernovae,
        },
    }
    return {
        "des_result": des_result,
        "pantheon_result": pantheon_result,
        "provenance": provenance,
    }


def binned_comparison_frame(comparison: Any) -> pd.DataFrame:
    """Create a display table while retaining minimum-count suppression."""
    rows = []
    for item in comparison.bins:
        sufficient = item.descriptive_median_difference is not None
        rows.append(
            {
                "Redshift bin": f"{item.lower_edge:.4f}–{item.upper_edge:.4f}",
                "DES rows": item.first_catalog.row_count,
                "DES median (mag)": (
                    item.first_catalog.median_residual if sufficient else None
                ),
                "Pantheon+ rows": item.second_catalog.row_count,
                "Pantheon+ median (mag)": (
                    item.second_catalog.median_residual if sufficient else None
                ),
                "DES − Pantheon+ (mag)": item.descriptive_median_difference,
                "Meets minimum": sufficient,
            }
        )
    return pd.DataFrame(rows)


def comparison_json(comparison: Any, provenance: Mapping[str, Any]) -> str:
    """Serialize comparison results and provenance without dense matrices."""
    payload = {
        "provenance": dict(provenance),
        "reference_cosmology": {"omega_m": 0.3, "reference_h0": 70.0},
        "comparison": comparison.to_dict(),
        "scientific_limitations": [
            "Independent nuisance intercepts; their difference is not H0.",
            "Descriptive residual comparison only; no joint likelihood.",
            "No cross-catalog covariance or formal tension significance.",
            "Percentile ranges are descriptive, not covariance-aware errors.",
        ],
    }
    return json.dumps(payload, indent=2, allow_nan=False)
