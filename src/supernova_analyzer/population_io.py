"""Validated input layer for the DES-SN5YR population catalog."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

PopulationSource = str | PathLike[str] | TextIO | BinaryIO

DES_SN5YR_RELEASE_NAME = "Legacy DES-SN5YR HD + metadata catalog"
DES_SN5YR_CATALOG_RELEASE = "Legacy DES-SN5YR (Vincenzi et al. 2024)"
DES_SN5YR_RELEASE_PRODUCT = "DES-SN5YR_HD+MetaData.csv"
DES_SN5YR_RELEASE_DOI = "10.5281/zenodo.12720778"
DES_SN5YR_RELEASE_URL = "https://github.com/des-science/DES-SN5YR"

REQUIRED_COLUMN_MAP = {
    "object_id": ("CID",),
    "z_hel": ("zHEL",),
    "z_cmb": ("zCMB",),
    "z_hd": ("zHD",),
    "distance_modulus": ("MU",),
    "distance_modulus_error": ("MUERR_FINAL",),
    "host_logmass": ("HOST_LOGMASS",),
    "x0": ("x0",),
    "x1": ("x1",),
    "color": ("c",),
    "peak_mjd": ("PKMJD",),
}
OPTIONAL_COLUMN_MAP = {
    "chi2_or_fit_probability": ("FITPROB", "FITCHI2"),
    "bias_correction": ("biasCor_mu",),
    "survey": ("IDSURVEY",),
    "type": ("TYPE",),
    "classification": (
        "PROB_SNNV19",
        "PROB_SNNDESCC",
        "PROB_SNNJ17",
        "PROB_SCONE",
        "PROB_SNIRFV19",
    ),
    "beams_core_collapse_probability": ("PROBCC_BEAMS",),
}
STRING_FIELDS = {"object_id"}
DOCUMENTED_SENTINELS = (-999.0, -9999.0)


class PopulationCatalogError(ValueError):
    """Raised when a population catalog cannot be read or validated."""


@dataclass(frozen=True, slots=True)
class PopulationDataQualitySummary:
    """Immutable data-quality accounting for a normalized population catalog."""

    source_row_count: int
    retained_row_count: int
    removed_row_count: int
    duplicate_object_id_count: int
    missingness_by_field: tuple[tuple[str, int], ...]
    sentinel_replacements_by_field: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DesSn5yrCatalog:
    """Immutable normalized DES-SN5YR population-catalog representation.

    Only normalized fields backed by an actual source column are present.
    ``to_pandas()`` returns a new mutable copy for downstream analysis.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    source_columns: tuple[str, ...]
    column_provenance: tuple[tuple[str, str, str], ...]
    metadata: tuple[tuple[str, str], ...]
    quality_summary: PopulationDataQualitySummary
    source_row_indices: tuple[int, ...] = ()

    @property
    def row_count(self) -> int:
        """Return the number of retained rows."""
        return len(self.rows)

    def to_pandas(self) -> pd.DataFrame:
        """Return a new DataFrame copy of the immutable normalized data."""
        return pd.DataFrame.from_records(self.rows, columns=self.columns)


def _read_catalog(source: PopulationSource) -> pd.DataFrame:
    """Read CSV input while preserving a seekable file-like position."""
    try:
        if isinstance(source, (str, PathLike)):
            return pd.read_csv(source)
        position = source.tell()
        try:
            source.seek(0)
            content = source.read()
        finally:
            source.seek(position)
        if isinstance(content, str):
            return pd.read_csv(BytesIO(content.encode("utf-8")))
        return pd.read_csv(BytesIO(content))
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise PopulationCatalogError(
            f"Unable to read DES-SN5YR catalog CSV: {error}"
        ) from error


def _resolve_columns(
    source_columns: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Resolve required and optional normalized fields to source columns."""
    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for normalized, candidates in REQUIRED_COLUMN_MAP.items():
        source_name = next(
            (candidate for candidate in candidates if candidate in source_columns),
            None,
        )
        if source_name is None:
            missing.append(f"{normalized} ({'/'.join(candidates)})")
        else:
            resolved.append((normalized, source_name))
    if missing:
        raise PopulationCatalogError(
            "DES-SN5YR catalog is missing required columns: "
            + ", ".join(missing)
            + "."
        )
    for normalized, candidates in OPTIONAL_COLUMN_MAP.items():
        source_name = next(
            (candidate for candidate in candidates if candidate in source_columns),
            None,
        )
        if source_name is not None:
            resolved.append((normalized, source_name))
    return tuple(resolved)


def _normalize_numeric_column(
    series: pd.Series,
    normalized_name: str,
) -> tuple[pd.Series, int]:
    """Coerce a numeric source column and replace only documented sentinels."""
    raw_missing = series.isna()
    numeric = pd.to_numeric(series, errors="coerce")
    coercion_failure = numeric.isna() & ~raw_missing
    if coercion_failure.any():
        first_index = coercion_failure[coercion_failure].index[0]
        invalid_value = series.loc[first_index]
        raise PopulationCatalogError(
            f"Column for normalized field '{normalized_name}' contains "
            f"non-numeric value {invalid_value!r} at row {first_index}."
        )
    sentinel_mask = numeric.isin(DOCUMENTED_SENTINELS)
    replacements = int(sentinel_mask.sum())
    return numeric.mask(sentinel_mask, np.nan).astype(float), replacements


def _normalize_catalog(
    source_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[tuple[str, str, str], ...], tuple[tuple[str, int], ...]]:
    """Normalize direct source fields without applying scientific cuts."""
    if source_frame.empty:
        raise PopulationCatalogError("DES-SN5YR catalog contains no data rows.")
    source_columns = tuple(str(column) for column in source_frame.columns)
    resolved = _resolve_columns(source_columns)
    normalized = pd.DataFrame(index=source_frame.index)
    sentinel_counts: list[tuple[str, int]] = []
    provenance: list[tuple[str, str, str]] = []

    for normalized_name, source_name in resolved:
        if normalized_name in STRING_FIELDS:
            values = source_frame[source_name]
            if values.isna().any() or (values.astype(str).str.strip() == "").any():
                raise PopulationCatalogError(
                    f"Source column '{source_name}' contains missing object IDs."
                )
            normalized[normalized_name] = values.astype(str).str.strip()
            sentinel_counts.append((normalized_name, 0))
        else:
            values, replacements = _normalize_numeric_column(
                source_frame[source_name],
                normalized_name,
            )
            normalized[normalized_name] = values
            sentinel_counts.append((normalized_name, replacements))
        provenance.append((normalized_name, source_name, "direct"))

    duplicate_mask = normalized["object_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = tuple(
            dict.fromkeys(normalized.loc[duplicate_mask, "object_id"].tolist())
        )
        preview = ", ".join(duplicate_ids[:5])
        raise PopulationCatalogError(
            f"DES-SN5YR catalog contains duplicate object IDs: {preview}."
        )
    return normalized, tuple(provenance), tuple(sentinel_counts)


def _quality_summary(
    frame: pd.DataFrame,
    source_row_count: int,
    sentinel_counts: tuple[tuple[str, int], ...],
) -> PopulationDataQualitySummary:
    """Build explicit row accounting and per-field missingness."""
    retained = len(frame)
    return PopulationDataQualitySummary(
        source_row_count=source_row_count,
        retained_row_count=retained,
        removed_row_count=source_row_count - retained,
        duplicate_object_id_count=0,
        missingness_by_field=tuple(
            (column, int(frame[column].isna().sum())) for column in frame.columns
        ),
        sentinel_replacements_by_field=sentinel_counts,
    )


def _immutable_rows(frame: pd.DataFrame) -> tuple[tuple[object, ...], ...]:
    """Convert a normalized frame to immutable Python/NumPy scalar rows."""
    return tuple(tuple(value for value in row) for row in frame.itertuples(index=False, name=None))


def load_des_sn5yr_catalog(source: PopulationSource) -> DesSn5yrCatalog:
    """Load and normalize the official DES-SN5YR HD + metadata CSV.

    The loader performs schema validation and sentinel conversion but applies no
    scientific quality cuts and removes no rows. The official release file is
    not bundled or downloaded automatically; callers must provide a local path
    or file-like object obtained from the DES-SN5YR release.

    Args:
        source: Local CSV path or readable text/binary file-like object.

    Returns:
        An immutable normalized catalog with provenance and quality accounting.

    Raises:
        PopulationCatalogError: If reading, schema, values, or object IDs are
            invalid.
    """
    source_frame = _read_catalog(source)
    normalized, provenance, sentinel_counts = _normalize_catalog(source_frame)
    quality = _quality_summary(normalized, len(source_frame), sentinel_counts)
    metadata = (
        ("release_name", DES_SN5YR_RELEASE_NAME),
        ("release_product", DES_SN5YR_RELEASE_PRODUCT),
        ("release_doi", DES_SN5YR_RELEASE_DOI),
        ("release_url", DES_SN5YR_RELEASE_URL),
        ("normalization_policy", "direct source columns; no derived fields"),
        ("quality_cut_policy", "none applied"),
        ("sentinel_policy", "-999 and -9999 converted to NaN"),
    )
    return DesSn5yrCatalog(
        columns=tuple(str(column) for column in normalized.columns),
        rows=_immutable_rows(normalized),
        source_columns=tuple(str(column) for column in source_frame.columns),
        column_provenance=provenance,
        metadata=metadata,
        quality_summary=quality,
        source_row_indices=tuple(range(len(source_frame))),
    )


def apply_des_sn5yr_quality_cuts(
    catalog: DesSn5yrCatalog,
    *,
    require_host_mass: bool = False,
    maximum_distance_modulus_error: float | None = None,
    minimum_fit_probability: float | None = None,
) -> DesSn5yrCatalog:
    """Apply explicitly requested, transparent population quality cuts.

    No defaults remove rows. A fit-probability threshold is available only when
    ``chi2_or_fit_probability`` came directly from ``FITPROB`` rather than
    ``FITCHI2``.
    """
    frame = catalog.to_pandas()
    mask = pd.Series(True, index=frame.index)
    cut_descriptions: list[str] = []
    if require_host_mass:
        mask &= frame["host_logmass"].notna()
        cut_descriptions.append("require_host_mass")
    if maximum_distance_modulus_error is not None:
        threshold = float(maximum_distance_modulus_error)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "maximum_distance_modulus_error must be finite and positive."
            )
        mask &= frame["distance_modulus_error"] <= threshold
        cut_descriptions.append(
            f"maximum_distance_modulus_error={threshold:g}"
        )
    if minimum_fit_probability is not None:
        threshold = float(minimum_fit_probability)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "minimum_fit_probability must be finite and between 0 and 1."
            )
        provenance = {
            normalized: source
            for normalized, source, _kind in catalog.column_provenance
        }
        if provenance.get("chi2_or_fit_probability") != "FITPROB":
            raise PopulationCatalogError(
                "minimum_fit_probability requires source column 'FITPROB'."
            )
        mask &= frame["chi2_or_fit_probability"] >= threshold
        cut_descriptions.append(f"minimum_fit_probability={threshold:g}")

    filtered = frame.loc[mask].reset_index(drop=True)
    source_row_indices = tuple(
        source_index
        for source_index, keep in zip(
            catalog.source_row_indices,
            mask,
            strict=True,
        )
        if bool(keep)
    )
    sentinel_counts = catalog.quality_summary.sentinel_replacements_by_field
    quality = _quality_summary(
        filtered,
        catalog.quality_summary.source_row_count,
        sentinel_counts,
    )
    metadata = catalog.metadata + (
        (
            "explicit_quality_cuts",
            ", ".join(cut_descriptions) if cut_descriptions else "none",
        ),
    )
    return DesSn5yrCatalog(
        columns=catalog.columns,
        rows=_immutable_rows(filtered),
        source_columns=catalog.source_columns,
        column_provenance=catalog.column_provenance,
        metadata=metadata,
        quality_summary=quality,
        source_row_indices=source_row_indices,
    )
