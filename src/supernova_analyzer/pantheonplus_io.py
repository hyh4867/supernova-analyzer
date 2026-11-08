"""Validated population adapter for the Pantheon+SH0ES distance vector."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from supernova_analyzer.hubble_analysis import NormalizedDistanceCatalog
from supernova_analyzer.covariance import catalog_order_fingerprint

PantheonPlusSource = str | PathLike[str] | TextIO | BinaryIO
PANTHEONPLUS_CATALOG_NAME = "Pantheon+"
PANTHEONPLUS_DATA_PRODUCT = "Pantheon+SH0ES"
PANTHEONPLUS_RELEASE = "Official Pantheon+SH0ES distance release"
REQUIRED_COLUMNS = (
    "CID",
    "IDSURVEY",
    "zHD",
    "zHEL",
    "MU_SH0ES",
    "MU_SH0ES_ERR_DIAG",
)
OPTIONAL_DIAGNOSTIC_COLUMNS = (
    "m_b_corr",
    "m_b_corr_err_DIAG",
    "IS_CALIBRATOR",
    "HOST_LOGMASS",
)


class PantheonPlusCatalogError(ValueError):
    """Raised when the Pantheon+SH0ES data vector is malformed."""


@dataclass(frozen=True, slots=True)
class PantheonPlusDuplicateCidSummary:
    """Immutable descriptive accounting for repeated physical CID values."""

    number_of_rows: int
    number_of_unique_cids: int
    repeated_cid_row_count: int
    number_of_repeated_cids: int
    maximum_multiplicity: int
    multiplicity_distribution: tuple[tuple[int, int], ...]
    repeated_cid_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PantheonPlusCatalog:
    """Immutable Pantheon+ released data-vector entries in exact row order.

    Rows are light-curve/survey entries and are not necessarily unique physical
    supernovae. ``CID`` may repeat; the pair ``(CID, IDSURVEY)`` is unique.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    source_columns: tuple[str, ...]
    column_provenance: tuple[tuple[str, str], ...]
    source_row_indices: tuple[int, ...]
    cid: tuple[str, ...]
    idsurvey: tuple[str, ...]
    duplicate_cid_summary: PantheonPlusDuplicateCidSummary
    calibrator_row_count: int
    non_calibrator_row_count: int
    catalog_name: str = PANTHEONPLUS_CATALOG_NAME
    data_product: str = PANTHEONPLUS_DATA_PRODUCT
    catalog_release: str = PANTHEONPLUS_RELEASE

    @property
    def row_count(self) -> int:
        """Return the exact released data-vector row count."""
        return len(self.rows)

    def to_pandas(self) -> pd.DataFrame:
        """Return a mutable copy without changing the stored release order."""
        return pd.DataFrame.from_records(self.rows, columns=self.columns)


@dataclass(frozen=True, slots=True)
class PantheonPlusHubbleFlowSubset:
    """Exact non-calibrator selection for redshift-based Hubble analysis."""

    full_catalog: NormalizedDistanceCatalog
    normalized_catalog: NormalizedDistanceCatalog
    retained_source_row_indices: tuple[int, ...]
    full_release_row_count: int
    calibrator_row_count: int
    hubble_flow_row_count: int
    full_catalog_fingerprint: str
    subset_catalog_fingerprint: str
    analysis_label: str = "Pantheon+ Hubble-flow subset (SN-only)"


def _read_table(source: PantheonPlusSource) -> pd.DataFrame:
    """Read whitespace data while preserving a seekable stream position."""
    try:
        if isinstance(source, (str, PathLike)):
            return pd.read_csv(
                source,
                sep=r"\s+",
                dtype={"CID": str, "IDSURVEY": str},
            )
        position = source.tell()
        try:
            source.seek(0)
            content = source.read()
        finally:
            source.seek(position)
        payload = content.encode("utf-8") if isinstance(content, str) else content
        return pd.read_csv(
            BytesIO(payload),
            sep=r"\s+",
            dtype={"CID": str, "IDSURVEY": str},
        )
    except (OSError, TypeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise PantheonPlusCatalogError(
            f"Unable to read Pantheon+SH0ES data vector: {error}"
        ) from error


def _duplicate_summary(cid: tuple[str, ...]) -> PantheonPlusDuplicateCidSummary:
    """Build machine-readable multiplicity diagnostics without deduplication."""
    counts = pd.Series(cid, dtype=str).value_counts()
    repeated = counts[counts > 1]
    multiplicities = repeated.value_counts().sort_index()
    return PantheonPlusDuplicateCidSummary(
        number_of_rows=len(cid),
        number_of_unique_cids=int(len(counts)),
        repeated_cid_row_count=int(len(cid) - len(counts)),
        number_of_repeated_cids=int(len(repeated)),
        maximum_multiplicity=int(counts.max()),
        multiplicity_distribution=tuple(
            (int(multiplicity), int(number_of_cids))
            for multiplicity, number_of_cids in multiplicities.items()
        ),
        repeated_cid_counts=tuple(
            (str(name), int(count)) for name, count in repeated.items()
        ),
    )


def load_pantheonplus_catalog(
    source: PantheonPlusSource,
) -> PantheonPlusCatalog:
    """Load all Pantheon+SH0ES rows without cuts, sorting, or deduplication."""
    source_frame = _read_table(source)
    if source_frame.empty:
        raise PantheonPlusCatalogError(
            "Pantheon+SH0ES data vector contains no rows."
        )
    missing = [name for name in REQUIRED_COLUMNS if name not in source_frame.columns]
    if missing:
        raise PantheonPlusCatalogError(
            "Pantheon+SH0ES data vector is missing required columns: "
            + ", ".join(missing)
            + "."
        )
    cid = source_frame["CID"].astype(str).str.strip()
    survey = source_frame["IDSURVEY"].astype(str).str.strip()
    if (cid == "").any() or (survey == "").any():
        raise PantheonPlusCatalogError(
            "CID and IDSURVEY must contain non-empty row identities."
        )
    identities = tuple(zip(cid, survey, strict=True))
    if len(set(identities)) != len(identities):
        raise PantheonPlusCatalogError(
            "Pantheon+SH0ES (CID, IDSURVEY) row identities must be unique."
        )
    selected_columns = tuple(
        name
        for name in (*REQUIRED_COLUMNS, *OPTIONAL_DIAGNOSTIC_COLUMNS)
        if name in source_frame.columns
    )
    normalized = pd.DataFrame(index=source_frame.index)
    normalized["CID"] = cid
    normalized["IDSURVEY"] = survey
    for name in selected_columns:
        if name in {"CID", "IDSURVEY"}:
            continue
        raw = source_frame[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna() & ~raw.isna()
        if invalid.any():
            row = int(invalid[invalid].index[0])
            raise PantheonPlusCatalogError(
                f"Pantheon+SH0ES column {name} contains a non-numeric value "
                f"at source row {row}."
            )
        values = numeric.to_numpy(dtype=float)
        if name in REQUIRED_COLUMNS and not np.isfinite(values).all():
            raise PantheonPlusCatalogError(
                f"Pantheon+SH0ES column {name} must contain finite values."
            )
        if name in {"zHD", "zHEL"} and np.any(values <= 0.0):
            raise PantheonPlusCatalogError(
                f"Pantheon+SH0ES column {name} must be greater than zero."
            )
        if name == "MU_SH0ES_ERR_DIAG" and np.any(values <= 0.0):
            raise PantheonPlusCatalogError(
                "Pantheon+SH0ES MU_SH0ES_ERR_DIAG must be greater than zero."
            )
        normalized[name] = values
    calibrator_count = (
        int(np.sum(normalized["IS_CALIBRATOR"] == 1.0))
        if "IS_CALIBRATOR" in normalized
        else 0
    )
    cid_tuple = tuple(str(value) for value in cid)
    survey_tuple = tuple(str(value) for value in survey)
    return PantheonPlusCatalog(
        columns=tuple(normalized.columns),
        rows=tuple(
            tuple(value for value in row)
            for row in normalized.itertuples(index=False, name=None)
        ),
        source_columns=tuple(str(name) for name in source_frame.columns),
        column_provenance=tuple((name, name) for name in normalized.columns),
        source_row_indices=tuple(range(len(normalized))),
        cid=cid_tuple,
        idsurvey=survey_tuple,
        duplicate_cid_summary=_duplicate_summary(cid_tuple),
        calibrator_row_count=calibrator_count,
        non_calibrator_row_count=len(normalized) - calibrator_count,
    )


def normalize_pantheonplus_distances(
    catalog: PantheonPlusCatalog,
) -> NormalizedDistanceCatalog:
    """Map exact Pantheon+ rows to the generic immutable distance protocol."""
    frame = catalog.to_pandas()
    identities = tuple(
        f"{cid}::{survey}"
        for cid, survey in zip(catalog.cid, catalog.idsurvey, strict=True)
    )
    return NormalizedDistanceCatalog(
        object_id=identities,
        redshift=tuple(float(value) for value in frame["zHD"]),
        heliocentric_redshift=tuple(float(value) for value in frame["zHEL"]),
        distance_modulus=tuple(float(value) for value in frame["MU_SH0ES"]),
        distance_modulus_error=tuple(
            float(value) for value in frame["MU_SH0ES_ERR_DIAG"]
        ),
        source_row_index=catalog.source_row_indices,
        catalog_name=PANTHEONPLUS_CATALOG_NAME,
        catalog_release=PANTHEONPLUS_RELEASE,
        redshift_definition="zHD",
        distance_definition="Released Pantheon+SH0ES distance modulus MU_SH0ES",
        reference_distance_convention=(
            "zHD cosmology redshift with zHEL frame correction"
        ),
    )


def select_pantheonplus_hubble_flow_subset(
    catalog: PantheonPlusCatalog,
) -> PantheonPlusHubbleFlowSubset:
    """Select ``IS_CALIBRATOR == 0`` rows without reordering or deduplication."""
    frame = catalog.to_pandas()
    if "IS_CALIBRATOR" not in frame:
        raise PantheonPlusCatalogError(
            "Pantheon+ Hubble-flow selection requires IS_CALIBRATOR."
        )
    calibrator = np.asarray(frame["IS_CALIBRATOR"], dtype=float)
    if not np.isfinite(calibrator).all() or not np.isin(
        calibrator,
        (0.0, 1.0),
    ).all():
        raise PantheonPlusCatalogError(
            "IS_CALIBRATOR must contain only finite zero or one values."
        )
    positions = np.flatnonzero(calibrator == 0.0)
    full = normalize_pantheonplus_distances(catalog)

    def selected(values: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(values[int(position)] for position in positions)

    subset = NormalizedDistanceCatalog(
        object_id=selected(full.object_id),
        redshift=selected(full.redshift),
        heliocentric_redshift=(
            selected(full.heliocentric_redshift)
            if full.heliocentric_redshift is not None
            else None
        ),
        distance_modulus=selected(full.distance_modulus),
        distance_modulus_error=selected(full.distance_modulus_error),
        source_row_index=selected(full.source_row_index),
        catalog_name="Pantheon+ Hubble-flow subset (SN-only)",
        catalog_release=full.catalog_release,
        redshift_definition=full.redshift_definition,
        distance_definition=(
            "Released Pantheon+SH0ES MU_SH0ES for non-calibrator rows"
        ),
        reference_distance_convention=full.reference_distance_convention,
    )
    return PantheonPlusHubbleFlowSubset(
        full_catalog=full,
        normalized_catalog=subset,
        retained_source_row_indices=subset.source_row_index,
        full_release_row_count=catalog.row_count,
        calibrator_row_count=int(np.sum(calibrator == 1.0)),
        hubble_flow_row_count=len(subset.object_id),
        full_catalog_fingerprint=catalog_order_fingerprint(full),
        subset_catalog_fingerprint=catalog_order_fingerprint(subset),
    )
