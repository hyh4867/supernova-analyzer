"""Validated input adapters for the current DES-Dovekie release."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from supernova_analyzer.hubble_analysis import NormalizedDistanceCatalog

DovekieSource = str | PathLike[str] | TextIO | BinaryIO
DOVEKIE_CATALOG_NAME = "DES-Dovekie"
DOVEKIE_RELEASE = "Current official DES-Dovekie release"

REQUIRED_HD_COLUMNS = ("CID", "IDSURVEY", "zHD", "zHEL", "MU", "MUERR")
OPTIONAL_HD_COLUMNS = ("MUERR_VPEC", "MUERR_SYS", "PROBIA_BEAMS")


class DovekieCatalogError(ValueError):
    """Raised when a DES-Dovekie release table is malformed."""


class DovekieMetadataAlignmentError(DovekieCatalogError):
    """Raised when metadata cannot be joined bijectively to HD order."""


@dataclass(frozen=True, slots=True)
class DesDovekieCatalog:
    """Immutable DES-Dovekie Hubble data vector in exact release row order."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    source_columns: tuple[str, ...]
    column_provenance: tuple[tuple[str, str], ...]
    source_row_indices: tuple[int, ...]
    catalog_name: str = DOVEKIE_CATALOG_NAME
    catalog_release: str = DOVEKIE_RELEASE

    @property
    def row_count(self) -> int:
        """Return the HD data-vector length."""
        return len(self.rows)

    def to_pandas(self) -> pd.DataFrame:
        """Return a mutable copy without changing stored release order."""
        return pd.DataFrame.from_records(self.rows, columns=self.columns)


@dataclass(frozen=True, slots=True)
class DesDovekieMetadata:
    """Immutable SNANA metadata with original-order provenance."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    original_row_indices: tuple[int, ...]
    source_columns: tuple[str, ...]
    hd_source_row_indices: tuple[int, ...] = ()
    order_mismatch_count: int = 0
    aligned_to_hd_order: bool = False
    catalog_release: str = DOVEKIE_RELEASE

    @property
    def row_count(self) -> int:
        """Return the number of metadata objects."""
        return len(self.rows)

    def to_pandas(self) -> pd.DataFrame:
        """Return a mutable metadata copy."""
        return pd.DataFrame.from_records(self.rows, columns=self.columns)


def _read_text(source: DovekieSource, label: str) -> str:
    """Read UTF-8 text while preserving seekable stream position."""
    if isinstance(source, (str, PathLike)):
        try:
            return Path(source).read_text(encoding="utf-8")
        except OSError as error:
            raise DovekieCatalogError(f"Unable to read {label}: {error}") from error
    position: int | None = None
    try:
        position = source.tell()
    except (AttributeError, OSError):
        pass
    try:
        if position is not None:
            source.seek(0)
        content = source.read()
    except (AttributeError, OSError, TypeError) as error:
        raise DovekieCatalogError(f"Unable to read {label}: {error}") from error
    finally:
        if position is not None:
            source.seek(position)
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DovekieCatalogError(f"{label} must be UTF-8 text.") from error
    return content


def _parse_snana_lines(
    text: str,
    *,
    label: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Parse exactly one VARNAMES header and prefixed data rows."""
    columns: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []
    header_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("VARNAMES:"):
            header_count += 1
            columns = tuple(line.split()[1:])
            continue
        if line.startswith("SN:"):
            if columns is None:
                raise DovekieCatalogError(
                    f"{label} contains an SN row before VARNAMES at line "
                    f"{line_number}."
                )
            values = tuple(line.split()[1:])
            if len(values) != len(columns):
                raise DovekieCatalogError(
                    f"{label} row at line {line_number} has {len(values)} "
                    f"values; expected {len(columns)} from VARNAMES."
                )
            rows.append(values)
            continue
        raise DovekieCatalogError(
            f"{label} contains unsupported content at line {line_number}."
        )
    if header_count != 1 or columns is None:
        raise DovekieCatalogError(
            f"{label} must contain exactly one VARNAMES line."
        )
    if not rows:
        raise DovekieCatalogError(f"{label} contains no SN data rows.")
    return columns, tuple(rows)


def _validate_unique_cid(frame: pd.DataFrame, label: str) -> None:
    """Require present, non-empty, unique CID values."""
    if "CID" not in frame.columns:
        raise DovekieCatalogError(f"{label} is missing required column CID.")
    cid = frame["CID"].astype(str).str.strip()
    if (cid == "").any():
        raise DovekieCatalogError(f"{label} contains an empty CID.")
    duplicates = cid[cid.duplicated()].unique()
    if len(duplicates):
        raise DovekieCatalogError(
            f"{label} contains duplicate CID values: "
            + ", ".join(str(value) for value in duplicates[:5])
            + "."
        )


def load_des_dovekie_hd(source: DovekieSource) -> DesDovekieCatalog:
    """Load the whitespace/SNANA-style DES-Dovekie HD data vector."""
    source_columns, source_rows = _parse_snana_lines(
        _read_text(source, "DES-Dovekie HD table"),
        label="DES-Dovekie HD table",
    )
    missing = [name for name in REQUIRED_HD_COLUMNS if name not in source_columns]
    if missing:
        raise DovekieCatalogError(
            "DES-Dovekie HD table is missing required columns: "
            + ", ".join(missing)
            + "."
        )
    selected = tuple(
        name
        for name in (*REQUIRED_HD_COLUMNS, *OPTIONAL_HD_COLUMNS)
        if name in source_columns
    )
    source_frame = pd.DataFrame.from_records(source_rows, columns=source_columns)
    _validate_unique_cid(source_frame, "DES-Dovekie HD table")
    normalized = pd.DataFrame(index=source_frame.index)
    normalized["CID"] = source_frame["CID"].astype(str).str.strip()
    for name in selected:
        if name == "CID":
            continue
        numeric = pd.to_numeric(source_frame[name], errors="coerce")
        invalid = numeric.isna()
        if invalid.any():
            row = int(invalid[invalid].index[0])
            raise DovekieCatalogError(
                f"DES-Dovekie HD column {name} contains a non-numeric value "
                f"at source row {row}."
            )
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DovekieCatalogError(
                f"DES-Dovekie HD column {name} must contain finite values."
            )
        if name in {"zHD", "zHEL", "MUERR"} and np.any(values <= 0.0):
            raise DovekieCatalogError(
                f"DES-Dovekie HD column {name} must be greater than zero."
            )
        normalized[name] = values
    return DesDovekieCatalog(
        columns=tuple(normalized.columns),
        rows=tuple(
            tuple(value for value in row)
            for row in normalized.itertuples(index=False, name=None)
        ),
        source_columns=source_columns,
        column_provenance=tuple((name, name) for name in normalized.columns),
        source_row_indices=tuple(range(len(normalized))),
    )


def load_des_dovekie_metadata(source: DovekieSource) -> DesDovekieMetadata:
    """Load DES-Dovekie SNANA metadata while preserving original order."""
    columns, rows = _parse_snana_lines(
        _read_text(source, "DES-Dovekie metadata"),
        label="DES-Dovekie metadata",
    )
    frame = pd.DataFrame.from_records(rows, columns=columns)
    _validate_unique_cid(frame, "DES-Dovekie metadata")
    return DesDovekieMetadata(
        columns=columns,
        rows=rows,
        original_row_indices=tuple(range(len(rows))),
        source_columns=columns,
    )


def join_dovekie_metadata_to_hd_order(
    hd: DesDovekieCatalog,
    metadata: DesDovekieMetadata,
) -> DesDovekieMetadata:
    """Join strictly by CID and return metadata in exact HD/precision order.

    Covariance and precision alignment follows the HD data-vector order, never
    the original metadata order.
    """
    hd_frame = hd.to_pandas()
    metadata_frame = metadata.to_pandas()
    hd_ids = tuple(str(value) for value in hd_frame["CID"])
    metadata_ids = tuple(str(value) for value in metadata_frame["CID"])
    hd_set = set(hd_ids)
    metadata_set = set(metadata_ids)
    missing = sorted(hd_set - metadata_set)
    extra = sorted(metadata_set - hd_set)
    if missing or extra or len(hd_ids) != len(metadata_ids):
        details = []
        if missing:
            details.append(f"missing CIDs: {', '.join(missing[:5])}")
        if extra:
            details.append(f"extra CIDs: {', '.join(extra[:5])}")
        raise DovekieMetadataAlignmentError(
            "DES-Dovekie metadata join is not bijective"
            + (f" ({'; '.join(details)})" if details else "")
            + "."
        )
    positions = {cid: index for index, cid in enumerate(metadata_ids)}
    mismatch_count = sum(
        positions[cid] != hd_index for hd_index, cid in enumerate(hd_ids)
    )
    ordered_positions = [positions[cid] for cid in hd_ids]
    ordered = metadata_frame.iloc[ordered_positions]
    return DesDovekieMetadata(
        columns=metadata.columns,
        rows=tuple(
            tuple(str(value) for value in row)
            for row in ordered.itertuples(index=False, name=None)
        ),
        original_row_indices=tuple(
            metadata.original_row_indices[index] for index in ordered_positions
        ),
        source_columns=metadata.source_columns,
        hd_source_row_indices=hd.source_row_indices,
        order_mismatch_count=mismatch_count,
        aligned_to_hd_order=True,
    )


def normalize_des_dovekie_distances(
    catalog: DesDovekieCatalog,
) -> NormalizedDistanceCatalog:
    """Convert DES-Dovekie HD rows to the generic immutable distance protocol."""
    frame = catalog.to_pandas()
    return NormalizedDistanceCatalog(
        object_id=tuple(str(value) for value in frame["CID"]),
        redshift=tuple(float(value) for value in frame["zHD"]),
        heliocentric_redshift=tuple(float(value) for value in frame["zHEL"]),
        distance_modulus=tuple(float(value) for value in frame["MU"]),
        distance_modulus_error=tuple(float(value) for value in frame["MUERR"]),
        source_row_index=catalog.source_row_indices,
        catalog_name=DOVEKIE_CATALOG_NAME,
        catalog_release=DOVEKIE_RELEASE,
        redshift_definition="zHD",
        distance_definition="Released DES-Dovekie distance modulus MU",
        reference_distance_convention=(
            "zHD cosmology redshift with zHEL frame correction"
        ),
    )
