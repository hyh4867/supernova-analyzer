"""Input utilities for scientific multi-band photometry."""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
from astropy.table import Table

ScientificSource = str | Path | TextIO | BinaryIO
REQUIRED_SCIENTIFIC_COLUMNS = ("time", "band", "flux", "fluxerr", "zp", "zpsys")
NUMERIC_COLUMNS = ("time", "flux", "fluxerr", "zp")
STRING_COLUMNS = ("band", "zpsys")


def _read_csv_preserving_position(source: ScientificSource) -> Table:
    """Read an Astropy Table and restore a seekable stream's position."""
    if isinstance(source, (str, Path)):
        return Table.read(source, format="ascii.csv")

    position = source.tell()
    try:
        content = source.read()
    finally:
        source.seek(position)

    if isinstance(content, str):
        return Table.read(content.splitlines(), format="ascii.csv")
    return Table.read(BytesIO(content), format="ascii.csv")


def load_scientific_photometry(source: ScientificSource) -> Table:
    """Load and validate multi-band photometry from a CSV source.

    Args:
        source: CSV path or readable text/binary file-like object.

    Returns:
        A new Astropy Table sorted by time. Duplicate times and negative fluxes
        are retained because both can be valid in multi-band photometry.

    Raises:
        ValueError: If required columns are absent, no rows are present, numeric
            data are invalid, flux errors are not positive, or string fields are
            empty.
    """
    table = _read_csv_preserving_position(source)
    return _validated_scientific_photometry_copy(table)


def _validated_scientific_photometry_copy(table: Table) -> Table:
    """Validate scientific photometry and return a time-sorted deep copy."""

    missing_columns = [
        column for column in REQUIRED_SCIENTIFIC_COLUMNS if column not in table.colnames
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Scientific photometry is missing required columns: {missing}.")

    if len(table) == 0:
        raise ValueError("Scientific photometry table is empty.")

    for column in NUMERIC_COLUMNS:
        values = np.ma.asarray(table[column])
        if np.ma.getmaskarray(values).any():
            raise ValueError(f"Column '{column}' contains missing values.")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"Column '{column}' must contain numeric values.")
        if not np.isfinite(values.data).all():
            raise ValueError(f"Column '{column}' must contain only finite values.")

    if not np.all(np.asarray(table["fluxerr"]) > 0):
        raise ValueError("Column 'fluxerr' must contain values greater than zero.")

    for column in STRING_COLUMNS:
        values = np.ma.asarray(table[column])
        if np.ma.getmaskarray(values).any():
            raise ValueError(f"Column '{column}' contains empty string values.")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"Column '{column}' must contain non-empty strings.")

    sorted_table = table.copy(copy_data=True)
    sorted_table.sort("time")
    return sorted_table
