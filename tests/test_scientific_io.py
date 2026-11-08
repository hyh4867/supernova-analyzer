"""Tests for scientific multi-band photometry loading."""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from supernova_analyzer import load_scientific_photometry

HEADER = "time,band,flux,fluxerr,zp,zpsys\n"
VALID_ROWS = (
    "2.0,sdssr,-0.2,0.3,25.0,ab\n"
    "1.0,sdssg,1.5,0.2,25.0,ab\n"
)


def test_load_scientific_photometry_from_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "photometry.csv"
    csv_path.write_text(HEADER + VALID_ROWS, encoding="utf-8")

    result = load_scientific_photometry(csv_path)

    assert isinstance(result, Table)
    assert result.colnames == ["time", "band", "flux", "fluxerr", "zp", "zpsys"]
    assert len(result) == 2


def test_load_scientific_photometry_rejects_missing_column() -> None:
    source = StringIO("time,band,flux,fluxerr,zp\n1.0,sdssg,1.5,0.2,25.0\n")

    with pytest.raises(ValueError, match="zpsys"):
        load_scientific_photometry(source)


def test_load_scientific_photometry_rejects_empty_table() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_scientific_photometry(StringIO(HEADER))


@pytest.mark.parametrize("column", ["time", "flux", "fluxerr", "zp"])
def test_load_scientific_photometry_rejects_non_numeric(column: str) -> None:
    values = {
        "time": "1.0",
        "band": "sdssg",
        "flux": "1.5",
        "fluxerr": "0.2",
        "zp": "25.0",
        "zpsys": "ab",
    }
    values[column] = "invalid"
    row = ",".join(values[name] for name in values)

    with pytest.raises(ValueError, match=rf"{column}.*numeric"):
        load_scientific_photometry(StringIO(HEADER + row + "\n"))


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("time", np.nan),
        ("time", np.inf),
        ("flux", np.nan),
        ("flux", np.inf),
        ("fluxerr", np.nan),
        ("fluxerr", np.inf),
        ("zp", np.nan),
        ("zp", np.inf),
    ],
)
def test_load_scientific_photometry_rejects_non_finite(
    column: str,
    invalid_value: float,
) -> None:
    values = {
        "time": 1.0,
        "band": "sdssg",
        "flux": 1.5,
        "fluxerr": 0.2,
        "zp": 25.0,
        "zpsys": "ab",
    }
    values[column] = invalid_value
    row = ",".join(str(values[name]) for name in values)

    with pytest.raises(ValueError, match=rf"{column}.*finite"):
        load_scientific_photometry(StringIO(HEADER + row + "\n"))


@pytest.mark.parametrize("fluxerr", [0.0, -0.1])
def test_load_scientific_photometry_requires_positive_fluxerr(
    fluxerr: float,
) -> None:
    row = f"1.0,sdssg,1.5,{fluxerr},25.0,ab\n"

    with pytest.raises(ValueError, match="fluxerr.*greater than zero"):
        load_scientific_photometry(StringIO(HEADER + row))


def test_load_scientific_photometry_allows_negative_flux() -> None:
    result = load_scientific_photometry(StringIO(HEADER + VALID_ROWS))

    assert np.any(result["flux"] < 0)


@pytest.mark.parametrize("column", ["band", "zpsys"])
def test_load_scientific_photometry_rejects_empty_strings(column: str) -> None:
    values = {
        "time": "1.0",
        "band": "sdssg",
        "flux": "1.5",
        "fluxerr": "0.2",
        "zp": "25.0",
        "zpsys": "ab",
    }
    values[column] = ""
    row = ",".join(values[name] for name in values)

    with pytest.raises(ValueError, match=column):
        load_scientific_photometry(StringIO(HEADER + row + "\n"))


def test_load_scientific_photometry_allows_same_time_across_bands() -> None:
    rows = (
        "1.0,sdssg,1.5,0.2,25.0,ab\n"
        "1.0,sdssr,1.2,0.2,25.0,ab\n"
    )

    result = load_scientific_photometry(StringIO(HEADER + rows))

    assert len(result) == 2
    assert list(result["band"]) == ["sdssg", "sdssr"]


def test_load_scientific_photometry_preserves_file_like_input() -> None:
    source = StringIO(HEADER + VALID_ROWS)
    original_text = source.getvalue()
    original_position = source.tell()

    load_scientific_photometry(source)

    assert source.getvalue() == original_text
    assert source.tell() == original_position


def test_load_scientific_photometry_sorts_by_time() -> None:
    result = load_scientific_photometry(StringIO(HEADER + VALID_ROWS))

    assert list(result["time"]) == [1.0, 2.0]
    assert list(result["band"]) == ["sdssg", "sdssr"]


def test_load_scientific_photometry_accepts_file_like_object() -> None:
    source = StringIO(HEADER + VALID_ROWS)

    result = load_scientific_photometry(source)

    assert isinstance(result, Table)
    assert len(result) == 2
