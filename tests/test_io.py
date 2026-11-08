"""Tests for light curve CSV loading."""

from pathlib import Path

import pandas as pd
import pytest

from supernova_analyzer import load_light_curve


def test_load_light_curve_returns_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "light_curve.csv"
    csv_path.write_text(
        "time,magnitude,magnitude_error\n0.0,18.7,0.12\n",
        encoding="utf-8",
    )

    result = load_light_curve(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["time", "magnitude", "magnitude_error"]
    assert len(result) == 1


def test_load_light_curve_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_column.csv"
    csv_path.write_text("time,magnitude\n0.0,18.7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="magnitude_error"):
        load_light_curve(csv_path)


def test_load_light_curve_rejects_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_light_curve(csv_path)
