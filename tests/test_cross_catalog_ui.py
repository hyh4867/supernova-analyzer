"""Pure presentation-helper tests for the cross-catalog Streamlit UI."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from cross_catalog_ui import (
    PATH_ENVIRONMENT_NAMES,
    binned_comparison_frame,
    catalog_availability,
    comparison_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


def test_catalog_availability_reports_each_catalog_independently(tmp_path) -> None:
    des_data = tmp_path / "des.csv"
    des_matrix = tmp_path / "des.npz"
    des_data.touch()
    des_matrix.touch()
    status = catalog_availability(
        {
            "DES_DOVEKIE_HD_PATH": str(des_data),
            "DES_DOVEKIE_STAT_SYS_PATH": str(des_matrix),
            "PANTHEONPLUS_DATA_PATH": "",
            "PANTHEONPLUS_STAT_SYS_PATH": str(tmp_path / "missing.cov"),
        }
    )

    assert status["des"].available
    assert not status["pantheon"].available
    assert status["pantheon"].missing == (
        "PANTHEONPLUS_DATA_PATH",
        "PANTHEONPLUS_STAT_SYS_PATH",
    )


def test_binned_frame_suppresses_underpopulated_bin_values() -> None:
    sparse = SimpleNamespace(
        lower_edge=0.01,
        upper_edge=0.1,
        first_catalog=SimpleNamespace(row_count=2, median_residual=0.4),
        second_catalog=SimpleNamespace(row_count=8, median_residual=-0.1),
        descriptive_median_difference=None,
    )
    valid = SimpleNamespace(
        lower_edge=0.1,
        upper_edge=0.2,
        first_catalog=SimpleNamespace(row_count=7, median_residual=0.03),
        second_catalog=SimpleNamespace(row_count=9, median_residual=0.01),
        descriptive_median_difference=0.02,
    )
    frame = binned_comparison_frame(SimpleNamespace(bins=(sparse, valid)))

    assert frame.loc[0, "DES median (mag)"] != frame.loc[0, "DES median (mag)"]
    assert frame.loc[0, "DES − Pantheon+ (mag)"] != frame.loc[
        0, "DES − Pantheon+ (mag)"
    ]
    assert frame.loc[1, "DES − Pantheon+ (mag)"] == 0.02


def test_comparison_download_is_json_safe_and_contains_no_matrix() -> None:
    comparison = SimpleNamespace(to_dict=lambda: {"bins": [], "kind": "descriptive"})
    encoded = comparison_json(
        comparison,
        {"des_dovekie": {"catalog_order_fingerprint": "abc"}},
    )
    payload = json.loads(encoded)

    assert payload["comparison"]["kind"] == "descriptive"
    assert payload["provenance"]["des_dovekie"][
        "catalog_order_fingerprint"
    ] == "abc"
    assert "matrix" not in encoded.lower()


def test_app_reports_missing_catalog_paths_without_exception(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    for name in PATH_ENVIRONMENT_NAMES:
        monkeypatch.setenv(name, str(tmp_path / name))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    warnings = [item.value for item in app.warning]
    assert any("DES-Dovekie data unavailable" in value for value in warnings)
    assert any("Pantheon+ data unavailable" in value for value in warnings)
    assert any(
        "Both validated datasets are required" in item.value for item in app.info
    )


def test_app_shows_invalid_catalog_error_instead_of_raising(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    for index, name in enumerate(PATH_ENVIRONMENT_NAMES):
        invalid_path = tmp_path / f"invalid-{index}.txt"
        invalid_path.write_text("not a valid catalog\n", encoding="utf-8")
        monkeypatch.setenv(name, str(invalid_path))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert any(
        "Cross-catalog data could not be validated:" in item.value
        for item in app.error
    )


@pytest.mark.integration
def test_app_renders_official_catalog_analysis_when_paths_are_available(
    monkeypatch,
) -> None:
    paths = {name: os.getenv(name, "") for name in PATH_ENVIRONMENT_NAMES}
    missing = [
        name
        for name, value in paths.items()
        if not value or not Path(value).is_file()
    ]
    if missing:
        pytest.skip(
            "Official cross-catalog inputs are unavailable for: "
            + ", ".join(missing)
        )

    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    for name, value in paths.items():
        monkeypatch.setenv(name, value)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert "Cross-Catalog Explorer" in [item.label for item in app.tabs]
    assert "Both catalogs available — analysis ready" in [
        item.value for item in app.success
    ]
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["DES fitted rows"] == "1820"
    assert metrics["Pantheon+ fitted rows"] == "1624"
