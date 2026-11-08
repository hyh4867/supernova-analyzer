"""Offline smoke tests for the Streamlit application."""

import os
from pathlib import Path
import tomllib

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_with_samples_and_does_not_fit() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    visible_text = " ".join(
        element.value
        for collection in (app.title, app.header, app.subheader, app.markdown)
        for element in collection
        if isinstance(element.value, str)
    )
    assert "Supernova Light-Curve Analyzer" in visible_text
    assert "Basic Analysis" in visible_text
    assert "Scientific SALT3 Fit" in visible_text
    assert any(metric.label == "Observations" for metric in app.metric)
    assert not any("Analysis complete" in item.value for item in app.success)
    assert any(
        "First use may download the pinned SALT3 model" in item.value
        for item in app.caption
    )


def test_basic_result_uses_primary_downloads_and_json_disclosure() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    app.button[0].click().run()

    assert not app.exception
    labels = {item.label for item in app.get("download_button")}
    assert "Download summary TXT" in labels
    assert "Download plot PNG" in labels
    assert "Download Basic Analysis JSON" in labels
    assert any(metric.label == "Observed peak magnitude" for metric in app.metric)


def test_invalid_advanced_input_remains_user_readable() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    app.text_input[0].set_value("not-a-number").run()

    assert not app.exception
    assert any("initial t0 must be numeric" in item.value for item in app.error)


def test_streamlit_light_theme_configuration() -> None:
    config_path = Path(__file__).parents[1] / ".streamlit" / "config.toml"
    with config_path.open("rb") as stream:
        theme = tomllib.load(stream)["theme"]

    assert theme["base"] == "light"
    assert theme["backgroundColor"] == "#FFFFFF"
    assert theme["font"] == "sans-serif"


def test_bundled_sample_paths_are_usable() -> None:
    project_root = Path(__file__).parents[1]
    assert (project_root / "data" / "sample_light_curve.csv").stat().st_size > 0
    assert (
        project_root / "data" / "scientific" / "sample_multiband_photometry.csv"
    ).stat().st_size > 0


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_SNCOSMO_INTEGRATION") != "1",
    reason="Set RUN_SNCOSMO_INTEGRATION=1 to exercise the real Streamlit fit.",
)
def test_scientific_result_downloads_and_state_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "DES_DOVEKIE_HD_PATH",
        "DES_DOVEKIE_STAT_SYS_PATH",
        "PANTHEONPLUS_DATA_PATH",
        "PANTHEONPLUS_STAT_SYS_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)

    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()

    app.button[1].click().run()

    assert not app.exception
    result = app.session_state["scientific_result"]
    original_fit = result["fit"]
    original_fingerprint = result["fingerprint"]
    original_mu = result["tripp"].distance_modulus
    labels = {item.label for item in app.get("download_button")}
    assert "Download combined summary TXT" in labels
    assert "Download SALT3 fit PNG" in labels
    assert "Download residual diagnostics PNG" in labels
    assert "Download combined fit + residual PNG" in labels
    assert {
        "Fit result JSON",
        "Diagnostics JSON",
        "Amplitude m_B JSON",
        "Synthetic B JSON",
        "Tripp result JSON",
    }.issubset(labels)
    visible_values = "\n".join(
        item.value
        for collection in (app.markdown, app.caption)
        for item in collection
        if isinstance(item.value, str)
    )
    assert "..." not in visible_values
    assert f"{original_fit.t0:.2f}" in visible_values
    assert f"{original_fit.x0:.3e}" in visible_values
    assert f"{original_fit.x1:.3f}" in visible_values
    assert f"{original_fit.c:.3f}" in visible_values
    assert f"{result['amplitude'].apparent_magnitude:.3f}" in visible_values
    assert f"{result['tripp'].distance_modulus:.3f}" in visible_values
    assert all(
        len(result[name]) > 0
        for name in ("plot", "light_curve_plot", "residual_plot")
    )

    alpha = next(
        item for item in app.number_input if item.label == "Illustrative alpha"
    )
    alpha.set_value(0.2).run()
    updated = app.session_state["scientific_result"]
    assert updated["fit"] is original_fit
    assert updated["fingerprint"] == original_fingerprint
    assert updated["tripp"].distance_modulus != original_mu

    redshift = next(item for item in app.number_input if item.label == "Fixed redshift")
    redshift.set_value(0.6).run()
    assert any("inputs changed" in item.value for item in app.warning)
    assert not app.success
