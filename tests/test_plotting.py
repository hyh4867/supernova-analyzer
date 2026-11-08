"""Tests for light curve plotting."""

from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import pandas as pd
from pandas.testing import assert_frame_equal

from supernova_analyzer import (
    calculate_dm15_linear,
    detect_observed_peak,
    plot_light_curve,
)


def make_light_curve() -> pd.DataFrame:
    """Create an unsorted light curve for plotting tests."""
    return pd.DataFrame(
        {
            "time": [2.0, 0.0, 1.0],
            "magnitude": [17.6, 18.7, 18.1],
            "magnitude_error": [0.08, 0.12, 0.10],
        }
    )


def test_plot_light_curve_saves_image(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "light_curve.png"

    result = plot_light_curve(make_light_curve(), output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_light_curve_inverts_y_axis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inversion_count = 0
    original_invert_yaxis = matplotlib.axes.Axes.invert_yaxis

    def record_inversion(axis: matplotlib.axes.Axes) -> None:
        nonlocal inversion_count
        inversion_count += 1
        original_invert_yaxis(axis)

    monkeypatch.setattr(matplotlib.axes.Axes, "invert_yaxis", record_inversion)

    plot_light_curve(make_light_curve(), tmp_path / "light_curve.png")

    assert inversion_count == 1


def test_plot_light_curve_does_not_modify_input(tmp_path: Path) -> None:
    light_curve = make_light_curve()
    original = light_curve.copy(deep=True)

    plot_light_curve(light_curve, tmp_path / "light_curve.png")

    assert_frame_equal(light_curve, original)


def test_plot_light_curve_saves_image_with_peak_annotation(tmp_path: Path) -> None:
    light_curve = make_light_curve()
    peak = detect_observed_peak(light_curve)
    output_path = tmp_path / "light_curve_with_peak.png"

    result = plot_light_curve(light_curve, output_path, peak=peak)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_light_curve_saves_image_with_peak_and_dm15(tmp_path: Path) -> None:
    light_curve = pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "magnitude": [17.0, 18.0, 19.0],
            "magnitude_error": [0.1, 0.1, 0.2],
        }
    )
    peak = detect_observed_peak(light_curve)
    dm15 = calculate_dm15_linear(light_curve, peak)
    output_path = tmp_path / "light_curve_with_dm15.png"

    result = plot_light_curve(light_curve, output_path, peak=peak, dm15=dm15)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_light_curve_uses_day_axis_without_floating_annotations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    light_curve = pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "magnitude": [17.0, 18.0, 19.0],
            "magnitude_error": [0.1, 0.1, 0.2],
        }
    )
    peak = detect_observed_peak(light_curve)
    dm15 = calculate_dm15_linear(light_curve, peak)
    captured_axes: list[matplotlib.axes.Axes] = []
    original_savefig = matplotlib.figure.Figure.savefig

    def inspect_before_save(figure, *args, **kwargs) -> None:
        captured_axes.extend(figure.axes)
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", inspect_before_save)

    plot_light_curve(
        light_curve,
        tmp_path / "light_curve.png",
        peak=peak,
        dm15=dm15,
    )

    assert captured_axes[0].get_xlabel() == "Time (days)"
    assert len(captured_axes[0].texts) == 0
