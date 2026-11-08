"""Plotting utilities for supernova light curve data."""

from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from supernova_analyzer.analysis import Dm15Result, PeakResult


def plot_light_curve(
    light_curve: pd.DataFrame,
    output_path: str | PathLike[str] = "outputs/light_curve.png",
    peak: PeakResult | None = None,
    dm15: Dm15Result | None = None,
) -> Path:
    """Plot a light curve and save it as an image.

    The observations are plotted in ascending time order without modifying the
    input DataFrame. The magnitude axis is inverted because smaller astronomical
    magnitudes represent brighter objects.

    Args:
        light_curve: Light curve data with ``time``, ``magnitude``, and
            ``magnitude_error`` columns.
        output_path: Destination path for the generated image.
        peak: Optional observed peak to highlight with a marker and annotation.
        dm15: Optional linear 15-day decline baseline result to visualize.

    Returns:
        The path where the image was saved.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sorted_light_curve = light_curve.sort_values("time")

    figure, axis = plt.subplots()
    try:
        axis.errorbar(
            sorted_light_curve["time"],
            sorted_light_curve["magnitude"],
            yerr=sorted_light_curve["magnitude_error"],
            marker="o",
            linestyle="-",
            capsize=3,
            label="Light curve",
        )
        if peak is not None:
            axis.scatter(
                peak.peak_time,
                peak.peak_magnitude,
                marker="*",
                s=180,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
                label="Observed peak",
            )
        if dm15 is not None:
            axis.axvline(
                dm15.peak_time,
                linestyle="--",
                linewidth=1.2,
                label="Peak time",
            )
            axis.axvline(
                dm15.target_time,
                linestyle=":",
                linewidth=1.5,
                label="Peak + 15 days",
            )
            axis.scatter(
                dm15.target_time,
                dm15.interpolated_magnitude,
                marker="s",
                s=70,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
                label="Peak + 15 interpolated",
            )
        axis.invert_yaxis()
        axis.set_title("Supernova Light Curve")
        axis.set_xlabel("Time (days)")
        axis.set_ylabel("Magnitude")
        axis.grid(visible=True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination)
    finally:
        plt.close(figure)

    return destination
