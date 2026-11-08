"""Run a demonstration SALT3 fit on the bundled sncosmo example data."""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from supernova_analyzer import (
    calculate_salt3_diagnostics,
    calculate_salt3_amplitude_magnitude,
    calculate_synthetic_restframe_b_magnitude,
    calculate_tripp_standardization,
    fit_salt3,
    load_scientific_photometry,
    plot_salt3_fit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "scientific" / "sample_multiband_photometry.csv"
FIT_RESULT_PATH = PROJECT_ROOT / "outputs" / "salt3_fit_result.json"
DIAGNOSTICS_PATH = PROJECT_ROOT / "outputs" / "salt3_diagnostics.json"
PLOT_PATH = PROJECT_ROOT / "outputs" / "salt3_fit.png"
AMPLITUDE_MAGNITUDE_PATH = PROJECT_ROOT / "outputs" / "salt3_amplitude_magnitude.json"
SYNTHETIC_MAGNITUDE_PATH = (
    PROJECT_ROOT / "outputs" / "synthetic_restframe_b_magnitude.json"
)
STANDARDIZATION_PATH = PROJECT_ROOT / "outputs" / "tripp_standardization.json"
DEMONSTRATION_REDSHIFT = 0.5
DEMONSTRATION_ALPHA = 0.14
DEMONSTRATION_BETA = 3.1
DEMONSTRATION_ABSOLUTE_MAGNITUDE = -19.3


def main() -> None:
    """Fit the sample photometry and write a JSON-friendly result summary."""
    print(
        "Using redshift=0.5 as a demonstration assumption; it is not an "
        "externally measured spectroscopic value."
    )
    photometry = load_scientific_photometry(INPUT_PATH)
    result = fit_salt3(
        photometry,
        redshift=DEMONSTRATION_REDSHIFT,
        include_model_covariance=True,
    )

    print("SALT3 source: salt3 version 2.0")
    print(f"t0 = {result.t0:.8g} ± {result.t0_error}")
    print(f"x0 = {result.x0:.8g} ± {result.x0_error}")
    print(f"x1 = {result.x1:.8g} ± {result.x1_error}")
    print(f"c = {result.c:.8g} ± {result.c_error}")
    print(f"chi-square = {result.chi2:.8g}")
    print(f"ndof = {result.ndof}")
    print(f"reduced chi-square = {result.reduced_chi2:.8g}")
    print(
        f"observations used = {result.n_observations_used}/"
        f"{result.n_observations_total}"
    )

    diagnostics = calculate_salt3_diagnostics(photometry, result)
    print(f"raw residual mean = {diagnostics.residual_mean:.8g}")
    print(f"raw residual RMS = {diagnostics.residual_rms:.8g}")
    print(
        "normalized residual mean = "
        f"{diagnostics.normalized_residual_mean:.8g}"
    )
    print(
        "normalized residual RMS = "
        f"{diagnostics.normalized_residual_rms:.8g}"
    )
    print(
        "Normalized residuals use the total diagonal variance for "
        "visualization; they do not decompose the full-covariance chi-square."
    )

    amplitude_magnitude = calculate_salt3_amplitude_magnitude(result)
    synthetic_magnitude = calculate_synthetic_restframe_b_magnitude(result)
    standardization = calculate_tripp_standardization(
        result,
        amplitude_magnitude,
        alpha=DEMONSTRATION_ALPHA,
        beta=DEMONSTRATION_BETA,
        absolute_magnitude=DEMONSTRATION_ABSOLUTE_MAGNITUDE,
    )
    print(
        "Tripp m_B convention: x0-only SALT3 amplitude magnitude with a "
        "project-derived model normalization."
    )
    print(
        "Derived normalization constant = "
        f"{amplitude_magnitude.normalization_constant:.8g} "
        f"({amplitude_magnitude.reference_band}, "
        f"{amplitude_magnitude.magnitude_system})"
    )
    print(
        f"Amplitude-based m_B = {amplitude_magnitude.apparent_magnitude:.8g} ± "
        f"{amplitude_magnitude.apparent_magnitude_error}"
    )
    print(
        "Full synthetic rest-frame B magnitude = "
        f"{synthetic_magnitude.apparent_magnitude:.8g} ± "
        f"{synthetic_magnitude.apparent_magnitude_error}"
    )
    print(
        "Synthetic minus amplitude magnitude = "
        f"{synthetic_magnitude.apparent_magnitude - amplitude_magnitude.apparent_magnitude:.8g}"
    )
    print(
        "Demonstration-only calibration: "
        f"alpha={DEMONSTRATION_ALPHA}, beta={DEMONSTRATION_BETA}, "
        f"M={DEMONSTRATION_ABSOLUTE_MAGNITUDE}"
    )
    print(
        f"Tripp-like mu = {standardization.distance_modulus:.8g} ± "
        f"{standardization.distance_modulus_error_statistical} "
        "(fit-statistical only)"
    )
    print(
        "WARNING: This is an illustrative single-object standardization, not "
        "a calibrated survey-level distance or cosmology measurement."
    )

    FIT_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_RESULT_PATH.write_text(
        json.dumps(asdict(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    DIAGNOSTICS_PATH.write_text(
        json.dumps(asdict(diagnostics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    AMPLITUDE_MAGNITUDE_PATH.write_text(
        json.dumps(asdict(amplitude_magnitude), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    SYNTHETIC_MAGNITUDE_PATH.write_text(
        json.dumps(asdict(synthetic_magnitude), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    STANDARDIZATION_PATH.write_text(
        json.dumps(asdict(standardization), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure = plot_salt3_fit(
        photometry,
        result,
        diagnostics=diagnostics,
        output_path=PLOT_PATH,
    )
    plt.close(figure)
    print(f"Fit result JSON: {FIT_RESULT_PATH}")
    print(f"Diagnostics JSON: {DIAGNOSTICS_PATH}")
    print(f"Amplitude magnitude JSON: {AMPLITUDE_MAGNITUDE_PATH}")
    print(f"Synthetic magnitude JSON: {SYNTHETIC_MAGNITUDE_PATH}")
    print(f"Tripp-like standardization JSON: {STANDARDIZATION_PATH}")
    print(f"Diagnostic plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
