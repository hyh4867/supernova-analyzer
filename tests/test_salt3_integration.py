"""Opt-in integration test for a real SALT3 fit."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
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

RUN_INTEGRATION = os.getenv("RUN_SNCOSMO_INTEGRATION") == "1"
SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "scientific"
    / "sample_multiband_photometry.csv"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set RUN_SNCOSMO_INTEGRATION=1 to run the real SALT3 fit.",
)
def test_real_salt3_fit_on_sncosmo_example_data(tmp_path: Path) -> None:
    photometry = load_scientific_photometry(SAMPLE_PATH)

    # This is a demonstration assumption for the sncosmo example dataset,
    # not an externally measured spectroscopic redshift.
    result = fit_salt3(photometry, redshift=0.5)

    assert all(np.isfinite([result.t0, result.x0, result.x1, result.c]))
    assert result.x0 > 0.0
    assert np.isfinite(result.chi2)
    assert np.isfinite(result.reduced_chi2)
    assert result.ndof > 0
    assert result.n_observations_used > 4

    diagnostics = calculate_salt3_diagnostics(photometry, result)
    diagnostic_values = (
        diagnostics.model_flux
        + diagnostics.raw_residual
        + diagnostics.normalized_residual_observation_only
    )
    assert all(np.isfinite(diagnostic_values))
    assert len(diagnostics.raw_residual) == result.n_observations_used

    output_path = tmp_path / "salt3_fit.png"
    figure = plot_salt3_fit(
        photometry,
        result,
        diagnostics=diagnostics,
        output_path=output_path,
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    plt.close(figure)

    amplitude_magnitude = calculate_salt3_amplitude_magnitude(result)
    synthetic_magnitude = calculate_synthetic_restframe_b_magnitude(result)
    assert np.isfinite(amplitude_magnitude.normalization_constant)
    assert np.isfinite(amplitude_magnitude.apparent_magnitude)
    assert amplitude_magnitude.apparent_magnitude_error is not None
    assert np.isfinite(amplitude_magnitude.apparent_magnitude_error)
    assert np.isfinite(synthetic_magnitude.apparent_magnitude)
    assert synthetic_magnitude.apparent_magnitude_error is not None
    assert np.isfinite(synthetic_magnitude.apparent_magnitude_error)

    shape_changed_result = replace(result, x1=result.x1 + 0.2, c=result.c - 0.05)
    changed_amplitude = calculate_salt3_amplitude_magnitude(shape_changed_result)
    changed_synthetic = calculate_synthetic_restframe_b_magnitude(
        shape_changed_result
    )
    assert changed_amplitude.apparent_magnitude == pytest.approx(
        amplitude_magnitude.apparent_magnitude
    )
    assert changed_synthetic.apparent_magnitude != pytest.approx(
        synthetic_magnitude.apparent_magnitude
    )

    shifted_synthetic = calculate_synthetic_restframe_b_magnitude(
        replace(result, t0=result.t0 + 10000.0)
    )
    assert shifted_synthetic.apparent_magnitude == pytest.approx(
        synthetic_magnitude.apparent_magnitude
    )

    # These nuisance parameters are demonstration values, not fitted or
    # externally calibrated truth for this example object.
    standardization = calculate_tripp_standardization(
        result,
        amplitude_magnitude,
        alpha=0.14,
        beta=3.1,
        absolute_magnitude=-19.3,
    )
    assert np.isfinite(standardization.distance_modulus)
    assert standardization.distance_modulus_error_statistical is not None
    assert np.isfinite(standardization.distance_modulus_error_statistical)

    amplitude_path = tmp_path / "salt3_amplitude_magnitude.json"
    synthetic_path = tmp_path / "synthetic_restframe_b_magnitude.json"
    standardization_path = tmp_path / "tripp_standardization.json"
    amplitude_path.write_text(
        json.dumps(asdict(amplitude_magnitude), allow_nan=False)
    )
    synthetic_path.write_text(
        json.dumps(asdict(synthetic_magnitude), allow_nan=False)
    )
    standardization_path.write_text(
        json.dumps(asdict(standardization), allow_nan=False)
    )
    assert np.isfinite(
        json.loads(amplitude_path.read_text())["normalization_constant"]
    )
    assert np.isfinite(
        json.loads(synthetic_path.read_text())["apparent_magnitude"]
    )
    assert np.isfinite(
        json.loads(standardization_path.read_text())["distance_modulus"]
    )
