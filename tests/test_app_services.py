"""Tests for Streamlit-independent application services."""

from dataclasses import FrozenInstanceError

import matplotlib.pyplot as plt
import pytest

from supernova_analyzer.analysis import Dm15Result, PeakResult
from supernova_analyzer.app_services import (
    basic_analysis_summary,
    basic_summary_text,
    figure_to_png_bytes,
    parse_bounds,
    parse_optional_number,
    result_to_json,
    scientific_combined_summary,
    scientific_input_fingerprint,
    validate_gui_number,
)
from supernova_analyzer.salt3_diagnostics import Salt3Diagnostics
from supernova_analyzer.salt3_fitting import Salt3FitResult
from supernova_analyzer.standardization import (
    Salt3AmplitudeMagnitudeResult,
    SyntheticRestFrameMagnitudeResult,
    TrippStandardizationResult,
)


def _fit_result() -> Salt3FitResult:
    return Salt3FitResult(
        source_name="salt3",
        source_version="2.0",
        sncosmo_version="2.13.0",
        redshift=0.5,
        t0=55000.0,
        t0_error=0.2,
        x0=1.0e-5,
        x0_error=1.0e-7,
        x1=0.3,
        x1_error=0.1,
        c=0.02,
        c_error=0.01,
        chi2=10.0,
        ndof=8,
        reduced_chi2=1.25,
        ncall=12,
        nfit=4,
        n_observations_total=3,
        n_observations_used=2,
        varied_parameter_names=("t0", "x0", "x1", "c"),
        covariance=None,
        data_mask=(True, False, True),
        model_covariance_included=True,
        fit_message="valid",
    )


def _diagnostics() -> Salt3Diagnostics:
    return Salt3Diagnostics(
        time=(1.0, 3.0),
        band=("sdssg", "sdssr"),
        observed_flux=(10.0, 12.0),
        observed_flux_error=(1.0, 1.0),
        model_flux=(9.5, 12.5),
        raw_residual=(0.5, -0.5),
        normalized_residual_observation_only=(0.5, -0.5),
        model_variance_diagonal=None,
        total_variance_diagonal=None,
        normalized_residual_total_diagonal=None,
        data_mask=(True, False, True),
        number_used=2,
        residual_mean=0.0,
        residual_rms=0.5,
        normalized_residual_mean=0.0,
        normalized_residual_rms=0.5,
    )


def _amplitude() -> Salt3AmplitudeMagnitudeResult:
    return Salt3AmplitudeMagnitudeResult(
        source_name="salt3",
        source_version="2.0",
        magnitude_system="ab",
        reference_band="bessellb",
        normalization_constant=10.6,
        x0=1.0e-5,
        apparent_magnitude=23.1,
        apparent_magnitude_error=0.02,
        method="x0-only",
        fit_fingerprint="fingerprint",
    )


def _synthetic() -> SyntheticRestFrameMagnitudeResult:
    return SyntheticRestFrameMagnitudeResult(
        band_name="bessellb",
        magnitude_system="ab",
        phase=0.0,
        observer_time=55000.0,
        apparent_magnitude=23.2,
        apparent_magnitude_error=0.03,
        method="spectral diagnostic",
        source_name="salt3",
        source_version="2.0",
        fit_fingerprint="fingerprint",
    )


def _tripp() -> TrippStandardizationResult:
    return TrippStandardizationResult(
        apparent_magnitude=23.1,
        apparent_magnitude_error=0.02,
        x1=0.3,
        x1_error=0.1,
        color=0.02,
        color_error=0.01,
        alpha=0.14,
        beta=3.1,
        absolute_magnitude=-19.3,
        distance_modulus=42.38,
        distance_modulus_error_statistical=0.04,
        covariance_included=True,
        formula="mu = m_B - M + alpha*x1 - beta*c",
    )


def _fingerprint(**overrides: object) -> str:
    arguments = {
        "redshift": 0.5,
        "include_model_covariance": True,
        "min_snr": 5.0,
        "initial_t0": None,
        "bounds": None,
    }
    arguments.update(overrides)
    return scientific_input_fingerprint(b"photometry", **arguments)


def test_scientific_fingerprint_is_deterministic() -> None:
    assert _fingerprint() == _fingerprint()
    assert _fingerprint(bounds={"x1": (-2.0, 2.0)}) == _fingerprint(
        bounds={"x1": (-2.0, 2.0)}
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("file", None),
        ("redshift", 0.6),
        ("include_model_covariance", False),
        ("min_snr", 4.0),
        ("initial_t0", 55000.0),
        ("bounds", {"x1": (-2.0, 2.0)}),
    ),
)
def test_scientific_fingerprint_changes_with_fit_inputs(
    name: str, value: object
) -> None:
    baseline = _fingerprint()
    if name == "file":
        changed = scientific_input_fingerprint(
            b"different photometry",
            redshift=0.5,
            include_model_covariance=True,
            min_snr=5.0,
            initial_t0=None,
            bounds=None,
        )
    else:
        changed = _fingerprint(**{name: value})
    assert changed != baseline


def test_standardization_controls_are_not_fingerprint_inputs() -> None:
    function_parameters = scientific_input_fingerprint.__annotations__
    assert "alpha" not in function_parameters
    assert "beta" not in function_parameters
    assert "absolute_magnitude" not in function_parameters
    assert _fingerprint() == _fingerprint()


def test_result_json_serializes_dataclasses_and_tuples() -> None:
    serialized = result_to_json(_fit_result())
    assert '"source_name": "salt3"' in serialized
    assert '"data_mask": [' in serialized
    assert "NaN" not in serialized


def test_result_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        result_to_json({"value": float("nan")})


def test_basic_summary_generation_and_text() -> None:
    peak = PeakResult(2, 4.0, 15.0)
    decline = Dm15Result(4.0, 15.0, 19.0, 16.1, 1.1, "linear")
    summary = basic_analysis_summary(peak, decline, 9)
    text = basic_summary_text(summary)
    assert summary["number_of_observations"] == 9
    assert summary["observed_peak"]["peak_time"] == 4.0
    assert "Linear 15-day decline baseline: 1.1 mag" in text


def test_scientific_combined_summary_contains_core_results() -> None:
    text = scientific_combined_summary(
        _fit_result(),
        _diagnostics(),
        _amplitude(),
        _synthetic(),
        _tripp(),
    )
    assert "Scientific SALT3 Analysis" in text
    assert "x0=1.00000000e-05" in text
    assert "Illustrative Tripp mu: 42.38" in text


def test_figure_to_png_bytes_returns_png_without_closing_figure() -> None:
    figure, axis = plt.subplots()
    try:
        axis.plot([0.0, 1.0], [1.0, 0.0])
        payload = figure_to_png_bytes(figure)
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert plt.fignum_exists(figure.number)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("value", (1, 2.5))
def test_validate_gui_number_accepts_valid_values(value: float) -> None:
    assert validate_gui_number(value, "value", minimum=0.0) == float(value)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (True, "numeric"),
        ("1", "numeric"),
        (float("inf"), "finite"),
        (-1.0, "at least"),
    ),
)
def test_validate_gui_number_rejects_invalid_values(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_gui_number(value, "value", minimum=0.0)


def test_optional_number_and_bounds_validation() -> None:
    assert parse_optional_number("", "initial t0") is None
    assert parse_optional_number(" 4.2 ", "initial t0") == 4.2
    assert parse_bounds({"x1": ("-2", "2"), "c": ("", "")}) == {
        "x1": (-2.0, 2.0)
    }
    with pytest.raises(ValueError, match="Both bounds"):
        parse_bounds({"x1": ("-2", "")})
    with pytest.raises(ValueError, match="less than"):
        parse_bounds({"x1": ("2", "-2")})


def test_summary_helpers_do_not_mutate_backend_results() -> None:
    fit = _fit_result()
    original = result_to_json(fit)
    scientific_combined_summary(
        fit,
        _diagnostics(),
        _amplitude(),
        _synthetic(),
        _tripp(),
    )
    assert result_to_json(fit) == original
    with pytest.raises(FrozenInstanceError):
        fit.x1 = 4.0  # type: ignore[misc]
