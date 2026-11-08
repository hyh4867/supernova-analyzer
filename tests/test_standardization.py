"""Offline tests for SALT3 magnitude conventions and standardization."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

import supernova_analyzer.standardization as standardization_module
from supernova_analyzer import (
    Salt3AmplitudeMagnitudeResult,
    Salt3FitResult,
    StandardizationError,
    SyntheticRestFrameMagnitudeResult,
    TrippStandardizationResult,
    calculate_salt3_amplitude_magnitude,
    calculate_salt3_peak_magnitude,
    calculate_synthetic_restframe_b_magnitude,
    calculate_tripp_standardization,
)


class FakeMagnitudeModel:
    """Analytic model double with a configurable reference normalization."""

    def __init__(self, normalization: float = 30.0) -> None:
        self.normalization = normalization
        self.values = {"z": 0.5, "t0": 10.0, "x0": 0.01, "x1": 0.5, "c": 0.1}
        self.calls: list[tuple[str, str, float, dict[str, float]]] = []

    def set(self, **parameters: float) -> None:
        self.values.update(parameters)

    def bandmag(self, band: str, magsys: str, time: float) -> float:
        self.calls.append((band, magsys, time, self.values.copy()))
        if band == "invalid":
            raise ValueError("unsupported band")
        return (
            self.normalization
            - 2.5 * np.log10(self.values["x0"])
            + 0.2 * self.values["x1"]
            + 0.3 * self.values["c"]
        )


def default_covariance() -> tuple[tuple[float, ...], ...]:
    """Return covariance ordered as t0, x0, x1, c."""
    return (
        (0.01, 0.0, 0.0, 0.0),
        (0.0, 1.0e-8, 0.0, 0.0),
        (0.0, 0.0, 0.04, 0.01),
        (0.0, 0.0, 0.01, 0.01),
    )


def make_fit_result(**overrides: Any) -> Salt3FitResult:
    """Create a deterministic immutable fit result."""
    values: dict[str, Any] = {
        "source_name": "salt3",
        "source_version": "2.0",
        "sncosmo_version": "2.13.0",
        "redshift": 0.5,
        "t0": 10.0,
        "t0_error": 0.1,
        "x0": 0.01,
        "x0_error": 0.0001,
        "x1": 0.5,
        "x1_error": 0.2,
        "c": 0.1,
        "c_error": 0.1,
        "chi2": 10.0,
        "ndof": 6,
        "reduced_chi2": 10.0 / 6.0,
        "ncall": 20,
        "nfit": 2,
        "n_observations_total": 10,
        "n_observations_used": 10,
        "varied_parameter_names": ("t0", "x0", "x1", "c"),
        "covariance": default_covariance(),
        "data_mask": (True,) * 10,
        "model_covariance_included": True,
        "fit_message": "success",
    }
    values.update(overrides)
    return Salt3FitResult(**values)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeMagnitudeModel:
    """Replace model reconstruction with an analytic model double."""
    model = FakeMagnitudeModel()
    monkeypatch.setattr(
        standardization_module,
        "build_fitted_salt3_model",
        lambda result: model,
    )
    return model


def dm_dx0(x0: float) -> float:
    """Return the analytic x0-only magnitude derivative."""
    return -2.5 / (np.log(10.0) * x0)


def test_amplitude_magnitude_depends_only_on_x0(
    fake_model: FakeMagnitudeModel,
) -> None:
    baseline = calculate_salt3_amplitude_magnitude(make_fit_result())
    changed = calculate_salt3_amplitude_magnitude(
        make_fit_result(x1=-2.0, c=0.8)
    )

    assert baseline.apparent_magnitude == pytest.approx(changed.apparent_magnitude)
    assert baseline.apparent_magnitude == pytest.approx(35.0)


def test_normalization_constant_is_derived_from_reference_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeMagnitudeModel(normalization=31.25)
    monkeypatch.setattr(
        standardization_module,
        "build_fitted_salt3_model",
        lambda result: model,
    )

    result = calculate_salt3_amplitude_magnitude(
        make_fit_result(), reference_band="customb", magnitude_system="vega"
    )

    assert result.normalization_constant == pytest.approx(31.25)
    assert result.apparent_magnitude == pytest.approx(36.25)
    reference_call = model.calls[0]
    assert reference_call[0:3] == ("customb", "vega", 0.0)
    assert reference_call[3] == {"z": 0.0, "t0": 0.0, "x0": 1.0, "x1": 0.0, "c": 0.0}


def test_amplitude_uncertainty_is_analytic_x0_only(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()

    result = calculate_salt3_amplitude_magnitude(fit_result)

    expected = abs(dm_dx0(fit_result.x0)) * np.sqrt(1.0e-8)
    assert result.apparent_magnitude_error == pytest.approx(expected)


def test_amplitude_uncertainty_respects_covariance_ordering(
    fake_model: FakeMagnitudeModel,
) -> None:
    names = ("c", "x1", "x0", "t0")
    covariance = np.diag([0.01, 0.04, 9.0e-8, 0.01])
    fit_result = make_fit_result(
        varied_parameter_names=names,
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
    )

    result = calculate_salt3_amplitude_magnitude(fit_result)

    assert result.apparent_magnitude_error == pytest.approx(
        abs(dm_dx0(fit_result.x0)) * 3.0e-4
    )


def test_off_diagonal_covariance_does_not_change_amplitude_uncertainty(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()
    full = calculate_salt3_amplitude_magnitude(fit_result)
    diagonal = np.diag(np.diag(np.asarray(fit_result.covariance)))
    diagonal_result = replace(
        fit_result,
        covariance=tuple(tuple(float(value) for value in row) for row in diagonal),
    )

    diagonal_magnitude = calculate_salt3_amplitude_magnitude(diagonal_result)

    assert full.apparent_magnitude_error == pytest.approx(
        diagonal_magnitude.apparent_magnitude_error
    )


def test_amplitude_magnitude_rejects_non_positive_x0(
    fake_model: FakeMagnitudeModel,
) -> None:
    with pytest.raises(StandardizationError, match="x0 must be positive"):
        calculate_salt3_amplitude_magnitude(make_fit_result(x0=0.0))


def test_amplitude_magnitude_rejects_invalid_reference_band(
    fake_model: FakeMagnitudeModel,
) -> None:
    with pytest.raises(StandardizationError, match="requested convention"):
        calculate_salt3_amplitude_magnitude(
            make_fit_result(),
            reference_band="invalid",
        )


def test_amplitude_without_covariance_has_no_uncertainty(
    fake_model: FakeMagnitudeModel,
) -> None:
    result = calculate_salt3_amplitude_magnitude(make_fit_result(covariance=None))

    assert result.apparent_magnitude_error is None


def test_amplitude_magnitude_rejects_duplicate_covariance_parameter_names(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result(
        varied_parameter_names=("t0", "x0", "x1", "x1"),
    )

    with pytest.raises(StandardizationError, match="must be unique"):
        calculate_salt3_amplitude_magnitude(fit_result)


def test_amplitude_magnitude_rejects_covariance_without_x0(
    fake_model: FakeMagnitudeModel,
) -> None:
    covariance = np.diag([0.01, 0.04, 0.01])
    fit_result = make_fit_result(
        varied_parameter_names=("t0", "x1", "c"),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
    )

    with pytest.raises(StandardizationError, match="does not contain parameter 'x0'"):
        calculate_salt3_amplitude_magnitude(fit_result)


def test_full_synthetic_magnitude_changes_with_x1_and_color(
    fake_model: FakeMagnitudeModel,
) -> None:
    baseline = calculate_synthetic_restframe_b_magnitude(make_fit_result())
    changed = calculate_synthetic_restframe_b_magnitude(
        make_fit_result(x1=1.5, c=-0.2)
    )

    assert baseline.apparent_magnitude != pytest.approx(changed.apparent_magnitude)


def test_full_synthetic_magnitude_rejects_unsupported_reference_band(
    fake_model: FakeMagnitudeModel,
) -> None:
    with pytest.raises(StandardizationError, match="outside model coverage"):
        calculate_synthetic_restframe_b_magnitude(
            make_fit_result(),
            band_name="invalid",
        )


def test_full_synthetic_magnitude_uses_phase_zero_and_z_zero(
    fake_model: FakeMagnitudeModel,
) -> None:
    result = calculate_synthetic_restframe_b_magnitude(make_fit_result())

    assert result.phase == 0.0
    assert result.observer_time == 10.0
    assert all(call[2] == call[3]["t0"] for call in fake_model.calls)
    assert all(call[3]["z"] == 0.0 for call in fake_model.calls)


def test_full_synthetic_magnitude_is_invariant_to_absolute_t0_shift(
    fake_model: FakeMagnitudeModel,
) -> None:
    baseline = calculate_synthetic_restframe_b_magnitude(make_fit_result(t0=10.0))
    shifted = calculate_synthetic_restframe_b_magnitude(make_fit_result(t0=55010.0))

    assert baseline.apparent_magnitude == pytest.approx(shifted.apparent_magnitude)


def test_full_synthetic_uncertainty_has_exactly_zero_t0_derivative(
    fake_model: FakeMagnitudeModel,
) -> None:
    covariance = np.diag([1.0e12, 0.0, 0.0, 0.0])
    fit_result = make_fit_result(
        covariance=tuple(tuple(float(value) for value in row) for row in covariance)
    )

    result = calculate_synthetic_restframe_b_magnitude(fit_result)

    assert result.apparent_magnitude_error == pytest.approx(0.0)


def test_deprecated_peak_wrapper_returns_synthetic_diagnostic(
    fake_model: FakeMagnitudeModel,
) -> None:
    with pytest.deprecated_call(match="calculate_salt3_peak_magnitude"):
        result = calculate_salt3_peak_magnitude(make_fit_result())

    assert isinstance(result, SyntheticRestFrameMagnitudeResult)


def test_deprecated_peak_result_cannot_be_used_for_tripp(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()
    with pytest.deprecated_call():
        deprecated_result = calculate_salt3_peak_magnitude(fit_result)

    with pytest.raises(TypeError, match="diagnostic-only"):
        calculate_tripp_standardization(
            fit_result,
            deprecated_result,  # type: ignore[arg-type]
            alpha=0.14,
            beta=3.1,
            absolute_magnitude=-19.3,
        )


def test_fit_fingerprint_is_stable() -> None:
    first = standardization_module._fit_fingerprint(make_fit_result())
    second = standardization_module._fit_fingerprint(make_fit_result())

    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize("field", ["t0", "x0", "x1", "c", "chi2", "ndof"])
def test_fit_fingerprint_changes_with_fit_content(field: str) -> None:
    fit_result = make_fit_result()
    changed_value = getattr(fit_result, field) + 1

    changed = replace(fit_result, **{field: changed_value})

    assert standardization_module._fit_fingerprint(fit_result) != (
        standardization_module._fit_fingerprint(changed)
    )


def test_tripp_uses_amplitude_magnitude(fake_model: FakeMagnitudeModel) -> None:
    fit_result = make_fit_result()
    amplitude = calculate_salt3_amplitude_magnitude(fit_result)

    result = calculate_tripp_standardization(
        fit_result, amplitude, alpha=0.14, beta=3.1, absolute_magnitude=-19.3
    )

    expected = amplitude.apparent_magnitude + 19.3 + 0.14 * 0.5 - 3.1 * 0.1
    assert result.distance_modulus == pytest.approx(expected)
    assert result.apparent_magnitude == amplitude.apparent_magnitude


def test_tripp_rejects_full_synthetic_magnitude(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()
    synthetic = calculate_synthetic_restframe_b_magnitude(fit_result)

    with pytest.raises(TypeError, match="diagnostic-only"):
        calculate_tripp_standardization(
            fit_result, synthetic, 0.14, 3.1, -19.3  # type: ignore[arg-type]
        )


def test_tripp_analytic_full_covariance_uncertainty(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()
    amplitude = calculate_salt3_amplitude_magnitude(fit_result)
    alpha = 0.14
    beta = 3.1
    jacobian = np.array([0.0, dm_dx0(fit_result.x0), alpha, -beta])
    covariance = np.asarray(fit_result.covariance)

    result = calculate_tripp_standardization(
        fit_result, amplitude, alpha, beta, -19.3
    )

    assert result.distance_modulus_error_statistical == pytest.approx(
        np.sqrt(jacobian @ covariance @ jacobian)
    )


def test_tripp_off_diagonal_covariance_contributes(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()
    amplitude = calculate_salt3_amplitude_magnitude(fit_result)
    full = calculate_tripp_standardization(fit_result, amplitude, 0.14, 3.1, -19.3)
    diagonal = np.diag(np.diag(np.asarray(fit_result.covariance)))
    diagonal_fit = replace(
        fit_result,
        covariance=tuple(tuple(float(value) for value in row) for row in diagonal),
    )
    diagonal_amplitude = calculate_salt3_amplitude_magnitude(diagonal_fit)
    diagonal_result = calculate_tripp_standardization(
        diagonal_fit, diagonal_amplitude, 0.14, 3.1, -19.3
    )

    assert full.distance_modulus_error_statistical != pytest.approx(
        diagonal_result.distance_modulus_error_statistical
    )


def test_tripp_does_not_double_count_reported_errors(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result(x1_error=999.0, c_error=999.0)
    amplitude = replace(
        calculate_salt3_amplitude_magnitude(fit_result),
        apparent_magnitude_error=999.0,
    )

    result = calculate_tripp_standardization(fit_result, amplitude, 0.14, 3.1, -19.3)

    assert result.distance_modulus_error_statistical is not None
    assert result.distance_modulus_error_statistical < 999.0


def test_tripp_rejects_fingerprint_mismatch(fake_model: FakeMagnitudeModel) -> None:
    fit_result = make_fit_result()
    amplitude = calculate_salt3_amplitude_magnitude(fit_result)

    with pytest.raises(ValueError, match="fingerprint"):
        calculate_tripp_standardization(
            replace(fit_result, x1=fit_result.x1 + 0.1),
            amplitude,
            0.14,
            3.1,
            -19.3,
        )


def test_tripp_rejects_source_mismatch(fake_model: FakeMagnitudeModel) -> None:
    fit_result = make_fit_result()
    amplitude = replace(
        calculate_salt3_amplitude_magnitude(fit_result), source_version="different"
    )

    with pytest.raises(ValueError, match="same source"):
        calculate_tripp_standardization(fit_result, amplitude, 0.14, 3.1, -19.3)


@pytest.mark.parametrize(
    ("name", "value"),
    [("alpha", np.nan), ("beta", np.inf), ("absolute_magnitude", "-19.3")],
)
def test_tripp_validates_calibration_inputs(
    fake_model: FakeMagnitudeModel,
    name: str,
    value: object,
) -> None:
    fit_result = make_fit_result()
    amplitude = calculate_salt3_amplitude_magnitude(fit_result)
    arguments: dict[str, object] = {
        "alpha": 0.14,
        "beta": 3.1,
        "absolute_magnitude": -19.3,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        calculate_tripp_standardization(
            fit_result, amplitude, **arguments  # type: ignore[arg-type]
        )


def test_results_are_frozen_and_inputs_unchanged(
    fake_model: FakeMagnitudeModel,
) -> None:
    fit_result = make_fit_result()
    original = replace(fit_result)
    amplitude = calculate_salt3_amplitude_magnitude(fit_result)
    synthetic = calculate_synthetic_restframe_b_magnitude(fit_result)
    tripp = calculate_tripp_standardization(
        fit_result, amplitude, 0.14, 3.1, -19.3
    )

    assert fit_result == original
    assert isinstance(amplitude, Salt3AmplitudeMagnitudeResult)
    assert isinstance(synthetic, SyntheticRestFrameMagnitudeResult)
    assert isinstance(tripp, TrippStandardizationResult)
    with pytest.raises(FrozenInstanceError):
        amplitude.x0 = 1.0  # type: ignore[misc]
