"""Offline unit tests for the SALT3 fitting core."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from astropy.table import Table

import supernova_analyzer.salt3_fitting as fitting
from supernova_analyzer import (
    Salt3FitError,
    Salt3FitResult,
    Salt3ModelUnavailableError,
    build_fitted_salt3_model,
    fit_salt3,
)


class FakeModel:
    """Minimal sncosmo.Model test double."""

    param_names = ["z", "t0", "x0", "x1", "c"]

    def __init__(self, source: object = None) -> None:
        self.source = source
        self.values = {"z": 0.0, "t0": 0.0, "x0": 1.0, "x1": 0.0, "c": 0.0}

    def set(self, **parameters: float) -> None:
        self.values.update(parameters)

    def get(self, name: str) -> float:
        return self.values[name]


def make_photometry() -> Table:
    """Create valid multi-band photometry without external registry access."""
    return Table(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "band": ["g", "r", "g", "r", "g", "r"],
            "flux": [1.0, 1.2, 1.5, 1.1, 0.8, -0.1],
            "fluxerr": [0.1] * 6,
            "zp": [25.0] * 6,
            "zpsys": ["ab"] * 6,
        }
    )


def make_fit_result(**overrides: Any) -> SimpleNamespace:
    """Create a successful sncosmo-like optimization result."""
    values = {
        "success": True,
        "message": "Optimization terminated successfully.",
        "ncall": 42,
        "chisq": 12.0,
        "ndof": 2,
        "param_names": ["z", "t0", "x0", "x1", "c"],
        "parameters": np.array([0.5, 2.5, 0.02, 0.8, -0.1]),
        "vparam_names": ["t0", "x0", "x1", "c"],
        "covariance": np.diag([0.04, 0.0001, 0.09, 0.0025]),
        "errors": {"t0": 0.2, "x0": 0.01, "x1": 0.3, "c": 0.05},
        "nfit": 2,
        "data_mask": np.array([True, True, False, True, True, False]),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def mocked_sncosmo(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace model registry and optimizer calls with offline test doubles."""
    calls: dict[str, Any] = {}
    source = object()

    def fake_get_source(name: str, version: str) -> object:
        calls["source_request"] = (name, version)
        return source

    def fake_model_factory(source: object) -> FakeModel:
        calls.setdefault("models", []).append(source)
        return FakeModel(source)

    def fake_fit_lc(
        data: Table,
        model: FakeModel,
        varied_parameters: list[str],
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, FakeModel]:
        calls["fit_data"] = data
        calls["input_model"] = model
        calls["varied_parameters"] = varied_parameters
        calls["fit_kwargs"] = kwargs
        fitted_model = FakeModel(source=model.source)
        fitted_model.set(z=0.5, t0=2.5, x0=0.02, x1=0.8, c=-0.1)
        return make_fit_result(), fitted_model

    monkeypatch.setattr(fitting.sncosmo, "get_source", fake_get_source)
    monkeypatch.setattr(fitting.sncosmo, "Model", fake_model_factory)
    monkeypatch.setattr(fitting.sncosmo, "fit_lc", fake_fit_lc)
    calls["source"] = source
    return calls


@pytest.mark.parametrize("redshift", ["0.5", True, np.nan, np.inf, -0.1])
def test_fit_salt3_validates_redshift(redshift: object) -> None:
    with pytest.raises(ValueError, match="redshift"):
        fit_salt3(make_photometry(), redshift)  # type: ignore[arg-type]


@pytest.mark.parametrize("min_snr", ["5", True, np.nan, np.inf, 0.0, -1.0])
def test_fit_salt3_validates_min_snr(min_snr: object) -> None:
    with pytest.raises(ValueError, match="min_snr"):
        fit_salt3(make_photometry(), 0.5, min_snr=min_snr)  # type: ignore[arg-type]


def test_fit_salt3_preserves_input_table(
    mocked_sncosmo: dict[str, Any],
) -> None:
    photometry = make_photometry()
    original = photometry.copy(copy_data=True)

    fit_salt3(photometry, 0.5)
    mocked_sncosmo["fit_data"]["flux"][0] = 999.0

    assert np.array_equal(photometry.as_array(), original.as_array())


def test_fit_salt3_pins_source_and_fixes_redshift(
    mocked_sncosmo: dict[str, Any],
) -> None:
    fit_salt3(make_photometry(), 0.5)

    assert mocked_sncosmo["source_request"] == ("salt3", "2.0")
    assert mocked_sncosmo["input_model"].get("z") == 0.5
    assert mocked_sncosmo["varied_parameters"] == ["t0", "x0", "x1", "c"]
    assert "z" not in mocked_sncosmo["varied_parameters"]


def test_fit_salt3_forwards_covariance_bounds_and_guesses(
    mocked_sncosmo: dict[str, Any],
) -> None:
    bounds = {"x1": (-3.0, 3.0), "c": (-0.3, 0.3)}

    fit_salt3(
        make_photometry(),
        0.5,
        include_model_covariance=False,
        bounds=bounds,
        min_snr=7.0,
    )

    kwargs = mocked_sncosmo["fit_kwargs"]
    assert kwargs["modelcov"] is False
    assert kwargs["bounds"] == bounds
    assert kwargs["guess_amplitude"] is True
    assert kwargs["guess_t0"] is True
    assert kwargs["guess_z"] is False
    assert kwargs["minsnr"] == 7.0


def test_fit_salt3_respects_initial_t0(mocked_sncosmo: dict[str, Any]) -> None:
    fit_salt3(make_photometry(), 0.5, initial_t0=55000.0)

    assert mocked_sncosmo["input_model"].get("t0") == 55000.0
    assert mocked_sncosmo["fit_kwargs"]["guess_t0"] is False


def test_fit_salt3_maps_successful_fitted_model_and_statistics(
    mocked_sncosmo: dict[str, Any],
) -> None:
    result = fit_salt3(make_photometry(), 0.5)

    assert isinstance(result, Salt3FitResult)
    assert result.source_name == "salt3"
    assert result.source_version == "2.0"
    assert result.redshift == 0.5
    assert (result.t0, result.x0, result.x1, result.c) == (2.5, 0.02, 0.8, -0.1)
    assert (result.t0_error, result.x0_error) == (0.2, 0.01)
    assert (result.x1_error, result.c_error) == (0.3, 0.05)
    assert result.chi2 == 12.0
    assert result.ndof == 2
    assert result.reduced_chi2 == 6.0
    assert result.ncall == 42
    assert result.nfit == 2
    assert result.model_covariance_included is True
    assert result.fit_message == "Optimization terminated successfully."


def test_fit_salt3_converts_covariance_and_mask_to_immutable_values(
    mocked_sncosmo: dict[str, Any],
) -> None:
    result = fit_salt3(make_photometry(), 0.5)

    assert isinstance(result.covariance, tuple)
    assert all(isinstance(row, tuple) for row in result.covariance or ())
    assert result.covariance == (
        (0.04, 0.0, 0.0, 0.0),
        (0.0, 0.0001, 0.0, 0.0),
        (0.0, 0.0, 0.09, 0.0),
        (0.0, 0.0, 0.0, 0.0025),
    )
    assert result.data_mask == (True, True, False, True, True, False)
    assert result.n_observations_total == 6
    assert result.n_observations_used == 4
    with pytest.raises(FrozenInstanceError):
        result.chi2 = 1.0  # type: ignore[misc]


def test_fit_salt3_allows_missing_uncertainties_and_covariance(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = fitting.sncosmo.fit_lc

    def fit_without_uncertainty(*args: Any, **kwargs: Any):
        result, model = original_fit(*args, **kwargs)
        result.errors = None
        result.covariance = None
        return result, model

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", fit_without_uncertainty)

    result = fit_salt3(make_photometry(), 0.5)

    assert result.covariance is None
    assert result.t0_error is None
    assert result.x0_error is None
    assert result.x1_error is None
    assert result.c_error is None


def test_fit_salt3_rejects_unsuccessful_result(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_fit(*args: Any, **kwargs: Any):
        return make_fit_result(success=False, message="Minimizer failed."), FakeModel()

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", failed_fit)

    with pytest.raises(Salt3FitError, match="Minimizer failed"):
        fit_salt3(make_photometry(), 0.5)


@pytest.mark.parametrize("parameter", ["t0", "x0", "x1", "c"])
def test_fit_salt3_rejects_non_finite_fitted_parameter(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    parameter: str,
) -> None:
    def non_finite_fit(*args: Any, **kwargs: Any):
        model = FakeModel()
        model.set(t0=2.5, x0=0.02, x1=0.8, c=-0.1)
        model.set(**{parameter: np.nan})
        return make_fit_result(), model

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", non_finite_fit)

    with pytest.raises(Salt3FitError, match=rf"{parameter}.*not finite"):
        fit_salt3(make_photometry(), 0.5)


def test_fit_salt3_rejects_non_finite_chi_square(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_fit(*args: Any, **kwargs: Any):
        model = FakeModel()
        model.set(t0=2.5, x0=0.02, x1=0.8, c=-0.1)
        return make_fit_result(chisq=np.inf), model

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", invalid_fit)

    with pytest.raises(Salt3FitError, match="chi-square.*not finite"):
        fit_salt3(make_photometry(), 0.5)


def test_fit_salt3_rejects_non_positive_ndof(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_fit(*args: Any, **kwargs: Any):
        model = FakeModel()
        model.set(t0=2.5, x0=0.02, x1=0.8, c=-0.1)
        return make_fit_result(ndof=0), model

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", invalid_fit)

    with pytest.raises(Salt3FitError, match="positive degrees of freedom"):
        fit_salt3(make_photometry(), 0.5)


def test_fit_salt3_rejects_wrong_covariance_shape(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = fitting.sncosmo.fit_lc

    def invalid_fit(*args: Any, **kwargs: Any):
        result, model = original_fit(*args, **kwargs)
        result.covariance = np.eye(3)
        return result, model

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", invalid_fit)

    with pytest.raises(Salt3FitError, match="covariance has shape"):
        fit_salt3(make_photometry(), 0.5)


def test_fit_salt3_rejects_missing_result_parameter(
    mocked_sncosmo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = fitting.sncosmo.fit_lc

    def invalid_fit(*args: Any, **kwargs: Any):
        result, model = original_fit(*args, **kwargs)
        result.param_names = ["z", "t0", "x0", "x1"]
        return result, model

    monkeypatch.setattr(fitting.sncosmo, "fit_lc", invalid_fit)

    with pytest.raises(Salt3FitError, match="missing parameters: c"):
        fit_salt3(make_photometry(), 0.5)


def test_fit_salt3_wraps_model_loading_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = OSError("cache unavailable")

    def fail_source(name: str, version: str) -> object:
        raise original_error

    monkeypatch.setattr(fitting.sncosmo, "get_source", fail_source)

    with pytest.raises(Salt3ModelUnavailableError) as captured:
        fit_salt3(make_photometry(), 0.5)

    assert captured.value.__cause__ is original_error
    assert "internet connection" in str(captured.value)
    assert "Astropy/sncosmo cache" in str(captured.value)


def test_build_fitted_salt3_model_restores_parameters(
    mocked_sncosmo: dict[str, Any],
) -> None:
    result = fit_salt3(make_photometry(), 0.5)

    rebuilt = build_fitted_salt3_model(result)

    assert mocked_sncosmo["source_request"] == ("salt3", "2.0")
    assert rebuilt is not mocked_sncosmo["input_model"]
    assert rebuilt.values == {
        "z": 0.5,
        "t0": 2.5,
        "x0": 0.02,
        "x1": 0.8,
        "c": -0.1,
    }
