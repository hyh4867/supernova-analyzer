"""Offline tests for SALT3 residual diagnostics and scientific plotting."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table
from matplotlib.figure import Figure

import supernova_analyzer.salt3_diagnostics as diagnostic_module
from supernova_analyzer import (
    Salt3DiagnosticError,
    Salt3FitResult,
    calculate_salt3_diagnostics,
    plot_salt3_fit,
    plot_salt3_light_curves,
    plot_salt3_residuals,
)


class FakeDiagnosticModel:
    """Predictable SALT3 model double for offline diagnostics."""

    param_names = ["z", "t0", "x0", "x1", "c"]

    def __init__(self) -> None:
        self.bandflux_calls: list[dict[str, Any]] = []
        self.raise_on_covariance = False

    def bandflux(
        self,
        band: object,
        time: object,
        zp: object = None,
        zpsys: object = None,
    ) -> np.ndarray:
        self.bandflux_calls.append(
            {"band": band, "time": time, "zp": zp, "zpsys": zpsys}
        )
        times = np.asarray(time, dtype=float)
        return 2.0 * times

    def bandfluxcov(
        self,
        band: object,
        time: object,
        zp: object = None,
        zpsys: object = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.raise_on_covariance:
            raise ValueError("covariance unavailable")
        flux = self.bandflux(band, time, zp=zp, zpsys=zpsys)
        return flux, np.eye(len(flux)) * 0.25


def make_photometry() -> Table:
    """Create sorted, calibrated multi-band photometry."""
    return Table(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "band": ["g", "r", "g", "r"],
            "flux": [3.0, 5.0, 7.0, 9.0],
            "fluxerr": [1.0, 1.0, 2.0, 1.0],
            "zp": [25.0, 25.0, 25.0, 25.0],
            "zpsys": ["ab", "ab", "ab", "ab"],
        }
    )


def make_fit_result(**overrides: Any) -> Salt3FitResult:
    """Create a compact immutable fit result for diagnostics."""
    values: dict[str, Any] = {
        "source_name": "salt3",
        "source_version": "2.0",
        "sncosmo_version": "2.13.0",
        "redshift": 0.5,
        "t0": 1.0,
        "t0_error": 0.1,
        "x0": 0.01,
        "x0_error": 0.001,
        "x1": 0.5,
        "x1_error": 0.2,
        "c": 0.1,
        "c_error": 0.03,
        "chi2": 4.0,
        "ndof": 2,
        "reduced_chi2": 2.0,
        "ncall": 10,
        "nfit": 2,
        "n_observations_total": 4,
        "n_observations_used": 3,
        "varied_parameter_names": ("t0", "x0", "x1", "c"),
        "covariance": None,
        "data_mask": (True, False, True, True),
        "model_covariance_included": True,
        "fit_message": "success",
    }
    values.update(overrides)
    return Salt3FitResult(**values)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeDiagnosticModel:
    """Replace fitted-model reconstruction with a deterministic model."""
    model = FakeDiagnosticModel()
    monkeypatch.setattr(
        diagnostic_module,
        "build_fitted_salt3_model",
        lambda result: model,
    )
    return model


def test_calculate_diagnostics_flux_and_residuals(
    fake_model: FakeDiagnosticModel,
) -> None:
    diagnostics = calculate_salt3_diagnostics(make_photometry(), make_fit_result())

    assert diagnostics.time == (1.0, 3.0, 4.0)
    assert diagnostics.band == ("g", "g", "r")
    assert diagnostics.observed_flux == (3.0, 7.0, 9.0)
    assert diagnostics.model_flux == (2.0, 6.0, 8.0)
    assert diagnostics.raw_residual == (1.0, 1.0, 1.0)
    assert diagnostics.normalized_residual_observation_only == (1.0, 0.5, 1.0)
    assert diagnostics.data_mask == (True, False, True, True)
    assert diagnostics.number_used == 3


def test_calculate_diagnostics_model_and_total_variance(
    fake_model: FakeDiagnosticModel,
) -> None:
    diagnostics = calculate_salt3_diagnostics(make_photometry(), make_fit_result())

    assert diagnostics.model_variance_diagonal == (0.25, 0.25, 0.25)
    assert diagnostics.total_variance_diagonal == (1.25, 4.25, 1.25)
    assert diagnostics.normalized_residual_total_diagonal == pytest.approx(
        (1.0 / np.sqrt(1.25), 1.0 / np.sqrt(4.25), 1.0 / np.sqrt(1.25))
    )


def test_calculate_diagnostics_summary_statistics(
    fake_model: FakeDiagnosticModel,
) -> None:
    diagnostics = calculate_salt3_diagnostics(make_photometry(), make_fit_result())
    expected_pull = np.array(diagnostics.normalized_residual_total_diagonal)

    assert diagnostics.residual_mean == pytest.approx(1.0)
    assert diagnostics.residual_rms == pytest.approx(1.0)
    assert diagnostics.normalized_residual_mean == pytest.approx(expected_pull.mean())
    assert diagnostics.normalized_residual_rms == pytest.approx(
        np.sqrt(np.mean(expected_pull**2))
    )


def test_calculate_diagnostics_can_use_observation_only_summary(
    fake_model: FakeDiagnosticModel,
) -> None:
    diagnostics = calculate_salt3_diagnostics(
        make_photometry(),
        make_fit_result(),
        include_model_covariance=False,
    )

    assert diagnostics.model_variance_diagonal is None
    assert diagnostics.total_variance_diagonal is None
    assert diagnostics.normalized_residual_total_diagonal is None
    assert diagnostics.normalized_residual_mean == pytest.approx(5.0 / 6.0)
    assert diagnostics.normalized_residual_rms == pytest.approx(
        np.sqrt((1.0 + 0.25 + 1.0) / 3.0)
    )


def test_calculate_diagnostics_rejects_mismatched_mask_length(
    fake_model: FakeDiagnosticModel,
) -> None:
    result = make_fit_result(data_mask=(True,), n_observations_used=1)

    with pytest.raises(Salt3DiagnosticError, match="data_mask length"):
        calculate_salt3_diagnostics(make_photometry(), result)


def test_calculate_diagnostics_rejects_non_finite_model_flux(
    fake_model: FakeDiagnosticModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fake_model,
        "bandfluxcov",
        lambda *args, **kwargs: (np.array([2.0, np.nan, 8.0]), np.eye(3)),
    )

    with pytest.raises(Salt3DiagnosticError, match="model flux.*finite"):
        calculate_salt3_diagnostics(make_photometry(), make_fit_result())


def test_calculate_diagnostics_rejects_negative_model_variance(
    fake_model: FakeDiagnosticModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    covariance = np.diag([0.25, -0.1, 0.25])
    monkeypatch.setattr(
        fake_model,
        "bandfluxcov",
        lambda *args, **kwargs: (np.array([2.0, 6.0, 8.0]), covariance),
    )

    with pytest.raises(Salt3DiagnosticError, match="variance.*non-negative"):
        calculate_salt3_diagnostics(make_photometry(), make_fit_result())


def test_calculate_diagnostics_reports_covariance_failure(
    fake_model: FakeDiagnosticModel,
) -> None:
    fake_model.raise_on_covariance = True

    with pytest.raises(Salt3DiagnosticError, match="model covariance"):
        calculate_salt3_diagnostics(make_photometry(), make_fit_result())


def test_calculate_diagnostics_preserves_inputs(
    fake_model: FakeDiagnosticModel,
) -> None:
    photometry = make_photometry()
    original_table = photometry.copy(copy_data=True)
    result = make_fit_result()
    original_result = deepcopy(result)

    calculate_salt3_diagnostics(photometry, result)

    assert np.array_equal(photometry.as_array(), original_table.as_array())
    assert result == original_result


def test_calculate_diagnostics_rejects_missing_model_parameters(
    fake_model: FakeDiagnosticModel,
) -> None:
    fake_model.param_names = ["z", "t0", "x0", "c"]

    with pytest.raises(Salt3DiagnosticError, match="missing.*x1"):
        calculate_salt3_diagnostics(make_photometry(), make_fit_result())


def test_rest_frame_phase_transformation() -> None:
    result = make_fit_result(t0=10.0, redshift=0.5)

    phase = diagnostic_module._plot_coordinate(
        np.array([10.0, 13.0]), result, phase_axis=True
    )

    assert phase == pytest.approx([0.0, 2.0])


def test_plot_salt3_fit_returns_figure_and_saves_output(
    fake_model: FakeDiagnosticModel,
    tmp_path: Path,
) -> None:
    photometry = make_photometry()
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)
    output_path = tmp_path / "nested" / "salt3_fit.png"

    figure = plot_salt3_fit(
        photometry,
        result,
        diagnostics=diagnostics,
        output_path=output_path,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert figure.axes[1].get_xlabel() == "Rest-frame phase (days)"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    plt.close(figure)


def test_standalone_salt3_plots_return_figures_and_nonempty_pngs(
    fake_model: FakeDiagnosticModel,
    tmp_path: Path,
) -> None:
    photometry = make_photometry()
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)
    light_curve_path = tmp_path / "salt3_light_curves.png"
    residual_path = tmp_path / "salt3_residuals.png"

    light_curve_figure = plot_salt3_light_curves(
        photometry,
        result,
        diagnostics=diagnostics,
        output_path=light_curve_path,
    )
    residual_figure = plot_salt3_residuals(
        photometry,
        result,
        diagnostics=diagnostics,
        output_path=residual_path,
    )

    assert isinstance(light_curve_figure, Figure)
    assert isinstance(residual_figure, Figure)
    assert len(light_curve_figure.axes) == 1
    assert len(residual_figure.axes) == 1
    assert light_curve_path.stat().st_size > 0
    assert residual_path.stat().st_size > 0
    assert light_curve_figure.get_figwidth() / light_curve_figure.get_figheight() == 2
    assert residual_figure.get_figwidth() / residual_figure.get_figheight() > 3
    plt.close(light_curve_figure)
    plt.close(residual_figure)


def test_standalone_residual_plot_uses_existing_diagnostic_values_exactly(
    fake_model: FakeDiagnosticModel,
) -> None:
    photometry = make_photometry()
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)
    expected = np.asarray(diagnostics.normalized_residual_total_diagonal)

    figure = plot_salt3_residuals(
        photometry,
        result,
        diagnostics=diagnostics,
    )

    plotted_by_band = {
        line.get_label(): np.asarray(line.get_ydata())
        for line in figure.axes[0].lines
        if line.get_label() in {"g", "r"}
    }
    diagnostic_bands = np.asarray(diagnostics.band)
    assert plotted_by_band["g"] == pytest.approx(expected[diagnostic_bands == "g"])
    assert plotted_by_band["r"] == pytest.approx(expected[diagnostic_bands == "r"])
    plt.close(figure)


def test_plot_salt3_fit_generates_curve_for_each_band(
    fake_model: FakeDiagnosticModel,
) -> None:
    photometry = make_photometry()
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)
    fake_model.bandflux_calls.clear()

    figure = plot_salt3_fit(photometry, result, diagnostics=diagnostics)

    curve_bands = {
        call["band"] for call in fake_model.bandflux_calls if isinstance(call["band"], str)
    }
    assert curve_bands == {"g", "r"}
    plt.close(figure)


def test_plot_salt3_fit_uses_distinct_band_markers(
    fake_model: FakeDiagnosticModel,
) -> None:
    photometry = make_photometry()
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)

    figure = plot_salt3_fit(photometry, result, diagnostics=diagnostics)

    used_containers = {
        container.get_label(): container
        for container in figure.axes[0].containers
        if container.get_label().endswith(" used")
    }
    assert used_containers["g used"].lines[0].get_marker() == "o"
    assert used_containers["r used"].lines[0].get_marker() == "s"
    assert "χ²/ndof" in figure._suptitle.get_text()
    plt.close(figure)


def test_plot_salt3_fit_rejects_inconsistent_zpsys(
    fake_model: FakeDiagnosticModel,
) -> None:
    photometry = make_photometry()
    photometry["zpsys"] = ["ab", "ab", "vega", "ab"]
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)

    with pytest.raises(Salt3DiagnosticError, match="inconsistent zpsys"):
        plot_salt3_fit(photometry, result, diagnostics=diagnostics)


def test_plot_salt3_fit_handles_varying_zp_with_reference_scale(
    fake_model: FakeDiagnosticModel,
) -> None:
    photometry = make_photometry()
    photometry["zp"][2] = 26.0
    result = make_fit_result()
    diagnostics = calculate_salt3_diagnostics(photometry, result)
    fake_model.bandflux_calls.clear()

    figure = plot_salt3_fit(photometry, result, diagnostics=diagnostics)

    g_curve_call = next(
        call for call in fake_model.bandflux_calls if call["band"] == "g"
    )
    assert g_curve_call["zp"] == 25.0
    assert g_curve_call["zpsys"] == "ab"
    plt.close(figure)
