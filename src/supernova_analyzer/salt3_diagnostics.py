"""Diagnostics and scientific plotting for fitted SALT3 light curves."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.figure import Figure

from supernova_analyzer.salt3_fitting import (
    Salt3FitResult,
    build_fitted_salt3_model,
)
from supernova_analyzer.scientific_io import (
    _validated_scientific_photometry_copy,
)

REQUIRED_MODEL_PARAMETERS = ("z", "t0", "x0", "x1", "c")
PLOT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


class Salt3DiagnosticError(RuntimeError):
    """Raised when SALT3 diagnostics or scientific plotting cannot be computed."""


@dataclass(frozen=True, slots=True)
class Salt3Diagnostics:
    """Immutable, JSON-friendly residual diagnostics for used observations.

    The total-diagonal normalized residual is a visualization approximation.
    It does not decompose or reconstruct the full-covariance fit chi-square.
    """

    time: tuple[float, ...]
    band: tuple[str, ...]
    observed_flux: tuple[float, ...]
    observed_flux_error: tuple[float, ...]
    model_flux: tuple[float, ...]
    raw_residual: tuple[float, ...]
    normalized_residual_observation_only: tuple[float, ...]
    model_variance_diagonal: tuple[float, ...] | None
    total_variance_diagonal: tuple[float, ...] | None
    normalized_residual_total_diagonal: tuple[float, ...] | None
    data_mask: tuple[bool, ...]
    number_used: int
    residual_mean: float
    residual_rms: float
    normalized_residual_mean: float
    normalized_residual_rms: float


def _finite_array(values: object, name: str) -> np.ndarray:
    """Convert values to a finite float array or raise a diagnostic error."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise Salt3DiagnosticError(f"{name} must contain only finite values.")
    return array


def _validate_model_parameters(model: object) -> None:
    """Ensure a reconstructed model exposes all required SALT3 parameters."""
    names = tuple(getattr(model, "param_names", ()))
    missing = [name for name in REQUIRED_MODEL_PARAMETERS if name not in names]
    if missing:
        raise Salt3DiagnosticError(
            "Fitted SALT3 model is missing required parameters: "
            + ", ".join(missing)
            + "."
        )


def calculate_salt3_diagnostics(
    photometry: Table,
    fit_result: Salt3FitResult,
    include_model_covariance: bool | None = None,
) -> Salt3Diagnostics:
    """Calculate residual diagnostics for observations used by a SALT3 fit.

    Observation-only normalized residuals use ``fluxerr``. When model
    covariance is requested, the model covariance diagonal is added to
    ``fluxerr**2`` and a total-diagonal normalized residual is also returned.
    That diagonal pull is intended for visualization only; it is not an exact
    decomposition of the full-covariance chi-square used during fitting.

    Args:
        photometry: Scientific multi-band flux photometry.
        fit_result: Immutable result from ``fit_salt3``.
        include_model_covariance: Whether to calculate model covariance. If
            omitted, use the setting recorded in the fit result.

    Returns:
        Immutable diagnostics for observations selected by ``data_mask``.

    Raises:
        ValueError: If the photometry schema is invalid.
        Salt3DiagnosticError: If masks, model values, or variances are invalid.
    """
    table = _validated_scientific_photometry_copy(photometry)
    if len(fit_result.data_mask) != len(table):
        raise Salt3DiagnosticError(
            "fit_result.data_mask length must match the photometry row count."
        )

    data_mask = tuple(bool(value) for value in fit_result.data_mask)
    mask = np.asarray(data_mask, dtype=bool)
    number_used = int(mask.sum())
    if number_used == 0:
        raise Salt3DiagnosticError("At least one observation must be used.")
    if number_used != fit_result.n_observations_used:
        raise Salt3DiagnosticError(
            "data_mask used count must match fit_result.n_observations_used."
        )

    used = table[mask]
    time = _finite_array(used["time"], "Observation time")
    observed_flux = _finite_array(used["flux"], "Observed flux")
    flux_error = _finite_array(used["fluxerr"], "Observed flux error")
    zp = _finite_array(used["zp"], "Zeropoint")
    band = np.asarray(used["band"], dtype=str)
    zpsys = np.asarray(used["zpsys"], dtype=str)

    model = build_fitted_salt3_model(fit_result)
    _validate_model_parameters(model)
    use_model_covariance = (
        fit_result.model_covariance_included
        if include_model_covariance is None
        else bool(include_model_covariance)
    )

    try:
        if use_model_covariance:
            model_values, covariance = model.bandfluxcov(
                band,
                time,
                zp=zp,
                zpsys=zpsys,
            )
        else:
            model_values = model.bandflux(band, time, zp=zp, zpsys=zpsys)
            covariance = None
    except (OSError, RuntimeError, ValueError) as error:
        operation = "model covariance" if use_model_covariance else "model flux"
        raise Salt3DiagnosticError(
            f"Unable to calculate SALT3 {operation}: {error}"
        ) from error

    model_flux = _finite_array(model_values, "SALT3 model flux")
    if model_flux.shape != observed_flux.shape:
        raise Salt3DiagnosticError(
            "SALT3 model flux shape must match the used observations."
        )
    raw_residual = observed_flux - model_flux
    observation_pull = raw_residual / flux_error

    model_variance: np.ndarray | None = None
    total_variance: np.ndarray | None = None
    total_pull: np.ndarray | None = None
    if covariance is not None:
        covariance_array = _finite_array(covariance, "SALT3 model covariance")
        expected_shape = (number_used, number_used)
        if covariance_array.shape != expected_shape:
            raise Salt3DiagnosticError(
                f"SALT3 model covariance has shape {covariance_array.shape}; "
                f"expected {expected_shape}."
            )
        model_variance = np.diag(covariance_array).copy()
        if np.any(model_variance < 0.0):
            raise Salt3DiagnosticError(
                "SALT3 model variance diagonal must be non-negative."
            )
        total_variance = flux_error**2 + model_variance
        if np.any(total_variance <= 0.0):
            raise Salt3DiagnosticError(
                "Total variance diagonal must be greater than zero."
            )
        total_pull = raw_residual / np.sqrt(total_variance)

    for values, name in (
        (raw_residual, "Raw residual"),
        (observation_pull, "Observation-only normalized residual"),
    ):
        if not np.isfinite(values).all():
            raise Salt3DiagnosticError(f"{name} must contain only finite values.")
    if total_pull is not None and not np.isfinite(total_pull).all():
        raise Salt3DiagnosticError(
            "Total-diagonal normalized residual must contain only finite values."
        )

    summary_pull = total_pull if total_pull is not None else observation_pull
    return Salt3Diagnostics(
        time=tuple(float(value) for value in time),
        band=tuple(str(value) for value in band),
        observed_flux=tuple(float(value) for value in observed_flux),
        observed_flux_error=tuple(float(value) for value in flux_error),
        model_flux=tuple(float(value) for value in model_flux),
        raw_residual=tuple(float(value) for value in raw_residual),
        normalized_residual_observation_only=tuple(
            float(value) for value in observation_pull
        ),
        model_variance_diagonal=(
            None
            if model_variance is None
            else tuple(float(value) for value in model_variance)
        ),
        total_variance_diagonal=(
            None
            if total_variance is None
            else tuple(float(value) for value in total_variance)
        ),
        normalized_residual_total_diagonal=(
            None if total_pull is None else tuple(float(value) for value in total_pull)
        ),
        data_mask=data_mask,
        number_used=number_used,
        residual_mean=float(np.mean(raw_residual)),
        residual_rms=float(np.sqrt(np.mean(raw_residual**2))),
        normalized_residual_mean=float(np.mean(summary_pull)),
        normalized_residual_rms=float(np.sqrt(np.mean(summary_pull**2))),
    )


def _parameter_text(name: str, value: float, error: float | None) -> str:
    """Format a fitted parameter with an optional uncertainty."""
    if error is None:
        return f"{name}={value:.4g}"
    return f"{name}={value:.4g} ± {error:.3g}"


def _plot_coordinate(
    time: np.ndarray,
    fit_result: Salt3FitResult,
    phase_axis: bool,
) -> np.ndarray:
    """Convert observer-frame time to the selected plot coordinate."""
    if phase_axis:
        return (time - fit_result.t0) / (1.0 + fit_result.redshift)
    return time


def _prepare_plot_inputs(
    photometry: Table,
    fit_result: Salt3FitResult,
    diagnostics: Salt3Diagnostics | None,
) -> tuple[Table, Salt3Diagnostics, np.ndarray, tuple[str, ...]]:
    """Validate shared plot inputs without changing scientific values."""
    table = _validated_scientific_photometry_copy(photometry)
    if len(fit_result.data_mask) != len(table):
        raise Salt3DiagnosticError(
            "fit_result.data_mask length must match the photometry row count."
        )
    diagnostics = diagnostics or calculate_salt3_diagnostics(table, fit_result)
    if diagnostics.data_mask != tuple(bool(value) for value in fit_result.data_mask):
        raise Salt3DiagnosticError(
            "Diagnostics data_mask does not match the SALT3 fit result."
        )
    mask = np.asarray(fit_result.data_mask, dtype=bool)
    bands = tuple(dict.fromkeys(str(value) for value in table["band"]))
    return table, diagnostics, mask, bands


def _selected_residuals(
    diagnostics: Salt3Diagnostics,
    show_normalized_residuals: bool,
) -> tuple[np.ndarray, str]:
    """Select existing diagnostic residual values for plotting."""
    if not show_normalized_residuals:
        return np.asarray(diagnostics.raw_residual), "Raw residual"
    selected = (
        diagnostics.normalized_residual_total_diagonal
        or diagnostics.normalized_residual_observation_only
    )
    label = (
        "Normalized residual\n(total diagonal)"
        if diagnostics.normalized_residual_total_diagonal is not None
        else "Normalized residual\n(observation only)"
    )
    return np.asarray(selected), label


def _draw_salt3_panels(
    table: Table,
    fit_result: Salt3FitResult,
    diagnostics: Salt3Diagnostics,
    mask: np.ndarray,
    bands: tuple[str, ...],
    *,
    flux_axis: object | None,
    residual_axis: object | None,
    show_normalized_residuals: bool,
    phase_axis: bool,
) -> None:
    """Draw requested SALT3 panels from one shared plotting implementation."""
    model = None
    if flux_axis is not None:
        model = build_fitted_salt3_model(fit_result)
        _validate_model_parameters(model)
    diagnostics_time = np.asarray(diagnostics.time)
    diagnostics_band = np.asarray(diagnostics.band)
    residual_values, residual_label = _selected_residuals(
        diagnostics,
        show_normalized_residuals,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for band_index, band_name in enumerate(bands):
        band_mask = np.asarray(table["band"], dtype=str) == band_name
        used_band_mask = band_mask & mask
        if not used_band_mask.any():
            continue

        zpsys_values = {str(value) for value in table["zpsys"][band_mask]}
        if len(zpsys_values) != 1:
            raise Salt3DiagnosticError(
                f"Band '{band_name}' has inconsistent zpsys values."
            )
        reference_zpsys = next(iter(zpsys_values))
        reference_zp = float(np.asarray(table["zp"])[used_band_mask][0])
        scale = 10.0 ** (
            0.4 * (reference_zp - np.asarray(table["zp"], dtype=float)[band_mask])
        )
        band_time = np.asarray(table["time"], dtype=float)[band_mask]
        band_flux = np.asarray(table["flux"], dtype=float)[band_mask] * scale
        band_error = np.asarray(table["fluxerr"], dtype=float)[band_mask] * scale
        band_used = mask[band_mask]
        band_x = _plot_coordinate(band_time, fit_result, phase_axis)
        marker = PLOT_MARKERS[band_index % len(PLOT_MARKERS)]
        color = colors[band_index % len(colors)]

        if flux_axis is not None:
            flux_axis.errorbar(
                band_x[band_used],
                band_flux[band_used],
                yerr=band_error[band_used],
                fmt=marker,
                color=color,
                capsize=2,
                label=f"{band_name} used",
            )
            if (~band_used).any():
                flux_axis.errorbar(
                    band_x[~band_used],
                    band_flux[~band_used],
                    yerr=band_error[~band_used],
                    fmt=marker,
                    linestyle="none",
                    color=color,
                    alpha=0.55,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    label=f"{band_name} excluded",
                )

            used_times = band_time[band_used]
            span = float(used_times.max() - used_times.min())
            padding = max(1.0, 0.05 * span)
            curve_time = np.linspace(
                used_times.min() - padding,
                used_times.max() + padding,
                200,
            )
            try:
                curve_flux = _finite_array(
                    model.bandflux(
                        band_name,
                        curve_time,
                        zp=reference_zp,
                        zpsys=reference_zpsys,
                    ),
                    f"SALT3 model curve for band '{band_name}'",
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise Salt3DiagnosticError(
                    f"Unable to calculate SALT3 curve for band "
                    f"'{band_name}': {error}"
                ) from error
            flux_axis.plot(
                _plot_coordinate(curve_time, fit_result, phase_axis),
                curve_flux,
                color=color,
                label=f"{band_name} SALT3",
            )

        if residual_axis is not None:
            diagnostic_mask = diagnostics_band == band_name
            band_residual_values = residual_values[diagnostic_mask]
            if not show_normalized_residuals:
                used_zp = np.asarray(table["zp"], dtype=float)[used_band_mask]
                residual_scale = 10.0 ** (0.4 * (reference_zp - used_zp))
                band_residual_values = band_residual_values * residual_scale
            residual_axis.plot(
                _plot_coordinate(
                    diagnostics_time[diagnostic_mask], fit_result, phase_axis
                ),
                band_residual_values,
                marker=marker,
                linestyle="none",
                color=color,
                label=band_name,
            )

    t0_coordinate = 0.0 if phase_axis else fit_result.t0
    x_label = "Rest-frame phase (days)" if phase_axis else "Observer-frame time"
    if flux_axis is not None:
        flux_axis.axvline(
            t0_coordinate,
            linestyle="--",
            color="black",
            label="t0",
        )
        flux_axis.set_ylabel("Flux")
        flux_axis.grid(visible=True, alpha=0.25)
        flux_axis.legend(ncols=2, fontsize="small")
    if residual_axis is not None:
        residual_axis.axvline(t0_coordinate, linestyle="--", color="black")
        residual_axis.axhline(0.0, color="black", linewidth=1.0)
        residual_axis.set_ylabel(residual_label)
        residual_axis.set_xlabel(x_label)
        residual_axis.grid(visible=True, alpha=0.25)
        residual_axis.legend(
            ncols=max(1, min(4, len(bands))),
            fontsize="small",
        )
    elif flux_axis is not None:
        flux_axis.set_xlabel(x_label)


def _fit_plot_title(fit_result: Salt3FitResult) -> str:
    """Return a standalone scientific title shared by fit figures."""
    return (
        f"{fit_result.source_name.upper()} {fit_result.source_version} fit | "
        f"z={fit_result.redshift:.4g} | "
        f"{_parameter_text('x1', fit_result.x1, fit_result.x1_error)} | "
        f"{_parameter_text('c', fit_result.c, fit_result.c_error)}\n"
        f"χ²/ndof={fit_result.chi2:.4g}/{fit_result.ndof} | "
        f"reduced χ²={fit_result.reduced_chi2:.4g}"
    )


def _save_optional_figure(
    figure: Figure,
    output_path: str | PathLike[str] | None,
) -> None:
    """Save a figure when requested without changing its open state."""
    if output_path is None:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)


def plot_salt3_fit(
    photometry: Table,
    fit_result: Salt3FitResult,
    diagnostics: Salt3Diagnostics | None = None,
    output_path: str | PathLike[str] | None = None,
    show_normalized_residuals: bool = True,
    phase_axis: bool = True,
) -> Figure:
    """Plot the backward-compatible combined SALT3 fit and residual panels."""
    table, diagnostics, mask, bands = _prepare_plot_inputs(
        photometry,
        fit_result,
        diagnostics,
    )
    figure, (flux_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    _draw_salt3_panels(
        table,
        fit_result,
        diagnostics,
        mask,
        bands,
        flux_axis=flux_axis,
        residual_axis=residual_axis,
        show_normalized_residuals=show_normalized_residuals,
        phase_axis=phase_axis,
    )
    figure.suptitle(_fit_plot_title(fit_result))
    figure.tight_layout()
    _save_optional_figure(figure, output_path)
    return figure


def plot_salt3_light_curves(
    photometry: Table,
    fit_result: Salt3FitResult,
    diagnostics: Salt3Diagnostics | None = None,
    output_path: str | PathLike[str] | None = None,
    phase_axis: bool = True,
) -> Figure:
    """Plot only multi-band observations and fitted SALT3 light curves."""
    table, diagnostics, mask, bands = _prepare_plot_inputs(
        photometry,
        fit_result,
        diagnostics,
    )
    figure, flux_axis = plt.subplots(figsize=(10, 5))
    _draw_salt3_panels(
        table,
        fit_result,
        diagnostics,
        mask,
        bands,
        flux_axis=flux_axis,
        residual_axis=None,
        show_normalized_residuals=True,
        phase_axis=phase_axis,
    )
    figure.suptitle(_fit_plot_title(fit_result))
    figure.tight_layout()
    _save_optional_figure(figure, output_path)
    return figure


def plot_salt3_residuals(
    photometry: Table,
    fit_result: Salt3FitResult,
    diagnostics: Salt3Diagnostics | None = None,
    output_path: str | PathLike[str] | None = None,
    phase_axis: bool = True,
) -> Figure:
    """Plot existing normalized residual diagnostics in a shallow figure."""
    table, diagnostics, mask, bands = _prepare_plot_inputs(
        photometry,
        fit_result,
        diagnostics,
    )
    figure, residual_axis = plt.subplots(figsize=(10, 3))
    _draw_salt3_panels(
        table,
        fit_result,
        diagnostics,
        mask,
        bands,
        flux_axis=None,
        residual_axis=residual_axis,
        show_normalized_residuals=True,
        phase_axis=phase_axis,
    )
    figure.suptitle(
        f"SALT3 Residual Diagnostics | "
        f"reduced χ²={fit_result.reduced_chi2:.4g}"
    )
    figure.tight_layout()
    _save_optional_figure(figure, output_path)
    return figure
