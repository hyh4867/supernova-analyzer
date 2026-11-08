from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
from scipy.linalg import cho_solve, solve_triangular

from supernova_analyzer import (
    build_fitted_salt3_model,
    calculate_dm15_linear,
    calculate_hubble_diagram,
    calculate_hubble_residuals_covariance,
    calculate_hubble_residuals_diagonal,
    calculate_hubble_residuals_precision,
    calculate_salt3_amplitude_magnitude,
    calculate_tripp_standardization,
    catalog_order_fingerprint,
    compare_catalog_residuals,
    detect_observed_peak,
    fit_salt3,
    load_des_dovekie_hd,
    load_light_curve,
    load_packed_precision_matrix,
    load_pantheonplus_catalog,
    load_scientific_photometry,
    maximum_residual_difference,
    normalize_des_dovekie_distances,
    normalize_pantheonplus_distances,
    select_pantheonplus_hubble_flow_subset,
    subset_pantheonplus_covariance,
)
from supernova_analyzer.cross_catalog import common_redshift_range
from supernova_analyzer.hubble_analysis import calculate_reference_distance_modulus
from supernova_analyzer.pantheonplus_analysis import load_pantheonplus_covariance
from supernova_analyzer.salt3_diagnostics import calculate_salt3_diagnostics
from supernova_analyzer.standardization import (
    calculate_salt3_amplitude_magnitude,
    calculate_tripp_standardization,
)

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "projects" / "supernova-analyzer"
REPORT = Path(__file__).resolve().parent
FIGURES = REPORT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 21,
    }
)

SUMMARY: dict[str, object] = {
    "date": "2026-11-06",
    "environment": {
        "python": os.sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sncosmo": sncosmo.__version__,
        "matplotlib_backend": os.environ.get("MPLBACKEND"),
        "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"),
        "astropy_cache_dir": os.environ.get("ASTROPY_CACHE_DIR"),
    },
    "steps": {},
    "figures": [],
}


def _native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_summary() -> None:
    (REPORT / "summary.json").write_text(
        json.dumps(_native(SUMMARY), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_figure(figure: plt.Figure, stem: str) -> None:
    png = FIGURES / f"{stem}.png"
    jpg = FIGURES / f"{stem}.jpg"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(
        jpg,
        dpi=200,
        bbox_inches="tight",
        format="jpg",
        pil_kwargs={"quality": 94},
    )
    plt.close(figure)
    SUMMARY["figures"].append(
        {"stem": stem, "png": f"figures/{png.name}", "jpg": f"figures/{jpg.name}"}
    )


def _step(name: str, function) -> object | None:
    started = time.perf_counter()
    print(f"START {name}", flush=True)
    try:
        result = function()
    except Exception as error:
        SUMMARY["steps"][name] = {
            "status": "FAIL",
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        _write_summary()
        print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
        return None
    SUMMARY["steps"][name] = {
        "status": "PASS",
        "runtime_seconds": time.perf_counter() - started,
        "results": _native(result),
    }
    _write_summary()
    print(f"PASS {name} ({SUMMARY['steps'][name]['runtime_seconds']:.2f}s)", flush=True)
    return result


def _figure_axes(title: str, xlabel: str, ylabel: str, *, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_title(title, pad=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    return fig, ax


def _basic_analysis() -> dict[str, object]:
    sample_path = PROJECT / "data" / "sample_light_curve.csv"
    data = load_light_curve(sample_path)
    peak = detect_observed_peak(data)
    decline = calculate_dm15_linear(data, peak)
    fig, ax = _figure_axes(
        "Basic sample: sampled peak and interpolated 15-day decline",
        "Rest-frame time from peak (days)",
        "Apparent magnitude (mag; brighter upward)",
        figsize=(12, 8),
    )
    ax.errorbar(
        data["time"],
        data["magnitude"],
        yerr=data["magnitude_error"],
        fmt="o",
        color="#1769aa",
        ecolor="#6b91ad",
        capsize=4,
        label="Bundled synthetic observations",
    )
    ax.plot(
        [peak.peak_time, decline.target_time],
        [peak.peak_magnitude, decline.interpolated_magnitude],
        "--",
        linewidth=3,
        color="#d04a35",
        label="15-day linear baseline",
    )
    ax.scatter([peak.peak_time], [peak.peak_magnitude], s=150, marker="*", color="#f2a900", zorder=5, label="Detected observed peak")
    ax.scatter([decline.target_time], [decline.interpolated_magnitude], s=85, marker="s", color="#d04a35", zorder=5, label="Interpolated at +15 d")
    ax.annotate(
        f"peak = ({peak.peak_time:.1f} d, {peak.peak_magnitude:.2f} mag)\n"
        f"Δm15 = {decline.delta_m15:.3f} mag",
        xy=(decline.target_time, decline.interpolated_magnitude),
        xytext=(0.55, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444"},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    _save_figure(fig, "01_basic_sample")
    return {
        "rows": len(data),
        "peak_time_days": peak.peak_time,
        "peak_magnitude_mag": peak.peak_magnitude,
        "dm15_target_time_days": decline.target_time,
        "interpolated_magnitude_mag": decline.interpolated_magnitude,
        "delta_m15_mag": decline.delta_m15,
        "peak_method": "minimum observed sample; earliest minimum after stable time sort",
        "decline_method": "linear interpolation at t_peak + 15 days; no extrapolation",
        "figure_status": "PASS",
    }


def _salt_fit() -> dict[str, object]:
    photometry = load_scientific_photometry(
        PROJECT / "data" / "scientific" / "sample_multiband_photometry.csv"
    )
    result = fit_salt3(
        photometry,
        redshift=0.5,
        include_model_covariance=True,
        bounds=None,
        initial_t0=None,
        min_snr=5.0,
    )
    model = build_fitted_salt3_model(result)
    frame = photometry.to_pandas()
    bands = sorted(frame["band"].astype(str).unique())
    fig, axes = plt.subplots(
        len(bands) + 1,
        1,
        figsize=(13, 3.25 * len(bands) + 3.2),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10").colors
    phases = (frame["time"].to_numpy(float) - result.t0) / (1.0 + result.redshift)
    for index, band in enumerate(bands):
        mask = frame["band"].astype(str).to_numpy() == band
        subset = frame.loc[mask]
        color = colors[index % len(colors)]
        times = subset["time"].to_numpy(float)
        zp = float(subset["zp"].iloc[0])
        zpsys = str(subset["zpsys"].iloc[0])
        prediction = np.asarray(
            model.bandflux(band, times, zp=zp, zpsys=zpsys), dtype=float
        )
        axes[index].errorbar(
            (times - result.t0) / (1.0 + result.redshift),
            subset["flux"],
            yerr=subset["fluxerr"],
            fmt="o",
            color=color,
            capsize=3,
            label=f"{band} observed",
        )
        order = np.argsort(times)
        axes[index].plot(
            (times[order] - result.t0) / (1.0 + result.redshift),
            prediction[order],
            color=color,
            linewidth=2.6,
            label="SALT3 model",
        )
        axes[index].set_ylabel("Flux (zp=25)")
        axes[index].set_title(f"{band} band", loc="left")
        axes[index].grid(alpha=0.2)
        axes[index].legend(loc="best", ncol=2)
        residual_ax = axes[-1]
        residual_ax.errorbar(
            (times - result.t0) / (1.0 + result.redshift),
            subset["flux"].to_numpy(float) - prediction,
            yerr=subset["fluxerr"].to_numpy(float),
            fmt="o",
            color=color,
            capsize=2,
            label=band,
            alpha=0.85,
        )
    axes[-1].axhline(0.0, color="black", linewidth=1.5)
    axes[-1].set_ylabel("Data − model\n(flux units)")
    axes[-1].set_xlabel("Rest-frame phase from fitted t0 (days)")
    axes[-1].set_title("Residuals by band", loc="left")
    axes[-1].grid(alpha=0.2)
    axes[-1].legend(title="Band", ncol=4)
    fig.suptitle("Bundled multi-band sample: SALT3 fit and residuals")
    _save_figure(fig, "02_salt3_fit")
    params = {
        name: {"value": getattr(result, name), "uncertainty": getattr(result, f"{name}_error")}
        for name in ("t0", "x0", "x1", "c")
    }
    standardization: dict[str, object]
    try:
        amplitude = calculate_salt3_amplitude_magnitude(result)
        tripp = calculate_tripp_standardization(
            result,
            amplitude,
            alpha=0.14,
            beta=3.1,
            absolute_magnitude=-19.3,
        )
        standardization = {
            "status": "PASS",
            "m_B_mag": amplitude.apparent_magnitude,
            "m_B_error_mag": amplitude.apparent_magnitude_error,
            "normalization_C_mag": amplitude.normalization_constant,
            "reference_band": amplitude.reference_band,
            "magnitude_system": amplitude.magnitude_system,
            "tripp_mu_demo_mag": tripp.distance_modulus,
            "alpha_demo": tripp.alpha,
            "beta_demo": tripp.beta,
            "M_demo_mag": tripp.absolute_magnitude,
            "origin": "app/script demonstration defaults, not a survey calibration",
        }
    except Exception as error:
        standardization = {
            "status": "CHECK",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    diagnostics = calculate_salt3_diagnostics(photometry, result)
    return {
        "sample_rows": len(photometry),
        "bands": bands,
        "redshift_fixed": result.redshift,
        "source": result.source_name,
        "source_version": result.source_version,
        "sncosmo_version": result.sncosmo_version,
        "parameters": params,
        "chi2": result.chi2,
        "ndof": result.ndof,
        "reduced_chi2": result.reduced_chi2,
        "observations_used": result.n_observations_used,
        "observations_total": result.n_observations_total,
        "model_covariance_included": result.model_covariance_included,
        "fit_message": result.fit_message,
        "bounds": "None passed; sncosmo backend defaults",
        "min_snr": 5.0,
        "diagnostic_raw_residual_rms_flux": diagnostics.residual_rms,
        "diagnostic_normalized_residual_rms": diagnostics.normalized_residual_rms,
        "standardization": standardization,
        "x1_c_screen": {
            "x1_within_broad_release_selection_range": -3.0 < result.x1 < 3.0,
            "c_within_broad_release_selection_range": -0.3 < result.c < 0.3,
            "note": "Pantheon+ broad sample selection ranges; not a universal definition of normal Ia",
        },
        "figure_status": "PASS",
    }


def _injection_recovery() -> dict[str, object]:
    photometry = load_scientific_photometry(
        PROJECT / "data" / "scientific" / "sample_multiband_photometry.csv"
    )
    frame = photometry.to_pandas()
    source = sncosmo.get_source("salt3", version="2.0")
    model = sncosmo.Model(source=source)
    z_true = 0.5
    t0_true = float(np.median(frame["time"].to_numpy(float)))
    x1_true = 0.4
    c_true = 0.05
    model.set(z=z_true, t0=t0_true, x0=1.0, x1=x1_true, c=c_true)
    base_flux = np.empty(len(frame), dtype=float)
    for band in sorted(frame["band"].astype(str).unique()):
        mask = frame["band"].astype(str).to_numpy() == band
        subset = frame.loc[mask]
        base_flux[mask] = model.bandflux(
            band,
            subset["time"].to_numpy(float),
            zp=float(subset["zp"].iloc[0]),
            zpsys=str(subset["zpsys"].iloc[0]),
        )
    target_rms = float(np.sqrt(np.mean(frame["flux"].to_numpy(float) ** 2)))
    x0_true = target_rms / float(np.sqrt(np.mean(base_flux**2)))
    model.set(x0=x0_true)
    noiseless = np.empty(len(frame), dtype=float)
    for band in sorted(frame["band"].astype(str).unique()):
        mask = frame["band"].astype(str).to_numpy() == band
        subset = frame.loc[mask]
        noiseless[mask] = model.bandflux(
            band,
            subset["time"].to_numpy(float),
            zp=float(subset["zp"].iloc[0]),
            zpsys=str(subset["zpsys"].iloc[0]),
        )
    seed = 20261106
    rng = np.random.default_rng(seed)
    error = frame["fluxerr"].to_numpy(float)
    recovered: dict[str, list[float]] = {name: [] for name in ("t0", "x1", "c", "m_B")}
    failures: list[str] = []
    for replicate in range(100):
        injected = photometry.copy(copy_data=True)
        injected["flux"] = noiseless + rng.normal(0.0, error)
        try:
            fit = fit_salt3(
                injected,
                redshift=z_true,
                include_model_covariance=True,
                bounds=None,
                initial_t0=None,
                min_snr=5.0,
            )
            recovered["t0"].append(fit.t0 - t0_true)
            recovered["x1"].append(fit.x1 - x1_true)
            recovered["c"].append(fit.c - c_true)
            recovered["m_B"].append(-2.5 * np.log10(fit.x0 / x0_true))
        except Exception as error:
            failures.append(f"replicate {replicate + 1}: {type(error).__name__}: {error}")
    if any(recovered.values()):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        labels = {
            "t0": ("t0 recovery − truth", "Δt0 (days)"),
            "x1": ("x1 recovery − truth", "Δx1"),
            "c": ("Color recovery − truth", "Δc"),
            "m_B": ("Amplitude magnitude recovery − truth", "ΔmB (mag)"),
        }
        for ax, (name, values) in zip(axes.flat, recovered.items(), strict=True):
            ax.hist(values, bins="auto", color="#2477a8", alpha=0.8, density=True)
            ax.axvline(0.0, color="#b5332b", linewidth=2.5, label="zero bias")
            if len(values) > 1:
                grid = np.linspace(min(values), max(values), 250)
                sd = float(np.std(values, ddof=1))
                if sd > 0:
                    from scipy.stats import norm

                    ax.plot(grid, norm.pdf(grid, float(np.mean(values)), sd), color="#ef8b22", linewidth=2, label="Gaussian using sample mean/SD")
            ax.set_title(labels[name][0])
            ax.set_xlabel(labels[name][1])
            ax.set_ylabel("Density")
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle("SALT3 injection–recovery (100 seeded synthetic light curves)")
        _save_figure(fig, "03_salt3_injection_recovery")
    statistics = {}
    for name, values in recovered.items():
        array = np.asarray(values, dtype=float)
        statistics[name] = {
            "successful_fits": int(len(array)),
            "bias": float(np.mean(array)) if len(array) else None,
            "scatter_sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else None,
            "median": float(np.median(array)) if len(array) else None,
        }
    return {
        "available_offline": True,
        "seed": seed,
        "replicates": 100,
        "truth": {"z": z_true, "t0": t0_true, "x0": x0_true, "x1": x1_true, "c": c_true},
        "noise": "independent Gaussian noise with each bundled sample row's fluxerr",
        "cadence_rows_bands": {"rows": len(frame), "bands": sorted(frame["band"].astype(str).unique())},
        "recovery": statistics,
        "failed_replicates": failures,
        "status": "PASS" if len(recovered["t0"]) >= 90 else "CHECK",
    }


def _load_dovekie():
    data_path = Path(os.environ["DES_DOVEKIE_HD_PATH"])
    matrix_path = Path(os.environ["DES_DOVEKIE_STAT_SYS_PATH"])
    hd_catalog = load_des_dovekie_hd(data_path)
    raw_frame = hd_catalog.to_pandas()
    catalog = normalize_des_dovekie_distances(hd_catalog)
    precision = load_packed_precision_matrix(
        matrix_path,
        precision_kind="dovekie_stat_plus_systematic_complete_precision",
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description="Complete local DES-Dovekie STAT+SYS precision; read-only.",
    )
    result = calculate_hubble_residuals_precision(catalog, precision)
    return catalog, precision, result, raw_frame


def _load_pantheon():
    release = load_pantheonplus_catalog(Path(os.environ["PANTHEONPLUS_DATA_PATH"]))
    selection = select_pantheonplus_hubble_flow_subset(release)
    full_catalog = normalize_pantheonplus_distances(release)
    full_covariance = load_pantheonplus_covariance(
        Path(os.environ["PANTHEONPLUS_STAT_SYS_PATH"]),
        release,
        covariance_kind="pantheonplus_stat_plus_systematic_complete_covariance",
        composition_description="Complete local Pantheon+ STAT+SYS covariance; read-only.",
    )
    covariance = subset_pantheonplus_covariance(full_covariance, selection)
    catalog = selection.normalized_catalog
    result = calculate_hubble_residuals_covariance(catalog, covariance)
    return release, full_catalog, selection, full_covariance, catalog, covariance, result


def _diag_result(catalog, marginal_error):
    diagonal_catalog = replace(
        catalog,
        distance_modulus_error=tuple(float(value) for value in marginal_error),
    )
    diagram = calculate_hubble_diagram(diagonal_catalog)
    return calculate_hubble_residuals_diagonal(diagram)


def _plot_hubble(
    stem: str,
    title: str,
    redshift,
    observed_mu,
    errors,
    fitted_reference,
    *,
    calibrator_redshift=None,
    calibrator_mu=None,
    calibrator_errors=None,
) -> None:
    fig, ax = _figure_axes(title, "Redshift (zHD)", "Distance modulus μ (mag)", figsize=(13, 8))
    ax.errorbar(redshift, observed_mu, yerr=errors, fmt="o", ms=3.3, alpha=0.45, capsize=0, color="#1769aa", label="Hubble-flow rows used in fit")
    order = np.argsort(np.asarray(redshift, dtype=float))
    ax.plot(np.asarray(redshift)[order], np.asarray(fitted_reference)[order], color="#d24a37", linewidth=3, label="Flat ΛCDM reference + fitted intercept")
    if calibrator_redshift is not None:
        ax.errorbar(calibrator_redshift, calibrator_mu, yerr=calibrator_errors, fmt="x", ms=5, alpha=0.75, color="#777777", label="77 Cepheid calibrators — excluded")
    ax.legend(loc="best")
    _save_figure(fig, stem)


def _plot_residuals(stem: str, title: str, redshift, residual, errors=None) -> None:
    fig, ax = _figure_axes(title, "Redshift (zHD)", "μ observed − μ reference − δ (mag)", figsize=(13, 7))
    ax.axhline(0.0, color="black", linewidth=1.5)
    if errors is None:
        ax.scatter(redshift, residual, s=14, alpha=0.45, color="#1769aa")
    else:
        ax.errorbar(redshift, residual, yerr=errors, fmt="o", ms=3, alpha=0.35, capsize=0, color="#1769aa")
    _save_figure(fig, stem)


def _plot_hist(stem: str, title: str, values, xlabel: str) -> None:
    fig, ax = _figure_axes(title, xlabel, "Supernova count", figsize=(11, 7))
    ax.hist(values, bins=45, color="#327e9e", alpha=0.83)
    ax.axvline(0.0, color="#a7372a", linewidth=2.2, linestyle="--", label="zero")
    ax.legend()
    _save_figure(fig, stem)


def _catalog_analyses() -> dict[str, object]:
    started = time.perf_counter()
    des_catalog, precision, des_result, des_frame = _load_dovekie()
    precision_values = np.asarray(precision.values, dtype=float)
    precision_cholesky = np.linalg.cholesky(precision_values)
    precision_inverse = cho_solve(
        (precision_cholesky, True), np.eye(precision.dimension), check_finite=False
    )
    des_marginal_error = np.sqrt(np.diag(precision_inverse))
    des_diagonal_result = _diag_result(des_catalog, des_marginal_error)
    des_whitened = precision_cholesky.T @ np.asarray(des_result.residual, dtype=float)
    des_covariance = precision_inverse
    des_eigen_min = float(np.linalg.eigvalsh(precision_values)[0])
    des_symmetry = float(np.max(np.abs(precision_values - precision_values.T)))
    des_covariance_ratio = des_marginal_error / des_frame["MUERR"].to_numpy(float)

    (
        pant_release,
        pant_full_catalog,
        pant_selection,
        pant_full_covariance,
        pant_catalog,
        pant_covariance,
        pant_result,
    ) = _load_pantheon()
    pant_full_frame = pant_release.to_pandas()
    pant_frame = pant_full_frame.loc[
        pant_full_frame["IS_CALIBRATOR"].to_numpy(float) == 0.0
    ].reset_index(drop=True)
    pant_values = np.asarray(pant_covariance.values, dtype=float)
    pant_cholesky = np.linalg.cholesky(pant_values)
    pant_eigenvalues = np.linalg.eigvalsh(pant_values)
    pant_marginal_error = np.sqrt(np.diag(pant_values))
    pant_diagonal_result = _diag_result(pant_catalog, pant_marginal_error)
    pant_whitened = solve_triangular(
        pant_cholesky,
        np.asarray(pant_result.residual, dtype=float),
        lower=True,
        check_finite=False,
    )
    pant_diag_column = pant_frame["MU_SH0ES_ERR_DIAG"].to_numpy(float)
    pant_covariance_ratio = pant_marginal_error / pant_diag_column
    pant_symmetry = float(np.max(np.abs(pant_values - pant_values.T)))

    # D. DES-Dovekie: full STAT+SYS precision and its marginal-diagonal approximation.
    _plot_hubble(
        "04a_dovekie_hubble",
        "DES-Dovekie Hubble diagram — STAT+SYS precision fit",
        des_result.redshift,
        des_catalog.distance_modulus,
        des_marginal_error,
        des_result.fitted_reference_distance_modulus,
    )
    _plot_residuals("04b_dovekie_residuals", "DES-Dovekie residuals vs redshift", des_result.redshift, des_result.residual, des_marginal_error)
    _plot_hist("04c_dovekie_residual_hist", "DES-Dovekie residual distribution", des_result.residual, "Residual (mag)")

    # E. Pantheon+: calibrators are plotted separately and are not fit rows.
    calibrator_mask = pant_full_frame["IS_CALIBRATOR"].to_numpy(float) == 1.0
    calibrator_z = pant_full_frame.loc[calibrator_mask, "zHD"].to_numpy(float)
    calibrator_mu = pant_full_frame.loc[calibrator_mask, "MU_SH0ES"].to_numpy(float)
    calibrator_error = pant_full_frame.loc[calibrator_mask, "MU_SH0ES_ERR_DIAG"].to_numpy(float)
    calibrator_ref = np.asarray(
        calculate_reference_distance_modulus(
            tuple(float(value) for value in calibrator_z),
            heliocentric_redshift=tuple(float(value) for value in pant_full_frame.loc[calibrator_mask, "zHEL"]),
        )
    )
    _plot_hubble(
        "05a_pantheon_hubble",
        "Pantheon+ Hubble flow — 1,624 fitted rows; 77 calibrators excluded",
        pant_result.redshift,
        pant_catalog.distance_modulus,
        pant_marginal_error,
        pant_result.fitted_reference_distance_modulus,
        calibrator_redshift=calibrator_z,
        calibrator_mu=calibrator_mu,
        calibrator_errors=calibrator_error,
    )
    _plot_residuals("05b_pantheon_residuals", "Pantheon+ Hubble-flow residuals vs redshift", pant_result.redshift, pant_result.residual, pant_marginal_error)
    _plot_hist("05c_pantheon_residual_hist", "Pantheon+ Hubble-flow residual distribution", pant_result.residual, "Residual (mag)")

    correlation = pant_values / np.sqrt(np.outer(np.diag(pant_values), np.diag(pant_values)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    full_image = axes[0].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    axes[0].set_title("Full Hubble-flow correlation\n(1,624 × 1,624)")
    axes[0].set_xlabel("Retained row position")
    axes[0].set_ylabel("Retained row position")
    zoom_n = min(120, len(correlation))
    axes[1].imshow(correlation[:zoom_n, :zoom_n], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    axes[1].set_title(f"First {zoom_n} retained rows (release order)")
    axes[1].set_xlabel("Retained row position")
    axes[1].set_ylabel("Retained row position")
    fig.colorbar(full_image, ax=axes, label="Correlation coefficient", shrink=0.85)
    fig.suptitle("Pantheon+ STAT+SYS covariance correlation matrix")
    _save_figure(fig, "05d_pantheon_correlation")

    fig, ax = _figure_axes("Pantheon+ Hubble-flow STAT+SYS covariance eigenvalues", "Eigenvalue index (ascending)", "Covariance eigenvalue (mag²; log scale)", figsize=(12, 7))
    ax.plot(np.arange(1, len(pant_eigenvalues) + 1), pant_eigenvalues, color="#1769aa", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title(f"1624 × 1624 subset; λmin={pant_eigenvalues[0]:.3e}, κ₂={pant_eigenvalues[-1]/pant_eigenvalues[0]:.3e}")
    _save_figure(fig, "05e_pantheon_eigenvalues")

    # F. Whitened residuals: C=L Lᵀ => L⁻¹r; P=L Lᵀ => Lᵀr.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, values, title in (
        (axes[0], des_whitened, "DES-Dovekie (P=L Lᵀ; w=Lᵀr)"),
        (axes[1], pant_whitened, "Pantheon+ (C=L Lᵀ; w=L⁻¹r)"),
    ):
        ax.hist(values, bins=55, density=True, alpha=0.8, color="#2477a8", label="Whitened residuals")
        grid = np.linspace(-5, 5, 500)
        from scipy.stats import norm

        ax.plot(grid, norm.pdf(grid), color="#d24a37", linewidth=2.5, label="N(0,1)")
        ax.set_title(title)
        ax.set_xlabel("Whitened residual (σ units)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Full-matrix whitened residual diagnostics")
    _save_figure(fig, "06_whitened_residuals")

    # H. Vary H0; fitted intercepts absorb this global normalization shift.
    h0_results = {}
    residual_changes = {}
    binned_changes = {}
    base_des = des_result
    base_pant = pant_result
    base_compare = compare_catalog_residuals(base_des, base_pant, bins=10, minimum_count_per_bin=5)
    for h0 in (67.0, 70.0, 73.0):
        d = calculate_hubble_residuals_precision(des_catalog, precision, reference_h0=h0)
        p = calculate_hubble_residuals_covariance(pant_catalog, pant_covariance, reference_h0=h0)
        comparison = compare_catalog_residuals(d, p, bins=10, minimum_count_per_bin=5)
        residual_changes[str(int(h0))] = max(
            maximum_residual_difference(base_des, d),
            maximum_residual_difference(base_pant, p),
        )
        valid = [
            (left.descriptive_median_difference, right.descriptive_median_difference)
            for left, right in zip(base_compare.bins, comparison.bins, strict=True)
            if left.descriptive_median_difference is not None
        ]
        binned_changes[str(int(h0))] = max(
            (abs(float(left) - float(right)) for left, right in valid), default=0.0
        )
        h0_results[str(int(h0))] = {
            "des_intercept_mag": d.fitted_intercept,
            "pantheon_intercept_mag": p.fitted_intercept,
        }
    fig, ax = _figure_axes("H0 normalization invariance after per-catalog intercept fit", "Reference H0 (km s⁻¹ Mpc⁻¹)", "Maximum absolute change (10⁻¹⁵ mag)", figsize=(11, 7))
    h0_values = np.array([67.0, 70.0, 73.0])
    ax.plot(h0_values, np.array([residual_changes[str(int(x))] for x in h0_values]) * 1e15, "o-", linewidth=3, markersize=8, label="Per-row residuals")
    ax.plot(h0_values, np.array([binned_changes[str(int(x))] for x in h0_values]) * 1e15, "s--", linewidth=3, markersize=8, label="Binned median differences")
    ax.axhline(0.01, color="#777", linewidth=1, linestyle=":", label="10⁻¹⁷ mag guide")
    ax.legend()
    _save_figure(fig, "08_h0_invariance")

    # I. Common redshift range, default bins, and two alternative bin counts.
    common = common_redshift_range(des_result, pant_result)
    default_comparison = base_compare
    fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    colors = ("#1769aa", "#d24a37")
    labels = ("DES-Dovekie", "Pantheon+")
    for axis, result, color, label in zip(axes, (des_result, pant_result), colors, labels, strict=True):
        z = np.asarray(result.redshift, dtype=float)
        residual = np.asarray(result.residual, dtype=float)
        mask = (z >= common[0]) & (z <= common[1])
        axis.scatter(z[mask], residual[mask], s=9, alpha=0.12, color=color, label=f"{label} rows in overlap")
        bins = default_comparison.bins
        series = bins[0].first_catalog if label == "DES-Dovekie" else bins[0].second_catalog
        all_series = [item.first_catalog if label == "DES-Dovekie" else item.second_catalog for item in bins]
        xs = np.array([item.representative_redshift if item.representative_redshift is not None else np.nan for item in all_series])
        med = np.array([item.median_residual if item.row_count >= 5 else np.nan for item in all_series])
        low = np.array([item.residual_percentile_16 if item.row_count >= 5 else np.nan for item in all_series])
        high = np.array([item.residual_percentile_84 if item.row_count >= 5 else np.nan for item in all_series])
        axis.plot(xs, med, "o-", color=color, linewidth=2.7, markersize=7, label=f"{label} 10-bin median")
        axis.fill_between(xs, low, high, color=color, alpha=0.2, label="16th–84th percentile band")
        for item in all_series:
            if item.row_count < 5:
                axis.axvspan(item.lower_edge, item.upper_edge, color="#999999", alpha=0.08)
                axis.text((item.lower_edge + item.upper_edge) / 2, 0.97, "excluded", transform=axis.get_xaxis_transform(), ha="center", va="top", rotation=90, fontsize=9, color="#555")
        axis.axhline(0.0, color="black", linewidth=1.1)
        axis.set_ylabel("Residual (mag)")
        axis.set_title(label, loc="left")
        axis.grid(alpha=0.15)
        axis.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("Redshift zHD (common overlap)")
    fig.suptitle(f"Cross-catalog residual comparison, z=[{common[0]:.5f}, {common[1]:.5f}]")
    _save_figure(fig, "09a_cross_catalog_residuals")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    sensitivity = {}
    for ax, bin_count in zip(axes, (10, 20, 30), strict=True):
        comparison = compare_catalog_residuals(des_result, pant_result, bins=bin_count, minimum_count_per_bin=5)
        eligible = []
        excluded = []
        for item in comparison.bins:
            x = (item.lower_edge + item.upper_edge) / 2
            if item.descriptive_median_difference is None:
                excluded.append(x)
            else:
                eligible.append((x, item.descriptive_median_difference))
        if eligible:
            x, y = np.asarray(eligible).T
            ax.plot(x, y, "o-", linewidth=2.4, markersize=6, label="DES − Pantheon+ median")
        if excluded:
            ax.scatter(excluded, np.zeros(len(excluded)), marker="x", s=60, color="#c53f33", label="excluded: either catalog n < 5")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("Median difference\n(mag)")
        ax.set_title(f"{bin_count} bins; eligible={len(eligible)}, excluded={len(excluded)}", loc="left")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        sensitivity[str(bin_count)] = {
            "eligible_bins": len(eligible),
            "excluded_bins": len(excluded),
            "differences_mag": [y for _, y in eligible],
        }
    axes[-1].set_xlabel("Redshift (bin center)")
    fig.suptitle("Cross-catalog bin-count sensitivity (minimum 5 rows per catalog)")
    _save_figure(fig, "09b_cross_catalog_bin_sensitivity")

    def whitened_stats(values, chi2):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(values)),
            "sd_sample": float(np.std(values, ddof=1)),
            "chi2_over_n_minus_1": float(chi2 / (len(values) - 1)),
            "sum_w2": float(values @ values),
        }

    pant_full_stats = {
        "dimension": pant_full_covariance.dimension,
        "symmetry_max_abs": float(np.max(np.abs(pant_full_covariance.values - pant_full_covariance.values.T))),
        "full_cholesky_succeeds": bool(np.linalg.cholesky(pant_full_covariance.values).shape == pant_full_covariance.values.shape),
    }
    outputs = {
        "runtime_load_factor_plot_seconds": time.perf_counter() - started,
        "des_dovekie": {
            "rows": len(des_catalog.object_id),
            "z_min": min(des_catalog.redshift),
            "z_max": max(des_catalog.redshift),
            "mu_column": "MU",
            "z_column": "zHD with zHEL frame correction",
            "full_statsys": {
                "intercept_mag": des_result.fitted_intercept,
                "intercept_sigma_mag": des_result.fitted_intercept_error,
                "chi2": des_result.chi2,
                "ndof": des_result.ndof,
                "chi2_per_dof": des_result.reduced_chi2,
                "matrix_type": "precision P, packed upper triangle; STAT+SYS",
            },
            "diagonal_only_marginal_errors": {
                "intercept_mag": des_diagonal_result.fitted_intercept,
                "intercept_sigma_mag": des_diagonal_result.fitted_intercept_error,
                "chi2": des_diagonal_result.chi2_diagonal,
                "ndof": des_diagonal_result.ndof,
                "chi2_per_dof": des_diagonal_result.reduced_chi2_diagonal,
            },
            "matrix_check": {
                "dimension": precision.dimension,
                "symmetry_max_abs": des_symmetry,
                "minimum_eigenvalue_precision": des_eigen_min,
                "cholesky_succeeds": True,
                "marginal_error_ratio_vs_MUERR": {
                    "median": float(np.median(des_covariance_ratio)),
                    "min": float(np.min(des_covariance_ratio)),
                    "max": float(np.max(des_covariance_ratio)),
                },
                "whitened_residual": whitened_stats(des_whitened, des_result.chi2),
            },
        },
        "pantheon_plus": {
            "full_rows": pant_selection.full_release_row_count,
            "calibrator_rows_excluded": pant_selection.calibrator_row_count,
            "hubble_flow_rows": pant_selection.hubble_flow_row_count,
            "z_min": min(pant_catalog.redshift),
            "z_max": max(pant_catalog.redshift),
            "mu_column": "MU_SH0ES",
            "z_column": "zHD with zHEL frame correction",
            "retained_source_row_indices_first_10": list(pant_selection.retained_source_row_indices[:10]),
            "full_statsys": {
                "intercept_mag": pant_result.fitted_intercept,
                "intercept_sigma_mag": pant_result.fitted_intercept_error,
                "chi2": pant_result.chi2,
                "ndof": pant_result.ndof,
                "chi2_per_dof": pant_result.reduced_chi2,
                "matrix_type": "covariance C; STAT+SYS; principal submatrix by source row position",
            },
            "diagonal_only_marginal_errors": {
                "intercept_mag": pant_diagonal_result.fitted_intercept,
                "intercept_sigma_mag": pant_diagonal_result.fitted_intercept_error,
                "chi2": pant_diagonal_result.chi2_diagonal,
                "ndof": pant_diagonal_result.ndof,
                "chi2_per_dof": pant_diagonal_result.reduced_chi2_diagonal,
            },
            "matrix_check": {
                **pant_full_stats,
                "subset_dimension": pant_covariance.dimension,
                "subset_symmetry_max_abs": pant_symmetry,
                "subset_cholesky_succeeds": True,
                "eigenvalue_min": float(pant_eigenvalues[0]),
                "eigenvalue_max": float(pant_eigenvalues[-1]),
                "condition_number_2": float(pant_eigenvalues[-1] / pant_eigenvalues[0]),
                "marginal_error_ratio_vs_MU_SH0ES_ERR_DIAG": {
                    "median": float(np.median(pant_covariance_ratio)),
                    "min": float(np.min(pant_covariance_ratio)),
                    "max": float(np.max(pant_covariance_ratio)),
                },
                "whitened_residual": whitened_stats(pant_whitened, pant_result.chi2),
            },
        },
        "h0_invariance": {
            "reference_h0_values_km_s_mpc": [67, 70, 73],
            "intercepts_by_h0": h0_results,
            "max_abs_per_row_residual_change_mag": residual_changes,
            "max_abs_10bin_median_difference_change_mag": binned_changes,
            "expected": "numerical roundoff, approximately <= 1e-14 mag",
        },
        "cross_catalog": {
            "common_redshift_interval": list(common),
            "rows_in_overlap": {
                "des_dovekie": default_comparison.first_catalog_rows_in_common_range,
                "pantheon_plus": default_comparison.second_catalog_rows_in_common_range,
            },
            "summary": default_comparison.to_dict(),
            "bin_sensitivity": sensitivity,
            "default_bins": 10,
            "minimum_count_per_catalog_per_bin": 5,
        },
        "matrix_whitening_figures": "PASS",
    }
    return outputs


def main() -> None:
    if os.environ.get("RUN_ONLY") == "official_retry":
        existing = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
        SUMMARY.clear()
        SUMMARY.update(existing)
        previous = SUMMARY.get("steps", {}).get("D_to_I_official_catalogs")
        if previous is not None:
            SUMMARY["steps"]["D_to_I_official_catalogs_first_attempt"] = previous
        _step("D_to_I_official_catalogs_retry", _catalog_analyses)
        return
    _step("A_basic_sample", _basic_analysis)
    salt_result = _step("B_salt3_sample", _salt_fit)
    if salt_result is not None:
        _step("C_salt3_injection_recovery", _injection_recovery)
    else:
        SUMMARY["steps"]["C_salt3_injection_recovery"] = {
            "status": "SKIP",
            "reason": "SALT3 sample fit did not complete; model-data availability is not assumed.",
        }
        _write_summary()
    _step("D_to_I_official_catalogs", _catalog_analyses)
    _write_summary()


if __name__ == "__main__":
    main()
