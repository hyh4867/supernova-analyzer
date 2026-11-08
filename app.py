"""Streamlit interface for the Supernova Analyzer scientific backend."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cross_catalog_ui import (
    PATH_ENVIRONMENT_NAMES,
    binned_comparison_frame,
    catalog_availability,
    comparison_json,
    load_validated_results,
)

from supernova_analyzer import (
    Salt3DiagnosticError,
    Salt3FitError,
    Salt3ModelUnavailableError,
    StandardizationError,
    calculate_dm15_linear,
    calculate_salt3_amplitude_magnitude,
    calculate_salt3_diagnostics,
    calculate_synthetic_restframe_b_magnitude,
    calculate_tripp_standardization,
    detect_observed_peak,
    fit_salt3,
    load_light_curve,
    load_scientific_photometry,
    plot_light_curve,
    plot_salt3_fit,
    plot_salt3_light_curves,
    plot_salt3_residuals,
)
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
from supernova_analyzer import (
    compare_catalog_residuals,
    plot_cross_catalog_binned_residuals,
    plot_cross_catalog_hubble_residuals,
    plot_cross_catalog_redshift_distribution,
    plot_cross_catalog_residual_distribution,
)

PROJECT_ROOT = Path(__file__).resolve().parent
BASIC_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_light_curve.csv"
SCIENTIFIC_SAMPLE_PATH = (
    PROJECT_ROOT / "data" / "scientific" / "sample_multiband_photometry.csv"
)
USER_ERRORS = (
    ValueError,
    Salt3FitError,
    Salt3ModelUnavailableError,
    Salt3DiagnosticError,
    StandardizationError,
)
DEFAULT_ALPHA = 0.14
DEFAULT_BETA = 3.1
DEFAULT_ABSOLUTE_MAGNITUDE = -19.3


@st.cache_resource(show_spinner=False)
def _load_cross_catalog_results(path_values: tuple[str, ...]) -> dict[str, object]:
    """Cache immutable validated results for an exact set of source paths."""
    return load_validated_results(dict(zip(PATH_ENVIRONMENT_NAMES, path_values)))


@st.cache_data
def _sample_bytes(path: str) -> bytes:
    """Load immutable bundled sample bytes."""
    return Path(path).read_bytes()


def _render_numeric_block(
    container: object,
    label: str,
    value: str,
    *,
    uncertainty: str | None = None,
) -> None:
    """Render a readable value and optional uncertainty on separate lines."""
    container.markdown(f"**{label}**")
    container.markdown(f"### {value}")
    if uncertainty is not None:
        container.caption(f"Statistical uncertainty: ±{uncertainty}")


def _initialize_standardization_controls() -> None:
    """Initialize illustrative nuisance controls without affecting fit state."""
    st.session_state.setdefault("illustrative_alpha", DEFAULT_ALPHA)
    st.session_state.setdefault("illustrative_beta", DEFAULT_BETA)
    st.session_state.setdefault(
        "illustrative_absolute_magnitude",
        DEFAULT_ABSOLUTE_MAGNITUDE,
    )


def _calculate_current_tripp(result: dict[str, object]) -> object:
    """Recalculate standardization from the current illustrative controls."""
    _initialize_standardization_controls()
    tripp = calculate_tripp_standardization(
        result["fit"],
        result["amplitude"],
        alpha=st.session_state["illustrative_alpha"],
        beta=st.session_state["illustrative_beta"],
        absolute_magnitude=st.session_state["illustrative_absolute_magnitude"],
    )
    result["tripp"] = tripp
    return tripp


def _basic_plot_bytes(data: pd.DataFrame, peak: object, decline: object) -> bytes:
    """Use the backend Basic plotter and return its output as in-memory bytes."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "basic_light_curve.png"
        plot_light_curve(data, output_path=path, peak=peak, dm15=decline)
        return path.read_bytes()


def _render_basic_tab() -> None:
    """Render the magnitude-based Basic Analysis workflow."""
    st.header("Basic Analysis")
    st.write(
        "Analyze a single-band magnitude CSV with columns `time`, `magnitude`, "
        "and `magnitude_error`. Lower magnitude means brighter."
    )
    sample = _sample_bytes(str(BASIC_SAMPLE_PATH))
    source = st.radio(
        "Basic data source",
        ("Use sample data", "Upload CSV"),
        horizontal=True,
        key="basic_source",
    )
    st.download_button(
        "Download basic sample CSV",
        sample,
        "sample_light_curve.csv",
        "text/csv",
    )
    upload = None
    if source == "Upload CSV":
        upload = st.file_uploader(
            "Upload Basic Analysis CSV",
            type="csv",
            key="basic_upload",
        )
    content = sample if source == "Use sample data" else upload.getvalue() if upload else None
    if content is None:
        st.caption("Upload a CSV file to continue.")
        return

    try:
        data = load_light_curve(io.BytesIO(content))
        with st.expander("Data preview"):
            st.dataframe(data, width="stretch")
        left, right = st.columns(2)
        left.metric("Observations", len(data))
        right.metric(
            "Time range",
            f"{data['time'].min():.5g} to {data['time'].max():.5g}",
        )
    except USER_ERRORS as error:
        st.error(str(error))
        return

    if st.button("Analyze Basic Light Curve", type="primary"):
        try:
            peak = detect_observed_peak(data)
            decline = calculate_dm15_linear(data, peak)
            summary = basic_analysis_summary(peak, decline, len(data))
            st.session_state["basic_result"] = {
                "summary": summary,
                "plot": _basic_plot_bytes(data, peak, decline),
            }
        except USER_ERRORS as error:
            st.error(str(error))

    result = st.session_state.get("basic_result")
    if result is None:
        return
    summary = result["summary"]
    peak = summary["observed_peak"]
    decline = summary["linear_15_day_decline_baseline"]
    columns = st.columns(3)
    columns[0].metric(
        "Observed peak magnitude", f"{peak['peak_magnitude']:.3f}"
    )
    columns[1].metric(
        "15-day decline", f"{decline['delta_m15']:.3f} mag"
    )
    columns[2].metric(
        "Number of observations", summary["number_of_observations"]
    )
    with st.container(border=True):
        st.caption("Secondary details")
        details = st.columns(2)
        details[0].metric("Observed peak time", f"{peak['peak_time']:.5g} days")
        details[1].metric(
            "Magnitude at peak + 15 days",
            f"{decline['interpolated_magnitude']:.3f}",
        )
    st.image(
        result["plot"],
        caption=(
            "Magnitude light curve with observation errors, observed peak, "
            "and the interpolated peak + 15 day reference."
        ),
        width="stretch",
    )
    st.caption(
        "The observed peak is the minimum sampled magnitude. The 15-day decline "
        "is a linear-interpolation baseline. It is not automatically a true "
        "Delta m15(B) measurement unless the data are explicitly rest-frame "
        "B-band data."
    )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download summary TXT",
        basic_summary_text(summary),
        "basic_analysis_summary.txt",
        "text/plain",
        type="primary",
    )
    downloads[1].download_button(
        "Download plot PNG",
        result["plot"],
        "basic_light_curve.png",
        "image/png",
        type="primary",
    )
    with st.expander("Individual result files (JSON)"):
        st.download_button(
            "Download Basic Analysis JSON",
            result_to_json(summary),
            "basic_analysis_summary.json",
            "application/json",
        )


def _scientific_inputs() -> tuple[bytes, str] | None:
    """Render the scientific data selector and return bytes plus identity."""
    sample = _sample_bytes(str(SCIENTIFIC_SAMPLE_PATH))
    source = st.radio(
        "Scientific data source",
        ("Use bundled scientific sample data", "Upload CSV"),
        horizontal=True,
        key="scientific_source",
    )
    st.download_button(
        "Download scientific sample CSV",
        sample,
        "sample_multiband_photometry.csv",
        "text/csv",
    )
    if source == "Use bundled scientific sample data":
        return sample, "bundled:sncosmo-example-photometry"
    upload = st.file_uploader(
        "Upload Scientific SALT3 CSV",
        type="csv",
        key="scientific_upload",
    )
    if upload is None:
        st.caption("Upload a CSV file to continue.")
        return None
    content = upload.getvalue()
    return content, content.hex()


def _render_scientific_headline(result: dict[str, object]) -> None:
    """Render the three primary scientific results."""
    fit = result["fit"]
    amplitude = result["amplitude"]
    tripp = result["tripp"]
    columns = st.columns(3)
    _render_numeric_block(
        columns[0],
        r"Reduced chi-square $\chi^2 / \mathrm{ndof}$",
        f"{fit.reduced_chi2:.3f}",
    )
    _render_numeric_block(
        columns[1],
        r"Amplitude-based magnitude $m_B$",
        f"{amplitude.apparent_magnitude:.3f}",
        uncertainty=(
            None
            if amplitude.apparent_magnitude_error is None
            else f"{amplitude.apparent_magnitude_error:.3f} mag"
        ),
    )
    columns[1].caption("Used in the Tripp-like calculation.")
    _render_numeric_block(
        columns[2],
        r"Illustrative distance modulus $\mu$",
        f"{tripp.distance_modulus:.3f}",
        uncertainty=(
            None
            if tripp.distance_modulus_error_statistical is None
            else f"{tripp.distance_modulus_error_statistical:.3f} mag"
        ),
    )
    distance = abs(fit.reduced_chi2 - 1.0)
    st.caption(
        f"The reduced χ² is {distance:.3f} from 1. Values near 1 indicate "
        "consistency at the supplied uncertainty and model-covariance scale; "
        "this statistic alone does not establish physical completeness."
    )


def _render_fit_parameters(result: dict[str, object]) -> None:
    """Render secondary fit parameters in one bordered section."""
    fit = result["fit"]
    with st.container(border=True):
        st.subheader("Fit Parameters")
        values = (
            (
                r"Peak time $t_0$",
                f"{fit.t0:.2f}",
                None if fit.t0_error is None else f"{fit.t0_error:.3f} days",
            ),
            (
                r"Amplitude $x_0$",
                f"{fit.x0:.3e}",
                None if fit.x0_error is None else f"{fit.x0_error:.3e}",
            ),
            (
                r"Shape $x_1$",
                f"{fit.x1:.3f}",
                None if fit.x1_error is None else f"{fit.x1_error:.3f}",
            ),
            (
                "Color c",
                f"{fit.c:.3f}",
                None if fit.c_error is None else f"{fit.c_error:.3f}",
            ),
            (r"Chi-square $\chi^2$", f"{fit.chi2:.3f}", None),
            ("Degrees of freedom", str(fit.ndof), None),
            (
                "Observations",
                f"{fit.n_observations_used} / {fit.n_observations_total}",
                None,
            ),
        )
        for start in range(0, len(values), 2):
            columns = st.columns(2)
            for column, (label, value, uncertainty) in zip(
                columns,
                values[start : start + 2],
                strict=False,
            ):
                _render_numeric_block(
                    column,
                    label,
                    value,
                    uncertainty=uncertainty,
                )


def _render_residual_diagnostics(result: dict[str, object]) -> None:
    """Render residual summaries in one bordered section."""
    diagnostics = result["diagnostics"]
    with st.container(border=True):
        groups = st.columns(2)
        groups[0].markdown("**Raw residual**")
        groups[0].write(f"Mean: `{diagnostics.residual_mean:.4f}`")
        groups[0].write(f"RMS: `{diagnostics.residual_rms:.4f}`")
        groups[1].markdown("**Normalized residual**")
        groups[1].write(
            f"Mean: `{diagnostics.normalized_residual_mean:.4f}`"
        )
        groups[1].write(f"RMS: `{diagnostics.normalized_residual_rms:.4f}`")
        st.caption(
            "The plotted total-diagonal normalized residual uses observational "
            "variance plus the model-covariance diagonal. It is a visualization "
            "approximation and does not decompose the full-covariance fit "
            "chi-square exactly."
        )


def _render_standardization(result: dict[str, object]) -> None:
    """Render magnitude conventions and illustrative nuisance controls."""
    amplitude = result["amplitude"]
    synthetic = result["synthetic"]
    with st.container(border=True):
        st.subheader("Standardization")
        first = st.columns(2)
        _render_numeric_block(
            first[0],
            r"Amplitude-based magnitude $m_B$",
            f"{amplitude.apparent_magnitude:.3f}",
            uncertainty=(
                None
                if amplitude.apparent_magnitude_error is None
                else f"{amplitude.apparent_magnitude_error:.3f} mag"
            ),
        )
        first[0].caption("Used in the Tripp-like calculation.")
        _render_numeric_block(
            first[1],
            "Full synthetic rest-frame B magnitude",
            f"{synthetic.apparent_magnitude:.3f}",
            uncertainty=(
                None
                if synthetic.apparent_magnitude_error is None
                else f"{synthetic.apparent_magnitude_error:.3f} mag"
            ),
        )
        first[1].caption("Spectral diagnostic only; not used as the Tripp input.")
        second = st.columns(2)
        _render_numeric_block(
            second[0],
            "Normalization constant",
            f"{amplitude.normalization_constant:.6f}",
        )
        _render_numeric_block(
            second[1],
            "Synthetic − amplitude",
            f"{synthetic.apparent_magnitude - amplitude.apparent_magnitude:.3f}",
        )
        controls = st.columns(3)
        controls[0].number_input(
            "Illustrative alpha",
            format="%.4f",
            key="illustrative_alpha",
        )
        controls[1].number_input(
            "Illustrative beta",
            format="%.4f",
            key="illustrative_beta",
        )
        controls[2].number_input(
            "Illustrative M",
            format="%.4f",
            key="illustrative_absolute_magnitude",
        )
        try:
            tripp = _calculate_current_tripp(result)
        except USER_ERRORS as error:
            st.error(str(error))
            return
        _render_numeric_block(
            st,
            r"Illustrative distance modulus $\mu$",
            f"{tripp.distance_modulus:.3f}",
            uncertainty=(
                None
                if tripp.distance_modulus_error_statistical is None
                else f"{tripp.distance_modulus_error_statistical:.3f} mag"
            ),
        )
        with st.expander("Uncertainty limitations"):
            st.caption(
                "The nuisance parameters shown here are illustrative and are "
                "not claimed to reproduce a specific survey calibration. The "
                "reported uncertainty excludes calibration uncertainty, "
                "intrinsic scatter, peculiar velocity, lensing, selection "
                "bias, and host-galaxy corrections."
            )


def _render_scientific_downloads(result: dict[str, object]) -> None:
    """Render all in-memory scientific downloads."""
    if "tripp" not in result:
        return
    fit = result["fit"]
    diagnostics = result["diagnostics"]
    amplitude = result["amplitude"]
    synthetic = result["synthetic"]
    tripp = result["tripp"]
    summary = scientific_combined_summary(
        fit,
        diagnostics,
        amplitude,
        synthetic,
        tripp,
    )
    json_payloads = (
        ("Fit result JSON", result_to_json(fit), "salt3_fit_result.json"),
        (
            "Diagnostics JSON",
            result_to_json(diagnostics),
            "salt3_diagnostics.json",
        ),
        (
            "Amplitude m_B JSON",
            result_to_json(amplitude),
            "salt3_amplitude_magnitude.json",
        ),
        (
            "Synthetic B JSON",
            result_to_json(synthetic),
            "synthetic_restframe_b_magnitude.json",
        ),
        (
            "Tripp result JSON",
            result_to_json(tripp),
            "tripp_standardization.json",
        ),
    )
    st.subheader("Downloads")
    primary = st.columns(2)
    primary[0].download_button(
        "Download combined summary TXT",
        summary,
        "salt3_analysis_summary.txt",
        "text/plain",
        type="primary",
    )
    primary[1].download_button(
        "Download SALT3 fit PNG",
        result["light_curve_plot"],
        "salt3_light_curves.png",
        "image/png",
        type="primary",
    )
    secondary = st.columns(2)
    secondary[0].download_button(
        "Download residual diagnostics PNG",
        result["residual_plot"],
        "salt3_residual_diagnostics.png",
        "image/png",
    )
    secondary[1].download_button(
        "Download combined fit + residual PNG",
        result["plot"],
        "salt3_fit.png",
        "image/png",
    )
    with st.expander("Individual result files (JSON)"):
        columns = st.columns(2)
        for index, (label, data, name) in enumerate(json_payloads):
            columns[index % 2].download_button(
                label,
                data,
                name,
                "application/json",
            )


def _render_scientific_tab() -> None:
    """Render the multi-band SALT3 fitting workflow."""
    st.header("Scientific SALT3 Fit")
    st.write(
        "Fit multi-band flux photometry with columns `time`, `band`, `flux`, "
        "`fluxerr`, `zp`, and `zpsys`."
    )
    selected = _scientific_inputs()
    if selected is None:
        return
    content, identity = selected
    try:
        photometry = load_scientific_photometry(io.BytesIO(content))
        with st.expander("Data preview"):
            st.dataframe(photometry.to_pandas(), width="stretch")
        summary_columns = st.columns(3)
        summary_columns[0].markdown(
            f"**Observations**  \n{len(photometry)}"
        )
        band_names = ", ".join(
            sorted({str(item) for item in photometry["band"]})
        )
        summary_columns[1].markdown(f"**Bands**  \n{band_names}")
        summary_columns[2].markdown(
            "**Observer-frame time**  \n"
            f"{min(photometry['time']):.2f}–{max(photometry['time']):.2f}"
        )
    except USER_ERRORS as error:
        st.error(str(error))
        return

    st.caption(
        "For the bundled sample, redshift 0.5 is a demonstration assumption, "
        "not an externally measured spectroscopic value."
    )
    controls = st.columns(3)
    redshift = controls[0].number_input(
        "Fixed redshift", min_value=0.0, value=0.5, format="%.5f"
    )
    include_covariance = controls[1].checkbox(
        "Include SALT3 model covariance", value=True
    )
    min_snr = controls[2].number_input(
        "Minimum S/N", min_value=0.0001, value=5.0, format="%.3f"
    )
    with st.expander("Advanced fitting settings"):
        initial_t0_text = st.text_input(
            "Optional initial t0", placeholder="Leave blank for backend guessing"
        )
        custom_bounds = st.checkbox("Enable custom fitting bounds", value=False)
        bound_values: dict[str, tuple[str, str]] = {}
        if custom_bounds:
            st.caption("Leave both fields blank to use the backend default.")
            for parameter in ("t0", "x0", "x1", "c"):
                lower_column, upper_column = st.columns(2)
                lower = lower_column.text_input(
                    f"{parameter} lower bound", key=f"bound_{parameter}_lower"
                )
                upper = upper_column.text_input(
                    f"{parameter} upper bound", key=f"bound_{parameter}_upper"
                )
                bound_values[parameter] = (lower, upper)

    try:
        validated_redshift = validate_gui_number(
            redshift, "redshift", minimum=0.0
        )
        validated_snr = validate_gui_number(
            min_snr,
            "minimum S/N",
            minimum=0.0,
            minimum_inclusive=False,
        )
        initial_t0 = parse_optional_number(initial_t0_text, "initial t0")
        bounds = parse_bounds(bound_values) if custom_bounds else None
        fingerprint = scientific_input_fingerprint(
            identity,
            redshift=validated_redshift,
            include_model_covariance=include_covariance,
            min_snr=validated_snr,
            initial_t0=initial_t0,
            bounds=bounds,
        )
    except ValueError as error:
        st.error(str(error))
        return

    stored = st.session_state.get("scientific_result")
    stale = stored is not None and stored["fingerprint"] != fingerprint
    if stale:
        st.warning(
            "Scientific inputs changed. Run SALT3 Analysis again to replace the "
            "stale result; the previous result is hidden."
        )

    st.caption(
        "First use may download the pinned SALT3 model and bandpass reference "
        "data to the local cache and may take longer."
    )
    if st.button("Run SALT3 Analysis", type="primary"):
        try:
            with st.spinner(
                "Preparing model resources and fitting the pinned SALT3 model..."
            ):
                fit = fit_salt3(
                    photometry,
                    redshift=validated_redshift,
                    include_model_covariance=include_covariance,
                    bounds=bounds,
                    initial_t0=initial_t0,
                    min_snr=validated_snr,
                )
                diagnostics = calculate_salt3_diagnostics(photometry, fit)
                combined_figure = plot_salt3_fit(
                    photometry,
                    fit,
                    diagnostics=diagnostics,
                    show_normalized_residuals=True,
                    phase_axis=True,
                )
                light_curve_figure = plot_salt3_light_curves(
                    photometry,
                    fit,
                    diagnostics=diagnostics,
                    phase_axis=True,
                )
                residual_figure = plot_salt3_residuals(
                    photometry,
                    fit,
                    diagnostics=diagnostics,
                    phase_axis=True,
                )
                try:
                    plot_bytes = figure_to_png_bytes(combined_figure)
                    light_curve_plot_bytes = figure_to_png_bytes(
                        light_curve_figure
                    )
                    residual_plot_bytes = figure_to_png_bytes(residual_figure)
                finally:
                    plt.close(combined_figure)
                    plt.close(light_curve_figure)
                    plt.close(residual_figure)
                amplitude = calculate_salt3_amplitude_magnitude(fit)
                synthetic = calculate_synthetic_restframe_b_magnitude(fit)
            stored = {
                "fingerprint": fingerprint,
                "fit": fit,
                "diagnostics": diagnostics,
                "amplitude": amplitude,
                "synthetic": synthetic,
                "plot": plot_bytes,
                "light_curve_plot": light_curve_plot_bytes,
                "residual_plot": residual_plot_bytes,
            }
            st.session_state["scientific_result"] = stored
            stale = False
        except USER_ERRORS as error:
            st.error(str(error))

    if stored is None or stale:
        return
    try:
        _calculate_current_tripp(stored)
    except USER_ERRORS as error:
        st.error(str(error))
        return
    st.success("Analysis complete.")
    _render_scientific_headline(stored)
    band_count = len({str(item) for item in photometry["band"]})
    st.subheader("SALT3 Multi-band Fit")
    st.image(
        stored["light_curve_plot"],
        caption=(
            f"SALT3 light curves across {band_count} bands in rest-frame phase; "
            f"reduced χ² = {stored['fit'].reduced_chi2:.3f}."
        ),
        width="stretch",
    )
    st.subheader("Residual Diagnostics")
    st.image(
        stored["residual_plot"],
        caption=(
            f"Normalized residuals across {band_count} bands using the existing "
            "diagnostic values."
        ),
        width="stretch",
    )
    _render_residual_diagnostics(stored)
    _render_fit_parameters(stored)
    _render_standardization(stored)
    _render_scientific_downloads(stored)


def _render_catalog_metrics(summary: object) -> None:
    """Render one catalog summary without modifying the frozen result."""
    st.subheader(summary.catalog_name)
    st.caption(summary.analysis_mode)
    redshift = st.columns(3)
    redshift[0].metric("Redshift min", f"{summary.minimum_redshift:.4f}")
    redshift[1].metric("Redshift median", f"{summary.median_redshift:.4f}")
    redshift[2].metric("Redshift max", f"{summary.maximum_redshift:.4f}")
    fit = st.columns(2)
    fit[0].metric(
        "Nuisance intercept",
        f"{summary.fitted_nuisance_intercept:.4f} ± "
        f"{summary.intercept_uncertainty:.4f} mag",
    )
    fit[1].metric(
        "Catalog reduced χ²",
        f"{summary.catalog_specific_reduced_chi2:.3f}",
    )
    residual = st.columns(3)
    residual[0].metric("Residual median", f"{summary.residual_median:.4f} mag")
    residual[1].metric("Residual RMS", f"{summary.residual_rms:.4f} mag")
    residual[2].metric(
        "MAD-based scatter", f"{summary.residual_mad_scatter:.4f} mag"
    )


def _render_cross_catalog_tab() -> None:
    """Render the DES-Dovekie versus Pantheon+ descriptive explorer."""
    st.header("Cross-Catalog Explorer")
    st.write(
        "Explore the validated DES-Dovekie and Pantheon+ residual analyses. "
        "This is a descriptive survey comparison, not a joint cosmology fit."
    )

    with st.expander("Official data locations"):
        st.caption(
            "Set these fields directly or use the matching environment variables. "
            "Official release files are never bundled or uploaded by this app."
        )
        paths = {
            name: st.text_input(
                name,
                value=os.getenv(name, ""),
                key=f"cross_catalog_{name.lower()}",
                placeholder="Local path to the official release product",
            ).strip()
            for name in PATH_ENVIRONMENT_NAMES
        }

    availability = catalog_availability(paths)
    status_columns = st.columns(2)
    if availability["des"].available:
        status_columns[0].success("DES-Dovekie data available")
    else:
        missing = ", ".join(availability["des"].missing)
        status_columns[0].warning(f"DES-Dovekie data unavailable: {missing}")
    if availability["pantheon"].available:
        status_columns[1].success("Pantheon+ data available")
    else:
        missing = ", ".join(availability["pantheon"].missing)
        status_columns[1].warning(f"Pantheon+ data unavailable: {missing}")
    if not all(item.available for item in availability.values()):
        st.info("Both validated datasets are required before analysis is ready.")
        return

    try:
        with st.spinner("Loading and validating official STAT+SYS products..."):
            loaded = _load_cross_catalog_results(
                tuple(paths[name] for name in PATH_ENVIRONMENT_NAMES)
            )
        des_result = loaded["des_result"]
        pantheon_result = loaded["pantheon_result"]
        preview = compare_catalog_residuals(des_result, pantheon_result)
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.error(f"Cross-catalog data could not be validated: {error}")
        return

    st.success("Both catalogs available — analysis ready")
    st.subheader("Comparison overview")
    overview = st.columns(4)
    overview[0].metric("DES fitted rows", preview.first_catalog.number_of_fitted_rows)
    overview[1].metric(
        "Pantheon+ fitted rows", preview.second_catalog.number_of_fitted_rows
    )
    overview[2].metric(
        "Common redshift range",
        f"{preview.common_redshift_minimum:.4f}–"
        f"{preview.common_redshift_maximum:.4f}",
    )
    overview[3].metric(
        "Rows in overlap",
        f"{preview.first_catalog_rows_in_common_range} DES / "
        f"{preview.second_catalog_rows_in_common_range} P+",
    )
    st.caption(
        "DES-Dovekie: validated STAT+SYS complete precision (1820-row release); "
        "Pantheon+: validated non-calibrator Hubble-flow subset with the "
        "STAT+SYS covariance principal submatrix."
    )

    st.subheader("Catalog summaries")
    catalog_columns = st.columns(2)
    with catalog_columns[0].container(border=True):
        _render_catalog_metrics(preview.first_catalog)
    with catalog_columns[1].container(border=True):
        _render_catalog_metrics(preview.second_catalog)
    st.info(
        "Each reduced χ² belongs to a different catalog-specific matrix and fit. "
        "Do not use these values to rank the surveys."
    )

    st.subheader("Redshift distributions")
    figure = plot_cross_catalog_redshift_distribution(des_result, pantheon_result)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.subheader("Residuals versus redshift")
    figure = plot_cross_catalog_hubble_residuals(des_result, pantheon_result)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Points show fitted residuals for visualization. Point scatter alone does "
        "not represent the full covariance- or precision-aware uncertainty."
    )

    st.subheader("Binned residual comparison")
    controls = st.columns(2)
    bins = controls[0].slider("Number of common bins", 2, 30, 10)
    minimum_count = controls[1].number_input(
        "Minimum rows per catalog/bin", min_value=1, value=5, step=1
    )
    comparison = compare_catalog_residuals(
        des_result,
        pantheon_result,
        bins=bins,
        minimum_count_per_bin=int(minimum_count),
    )
    figure = plot_cross_catalog_binned_residuals(comparison)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Only bins meeting the selected minimum in both catalogs are emphasized "
        "and assigned a DES − Pantheon+ median difference. Percentile ranges are "
        "descriptive, not covariance-aware parameter errors."
    )
    st.dataframe(
        binned_comparison_frame(comparison),
        hide_index=True,
        width="stretch",
        column_config={
            "DES median (mag)": st.column_config.NumberColumn(format="%.4f"),
            "Pantheon+ median (mag)": st.column_config.NumberColumn(format="%.4f"),
            "DES − Pantheon+ (mag)": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.subheader("Residual distributions")
    figure = plot_cross_catalog_residual_distribution(des_result, pantheon_result)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption("Distribution shapes are descriptive only.")

    with st.expander("Scientific limitations / interpretation", expanded=True):
        st.markdown(
            "- Each catalog has an independent nuisance intercept.\n"
            "- The intercept difference is not an H0 measurement.\n"
            "- No joint likelihood or cross-catalog covariance is constructed.\n"
            "- No formal survey-tension significance is calculated.\n"
            "- No object cross-match or cosmological-parameter inference is performed.\n"
            "- Percentile ranges are descriptive, not covariance-aware parameter errors."
        )

    st.download_button(
        "Download comparison JSON",
        comparison_json(comparison, loaded["provenance"]),
        "cross_catalog_comparison.json",
        "application/json",
        type="primary",
    )


st.set_page_config(
    page_title="Supernova Light-Curve Analyzer",
    layout="wide",
)
st.title("Supernova Light-Curve Analyzer")
st.write(
    "Explore a magnitude-based baseline analysis or fit validated multi-band "
    "photometry with SALT3."
)
st.caption(
    "This application is an educational single-object analysis tool. "
    "It is not a survey-level cosmology pipeline."
)
basic_tab, scientific_tab, cross_catalog_tab = st.tabs(
    ("Basic Analysis", "Scientific SALT3 Fit", "Cross-Catalog Explorer")
)
with basic_tab:
    _render_basic_tab()
with scientific_tab:
    _render_scientific_tab()
with cross_catalog_tab:
    _render_cross_catalog_tab()
