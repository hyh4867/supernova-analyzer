"""Tests for dataset-independent Hubble-diagram analysis."""

import json
import os
from dataclasses import FrozenInstanceError, asdict, fields
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from supernova_analyzer import (
    CovarianceAlignmentError,
    CovarianceMatrix,
    HubbleAnalysisError,
    NormalizedDistanceCatalog,
    apply_des_sn5yr_quality_cuts,
    calculate_hubble_diagram,
    calculate_hubble_residuals_diagonal,
    calculate_hubble_residuals_covariance,
    calculate_reference_distance_modulus,
    calculate_uncertainty_display_limit,
    catalog_order_fingerprint,
    compare_diagonal_and_covariance_results,
    fit_additive_intercept_covariance,
    fit_additive_intercept_diagonal,
    load_des_sn5yr_catalog,
    normalize_des_sn5yr_distances,
    plot_hubble_diagram,
    plot_hubble_residuals,
    plot_diagonal_vs_covariance_residuals,
    plot_distance_uncertainty_distribution,
)


def make_catalog(
    *,
    object_id: tuple[str, ...] = ("SN-A", "SN-B", "SN-C"),
    redshift: tuple[float, ...] = (0.03, 0.1, 0.3),
    distance_modulus: tuple[float, ...] | None = None,
    distance_modulus_error: tuple[float, ...] = (0.1, 0.2, 0.15),
    source_row_index: tuple[int, ...] = (3, 7, 11),
) -> NormalizedDistanceCatalog:
    """Create a valid generic distance catalog."""
    if distance_modulus is None:
        reference = calculate_reference_distance_modulus(redshift)
        distance_modulus = tuple(value + 0.4 for value in reference)
    return NormalizedDistanceCatalog(
        object_id=object_id,
        redshift=redshift,
        distance_modulus=distance_modulus,
        distance_modulus_error=distance_modulus_error,
        source_row_index=source_row_index,
        catalog_name="Synthetic public release adapter",
        catalog_release="Synthetic test release",
        redshift_definition="test redshift",
        distance_definition="synthetic distance modulus",
    )


def legacy_csv() -> StringIO:
    """Return a minimal legacy DES release-shaped fixture."""
    return StringIO(
        "\n".join(
            (
                "CID,zHEL,zCMB,zHD,MU,MUERR_FINAL,HOST_LOGMASS,x0,x1,c,PKMJD",
                "DES1,0.09,0.10,0.11,38.1,0.12,-999,0.01,0.2,-0.03,59000",
                "DES2,0.19,0.20,0.21,39.5,0.15,10.1,0.02,-0.1,0.04,59100",
            )
        )
    )


def test_normalized_catalog_validation_accepts_aligned_values() -> None:
    catalog = make_catalog()

    assert catalog.object_id == ("SN-A", "SN-B", "SN-C")
    assert catalog.source_row_index == (3, 7, 11)


def test_normalized_catalog_requires_immutable_tuple_fields() -> None:
    with pytest.raises(HubbleAnalysisError, match="redshift.*immutable tuple"):
        make_catalog(redshift=[0.03, 0.1, 0.3])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"redshift": (0.03, 0.0, 0.3)}, "redshift.*greater than zero"),
        (
            {"distance_modulus": (38.0, np.nan, 41.0)},
            "distance_modulus.*finite",
        ),
        (
            {"distance_modulus_error": (0.1, -0.2, 0.15)},
            "distance_modulus_error.*greater than zero",
        ),
        ({"source_row_index": (3, 3, 11)}, "unique row correspondence"),
        ({"source_row_index": (3, 7)}, "equal lengths"),
    ),
)
def test_normalized_catalog_rejects_invalid_rows(
    override: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(HubbleAnalysisError, match=match):
        make_catalog(**override)


def test_legacy_des_conversion_preserves_ids_rows_and_z_hd() -> None:
    legacy = load_des_sn5yr_catalog(legacy_csv())
    original_rows = legacy.rows

    normalized = normalize_des_sn5yr_distances(legacy)

    assert normalized.object_id == ("DES1", "DES2")
    assert normalized.source_row_index == (0, 1)
    assert normalized.redshift == pytest.approx((0.11, 0.21))
    assert normalized.heliocentric_redshift == pytest.approx((0.09, 0.19))
    assert normalized.distance_modulus == pytest.approx((38.1, 39.5))
    assert normalized.distance_modulus_error == pytest.approx((0.12, 0.15))
    assert normalized.redshift_definition == "zHD"
    assert normalized.reference_distance_convention == (
        "zHD cosmology redshift with zHEL frame correction"
    )
    assert normalized.catalog_name == "Legacy DES-SN5YR HD + metadata catalog"
    assert "Vincenzi et al. 2024" in normalized.catalog_release
    assert legacy.rows == original_rows
    assert tuple(zip(normalized.object_id, normalized.source_row_index)) == (
        ("DES1", 0),
        ("DES2", 1),
    )


def test_explicit_legacy_cut_preserves_original_source_row_mapping() -> None:
    legacy = load_des_sn5yr_catalog(legacy_csv())
    filtered = apply_des_sn5yr_quality_cuts(
        legacy,
        require_host_mass=True,
    )

    normalized = normalize_des_sn5yr_distances(filtered)

    assert normalized.object_id == ("DES2",)
    assert normalized.source_row_index == (1,)


def test_intercept_requires_at_least_two_supernovae() -> None:
    reference = calculate_reference_distance_modulus((0.1,))
    catalog = make_catalog(
        object_id=("SN-A",),
        redshift=(0.1,),
        distance_modulus=(reference[0] + 0.2,),
        distance_modulus_error=(0.1,),
        source_row_index=(0,),
    )

    with pytest.raises(HubbleAnalysisError, match="At least two"):
        fit_additive_intercept_diagonal(catalog)
    with pytest.raises(HubbleAnalysisError, match="At least two"):
        calculate_hubble_diagram(catalog)


def test_exact_intercept_recovery_and_uncertainty() -> None:
    catalog = make_catalog()

    intercept, uncertainty = fit_additive_intercept_diagonal(catalog)
    diagram = calculate_hubble_diagram(catalog)

    expected_uncertainty = 1.0 / np.sqrt(
        np.sum(1.0 / np.asarray(catalog.distance_modulus_error) ** 2)
    )
    assert intercept == pytest.approx(0.4)
    assert uncertainty == pytest.approx(expected_uncertainty)
    assert diagram.fitted_intercept == pytest.approx(0.4)
    assert diagram.fitted_intercept_error == pytest.approx(expected_uncertainty)


def test_residual_statistics_match_diagonal_formulas() -> None:
    redshift = (0.03, 0.1, 0.3)
    reference = np.asarray(calculate_reference_distance_modulus(redshift))
    raw_offsets = np.array([0.5, 0.2, 0.35])
    errors = np.array([0.1, 0.2, 0.15])
    catalog = make_catalog(
        redshift=redshift,
        distance_modulus=tuple(reference + raw_offsets),
        distance_modulus_error=tuple(errors),
    )

    diagram = calculate_hubble_diagram(catalog)
    result = calculate_hubble_residuals_diagonal(diagram)

    weights = 1.0 / errors**2
    expected_intercept = np.sum(weights * raw_offsets) / np.sum(weights)
    expected_residual = raw_offsets - expected_intercept
    expected_chi2 = np.sum((expected_residual / errors) ** 2)
    assert result.residual == pytest.approx(expected_residual)
    assert result.weighted_residual_mean == pytest.approx(0.0, abs=1e-14)
    assert result.residual_mean == pytest.approx(np.mean(expected_residual))
    assert result.residual_rms == pytest.approx(
        np.sqrt(np.mean(expected_residual**2))
    )
    assert result.weighted_residual_rms == pytest.approx(
        np.sqrt(np.sum(weights * expected_residual**2) / np.sum(weights))
    )
    assert result.chi2_diagonal == pytest.approx(expected_chi2)
    assert result.ndof == 2
    assert result.reduced_chi2_diagonal == pytest.approx(expected_chi2 / 2)
    assert result.covariance_mode == "diagonal_statistical"


def test_extreme_positive_uncertainties_remain_with_negligible_weight() -> None:
    redshift = (0.02, 0.05, 0.1, 0.3, 0.8)
    reference = np.asarray(calculate_reference_distance_modulus(redshift))
    errors = (0.1, 0.2, 0.3, 150.0, 300.0)
    catalog = NormalizedDistanceCatalog(
        object_id=("A", "B", "C", "TAIL-1", "TAIL-2"),
        redshift=redshift,
        distance_modulus=tuple(reference + np.array([0.3, 0.35, 0.25, 20.0, -30.0])),
        distance_modulus_error=errors,
        source_row_index=(0, 1, 2, 3, 4),
        catalog_name="Extreme uncertainty fixture",
        catalog_release="Offline test",
        redshift_definition="test",
        distance_definition="test MU",
    )

    diagram = calculate_hubble_diagram(catalog)
    residuals = calculate_hubble_residuals_diagonal(diagram)
    weights = 1.0 / np.asarray(errors) ** 2

    assert diagram.number_of_supernovae == 5
    assert residuals.object_id == catalog.object_id
    assert residuals.source_row_index == catalog.source_row_index
    assert weights[3:].sum() / weights.sum() < 1.0e-6
    assert np.isfinite(
        (
            diagram.fitted_intercept,
            *residuals.residual,
            residuals.chi2_diagonal,
            residuals.residual_rms,
            residuals.weighted_residual_rms,
        )
    ).all()


def test_inputs_and_results_are_immutable_and_json_friendly() -> None:
    catalog = make_catalog()
    diagram = calculate_hubble_diagram(catalog)
    residuals = calculate_hubble_residuals_diagonal(diagram)

    json.dumps(asdict(diagram), allow_nan=False)
    json.dumps(asdict(residuals), allow_nan=False)
    assert isinstance(diagram.redshift, tuple)
    assert isinstance(residuals.residual, tuple)
    with pytest.raises(FrozenInstanceError):
        catalog.catalog_name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        diagram.fitted_intercept = 2.0  # type: ignore[misc]


def test_reference_h0_changes_normalization_but_not_final_residuals() -> None:
    catalog = make_catalog()

    diagram_60 = calculate_hubble_diagram(catalog, reference_h0=60.0)
    diagram_80 = calculate_hubble_diagram(catalog, reference_h0=80.0)
    residual_60 = calculate_hubble_residuals_diagonal(diagram_60)
    residual_80 = calculate_hubble_residuals_diagonal(diagram_80)

    assert diagram_60.provisional_reference_distance_modulus != pytest.approx(
        diagram_80.provisional_reference_distance_modulus
    )
    assert diagram_60.fitted_intercept != pytest.approx(
        diagram_80.fitted_intercept
    )
    assert residual_60.residual == pytest.approx(residual_80.residual, abs=1e-12)


def test_omega_m_changes_intercept_corrected_residual_shape() -> None:
    catalog = make_catalog()

    result_03 = calculate_hubble_residuals_diagonal(
        calculate_hubble_diagram(catalog, omega_m=0.3)
    )
    result_04 = calculate_hubble_residuals_diagonal(
        calculate_hubble_diagram(catalog, omega_m=0.4)
    )

    assert result_03.residual != pytest.approx(result_04.residual)


def test_object_ordering_is_preserved() -> None:
    catalog = make_catalog(
        object_id=("SN-C", "SN-A", "SN-B"),
        source_row_index=(11, 3, 7),
    )

    diagram = calculate_hubble_diagram(catalog)
    residuals = calculate_hubble_residuals_diagonal(diagram)

    assert diagram.object_id == catalog.object_id
    assert residuals.source_row_index == catalog.source_row_index


def test_plotting_returns_figures_and_nonempty_files(tmp_path: Path) -> None:
    diagram = calculate_hubble_diagram(make_catalog())
    residuals = calculate_hubble_residuals_diagonal(diagram)
    diagram_path = tmp_path / "diagram.png"
    residual_path = tmp_path / "residuals.png"

    diagram_figure = plot_hubble_diagram(diagram, diagram_path)
    residual_figure = plot_hubble_residuals(residuals, residual_path)

    assert isinstance(diagram_figure, Figure)
    assert isinstance(residual_figure, Figure)
    assert diagram_path.stat().st_size > 0
    assert residual_path.stat().st_size > 0
    assert diagram_figure.axes[0].get_xscale() == "log"
    assert residual_figure.axes[0].get_xscale() == "log"
    assert "nuisance offset" in diagram_figure.axes[0].get_title()
    plt.close(diagram_figure)
    plt.close(residual_figure)


def test_extreme_uncertainty_plotting_is_display_only(
    tmp_path: Path,
) -> None:
    redshift = (0.02, 0.05, 0.1, 0.3, 0.8)
    reference = np.asarray(calculate_reference_distance_modulus(redshift))
    catalog = NormalizedDistanceCatalog(
        object_id=("A", "B", "C", "TAIL-1", "TAIL-2"),
        redshift=redshift,
        distance_modulus=tuple(reference + 0.3),
        distance_modulus_error=(0.1, 0.2, 0.3, 150.0, 300.0),
        source_row_index=(0, 1, 2, 3, 4),
        catalog_name="Extreme uncertainty fixture",
        catalog_release="Offline test",
        redshift_definition="test",
        distance_definition="test MU",
    )
    diagram = calculate_hubble_diagram(catalog)
    residuals = calculate_hubble_residuals_diagonal(diagram)
    original_diagram = diagram
    original_residuals = residuals
    diagram_path = tmp_path / "display_limited_diagram.png"
    residual_path = tmp_path / "display_limited_residuals.png"
    distribution_path = tmp_path / "uncertainty_distribution.png"

    diagram_figure = plot_hubble_diagram(
        diagram,
        diagram_path,
        display_error_limit=1.0,
    )
    residual_figure = plot_hubble_residuals(
        residuals,
        residual_path,
        display_error_limit=1.0,
    )
    distribution_figure = plot_distance_uncertainty_distribution(
        diagram,
        distribution_path,
    )

    diagram_points = next(
        container
        for container in diagram_figure.axes[0].containers
        if container.get_label() == "Observed distance modulus"
    )
    residual_points = next(
        container
        for container in residual_figure.axes[0].containers
        if container.get_label() == "Hubble residual"
    )
    diagram_annotation = " ".join(
        text.get_text() for text in diagram_figure.axes[0].texts
    )
    residual_annotation = " ".join(
        text.get_text() for text in residual_figure.axes[0].texts
    )
    assert len(diagram_points.lines[0].get_xdata()) == 5
    assert len(residual_points.lines[0].get_xdata()) == 5
    assert "Display-only error limit: 1 mag" in diagram_annotation
    assert "2 error bars exceed it" in diagram_annotation
    assert "2 error bars exceed it" in residual_annotation
    assert distribution_figure.axes[0].get_xscale() == "log"
    assert diagram_path.stat().st_size > 0
    assert residual_path.stat().st_size > 0
    assert distribution_path.stat().st_size > 0
    assert diagram == original_diagram
    assert residuals == original_residuals
    plt.close(diagram_figure)
    plt.close(residual_figure)
    plt.close(distribution_figure)


def test_default_uncertainty_display_rule_is_deterministic_p95() -> None:
    errors = (0.1, 0.2, 0.3, 1.0, 200.0)

    first = calculate_uncertainty_display_limit(errors)
    second = calculate_uncertainty_display_limit(errors)

    assert first == second
    assert first == pytest.approx(np.percentile(errors, 95.0))


def test_result_schema_does_not_claim_an_h0_measurement() -> None:
    diagram_fields = {field.name for field in fields(calculate_hubble_diagram(make_catalog()))}
    residual_fields = {
        field.name
        for field in fields(
            calculate_hubble_residuals_diagonal(
                calculate_hubble_diagram(make_catalog())
            )
        )
    }

    assert "fitted_h0" not in diagram_fields
    assert "measured_h0" not in diagram_fields
    assert "fitted_h0" not in residual_fields
    assert "measured_h0" not in residual_fields


def covariance_for(
    catalog: NormalizedDistanceCatalog,
    values: np.ndarray,
) -> CovarianceMatrix:
    """Build an aligned positive-definite covariance fixture."""
    return CovarianceMatrix(
        values=values,
        dimension=len(catalog.object_id),
        covariance_kind="synthetic_full_covariance",
        source_path_name="synthetic.txt",
        source_sha256="a" * 64,
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description="Synthetic total covariance.",
    )


def test_zhel_reference_correction_matches_independent_formula() -> None:
    from astropy.cosmology import FlatLambdaCDM

    z_hd = (0.1, 0.3)
    z_hel = (0.09, 0.28)
    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
    distance = (
        cosmology.luminosity_distance(np.asarray(z_hd)).value
        * (1.0 + np.asarray(z_hel))
        / (1.0 + np.asarray(z_hd))
    )
    expected = 5.0 * np.log10(distance) + 25.0

    corrected = calculate_reference_distance_modulus(
        z_hd,
        heliocentric_redshift=z_hel,
    )
    generic = calculate_reference_distance_modulus(z_hd)

    assert corrected == pytest.approx(expected)
    assert generic == pytest.approx(cosmology.distmod(z_hd).value)
    assert corrected != pytest.approx(generic)


def test_full_covariance_intercept_and_chi_square_match_solve() -> None:
    redshift = (0.03, 0.1, 0.3)
    reference = np.asarray(calculate_reference_distance_modulus(redshift))
    raw = np.array([0.5, 0.2, 0.35])
    catalog = make_catalog(
        redshift=redshift,
        distance_modulus=tuple(reference + raw),
    )
    matrix = np.array(
        [[0.04, 0.01, 0.002], [0.01, 0.09, 0.005], [0.002, 0.005, 0.06]]
    )
    covariance = covariance_for(catalog, matrix)
    one = np.ones(3)
    solved_one = np.linalg.solve(matrix, one)
    solved_raw = np.linalg.solve(matrix, raw)
    expected_intercept = (one @ solved_raw) / (one @ solved_one)
    expected_error = np.sqrt(1.0 / (one @ solved_one))
    expected_residual = raw - expected_intercept
    expected_chi2 = expected_residual @ np.linalg.solve(
        matrix,
        expected_residual,
    )

    intercept, error = fit_additive_intercept_covariance(catalog, covariance)
    result = calculate_hubble_residuals_covariance(catalog, covariance)

    assert intercept == pytest.approx(expected_intercept)
    assert error == pytest.approx(expected_error)
    assert result.residual == pytest.approx(expected_residual)
    assert result.chi2 == pytest.approx(expected_chi2)
    assert result.ndof == 2
    assert result.reduced_chi2 == pytest.approx(expected_chi2 / 2)
    assert result.generalized_residual_mean == pytest.approx(0.0, abs=1e-14)


def test_diagonal_covariance_reproduces_diagonal_engine() -> None:
    catalog = make_catalog()
    covariance = covariance_for(
        catalog,
        np.diag(np.asarray(catalog.distance_modulus_error) ** 2),
    )
    diagonal = calculate_hubble_residuals_diagonal(
        calculate_hubble_diagram(catalog)
    )
    full = calculate_hubble_residuals_covariance(catalog, covariance)
    comparison = compare_diagonal_and_covariance_results(diagonal, full)

    assert full.fitted_intercept == pytest.approx(diagonal.fitted_intercept)
    assert full.fitted_intercept_error == pytest.approx(
        diagonal.fitted_intercept_error
    )
    assert full.residual == pytest.approx(diagonal.residual)
    assert full.chi2 == pytest.approx(diagonal.chi2_diagonal)
    assert comparison.maximum_residual_difference == pytest.approx(0.0)


def test_off_diagonal_covariance_changes_fit_and_h0_remains_invariant() -> None:
    redshift = (0.03, 0.1, 0.3)
    reference = np.asarray(calculate_reference_distance_modulus(redshift))
    catalog = make_catalog(
        redshift=redshift,
        distance_modulus=tuple(reference + np.array([0.6, 0.1, 0.4])),
    )
    diagonal_covariance = covariance_for(
        catalog,
        np.diag(np.asarray(catalog.distance_modulus_error) ** 2),
    )
    full_covariance = covariance_for(
        catalog,
        np.array(
            [
                [0.01, 0.006, 0.002],
                [0.006, 0.04, 0.012],
                [0.002, 0.012, 0.0225],
            ]
        ),
    )
    diagonal_like = calculate_hubble_residuals_covariance(
        catalog,
        diagonal_covariance,
    )
    full_60 = calculate_hubble_residuals_covariance(
        catalog,
        full_covariance,
        reference_h0=60.0,
    )
    full_80 = calculate_hubble_residuals_covariance(
        catalog,
        full_covariance,
        reference_h0=80.0,
    )

    assert full_60.fitted_intercept_error != pytest.approx(
        diagonal_like.fitted_intercept_error
    )
    assert full_60.residual == pytest.approx(full_80.residual, abs=1e-12)


def test_covariance_row_order_mismatch_is_rejected() -> None:
    catalog = make_catalog()
    covariance = covariance_for(catalog, np.eye(3))
    reordered = make_catalog(
        object_id=("SN-B", "SN-A", "SN-C"),
        source_row_index=(7, 3, 11),
    )
    with pytest.raises(CovarianceAlignmentError, match="fingerprint"):
        calculate_hubble_residuals_covariance(reordered, covariance)


def test_full_covariance_results_are_immutable_json_friendly_and_plottable(
    tmp_path: Path,
) -> None:
    catalog = make_catalog()
    covariance = covariance_for(catalog, np.eye(3) * 0.04)
    diagonal = calculate_hubble_residuals_diagonal(
        calculate_hubble_diagram(catalog)
    )
    result = calculate_hubble_residuals_covariance(catalog, covariance)
    path = tmp_path / "comparison.png"
    figure = plot_diagonal_vs_covariance_residuals(diagonal, result, path)

    json.dumps(asdict(result), allow_nan=False)
    assert path.stat().st_size > 0
    with pytest.raises(FrozenInstanceError):
        result.fitted_intercept = 1.0  # type: ignore[misc]
    plt.close(figure)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DES_SN5YR_CATALOG_PATH"),
    reason="Set DES_SN5YR_CATALOG_PATH to the official legacy catalog.",
)
def test_real_legacy_des_catalog_integration(tmp_path: Path) -> None:
    path = Path(os.environ["DES_SN5YR_CATALOG_PATH"])
    legacy = load_des_sn5yr_catalog(path)
    catalog = normalize_des_sn5yr_distances(legacy)

    diagram = calculate_hubble_diagram(catalog)
    residuals = calculate_hubble_residuals_diagonal(diagram)
    diagram_path = tmp_path / "legacy_des_hubble_diagram.png"
    residual_path = tmp_path / "legacy_des_hubble_residuals.png"
    diagram_figure = plot_hubble_diagram(diagram, diagram_path)
    residual_figure = plot_hubble_residuals(residuals, residual_path)

    assert diagram.number_of_supernovae == 1829
    assert np.isfinite(catalog.redshift).all()
    assert min(catalog.redshift) > 0.0
    assert np.isfinite(diagram.fitted_intercept)
    assert np.isfinite(residuals.residual).all()
    assert residuals.ndof == diagram.number_of_supernovae - 1
    assert diagram_path.stat().st_size > 0
    assert residual_path.stat().st_size > 0
    plt.close(diagram_figure)
    plt.close(residual_figure)
