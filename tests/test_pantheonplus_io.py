"""Tests for Pantheon+ row identity and full-covariance analysis."""

import inspect
import os
from dataclasses import FrozenInstanceError, asdict, replace
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from matplotlib.figure import Figure

import supernova_analyzer.covariance as covariance_module
import supernova_analyzer.hubble_analysis as hubble_module
import supernova_analyzer.pantheonplus_analysis as pantheon_analysis_module
from supernova_analyzer import (
    CovarianceNotPositiveDefiniteError,
    CovarianceAlignmentError,
    CovarianceError,
    PantheonPlusCatalogError,
    calculate_hubble_residuals_covariance,
    calculate_reference_distance_modulus,
    catalog_order_fingerprint,
    compare_pantheonplus_covariance_diagonal,
    compare_pantheonplus_covariance_results,
    covariance_summary,
    load_flattened_covariance,
    load_pantheonplus_catalog,
    load_pantheonplus_covariance,
    normalize_pantheonplus_distances,
    plot_pantheonplus_covariance_comparison,
    select_pantheonplus_hubble_flow_subset,
    subset_pantheonplus_covariance,
)


def table(
    rows: tuple[str, ...] | None = None,
    header: str | None = None,
) -> StringIO:
    """Return a small Pantheon+ whitespace data vector."""
    selected_header = header or (
        "CID IDSURVEY zHD zHEL MU_SH0ES MU_SH0ES_ERR_DIAG "
        "m_b_corr m_b_corr_err_DIAG IS_CALIBRATOR HOST_LOGMASS"
    )
    selected_rows = rows or (
        "SN-A 1 0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
        "SN-A 2 0.05 0.04 36.9 0.12 15.2 0.13 0 10.1",
        "SN-B 3 0.20 0.19 40.0 0.20 18.0 0.21 0 9.8",
    )
    return StringIO("\n".join((selected_header, *selected_rows)))


def covariance_text(matrix: np.ndarray) -> StringIO:
    """Return the released leading-N sequential row-major convention."""
    values = "\n".join(f"{value:.17g}" for value in matrix.reshape(-1))
    return StringIO(f"{len(matrix)}\n{values}\n")


def load_covariance(
    matrix: np.ndarray,
    kind: str = "pantheonplus_statonly_complete_covariance",
):
    """Load one aligned complete covariance fixture."""
    catalog = load_pantheonplus_catalog(table())
    return load_pantheonplus_covariance(
        covariance_text(matrix),
        catalog,
        covariance_kind=kind,
        composition_description="Complete released test covariance.",
    )


def test_whitespace_parser_preserves_repeated_cid_and_exact_order() -> None:
    catalog = load_pantheonplus_catalog(table())

    assert catalog.row_count == 3
    assert catalog.cid == ("SN-A", "SN-A", "SN-B")
    assert catalog.idsurvey == ("1", "2", "3")
    assert catalog.source_row_indices == (0, 1, 2)
    assert catalog.duplicate_cid_summary.number_of_unique_cids == 2
    assert catalog.duplicate_cid_summary.repeated_cid_row_count == 1
    assert catalog.duplicate_cid_summary.maximum_multiplicity == 2
    assert catalog.calibrator_row_count == 1
    assert catalog.non_calibrator_row_count == 2


def test_same_cid_different_surveys_remains_distinct_after_normalization() -> None:
    catalog = load_pantheonplus_catalog(table())
    normalized = normalize_pantheonplus_distances(catalog)

    assert normalized.object_id == ("SN-A::1", "SN-A::2", "SN-B::3")
    assert normalized.source_row_index == (0, 1, 2)
    assert len(normalized.object_id) == catalog.row_count
    assert normalized.catalog_name == "Pantheon+"
    assert "MU_SH0ES" in normalized.distance_definition


def test_hubble_flow_subset_preserves_order_repeated_cids_and_indices() -> None:
    rows = (
        "SN-A 1 0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
        "SN-A 2 0.05 0.04 36.9 0.12 15.2 0.13 0 10.1",
        "SN-B 3 0.20 0.19 40.0 0.20 18.0 0.21 0 9.8",
        "SN-A 4 0.06 0.05 37.0 0.14 15.3 0.15 0 10.1",
    )
    catalog = load_pantheonplus_catalog(table(rows))
    selection = select_pantheonplus_hubble_flow_subset(catalog)

    assert selection.full_release_row_count == 4
    assert selection.calibrator_row_count == 1
    assert selection.hubble_flow_row_count == 3
    assert selection.retained_source_row_indices == (1, 2, 3)
    assert selection.normalized_catalog.object_id == (
        "SN-A::2",
        "SN-B::3",
        "SN-A::4",
    )
    assert selection.full_catalog_fingerprint != (
        selection.subset_catalog_fingerprint
    )
    assert sum(
        identity.startswith("SN-A::")
        for identity in selection.normalized_catalog.object_id
    ) == 2


def test_principal_covariance_subset_uses_identical_source_rows() -> None:
    rows = (
        "SN-A 1 0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
        "SN-A 2 0.05 0.04 36.9 0.12 15.2 0.13 0 10.1",
        "SN-B 3 0.20 0.19 40.0 0.20 18.0 0.21 0 9.8",
        "SN-A 4 0.06 0.05 37.0 0.14 15.3 0.15 0 10.1",
    )
    catalog = load_pantheonplus_catalog(table(rows))
    selection = select_pantheonplus_hubble_flow_subset(catalog)
    matrix = np.array(
        [
            [1.0, 0.01, 0.02, 0.03],
            [0.01, 2.0, 0.04, 0.05],
            [0.02, 0.04, 3.0, 0.06],
            [0.03, 0.05, 0.06, 4.0],
        ]
    )
    parent = load_pantheonplus_covariance(
        covariance_text(matrix),
        catalog,
        covariance_kind="pantheonplus_statonly_complete_covariance",
        composition_description="Complete test covariance.",
    )
    subset = subset_pantheonplus_covariance(parent, selection)

    assert np.array_equal(subset.values, matrix[np.ix_((1, 2, 3), (1, 2, 3))])
    assert subset.dimension == 3
    assert subset.original_dimension == 4
    assert subset.parent_source_sha256 == parent.source_sha256
    assert subset.retained_source_row_indices == (1, 2, 3)
    assert subset.catalog_order_fingerprint == selection.subset_catalog_fingerprint
    assert not subset.values.flags.writeable

    normalized = selection.normalized_catalog
    reversed_catalog = replace(
        normalized,
        object_id=normalized.object_id[::-1],
        redshift=normalized.redshift[::-1],
        heliocentric_redshift=normalized.heliocentric_redshift[::-1],
        distance_modulus=normalized.distance_modulus[::-1],
        distance_modulus_error=normalized.distance_modulus_error[::-1],
        source_row_index=normalized.source_row_index[::-1],
    )
    with pytest.raises(CovarianceAlignmentError, match="fingerprint"):
        calculate_hubble_residuals_covariance(reversed_catalog, subset)


def test_duplicate_cid_survey_pair_is_rejected() -> None:
    rows = (
        "SN-A 1 0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
        "SN-A 1 0.05 0.04 36.9 0.12 15.2 0.13 1 10.1",
    )
    with pytest.raises(PantheonPlusCatalogError, match="must be unique"):
        load_pantheonplus_catalog(table(rows))


def test_missing_and_invalid_required_values_are_rejected() -> None:
    with pytest.raises(PantheonPlusCatalogError, match="missing required"):
        load_pantheonplus_catalog(
            table(header="CID IDSURVEY zHD zHEL MU_SH0ES")
        )
    invalid_rows = (
        "SN-A 1 -0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
    )
    with pytest.raises(PantheonPlusCatalogError, match="zHD must be greater"):
        load_pantheonplus_catalog(table(invalid_rows))
    error_rows = (
        "SN-A 1 0.05 0.04 36.8 0.0 15.1 0.11 1 10.1",
    )
    with pytest.raises(PantheonPlusCatalogError, match="must be greater"):
        load_pantheonplus_catalog(table(error_rows))


def test_fingerprint_is_order_sensitive_to_cid_survey_rows() -> None:
    first = normalize_pantheonplus_distances(load_pantheonplus_catalog(table()))
    shuffled_rows = (
        "SN-A 2 0.05 0.04 36.9 0.12 15.2 0.13 1 10.1",
        "SN-A 1 0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
        "SN-B 3 0.20 0.19 40.0 0.20 18.0 0.21 0 9.8",
    )
    second = normalize_pantheonplus_distances(
        load_pantheonplus_catalog(table(shuffled_rows))
    )
    assert catalog_order_fingerprint(first) != catalog_order_fingerprint(second)


def test_flattened_covariance_reconstruction_is_read_only_and_unmodified() -> None:
    matrix = np.array(
        [[0.04, 0.01, 0.002], [0.01, 0.05, 0.003], [0.002, 0.003, 0.06]]
    )
    covariance = load_covariance(matrix)

    assert np.allclose(covariance.values, matrix)
    assert not covariance.values.flags.writeable
    assert covariance.covariance_kind == (
        "pantheonplus_statonly_complete_covariance"
    )
    with pytest.raises(ValueError):
        covariance.values[0, 0] = 1.0


def test_pantheon_tolerance_is_explicit_and_legacy_default_stays_tight() -> None:
    matrix = np.array(
        [[1.0, 5.0e-8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    catalog = load_pantheonplus_catalog(table())
    pantheon = load_pantheonplus_covariance(
        covariance_text(matrix),
        catalog,
        covariance_kind="pantheonplus_test_covariance",
        composition_description="Tolerance test.",
    )

    assert pantheon.symmetry_rtol == pytest.approx(1.0e-7)
    assert pantheon.symmetry_atol == pytest.approx(1.0e-12)
    assert pantheon.values[0, 1] == pytest.approx(5.0e-8)
    assert pantheon.values[1, 0] == pytest.approx(0.0)
    with pytest.raises(CovarianceError, match="rtol=1e-10"):
        load_flattened_covariance(
            covariance_text(matrix),
            covariance_kind="legacy_tight_test",
            catalog_order_fingerprint="test-fingerprint",
            release_name="Legacy test",
            composition_description="Tight default test.",
        )


def test_wrong_length_alignment_and_non_positive_definite_are_rejected() -> None:
    catalog = load_pantheonplus_catalog(table())
    with pytest.raises(ValueError, match="truncated"):
        load_pantheonplus_covariance(
            StringIO("3\n1 0 0\n"),
            catalog,
            covariance_kind="test",
            composition_description="test",
        )
    shuffled = load_pantheonplus_catalog(
        table(
            (
                "SN-A 2 0.05 0.04 36.9 0.12 15.2 0.13 1 10.1",
                "SN-A 1 0.05 0.04 36.8 0.10 15.1 0.11 1 10.1",
                "SN-B 3 0.20 0.19 40.0 0.20 18.0 0.21 0 9.8",
            )
        )
    )
    covariance = load_covariance(np.eye(3))
    with pytest.raises(ValueError, match="fingerprint"):
        calculate_hubble_residuals_covariance(
            normalize_pantheonplus_distances(shuffled),
            covariance,
        )
    with pytest.raises(CovarianceNotPositiveDefiniteError):
        load_covariance(np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_diagonal_errors_are_diagnostic_and_never_added() -> None:
    matrix = np.diag([0.04, 0.05, 0.06])
    catalog = load_pantheonplus_catalog(table())
    covariance = load_covariance(matrix)
    diagnostics = compare_pantheonplus_covariance_diagonal(covariance, catalog)

    assert np.allclose(covariance.values, matrix)
    assert diagnostics.mu_error_diagonal_rms_difference > 0.0
    assert diagnostics.mb_error_diagonal_rms_difference is not None


def test_covariance_analysis_preserves_repeated_rows_and_h0_invariance() -> None:
    catalog = load_pantheonplus_catalog(table())
    selection = select_pantheonplus_hubble_flow_subset(catalog)
    normalized = selection.normalized_catalog
    first_covariance = subset_pantheonplus_covariance(load_covariance(
        np.diag([0.04, 0.05, 0.06]),
        "pantheonplus_statonly_complete_covariance",
    ), selection)
    second_covariance = subset_pantheonplus_covariance(load_covariance(
        np.array(
            [[0.05, 0.01, 0.002], [0.01, 0.06, 0.003], [0.002, 0.003, 0.07]]
        ),
        "pantheonplus_stat_plus_systematic_complete_covariance",
    ), selection)
    first = calculate_hubble_residuals_covariance(normalized, first_covariance)
    second = calculate_hubble_residuals_covariance(normalized, second_covariance)
    comparison = compare_pantheonplus_covariance_results(first, second)
    shifted = calculate_hubble_residuals_covariance(
        normalized,
        second_covariance,
        reference_h0=80.0,
    )

    assert first.object_id == ("SN-A::2", "SN-B::3")
    assert "SN-A::1" not in first.object_id
    assert first.number_of_supernovae == 2
    assert first.generalized_residual_mean == pytest.approx(0.0, abs=1e-13)
    assert first.residual == pytest.approx(
        calculate_hubble_residuals_covariance(
            normalized,
            first_covariance,
            reference_h0=60.0,
        ).residual,
        abs=1e-12,
    )
    assert second.residual == pytest.approx(shifted.residual, abs=1e-12)
    assert comparison.maximum_residual_difference > 0.0
    assert first.covariance_mode != second.covariance_mode


def test_zhel_correction_and_immutable_catalog_result() -> None:
    catalog = load_pantheonplus_catalog(table())
    normalized = normalize_pantheonplus_distances(catalog)
    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
    distance = (
        cosmology.luminosity_distance(np.asarray(normalized.redshift)).value
        * (1.0 + np.asarray(normalized.heliocentric_redshift))
        / (1.0 + np.asarray(normalized.redshift))
    )
    expected = 5.0 * np.log10(distance) + 25.0

    assert calculate_reference_distance_modulus(
        normalized.redshift,
        heliocentric_redshift=normalized.heliocentric_redshift,
    ) == pytest.approx(expected)
    asdict(catalog.duplicate_cid_summary)
    with pytest.raises(FrozenInstanceError):
        catalog.catalog_name = "changed"  # type: ignore[misc]


def test_covariance_comparison_plot_is_nonempty(tmp_path: Path) -> None:
    catalog = load_pantheonplus_catalog(table())
    selection = select_pantheonplus_hubble_flow_subset(catalog)
    normalized = selection.normalized_catalog
    first = calculate_hubble_residuals_covariance(
        normalized,
        subset_pantheonplus_covariance(
            load_covariance(np.diag([0.04, 0.05, 0.06]), "statonly"),
            selection,
        ),
    )
    second = calculate_hubble_residuals_covariance(
        normalized,
        subset_pantheonplus_covariance(
            load_covariance(
                np.array(
                    [
                        [0.05, 0.01, 0.0],
                        [0.01, 0.06, 0.0],
                        [0.0, 0.0, 0.07],
                    ]
                ),
                "statsys",
            ),
            selection,
        ),
    )
    destination = tmp_path / "comparison.png"
    figure = plot_pantheonplus_covariance_comparison(
        first,
        second,
        destination,
    )

    assert isinstance(figure, Figure)
    assert destination.stat().st_size > 0
    plt.close(figure)


def test_no_explicit_inverse_is_used() -> None:
    source = (
        inspect.getsource(pantheon_analysis_module)
        + inspect.getsource(covariance_module)
        + inspect.getsource(hubble_module)
    )
    assert "np.linalg.inv(" not in source
    assert "scipy.linalg.inv(" not in source


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        os.getenv(name)
        for name in (
            "PANTHEONPLUS_DATA_PATH",
            "PANTHEONPLUS_STATONLY_PATH",
            "PANTHEONPLUS_STAT_SYS_PATH",
        )
    ),
    reason="Set Pantheon+ data and both covariance path variables.",
)
def test_real_pantheonplus_integration() -> None:
    catalog = load_pantheonplus_catalog(Path(os.environ["PANTHEONPLUS_DATA_PATH"]))
    full_normalized = normalize_pantheonplus_distances(catalog)
    selection = select_pantheonplus_hubble_flow_subset(catalog)
    normalized = selection.normalized_catalog
    fingerprint = catalog_order_fingerprint(full_normalized)
    assert catalog.row_count == 1701
    assert catalog.calibrator_row_count == 77
    assert catalog.non_calibrator_row_count == 1624
    assert selection.hubble_flow_row_count == 1624
    assert catalog.duplicate_cid_summary.number_of_unique_cids == 1543
    assert catalog.duplicate_cid_summary.repeated_cid_row_count == 158
    assert len(set(zip(catalog.cid, catalog.idsurvey, strict=True))) == 1701
    assert fingerprint == catalog_order_fingerprint(full_normalized)
    assert fingerprint != catalog_order_fingerprint(normalized)

    results = []
    for environment_name, kind in (
        ("PANTHEONPLUS_STATONLY_PATH", "pantheonplus_statonly_complete_covariance"),
        (
            "PANTHEONPLUS_STAT_SYS_PATH",
            "pantheonplus_stat_plus_systematic_complete_covariance",
        ),
    ):
        covariance = load_pantheonplus_covariance(
            Path(os.environ[environment_name]),
            catalog,
            covariance_kind=kind,
            composition_description="Complete released covariance product.",
        )
        subset_covariance = subset_pantheonplus_covariance(
            covariance,
            selection,
        )
        result = calculate_hubble_residuals_covariance(
            normalized,
            subset_covariance,
        )
        assert covariance.dimension == 1701
        assert subset_covariance.dimension == 1624
        assert covariance_summary(covariance).cholesky_success
        assert covariance_summary(subset_covariance).cholesky_success
        assert np.isfinite((result.fitted_intercept, result.chi2)).all()
        shifted = calculate_hubble_residuals_covariance(
            normalized,
            subset_covariance,
            reference_h0=80.0,
        )
        assert result.residual == pytest.approx(shifted.residual, abs=1e-11)
        results.append(result)
    assert results[0].covariance_mode != results[1].covariance_mode
