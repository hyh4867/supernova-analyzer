"""Offline tests for covariance loading, composition, and provenance."""

import gzip
import inspect
import os
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pytest

from supernova_analyzer import (
    CovarianceAlignmentError,
    CovarianceError,
    CovarianceMatrix,
    CovarianceNotPositiveDefiniteError,
    NormalizedDistanceCatalog,
    catalog_order_fingerprint,
    compose_covariance_with_statistical_diagonal,
    compose_legacy_des_total_covariance,
    covariance_summary,
    load_flattened_covariance,
    load_des_sn5yr_catalog,
    normalize_des_sn5yr_distances,
    calculate_hubble_residuals_covariance,
)
from supernova_analyzer.covariance import require_positive_definite
import supernova_analyzer.covariance as covariance_module
import supernova_analyzer.hubble_analysis as hubble_module


def catalog(order: tuple[str, ...] = ("A", "B")) -> NormalizedDistanceCatalog:
    """Return a two-row legacy-shaped normalized catalog."""
    return NormalizedDistanceCatalog(
        object_id=order,
        redshift=(0.1, 0.2),
        distance_modulus=(38.0, 40.0),
        distance_modulus_error=(0.2, 150.0),
        source_row_index=(0, 1),
        catalog_name="Legacy DES-SN5YR HD + metadata catalog",
        catalog_release="Legacy DES-SN5YR (Vincenzi et al. 2024)",
        redshift_definition="zHD",
        distance_definition="MU with MUERR_FINAL",
        heliocentric_redshift=(0.09, 0.19),
        reference_distance_convention=(
            "zHD cosmology redshift with zHEL frame correction"
        ),
    )


def load(source: object, cat: NormalizedDistanceCatalog | None = None) -> CovarianceMatrix:
    """Load a test covariance with explicit semantics."""
    selected = cat or catalog()
    return load_flattened_covariance(
        source,  # type: ignore[arg-type]
        covariance_kind="systematic_component",
        catalog_order_fingerprint=catalog_order_fingerprint(selected),
        release_name=selected.catalog_release,
        composition_description="Synthetic component for offline tests.",
    )


def test_plain_text_and_gzip_loading_preserve_stream_position() -> None:
    text = "2\n1 0.1 0.1 2\n"
    stream = StringIO(text)
    stream.seek(3)
    plain = load(stream)
    compressed = load(BytesIO(gzip.compress(text.encode())))

    assert stream.tell() == 3
    assert plain.dimension == 2
    assert np.allclose(plain.values, [[1.0, 0.1], [0.1, 2.0]])
    assert np.allclose(compressed.values, plain.values)
    assert len(plain.source_sha256) == 64


@pytest.mark.parametrize(
    ("content", "match"),
    (
        ("two 1 0 0 1", "integer dimension"),
        ("2.5 1 0 0 1", "integer dimension"),
        ("2 1 0 0", "truncated"),
        ("2 1 0 0 1 2", "oversized"),
        ("2 1 bad 0 1", "numeric"),
        ("2 1 nan nan 1", "finite"),
    ),
)
def test_loader_rejects_malformed_content(content: str, match: str) -> None:
    with pytest.raises(CovarianceError, match=match):
        load(StringIO(content))


def test_matrix_is_copied_read_only_and_hash_is_stable() -> None:
    first = load(StringIO("2 1 0 0 2"))
    second = load(StringIO("2 1 0 0 2"))

    assert first.source_sha256 == second.source_sha256
    assert not first.values.flags.writeable
    with pytest.raises(ValueError):
        first.values[0, 0] = 2.0


def test_symmetry_and_positive_definiteness_validation() -> None:
    with pytest.raises(CovarianceError, match="symmetric"):
        load(StringIO("2 1 0.2 0.1 1"))
    indefinite = load(StringIO("2 1 2 2 1"))
    with pytest.raises(CovarianceNotPositiveDefiniteError):
        require_positive_definite(indefinite)


def test_catalog_fingerprint_is_stable_and_order_sensitive() -> None:
    first = catalog_order_fingerprint(catalog())
    assert first == catalog_order_fingerprint(catalog())
    assert first != catalog_order_fingerprint(catalog(("B", "A")))


def test_alignment_rejects_same_dimension_in_different_order() -> None:
    component = load(StringIO("2 0 0 0 0"))
    with pytest.raises(CovarianceAlignmentError, match="fingerprint"):
        compose_legacy_des_total_covariance(
            component,
            catalog(("B", "A")),
            component_convention="official_tutorial_component",
        )


def test_statistical_diagonal_is_added_exactly_once_and_extreme_is_retained() -> None:
    cat = catalog()
    component = load(StringIO("2 0.01 0.002 0.002 0.03"), cat)
    total = compose_legacy_des_total_covariance(
        component,
        cat,
        component_convention="official_tutorial_component",
    )

    assert np.diag(total.values) == pytest.approx(
        np.diag(component.values)
        + np.asarray(cat.distance_modulus_error) ** 2
    )
    assert total.values[1, 1] == pytest.approx(22500.03)


def test_verified_released_total_does_not_add_statistical_diagonal() -> None:
    cat = catalog()
    released = load(StringIO("2 1 0.1 0.1 2"), cat)
    total = compose_legacy_des_total_covariance(
        released,
        cat,
        component_convention="verified_released_total",
    )
    assert total.values == pytest.approx(released.values)


def test_generic_composition_rejects_nonpositive_total_diagonal() -> None:
    component = load(StringIO("2 -1 0 0 -1"))
    with pytest.raises(CovarianceError, match="strictly positive"):
        compose_covariance_with_statistical_diagonal(
            component,
            (0.1, 0.1),
            covariance_kind="test",
            composition_description="test",
        )


def test_covariance_summary_is_json_friendly_and_reports_cholesky() -> None:
    matrix = load(StringIO("2 1 0.1 0.1 2"))
    summary = covariance_summary(matrix)
    assert summary.dimension == 2
    assert summary.diagonal_min == pytest.approx(1.0)
    assert summary.off_diagonal_rms == pytest.approx(0.1)
    assert summary.cholesky_success
    assert summary.condition_estimate is None


def test_production_covariance_code_does_not_use_explicit_inverse() -> None:
    source = inspect.getsource(covariance_module) + inspect.getsource(
        hubble_module
    )
    assert "np.linalg.inv(" not in source
    assert "scipy.linalg.inv(" not in source
    assert " linalg.inv(" not in source


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        os.getenv(name)
        for name in (
            "DES_SN5YR_CATALOG_PATH",
            "DES_SN5YR_STATONLY_PATH",
            "DES_SN5YR_STAT_SYS_PATH",
        )
    ),
    reason="Set all three legacy DES catalog and covariance path variables.",
)
def test_real_legacy_des_full_covariance_integration() -> None:
    catalog_path = Path(os.environ["DES_SN5YR_CATALOG_PATH"])
    stat_sys_path = Path(os.environ["DES_SN5YR_STAT_SYS_PATH"])
    statonly_path = Path(os.environ["DES_SN5YR_STATONLY_PATH"])
    catalog = normalize_des_sn5yr_distances(
        load_des_sn5yr_catalog(catalog_path)
    )
    component = load_flattened_covariance(
        stat_sys_path,
        covariance_kind="legacy_des_stat_plus_systematic_component",
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description="Official released component.",
    )
    statonly = load_flattened_covariance(
        statonly_path,
        covariance_kind="legacy_des_statistical_component",
        catalog_order_fingerprint=catalog_order_fingerprint(catalog),
        release_name=catalog.catalog_release,
        composition_description="Official released component.",
    )
    total = compose_legacy_des_total_covariance(
        component,
        catalog,
        component_convention="official_tutorial_component",
    )
    result = calculate_hubble_residuals_covariance(catalog, total)
    summary = covariance_summary(total)

    assert len(catalog.object_id) == 1829
    assert component.dimension == 1829
    assert statonly.dimension == 1829
    assert len(component.source_sha256) == 64
    assert len(statonly.source_sha256) == 64
    assert np.all(np.diag(total.values) > 0.0)
    assert summary.cholesky_success
    assert np.isfinite(
        (
            result.fitted_intercept,
            result.chi2,
            result.reduced_chi2,
            *result.residual,
        )
    ).all()
