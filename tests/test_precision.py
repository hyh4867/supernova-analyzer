"""Tests for packed precision loading and precision-weighted residuals."""

import inspect
import os
import zipfile
from dataclasses import FrozenInstanceError, asdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import supernova_analyzer.precision as precision_module
from supernova_analyzer import (
    NormalizedDistanceCatalog,
    PrecisionMatrix,
    PrecisionMatrixError,
    calculate_hubble_residuals_precision,
    calculate_profiled_precision_chi2,
    calculate_reference_distance_modulus,
    catalog_order_fingerprint,
    compare_precision_results,
    fit_additive_intercept_precision,
    load_des_dovekie_hd,
    load_packed_precision_matrix,
    normalize_des_dovekie_distances,
    precision_summary,
    plot_precision_residual_comparison,
    validate_precision_alignment,
)


def catalog(
    object_id: tuple[str, ...] = ("A", "B", "C"),
) -> NormalizedDistanceCatalog:
    """Return a small immutable DES-Dovekie-shaped catalog."""
    redshift = (0.05, 0.1, 0.2)
    reference = calculate_reference_distance_modulus(redshift)
    return NormalizedDistanceCatalog(
        object_id=object_id,
        redshift=redshift,
        heliocentric_redshift=None,
        distance_modulus=tuple(
            value + offset
            for value, offset in zip(reference, (0.4, 0.2, 0.5), strict=True)
        ),
        distance_modulus_error=(0.1, 0.2, 0.15),
        source_row_index=(0, 1, 2),
        catalog_name="DES-Dovekie",
        catalog_release="Current official DES-Dovekie release",
        redshift_definition="zHD",
        distance_definition="Released DES-Dovekie distance modulus MU",
    )


def write_npz(
    path: Path,
    matrix: np.ndarray,
    *,
    keys: bool = True,
    packed: np.ndarray | None = None,
    stored_allow_pickle: bool = False,
) -> None:
    """Write the official packed-upper-triangle NPZ shape."""
    values = (
        np.asarray(packed)
        if packed is not None
        else matrix[np.triu_indices(len(matrix))]
    )
    if keys:
        np.savez(path, nsn=np.array([len(matrix)]), cov=values)
        buffer = BytesIO()
        np.save(buffer, np.array(stored_allow_pickle), allow_pickle=False)
        with zipfile.ZipFile(path, mode="a") as archive:
            archive.writestr("allow_pickle.npy", buffer.getvalue())
    else:
        np.savez(path, nsn=np.array([len(matrix)]), cov=values)


def load(path: Path, selected: NormalizedDistanceCatalog | None = None):
    """Load one explicit complete precision fixture."""
    chosen = selected or catalog()
    return load_packed_precision_matrix(
        path,
        precision_kind="synthetic_complete_precision",
        catalog_order_fingerprint=catalog_order_fingerprint(chosen),
        release_name=chosen.catalog_release,
        composition_description="Complete synthetic inverse covariance.",
    )


def precision_for(
    selected: NormalizedDistanceCatalog,
    values: np.ndarray,
    kind: str = "synthetic_complete_precision",
) -> PrecisionMatrix:
    """Construct an aligned precision fixture."""
    return PrecisionMatrix(
        values=values,
        dimension=len(selected.object_id),
        precision_kind=kind,
        source_path_name="synthetic.npz",
        source_sha256="a" * 64,
        catalog_order_fingerprint=catalog_order_fingerprint(selected),
        release_name=selected.catalog_release,
        composition_description="Complete synthetic inverse covariance.",
        npz_keys=("nsn", "cov", "allow_pickle"),
    )


def test_packed_upper_triangle_is_reconstructed_and_reflected(
    tmp_path: Path,
) -> None:
    matrix = np.array(
        [[4.0, 0.5, 0.2], [0.5, 5.0, -0.1], [0.2, -0.1, 6.0]]
    )
    path = tmp_path / "precision.npz"
    write_npz(path, matrix)

    precision = load(path)

    assert np.allclose(precision.values, matrix)
    assert np.allclose(precision.values, precision.values.T)
    assert not precision.values.flags.writeable
    assert precision.npz_keys == ("nsn", "cov", "allow_pickle")
    with pytest.raises(ValueError):
        precision.values[0, 0] = 2.0


def test_loader_hash_is_stable_and_stored_allow_pickle_is_not_permission(
    tmp_path: Path,
) -> None:
    matrix = np.eye(3)
    path = tmp_path / "precision.npz"
    write_npz(path, matrix, stored_allow_pickle=True)

    first = load(path)
    second = load(path)

    assert first.source_sha256 == second.source_sha256
    assert len(first.source_sha256) == 64


def test_loader_rejects_wrong_keys_and_packed_length(tmp_path: Path) -> None:
    matrix = np.eye(3)
    missing_key = tmp_path / "missing.npz"
    write_npz(missing_key, matrix, keys=False)
    with pytest.raises(PrecisionMatrixError, match="keys must be exactly"):
        load(missing_key)

    truncated = tmp_path / "truncated.npz"
    write_npz(truncated, matrix, packed=np.ones(5))
    with pytest.raises(PrecisionMatrixError, match="Packed precision length"):
        load(truncated)


@pytest.mark.parametrize(
    ("matrix", "match"),
    (
        (
            np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "finite",
        ),
        (
            np.diag([1.0, 0.0, 1.0]),
            "diagonal must be strictly positive",
        ),
        (
            np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "not positive definite",
        ),
    ),
)
def test_loader_rejects_invalid_precision(
    tmp_path: Path,
    matrix: np.ndarray,
    match: str,
) -> None:
    path = tmp_path / "invalid.npz"
    write_npz(path, matrix)
    with pytest.raises(PrecisionMatrixError, match=match):
        load(path)


def test_precision_summary_and_frozen_result_are_json_friendly() -> None:
    selected = catalog()
    precision = precision_for(selected, np.diag([100.0, 25.0, 44.0]))
    summary = precision_summary(precision)
    result = calculate_hubble_residuals_precision(selected, precision)

    assert summary.cholesky_success
    assert summary.minimum_eigenvalue == pytest.approx(25.0)
    assert isinstance(asdict(result)["residual"], tuple)
    with pytest.raises(FrozenInstanceError):
        result.chi2 = 2.0  # type: ignore[misc]


def test_fingerprint_rejects_metadata_or_sorted_order() -> None:
    selected = catalog()
    precision = precision_for(selected, np.eye(3))
    reordered = catalog(("B", "A", "C"))

    with pytest.raises(PrecisionMatrixError, match="fingerprint"):
        validate_precision_alignment(precision, reordered)


def test_precision_intercept_and_profiled_chi2_identity() -> None:
    selected = catalog()
    matrix = np.array(
        [[80.0, 4.0, 1.0], [4.0, 30.0, 3.0], [1.0, 3.0, 50.0]]
    )
    precision = precision_for(selected, matrix)
    reference = np.asarray(calculate_reference_distance_modulus(selected.redshift))
    raw = np.asarray(selected.distance_modulus) - reference
    one = np.ones(3)
    expected_intercept = (one @ matrix @ raw) / (one @ matrix @ one)
    expected_error = np.sqrt(1.0 / (one @ matrix @ one))
    expected_residual = raw - expected_intercept
    expected_chi2 = expected_residual @ matrix @ expected_residual

    intercept, error = fit_additive_intercept_precision(selected, precision)
    result = calculate_hubble_residuals_precision(selected, precision)

    assert intercept == pytest.approx(expected_intercept)
    assert error == pytest.approx(expected_error)
    assert result.residual == pytest.approx(expected_residual)
    assert result.chi2 == pytest.approx(expected_chi2)
    assert result.profiled_chi2 == pytest.approx(expected_chi2)
    assert calculate_profiled_precision_chi2(raw, precision) == pytest.approx(
        expected_chi2
    )
    assert result.generalized_residual_mean == pytest.approx(0.0, abs=1e-14)
    assert result.ndof == 2


def test_diagonal_precision_equivalence_and_off_diagonal_effect() -> None:
    selected = catalog()
    diagonal = precision_for(
        selected,
        np.diag(1.0 / np.asarray(selected.distance_modulus_error) ** 2),
        "statonly_complete_precision",
    )
    off_diagonal = precision_for(
        selected,
        np.array(
            [[100.0, 8.0, 2.0], [8.0, 25.0, 4.0], [2.0, 4.0, 44.444]]
        ),
        "statsys_complete_precision",
    )
    first = calculate_hubble_residuals_precision(selected, diagonal)
    second = calculate_hubble_residuals_precision(selected, off_diagonal)
    comparison = compare_precision_results(first, second)
    weights = 1.0 / np.asarray(selected.distance_modulus_error) ** 2
    reference = np.asarray(calculate_reference_distance_modulus(selected.redshift))
    raw = np.asarray(selected.distance_modulus) - reference

    assert first.fitted_intercept == pytest.approx(
        np.sum(weights * raw) / np.sum(weights)
    )
    assert second.fitted_intercept_error != pytest.approx(
        first.fitted_intercept_error
    )
    assert comparison.first_kind == "statonly_complete_precision"
    assert comparison.second_kind == "statsys_complete_precision"
    assert comparison.maximum_residual_difference > 0.0


def test_precision_h0_invariance() -> None:
    selected = catalog()
    precision = precision_for(selected, np.eye(3) * 40.0)
    low = calculate_hubble_residuals_precision(
        selected,
        precision,
        reference_h0=60.0,
    )
    high = calculate_hubble_residuals_precision(
        selected,
        precision,
        reference_h0=80.0,
    )
    assert low.residual == pytest.approx(high.residual, abs=1e-12)


def test_precision_comparison_plot_is_nonempty(tmp_path: Path) -> None:
    selected = catalog()
    first = calculate_hubble_residuals_precision(
        selected,
        precision_for(selected, np.eye(3) * 40.0, "statonly"),
    )
    second = calculate_hubble_residuals_precision(
        selected,
        precision_for(
            selected,
            np.array(
                [[40.0, 2.0, 0.0], [2.0, 35.0, 1.0], [0.0, 1.0, 45.0]]
            ),
            "statsys",
        ),
    )
    destination = tmp_path / "precision_comparison.png"

    figure = plot_precision_residual_comparison(
        first,
        second,
        destination,
    )

    assert isinstance(figure, Figure)
    assert destination.stat().st_size > 0
    plt.close(figure)


def test_production_precision_code_has_no_explicit_inverse() -> None:
    source = inspect.getsource(precision_module)
    assert "np.linalg.inv(" not in source
    assert "scipy.linalg.inv(" not in source
    assert " linalg.inv(" not in source


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        os.getenv(name)
        for name in (
            "DES_DOVEKIE_HD_PATH",
            "DES_DOVEKIE_STATONLY_PATH",
            "DES_DOVEKIE_STAT_SYS_PATH",
        )
    ),
    reason="Set DES-Dovekie HD and both precision path variables.",
)
def test_real_dovekie_precision_integration() -> None:
    selected = normalize_des_dovekie_distances(
        load_des_dovekie_hd(Path(os.environ["DES_DOVEKIE_HD_PATH"]))
    )
    fingerprint = catalog_order_fingerprint(selected)
    matrices = []
    for environment_name, kind in (
        ("DES_DOVEKIE_STATONLY_PATH", "dovekie_statonly_precision"),
        ("DES_DOVEKIE_STAT_SYS_PATH", "dovekie_stat_plus_systematic_precision"),
    ):
        matrix = load_packed_precision_matrix(
            Path(os.environ[environment_name]),
            precision_kind=kind,
            catalog_order_fingerprint=fingerprint,
            release_name=selected.catalog_release,
            composition_description="Complete released precision matrix.",
        )
        result = calculate_hubble_residuals_precision(selected, matrix)
        assert matrix.dimension == 1820
        assert precision_summary(matrix).cholesky_success
        if kind == "dovekie_statonly_precision":
            expected_diagonal = (
                1.0 / np.asarray(selected.distance_modulus_error) ** 2
            )
            assert np.allclose(
                np.diag(matrix.values),
                expected_diagonal,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        assert np.isfinite(
            (result.fitted_intercept, result.chi2, result.reduced_chi2)
        ).all()
        invariant = calculate_hubble_residuals_precision(
            selected,
            matrix,
            reference_h0=75.0,
        )
        assert result.residual == pytest.approx(invariant.residual, abs=1e-11)
        matrices.append(matrix)
    assert matrices[0].precision_kind != matrices[1].precision_kind
