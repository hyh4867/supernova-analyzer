"""Tests for DES-Dovekie HD and SNANA metadata adapters."""

import os
from io import StringIO
from pathlib import Path

import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from supernova_analyzer import (
    DovekieCatalogError,
    DovekieMetadataAlignmentError,
    join_dovekie_metadata_to_hd_order,
    load_des_dovekie_hd,
    load_des_dovekie_metadata,
    normalize_des_dovekie_distances,
    calculate_reference_distance_modulus,
)


def hd_text(rows: tuple[str, ...] | None = None) -> StringIO:
    """Return a small release-shaped HD table."""
    selected = rows or (
        "SN: B 2 0.20 0.19 40.0 0.20 0.03 0.04 0.90",
        "SN: A 1 0.10 0.09 38.0 0.10 0.02 0.03 0.80",
    )
    return StringIO(
        "\n".join(
            (
                "# release comment",
                "VARNAMES: CID IDSURVEY zHD zHEL MU MUERR "
                "MUERR_VPEC MUERR_SYS PROBIA_BEAMS",
                *selected,
            )
        )
    )


def metadata_text(rows: tuple[str, ...] | None = None) -> StringIO:
    """Return small metadata in a deliberately different order."""
    selected = rows or (
        "SN: A 10.1 0.2 -0.03 0.8 0.1 1",
        "SN: B 9.8 -0.1 0.05 0.9 0.2 2",
    )
    return StringIO(
        "\n".join(
            (
                "VARNAMES: CID HOST_LOGMASS x1 c PROB_SNNV19 "
                "PROBCC_BEAMS IDSAMPLE",
                *selected,
            )
        )
    )


def test_hd_parser_preserves_whitespace_release_order_and_provenance() -> None:
    catalog = load_des_dovekie_hd(hd_text())
    frame = catalog.to_pandas()

    assert catalog.row_count == 2
    assert tuple(frame["CID"]) == ("B", "A")
    assert tuple(frame["zHD"]) == pytest.approx((0.2, 0.1))
    assert catalog.source_row_indices == (0, 1)
    assert ("MU", "MU") in catalog.column_provenance
    assert catalog.catalog_name == "DES-Dovekie"
    assert "legacy" not in catalog.catalog_release.lower()


@pytest.mark.parametrize(
    ("rows", "match"),
    (
        (
            (
                "SN: A 1 0.1 0.09 38 0.1 0.02 0.03 0.8",
                "SN: A 2 0.2 0.19 40 0.2 0.02 0.03 0.9",
            ),
            "duplicate CID",
        ),
        (
            ("SN: A 1 0.0 0.09 38 0.1 0.02 0.03 0.8",),
            "zHD must be greater",
        ),
        (
            ("SN: A 1 0.1 0.09 38 -0.1 0.02 0.03 0.8",),
            "MUERR must be greater",
        ),
    ),
)
def test_hd_parser_rejects_invalid_rows(
    rows: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(DovekieCatalogError, match=match):
        load_des_dovekie_hd(hd_text(rows))


def test_metadata_parser_preserves_original_order_and_selected_fields() -> None:
    metadata = load_des_dovekie_metadata(metadata_text())

    assert metadata.row_count == 2
    assert tuple(metadata.to_pandas()["CID"]) == ("A", "B")
    assert metadata.original_row_indices == (0, 1)
    assert "HOST_LOGMASS" in metadata.columns
    assert "x1" in metadata.columns
    assert "PROBCC_BEAMS" in metadata.columns


def test_metadata_requires_one_header_and_matching_row_lengths() -> None:
    with pytest.raises(DovekieCatalogError, match="before VARNAMES"):
        load_des_dovekie_metadata(StringIO("SN: A 1"))
    with pytest.raises(DovekieCatalogError, match="expected 2"):
        load_des_dovekie_metadata(StringIO("VARNAMES: CID x1\nSN: A"))
    with pytest.raises(DovekieCatalogError, match="duplicate CID"):
        load_des_dovekie_metadata(
            StringIO("VARNAMES: CID x1\nSN: A 1\nSN: A 2")
        )


def test_metadata_join_uses_cid_and_returns_exact_hd_order() -> None:
    hd = load_des_dovekie_hd(hd_text())
    metadata = load_des_dovekie_metadata(metadata_text())

    joined = join_dovekie_metadata_to_hd_order(hd, metadata)

    assert tuple(joined.to_pandas()["CID"]) == ("B", "A")
    assert joined.original_row_indices == (1, 0)
    assert joined.hd_source_row_indices == (0, 1)
    assert joined.order_mismatch_count == 2
    assert joined.aligned_to_hd_order
    assert tuple(hd.to_pandas()["CID"]) == ("B", "A")


@pytest.mark.parametrize(
    "rows",
    (
        ("SN: A 10 0 0 0 0 1",),
        (
            "SN: A 10 0 0 0 0 1",
            "SN: B 10 0 0 0 0 1",
            "SN: C 10 0 0 0 0 1",
        ),
    ),
)
def test_metadata_join_rejects_missing_or_extra_cids(
    rows: tuple[str, ...],
) -> None:
    with pytest.raises(DovekieMetadataAlignmentError, match="not bijective"):
        join_dovekie_metadata_to_hd_order(
            load_des_dovekie_hd(hd_text()),
            load_des_dovekie_metadata(metadata_text(rows)),
        )


def test_normalized_adapter_maps_hd_order_and_frame_correction() -> None:
    catalog = normalize_des_dovekie_distances(
        load_des_dovekie_hd(hd_text())
    )

    assert catalog.object_id == ("B", "A")
    assert catalog.redshift == pytest.approx((0.2, 0.1))
    assert catalog.heliocentric_redshift == pytest.approx((0.19, 0.09))
    assert catalog.distance_modulus == pytest.approx((40.0, 38.0))
    assert catalog.distance_modulus_error == pytest.approx((0.2, 0.1))
    assert catalog.source_row_index == (0, 1)
    assert catalog.reference_distance_convention == (
        "zHD cosmology redshift with zHEL frame correction"
    )
    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
    expected_distance = (
        cosmology.luminosity_distance(np.asarray(catalog.redshift)).value
        * (1.0 + np.asarray(catalog.heliocentric_redshift))
        / (1.0 + np.asarray(catalog.redshift))
    )
    expected_mu = 5.0 * np.log10(expected_distance) + 25.0
    assert calculate_reference_distance_modulus(
        catalog.redshift,
        heliocentric_redshift=catalog.heliocentric_redshift,
    ) == pytest.approx(expected_mu)


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("DES_DOVEKIE_HD_PATH")
        and os.getenv("DES_DOVEKIE_METADATA_PATH")
    ),
    reason="Set DES_DOVEKIE_HD_PATH and DES_DOVEKIE_METADATA_PATH.",
)
def test_real_dovekie_hd_metadata_alignment() -> None:
    hd = load_des_dovekie_hd(Path(os.environ["DES_DOVEKIE_HD_PATH"]))
    metadata = load_des_dovekie_metadata(
        Path(os.environ["DES_DOVEKIE_METADATA_PATH"])
    )
    joined = join_dovekie_metadata_to_hd_order(hd, metadata)

    assert hd.row_count == 1820
    assert metadata.row_count == 1820
    assert set(hd.to_pandas()["CID"]) == set(metadata.to_pandas()["CID"])
    assert joined.order_mismatch_count == 1819
    assert tuple(joined.to_pandas()["CID"]) == tuple(hd.to_pandas()["CID"])
