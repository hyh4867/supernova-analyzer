"""Offline tests for the DES-SN5YR population-catalog input layer."""

from dataclasses import FrozenInstanceError
from io import StringIO

import numpy as np
import pytest

from supernova_analyzer import (
    DesSn5yrCatalog,
    PopulationCatalogError,
    apply_des_sn5yr_quality_cuts,
    load_des_sn5yr_catalog,
)

HEADER = (
    "CID,zHEL,zCMB,zHD,MU,MUERR_FINAL,HOST_LOGMASS,x0,x1,c,PKMJD,"
    "FITPROB,biasCor_mu,IDSURVEY,TYPE,PROB_SNNV19"
)
ROWS = (
    "1001,0.100,0.101,0.102,38.10,0.12,10.2,1.2e-4,0.5,-0.03,"
    "59000.2,0.95,-0.012,10,1,0.99",
    "1002,0.200,0.201,0.202,39.70,0.15,9.7,8.0e-5,-0.4,0.08,"
    "59100.4,0.80,0.021,10,1,0.96",
)


def make_csv(*rows: str, header: str = HEADER) -> StringIO:
    """Create an in-memory release-shaped CSV fixture."""
    selected_rows = rows or ROWS
    return StringIO("\n".join((header, *selected_rows)))


def test_load_valid_des_sn5yr_catalog() -> None:
    catalog = load_des_sn5yr_catalog(make_csv())

    assert isinstance(catalog, DesSn5yrCatalog)
    assert catalog.row_count == 2
    assert catalog.columns[:6] == (
        "object_id",
        "z_hel",
        "z_cmb",
        "z_hd",
        "distance_modulus",
        "distance_modulus_error",
    )
    frame = catalog.to_pandas()
    assert frame["object_id"].tolist() == ["1001", "1002"]
    assert frame["z_hd"].tolist() == pytest.approx([0.102, 0.202])
    assert frame["distance_modulus"].tolist() == pytest.approx([38.10, 39.70])
    assert frame["distance_modulus_error"].tolist() == pytest.approx([0.12, 0.15])
    assert frame["color"].tolist() == pytest.approx([-0.03, 0.08])
    assert frame["bias_correction"].tolist() == pytest.approx([-0.012, 0.021])


def test_missing_required_column_is_rejected() -> None:
    header = HEADER.replace(",zHD", "")
    row = ROWS[0].replace(",0.102", "", 1)

    with pytest.raises(PopulationCatalogError, match=r"z_hd \(zHD\)"):
        load_des_sn5yr_catalog(make_csv(row, header=header))


def test_documented_sentinel_is_converted_without_removing_row() -> None:
    row = ROWS[0].replace(",10.2,", ",-999,")

    catalog = load_des_sn5yr_catalog(make_csv(row, ROWS[1]))
    frame = catalog.to_pandas()

    assert np.isnan(frame.loc[0, "host_logmass"])
    assert catalog.row_count == 2
    assert catalog.quality_summary.source_row_count == 2
    assert catalog.quality_summary.retained_row_count == 2
    assert catalog.quality_summary.removed_row_count == 0
    replacements = dict(
        catalog.quality_summary.sentinel_replacements_by_field
    )
    assert replacements["host_logmass"] == 1


def test_duplicate_object_ids_are_rejected() -> None:
    duplicate = ROWS[1].replace("1002,", "1001,", 1)

    with pytest.raises(PopulationCatalogError, match="duplicate object IDs.*1001"):
        load_des_sn5yr_catalog(make_csv(ROWS[0], duplicate))


def test_numeric_coercion_failure_names_field_and_value() -> None:
    invalid = ROWS[0].replace(",0.102,", ",not-a-redshift,")

    with pytest.raises(
        PopulationCatalogError,
        match="normalized field 'z_hd'.*not-a-redshift",
    ):
        load_des_sn5yr_catalog(make_csv(invalid))


def test_missing_host_mass_is_reported_per_field() -> None:
    missing = ROWS[0].replace(",10.2,", ",-999,")

    catalog = load_des_sn5yr_catalog(make_csv(missing))

    missingness = dict(catalog.quality_summary.missingness_by_field)
    assert missingness["host_logmass"] == 1
    assert all(
        count == 0
        for field, count in missingness.items()
        if field != "host_logmass"
    )


def test_catalog_output_is_immutable_and_dataframe_is_a_copy() -> None:
    catalog = load_des_sn5yr_catalog(make_csv())
    frame = catalog.to_pandas()
    frame.loc[0, "x1"] = 99.0

    assert catalog.to_pandas().loc[0, "x1"] == pytest.approx(0.5)
    with pytest.raises(FrozenInstanceError):
        catalog.columns = ("changed",)  # type: ignore[misc]
    with pytest.raises(TypeError):
        catalog.rows[0][0] = "changed"  # type: ignore[index]


def test_release_and_column_provenance_are_preserved() -> None:
    catalog = load_des_sn5yr_catalog(make_csv())
    metadata = dict(catalog.metadata)
    provenance = {
        normalized: (source, kind)
        for normalized, source, kind in catalog.column_provenance
    }

    assert metadata["release_product"] == "DES-SN5YR_HD+MetaData.csv"
    assert metadata["release_name"] == "Legacy DES-SN5YR HD + metadata catalog"
    assert metadata["release_doi"] == "10.5281/zenodo.12720778"
    assert provenance["object_id"] == ("CID", "direct")
    assert provenance["chi2_or_fit_probability"] == ("FITPROB", "direct")
    assert catalog.source_columns == tuple(HEADER.split(","))
    assert catalog.source_row_indices == (0, 1)


def test_optional_absent_fields_are_not_claimed_in_normalized_schema() -> None:
    required_header = ",".join(HEADER.split(",")[:11])
    required_rows = tuple(",".join(row.split(",")[:11]) for row in ROWS)

    catalog = load_des_sn5yr_catalog(
        make_csv(*required_rows, header=required_header)
    )

    assert "chi2_or_fit_probability" not in catalog.columns
    assert "bias_correction" not in catalog.columns
    assert "survey" not in catalog.columns
    assert "type" not in catalog.columns
    assert "classification" not in catalog.columns
    assert "beams_core_collapse_probability" not in catalog.columns


def test_optional_beams_probability_is_distinct_and_preserves_provenance() -> None:
    header = f"{HEADER},PROBCC_BEAMS"
    rows = tuple(f"{row},{value}" for row, value in zip(ROWS, (0.01, 0.8)))

    catalog = load_des_sn5yr_catalog(make_csv(*rows, header=header))
    frame = catalog.to_pandas()
    provenance = {
        normalized: source
        for normalized, source, _kind in catalog.column_provenance
    }

    assert frame["beams_core_collapse_probability"].tolist() == pytest.approx(
        [0.01, 0.8]
    )
    assert provenance["beams_core_collapse_probability"] == "PROBCC_BEAMS"
    assert provenance["classification"] == "PROB_SNNV19"


def test_loader_preserves_file_like_position() -> None:
    source = make_csv()
    source.seek(4)

    load_des_sn5yr_catalog(source)

    assert source.tell() == 4


def test_quality_cuts_are_explicit_and_report_row_removal() -> None:
    missing = ROWS[0].replace(",10.2,", ",-999,")
    catalog = load_des_sn5yr_catalog(make_csv(missing, ROWS[1]))

    unchanged = apply_des_sn5yr_quality_cuts(catalog)
    filtered = apply_des_sn5yr_quality_cuts(
        catalog,
        require_host_mass=True,
        maximum_distance_modulus_error=0.2,
        minimum_fit_probability=0.7,
    )

    assert unchanged.row_count == 2
    assert filtered.row_count == 1
    assert filtered.quality_summary.source_row_count == 2
    assert filtered.quality_summary.retained_row_count == 1
    assert filtered.quality_summary.removed_row_count == 1
    assert catalog.row_count == 2
