# DES-SN5YR population catalog

## Scope and source

The population input layer targets the official
`DES-SN5YR_HD+MetaData.csv` product from the DES Supernova 5YR Data Release.
The file belongs to the legacy Vincenzi et al. (2024) DES-SN5YR cosmology
release and is available through the
[DES-SN5YR repository](https://github.com/des-science/DES-SN5YR) and
[Zenodo record 12720778](https://doi.org/10.5281/zenodo.12720778).

This supported file is a legacy DES-SN5YR product, not the latest DES
supernova release. The current DES repository supersedes the original result
with DES-Dovekie products. DES-Dovekie is supported by a separate release
adapter and is not interpreted by `load_des_sn5yr_catalog()`. See
[DES-Dovekie precision-matrix analysis](dovekie_analysis.md) for its
catalog-order and STAT+SYS precision handling.

The CSV is not redistributed by this project. Users must obtain it from the
official release and pass its local path or a file-like object to
`load_des_sn5yr_catalog()`. Loading and tests never initiate a network request.

When using the release, follow its citation guidance. At minimum, consult and
cite the DES-SN5YR data release paper (Sánchez et al. 2024), the DES 5-year
cosmology analysis (DES Collaboration 2024), and the cosmology/systematics
analysis associated with the released distances (Vincenzi et al. 2024), as
appropriate for the data product and analysis.

## Photometry-level versus catalog-level data

Photometry-level data contain repeated time-series measurements in multiple
bands for each transient. They support light-curve fitting and diagnostics.

`DES-SN5YR_HD+MetaData.csv` is catalog-level data: one row represents one
object in the released Hubble-diagram sample, with fitted SALT3 parameters,
redshifts, standardized distance information, host properties, classification
information, and fit metadata where supplied. The catalog loader does not
reconstruct photometry or repeat the DES light-curve and distance analysis.

## Normalized schema

Every normalized field is copied directly from one source column. No normalized
field is derived in this milestone.

| Normalized field | Release source | Availability |
|---|---|---|
| `object_id` | `CID` | Required, direct |
| `z_hel` | `zHEL` | Required, direct |
| `z_cmb` | `zCMB` | Required, direct |
| `z_hd` | `zHD` | Required, direct |
| `distance_modulus` | `MU` | Required, direct |
| `distance_modulus_error` | `MUERR_FINAL` | Required, direct |
| `host_logmass` | `HOST_LOGMASS` | Required, direct; may be missing |
| `x0` | `x0` | Required, direct |
| `x1` | `x1` | Required, direct |
| `color` | `c` | Required, direct |
| `peak_mjd` | `PKMJD` | Required, direct |
| `chi2_or_fit_probability` | `FITPROB`, otherwise `FITCHI2` | Optional, direct |
| `bias_correction` | `biasCor_mu` | Optional, direct |
| `survey` | `IDSURVEY` | Optional, direct |
| `type` | `TYPE` | Optional, direct |
| `classification` | first available documented classifier probability | Optional, direct |
| `beams_core_collapse_probability` | `PROBCC_BEAMS` | Optional, direct |

Optional fields absent from the source are absent from the normalized catalog;
they are not created as all-missing placeholders. `column_provenance` records
the exact source chosen for every normalized field, and `source_columns`
preserves the full input header.

`distance_modulus_error` maps directly from `MUERR_FINAL`, the released final
DES distance uncertainty. The release may assign strongly enlarged
uncertainties to objects downweighted by BEAMS core-collapse contamination
probabilities, so the legacy catalog can have a strongly bimodal uncertainty
distribution. Large positive `MUERR_FINAL` values are not automatically
sentinels or parsing errors. Their inverse-variance weights are nearly zero in
the diagonal Hubble analysis, but the rows and original uncertainties remain
present. The source metadata does not justify attributing every large
uncertainty to one specific classifier.

When present, `PROBCC_BEAMS` is preserved separately as
`beams_core_collapse_probability`; it is not merged into the generic
classification field and does not alter current Hubble statistics. A future
diagnostic may use it to identify BEAMS-downweighted objects.

`normalize_des_sn5yr_distances()` converts the legacy catalog to the common
`NormalizedDistanceCatalog` protocol. It maps `zHD` explicitly, preserves
source-row correspondence, and does not substitute heliocentric or CMB
redshift.

## Validation and data quality

The loader:

- validates required columns;
- rejects non-numeric content in normalized numeric fields;
- rejects missing or duplicate object IDs;
- converts documented numeric sentinels `-999` and `-9999` to `NaN`;
- retains valid negative values, including negative color, stretch, or bias
  correction values;
- preserves every input row;
- reports source, retained, and removed row counts;
- reports missingness and sentinel replacements per normalized field.

No scientific quality cut is applied automatically. The separate
`apply_des_sn5yr_quality_cuts()` helper applies only caller-requested cuts and
returns a new immutable catalog with explicit row accounting. A fit-probability
cut is accepted only when the normalized fit-quality field came from
`FITPROB`, not `FITCHI2`.

## Known limitations

- The loader does not include or combine the released statistical or systematic
  covariance matrices.
- It does not reproduce DES bias corrections, standardization, classification,
  or cosmology fitting.
- It does not infer missing host properties.
- It does not reinterpret `IDSURVEY`, `TYPE`, or classifier probabilities.
- Sentinel handling is intentionally limited to documented release-style
  values; arbitrary negative measurements are retained.
- Release provenance identifies the source product but does not verify a local
  file checksum.
