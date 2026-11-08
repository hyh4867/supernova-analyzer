# Pantheon+SH0ES covariance analysis

## Release scope and row identity

The adapter targets the official
[Pantheon+SH0ES distance release](https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR).
Official files are user-supplied and are not bundled or downloaded by the
project.

The released data vector has 1,701 rows but 1,543 unique `CID` values. Some
physical supernovae have multiple survey or light-curve entries. The pair
`(CID, IDSURVEY)` is unique in the release and, together with exact source-row
index, defines project row identity. The adapter never deduplicates, averages,
or sorts these entries.

Full-release covariance alignment follows the exact 1,701-row released ordering. The
order-sensitive fingerprint includes display identity `CID::IDSURVEY`, source
row indices, catalog/release provenance, redshift definition, and row count.
CID alone is insufficient.

## Distance and calibrator columns

The normalized distance mapping is:

- redshift: `zHD`;
- heliocentric redshift: `zHEL`;
- distance modulus: `MU_SH0ES`;
- display uncertainty: `MU_SH0ES_ERR_DIAG`.

Reference distance retains the shared frame correction:

```text
dL_corrected = dL(zHD) * (1 + zHEL) / (1 + zHD)
mu_reference = 5 log10(dL_corrected / Mpc) + 25
```

All `IS_CALIBRATOR` rows are retained and counted in the full catalog.
`MU_SH0ES` uses a fiducial
supernova magnitude determined from SH0ES Cepheid-host distances, and Cepheid
host uncertainty is included in the released covariance according to the
official README. This milestone does not repeat the Cepheid calibration or
infer H0.

The full-covariance release is statistical plus systematics. `MU_SH0ES` and
DES-Dovekie `MU` therefore have different absolute-magnitude conventions. A
roughly 0.1 mag difference between their default fitted intercepts is a
zero-point convention, not an H0 measurement. With reference H0=73.04 and
Ωm=0.334, an independent reviewer reports Pantheon+ δ=−0.006 ± 0.004 mag.

## Non-calibrator subset used by this project

The full release contains 1,701 rows, including 77 Cepheid calibrators. The
current residual analysis selects rows with `IS_CALIBRATOR == 0`, preserving
their exact released order. For the official files this produces 1,624
non-calibrator rows with `zHD=0.00299–2.26137`; the covariance is reduced using
the same retained row positions as an exact principal submatrix. Repeated CIDs
are not deduplicated. This non-calibrator sample includes 44 rows with
`zHD <= 0.01`. Standard Pantheon+ cosmology uses `zHD > 0.01` and 1,580 rows;
including the 44 low-redshift rows adds Δχ²=70.3 and shifts the intercept from
−0.10900 to −0.11041 mag in the reviewer's comparison. The cross-catalog
interval starts at z=0.02509 and excludes those 44 rows.

An independent review rejected the previous all-1,701 redshift-residual
analysis because it mixed Cepheid-anchored calibrator rows and their covariance
with cosmological-redshift residuals. That result is not scientific output of
the project. A true joint SH0ES likelihood using `CEPH_DIST` is a possible
future extension, not part of this milestone.

## Complete covariance products

The official README describes `Pantheon+SH0ES_STATONLY.cov` as the statistical
covariance and `Pantheon+SH0ES_STAT+SYS.cov` as statistical plus all systematic
covariance. Each file starts with N and contains N*N sequential matrix values.
The adapter reconstructs them in sequential row-major order, validates
symmetry and positive diagonal, and requires Cholesky factorization without
jitter.

The released decimal text is symmetric to finite output precision rather than
bit-for-bit. Pantheon+ explicitly opts into `rtol=1e-7` and `atol=1e-12`
while retaining the original matrix entries. This product-specific tolerance
does not weaken the tighter legacy DES default. No matrix is averaged or
silently symmetrized, and the maximum observed transpose difference is
reported in every covariance summary.

STATONLY includes off-diagonal intrinsic-scatter terms when different surveys
observe the same physical supernova. STAT+SYS additionally includes covariance
between supernovae and Cepheid hosts from systematic uncertainty. They are
separate complete released products: they are not combined, and no diagonal
is added to either.

The nuisance intercept and chi-square use Cholesky solves and never form an
explicit inverse. A lower chi-square for one covariance product does not by
itself mean a better physical model.

## Diagonal-error caution

The official
[release README](https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/README)
labels `m_b_corr_err_DIAG` and `MU_SH0ES_ERR_DIAG` as plotting/visual values and
explicitly warns: “DO NOT FIT COSMOLOGICAL PARAMETERS WITH THESE UNCERTAINTIES.
YOU MUST USE THE FULL COVARIANCE.” The app uses `MU_SH0ES_ERR_DIAG` only for
display. A diagonal approximation to a full STAT+SYS fit instead uses the
per-row marginal variance `diag(C)`.

The variance column approximately satisfies
`MU_SH0ES_ERR_DIAG**2 = diag(STAT+SYS) + diag(STATONLY)` (median variance ratio
0.992). Therefore its square is not the diagonal of the STAT+SYS matrix alone.
The app does not add either error-column variance to the released matrices.

## Duplicate-CID diagnostics

The immutable catalog reports row count, unique CID count, repeated-CID row
count, multiplicity distribution, maximum multiplicity, and machine-readable
counts per repeated CID. These are full-release descriptive diagnostics. All
1,701 rows remain in the immutable catalog; covariance Hubble fitting uses the
1,624 non-calibrator rows only.

## Command line

```bash
python scripts/run_pantheonplus_covariance_analysis.py \
  /path/to/Pantheon+SH0ES.dat \
  /path/to/Pantheon+SH0ES_STAT+SYS.cov \
  --statonly /path/to/Pantheon+SH0ES_STATONLY.cov
```

The script writes ignored figures and JSON summaries, including full-release
duplicate-CID and diagonal-error diagnostics plus explicit subset provenance.
It fits only an additive nuisance intercept on the SN-only Hubble-flow subset.
It does not infer H0, `Omega_m`, dark-energy parameters, host-mass effects, or
cross-catalog offsets.
