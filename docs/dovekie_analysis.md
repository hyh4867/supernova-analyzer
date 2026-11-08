# DES-Dovekie precision-matrix analysis

## Current release scope

DES-Dovekie is the current supernova distance product in the official
[DES-SN5YR repository](https://github.com/des-science/DES-SN5YR). It
supersedes the legacy DES-SN5YR result supported by the separate legacy
adapter. Official files are user-supplied and are never bundled or downloaded
by Supernova Analyzer.

This adapter provides residual-analysis infrastructure. It does not infer H0,
matter density, dark-energy parameters, or host-mass effects.

The released distance modulus `MU` follows the DES-Dovekie absolute-magnitude
convention at H0=70. This differs from Pantheon+ `MU_SH0ES`, whose fiducial SN
magnitude is Cepheid-calibrated by SH0ES. Their roughly 0.1 mag nuisance-
intercept difference is therefore a zero-point convention, not an H0
measurement. With reference H0=73.04 and Ωm=0.334, an independent reviewer
reports Pantheon+ δ=−0.006 ± 0.004 mag.

## Release table formats and ordering

`DES-Dovekie_HD.csv` is not an ordinary comma-separated file. It contains
leading `#` comments followed by whitespace-separated `VARNAMES:` and `SN:`
records. Its 1,820 rows define the Hubble data vector and the exact ordering of
the released precision matrices.

`DES-Dovekie_Metadata.csv` uses the SNANA `VARNAMES:`/`SN:` text format. It
contains the same CID set but a different row order. Metadata is joined
strictly by CID and reordered to the HD order. Both the HD source-row index
and metadata original-row index are retained. Sorting by CID or using metadata
order would invalidate precision alignment.

Selected host, SALT, classification, BEAMS, survey, and sample fields remain
available in the joined metadata for future diagnostics. They do not affect
current Hubble statistics and no host cut is applied.

## Packed complete precision matrices

`STATONLY.npz` and `STAT+SYS.npz` each contain:

- `nsn`: one positive dimension `N`;
- `cov`: `N*(N+1)/2` packed upper-triangular precision values;
- `allow_pickle`: release metadata that is not treated as permission.

Despite the key name `cov`, the packed values are inverse covariance
matrices. They are unpacked with `numpy.triu_indices(N)` and reflected to the
lower triangle. The loader always uses `allow_pickle=False`, validates
symmetry, positive diagonal, and Cholesky factorization, and makes the dense
matrix read-only.

No explicit inverse is formed. The dimension alone does not establish row
alignment: analysis also requires the exact HD-order catalog fingerprint and
matching release provenance.

`STATONLY` and `STAT+SYS` are separate, complete released precision products.
They are not combined, and `MUERR**2` is not added to either one. A lower
chi-square under one uncertainty product is not automatically a better
physical fit.
The larger STAT+SYS intercept uncertainty reflects reduced effective
information from correlated systematics and does not mean the physical model
is worse.

For a diagonal approximation to the full STAT+SYS fit, the per-row marginal
variance is `diag(P**-1)`, which equals `MUERR**2 + MUERR_SYS**2` in the
release. `MUERR` alone omits the systematic diagonal. The production fit uses P
directly and never substitutes a diagonal approximation.

## Precision-weighted nuisance intercept

For raw residual `r`, unit vector `1`, and precision matrix `P`:

```text
delta = (1.T P r) / (1.T P 1)
sigma_delta = sqrt(1 / (1.T P 1))
residual = r - delta * 1
chi2 = residual.T P residual
```

The generalized residual mean is approximately zero by construction. The
independent profiled quadratic cross-check is:

```text
A = r.T P r
B = 1.T P r
C = 1.T P 1
profiled_chi2 = A - B**2 / C
```

It equals the residual quadratic form within numerical tolerance. The project
does not add the official likelihood normalization or log-determinant term to
this residual chi-square.

Reference distance uses `zHD` with the frame correction
`(1 + zHEL) / (1 + zHD)`. The reference H0 remains only a provisional
normalization absorbed by the nuisance intercept.

## Command line

```bash
python scripts/run_dovekie_precision_analysis.py \
  /path/to/DES-Dovekie_HD.csv \
  /path/to/STAT+SYS.npz \
  --statonly /path/to/STATONLY.npz \
  --metadata /path/to/DES-Dovekie_Metadata.csv
```

The script prints hashes, HD/metadata alignment, precision summaries,
STATONLY/STAT+SYS comparisons, and reference-H0 invariance. Figures and
JSON summaries are written only under ignored `outputs/`.
