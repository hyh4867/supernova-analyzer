# Legacy DES-SN5YR covariance analysis

## Scope and provenance

This module supports the covariance format released with the legacy
DES-SN5YR Hubble-diagram product. It does not support current DES-Dovekie
products, infer cosmological parameters, or reproduce an official survey
likelihood beyond the explicitly selected covariance treatment.

DES-Dovekie uses separate packed complete precision products rather than this
legacy flattened covariance-component convention. See
[DES-Dovekie precision-matrix analysis](dovekie_analysis.md); the two formats
must not be interchanged.

Pantheon+SH0ES returns to a leading-N plus sequential N*N full-covariance text
format, but its STATONLY and STAT+SYS files are separate complete products and
must not use the legacy DES diagonal-composition rule. See
[Pantheon+SH0ES covariance analysis](pantheonplus_analysis.md).

The shared covariance default retains the tighter legacy DES tolerance
(`rtol=1e-10`, `atol=1e-12`). Pantheon+ explicitly supplies its product-level
`rtol=1e-7`, `atol=1e-12` tolerance for the released decimal text. Neither
path rewrites, averages, or silently symmetrizes matrix entries.

Pantheon+ full-release covariances include calibrator rows. This project's
Hubble residual analysis uses exact principal submatrices for the non-calibrator
Hubble-flow rows; the parent hash, original dimension, retained source-row
indices, and subset fingerprint are preserved as covariance provenance.

Official files are not bundled, copied, or downloaded. Obtain the catalog and
covariance components from the
[official DES-SN5YR release](https://github.com/des-science/DES-SN5YR) and
follow its citation and tutorial guidance.

## Flattened file format and ordering

A covariance text file contains one leading integer dimension `N`, followed by
exactly `N*N` row-major values. Plain text and gzip-compressed text are
supported. The loader validates element count, finite values, shape, and
symmetry.

The files contain no row labels. A matching dimension therefore cannot prove
catalog identity or row order. The analysis requires a SHA-256 fingerprint of
the ordered object IDs, source-row indices, catalog identity, release,
redshift definition, and row count. Matching release provenance remains a
scientific requirement even when the fingerprint supplied during loading
matches.

## Component and total covariance

The filename does not define covariance semantics. Callers explicitly label a
matrix as a component or verified released total. Under the official legacy
DES tutorial convention, the released `STAT+SYS` matrix is treated as a
covariance component and the released final distance variance is added once:

```text
C_total = C_STAT+SYS + diag(MUERR_FINAL**2)
```

Adding that diagonal twice would double count statistical uncertainty.
Conversely, a caller-declared, independently verified released-total matrix is
used without adding it. Composition metadata records which path was selected.
Extreme BEAMS-renormalized `MUERR_FINAL` values are retained without display
thresholds or quality cuts.

The names `STATONLY` and `STAT+SYS` are not reinterpreted from numerical
heuristics. Diagnostics may compare their diagonals, off-diagonal RMS, and the
minimum eigenvalue of `STAT+SYS - STATONLY`, but production semantics follow
the documented tutorial convention.

## Reference distance

The legacy adapter uses `zHD` for the FlatLambdaCDM luminosity distance and
applies the released frame correction with `zHEL`:

```text
dL_corrected = dL(zHD) * (1 + zHEL) / (1 + zHD)
mu_reference = 5 log10(dL_corrected / Mpc) + 25
```

Generic catalogs without a heliocentric redshift retain the redshift-only
Astropy distance-modulus behavior.

## Cholesky likelihood calculations

The covariance-aware nuisance intercept uses `scipy.linalg.cho_factor` and
`cho_solve`. It never forms an explicit inverse and never silently adds jitter.
A non-positive-definite covariance raises a dedicated error.

For raw residual vector `r`, unit vector `1`, and covariance `C`:

```text
delta = (1.T C^-1 r) / (1.T C^-1 1)
sigma_delta = sqrt(1 / (1.T C^-1 1))
```

The final residual is `r - delta`. Its generalized weighted mean is
approximately zero by construction, not independent evidence of fit quality.
The full-covariance chi-square and residual structure are the diagnostics.

Diagonal and full-covariance results answer the same residual question under
different uncertainty models. Neither is automatically described as better.
Ordinary plot error bars continue to show released `MUERR_FINAL`; plot
thresholding is display-only, while fitting uses the complete covariance.

This milestone does not measure H0, infer `Omega_m` or dark-energy parameters,
run MCMC, include host-mass steps, or implement a DES-Dovekie adapter.

## Command line

```bash
python scripts/run_hubble_covariance_analysis.py \
  /path/to/DES-SN5YR_HD+MetaData.csv \
  /path/to/STAT+SYS.txt.gz \
  --statonly /path/to/STATONLY.txt.gz
```

The command prints hashes, fingerprints, covariance summaries, the explicit
composition, and diagonal/full-covariance comparisons. Generated figures and
JSON summaries are written under ignored `outputs/`.
