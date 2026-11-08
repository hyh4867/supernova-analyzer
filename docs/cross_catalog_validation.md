# Cross-catalog validation boundary

This milestone independently validates software quality and reproducibility for
the existing DES-Dovekie versus Pantheon+ descriptive residual comparison. It
does not introduce a new scientific analysis or reinterpret the measured
differences.

## Independently validated behavior

Offline synthetic regressions cover full-catalog row accounting, closed
common-redshift overlap, exclusive bin membership (including the final upper
edge), configurable bin count and minimum population, and the sign convention
`DES-Dovekie minus Pantheon+`. They also verify suppression of underpopulated
bin differences, the scaled MAD convention, immutable JSON-safe summaries, and
preservation of input row order.

Both catalog likelihood paths are exercised separately with synthetic complete
uncertainty matrices. Tests distinguish the ordinary arithmetic residual mean
from the generalized matrix-weighted mean. They also verify that changing the
reference H0 leaves the final DES-Dovekie residuals, final Pantheon+ residuals,
and their binned descriptive differences invariant to floating-point precision.

## Provenance and catalog-order fingerprints

The read-only validation script hashes all four official inputs before loading
them and hashes them again after the analysis. Any change is an error. It
reports source and matrix SHA-256 values, row counts, release-order fingerprints,
the Pantheon+ full and Hubble-flow subset fingerprints, overlap accounting,
binning configuration, summary statistics, and H0-invariance diagnostics.

A catalog-order fingerprint is a stable SHA-256 digest of ordered row identity,
source-row correspondence, release metadata, and row count. It is an alignment
guard, not a digest of every scientific value. Source SHA-256 values therefore
remain necessary to identify the exact official files.

If the four documented environment variables are absent, real-data validation
skips successfully. An optional manifest contains only JSON-safe provenance and
summary values; it never contains a catalog, covariance matrix, or precision
matrix.

## Scientific interpretation boundary

DES-Dovekie and Pantheon+ cannot be concatenated into one likelihood here.
Their released uncertainty products describe different ordered data vectors,
and no validated cross-catalog covariance model is available. Concatenation
would silently assert unproven independence and common nuisance treatment.

Each catalog must retain its own additive nuisance intercept because its
distance-modulus zero point and released likelihood are catalog-specific. The
matrix-weighted normal equation makes the generalized residual mean nearly zero
after fitting; it does not require the ordinary, unweighted arithmetic residual
mean to vanish when weights and correlations differ by row.

Binned median DES-Dovekie-minus-Pantheon+ residual differences are descriptive
summaries of two independently fitted samples. Their percentile ranges are
distributions, not covariance-aware uncertainties or significance intervals.
They do not establish a calibration offset, survey tension, or cosmological
tension.

H0 invariance proves that an arbitrary reference-H0 shift is absorbed by each
catalog's fitted additive intercept and does not leak into final residuals. It
does not measure H0, validate a cosmological model, establish agreement between
surveys, or test systematic-error completeness.

Outside project scope are cross-catalog covariance construction, object-level
cross-matching, a shared-intercept or joint likelihood, H0 inference,
Omega_m or w fitting, and KS/tension/significance testing. Real-data execution
also depends on separately supplied official files and cannot by itself verify
their external publication or distribution history.
