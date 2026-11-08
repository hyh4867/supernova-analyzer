# DES-Dovekie and Pantheon+ cross-catalog residual comparison

This milestone compares the descriptive Hubble-residual structure of two
independently constructed modern SN Ia compilations. DES-Dovekie uses its
released complete STAT+SYS precision matrix. Pantheon+ uses the validated
1,624 non-calibrator rows (`IS_CALIBRATOR == 0`, zHD 0.00299–2.26137) and the
matching principal submatrix of its released complete STAT+SYS covariance.
This sample includes 44 rows with zHD <= 0.01; the standard Pantheon+
cosmology sample uses zHD > 0.01 and 1,580 rows. The 44 low-redshift rows add
Δχ²=70.3 and shift the intercept from −0.10900 to −0.11041 mag in the
reviewer's comparison. The cross-catalog interval begins at z=0.02509 and
excludes all 44 rows.

The catalogs cannot be concatenated. They have different selection,
calibration, row definitions, and uncertainty structures, and no validated
cross-catalog covariance is available. A precision matrix is an inverse
covariance representation used directly in a quadratic form; the Pantheon+
pipeline instead solves with its covariance. The comparison neither converts
these into a joint matrix nor builds a joint likelihood.

Both analyses use the project's common fixed reference-cosmology
implementation. Each catalog independently profiles its own additive nuisance
intercept before residuals are compared. This absorbs the arbitrary reference
normalization. An intercept, and the difference between the two fitted
intercepts, is not an H0 measurement. Changing the arbitrary reference H0 must
leave final residuals and their binned comparison invariant to floating-point
precision.

The releases also use different absolute-magnitude conventions: Pantheon+
`MU_SH0ES` uses the SH0ES Cepheid-calibrated fiducial SN magnitude, while
DES-Dovekie `MU` follows its H0=70 release convention. Their roughly 0.1 mag
intercept difference is a zero-point convention, not an H0 measurement. At
reference H0=73.04 and Ωm=0.334, an independent reviewer reports Pantheon+
δ=−0.006 ± 0.004 mag.

The fitted intercept satisfies the precision- or covariance-weighted normal
equation. It therefore forces the weighted generalized residual mean to zero
to numerical precision, not the ordinary arithmetic residual mean. Correlated
weights need not give every row equal influence. Consequently, a nonzero
ordinary mean—such as approximately 0.0469 mag for the validated DES-Dovekie
result—is not evidence that the nuisance-intercept fit failed.

Full-catalog summaries are retained. Common bins are constructed only over the
closed overlap of the two redshift ranges. Each bin reports its row count,
median redshift, median residual, and 16th and 84th residual percentiles. The
percentile span describes the observed residual distribution; it is not a
covariance-aware uncertainty. The DES-Dovekie minus Pantheon+ median is emitted
only when both bins meet the configured minimum count, and is descriptive only.
The default minimum of 5 is permissive: the z≈0.98 bin has only 6 Pantheon+
rows. Removing the 101 DES rows with PROBIA_BEAMS < 0.5 changes the default 10-bin DES−Pantheon+ median differences by at most 0.0082 mag in the seven bins with ≥51 Pantheon+ rows, and by 0.0192 mag in the z≈0.98 bin, which has only 6 Pantheon+ rows. The releases contain 339 supernova identities in common; the
application does not cross-match or use that overlap in its comparison.

No significance or tension test is performed. In particular, there is no KS
test, cross-catalog chi-square, joint-survey likelihood, cosmological parameter
inference, H0 inference, host-mass inference, or MCMC. Reduced chi-square is
catalog-specific and must not be used to rank the surveys.

CID values are not assumed to identify the same physical supernova across
releases. No alias, coordinate, fuzzy-name, or other object-level cross-match
is attempted. That requires a separate validated milestone.

## Reproduction

Official release products remain local, unmodified, and ignored by Git. Supply
explicit CLI options or these environment variables:

- `DES_DOVEKIE_HD_PATH`
- `DES_DOVEKIE_STAT_SYS_PATH`
- `PANTHEONPLUS_DATA_PATH`
- `PANTHEONPLUS_STAT_SYS_PATH`

Run:

```bash
python scripts/run_cross_catalog_comparison.py --bins 10 --minimum-count 5
```

The ignored `outputs/` directory receives provenance-rich JSON summaries and
four diagnostic figures. JSON records source hashes, catalog-order
fingerprints, Pantheon+ retained source-row indices, analysis modes, and the H0
invariance diagnostic, but never serializes covariance or precision matrices.
