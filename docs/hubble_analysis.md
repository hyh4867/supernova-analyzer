# Population Hubble-diagram analysis

## Dataset-independent input

The analysis engine consumes `NormalizedDistanceCatalog`, not a release-specific
loader type. The immutable representation contains aligned tuples for object
ID, redshift, distance modulus, distance-modulus uncertainty, and source-row
index, plus catalog, release, redshift, and distance-definition provenance.

Release adapters are responsible for choosing scientifically appropriate source
columns. The legacy DES-SN5YR adapter maps `zHD` to the common redshift field
and never substitutes `zHEL` or `zCMB`. It preserves original row ordering and
source-row indices and applies no automatic quality cut.

For the legacy adapter, `distance_modulus_error` maps directly from
`MUERR_FINAL`, the released final DES distance uncertainty. Some objects
downweighted by BEAMS core-collapse contamination probabilities can have
strongly enlarged uncertainties, producing a strongly bimodal distribution.
Large positive values are retained as valid release values rather than treated
as sentinels. They receive nearly zero inverse-variance weight, but remain in
the statistical arrays and outputs. Not every large uncertainty is attributed
to one classifier without supporting release metadata.

Photometry-level data contain repeated flux measurements used to fit individual
light curves. Population distance catalogs contain one released distance
estimate per selected object. This module analyzes the latter and does not
repeat object-level SALT3 fitting.

## Provisional reference cosmology

`calculate_reference_distance_modulus()` evaluates Astropy
`FlatLambdaCDM` with fixed `omega_m` and a provisional `reference_h0`:

```text
mu_ref(z) = 5 log10(D_L / 10 pc)
```

The default values are `omega_m=0.3` and `reference_h0=70`. Matter density is
fixed, not inferred. The reference H0 sets only an arbitrary normalization.
Supernova distances retain a degeneracy between absolute magnitude and H0, so
the value is not an H0 measurement.

For the legacy DES adapter, the luminosity distance uses `zHD` and is corrected
by `(1 + zHEL) / (1 + zHD)` before conversion to distance modulus. Generic
catalogs without heliocentric redshift preserve the redshift-only behavior.

## Additive nuisance intercept

For diagonal distance-modulus uncertainties, the engine calculates:

```text
raw residual_i = mu_observed_i - mu_reference_i
weight_i = 1 / sigma_i^2
delta = sum(weight_i * raw residual_i) / sum(weight_i)
sigma_delta = 1 / sqrt(sum(weight_i))
residual_i = mu_observed_i - mu_reference_i - delta
```

The additive `delta` absorbs the provisional absolute normalization. It is
described only as a nuisance intercept, not H0 and not an absolute-magnitude
measurement. Input distance moduli are never overwritten.

Changing `reference_h0` shifts both the provisional reference curve and fitted
intercept. The final intercept-corrected residuals are invariant to that change
within numerical precision.

## Diagonal residual statistics

After fitting one intercept:

```text
ndof = N - 1
chi2_diagonal = sum((residual_i / sigma_i)^2)
reduced_chi2_diagonal = chi2_diagonal / ndof
residual_rms = sqrt(mean(residual_i^2))
weighted_residual_rms =
    sqrt(sum(weight_i * residual_i^2) / sum(weight_i))
```

The result records `covariance_mode="diagonal_statistical"`. These statistics
use only per-object diagonal uncertainties. They do not reproduce an official
survey likelihood, which requires the release-specific statistical and
systematic covariance matrices and their documented likelihood treatment.

The weighted residual mean is approximately zero by construction after the
weighted additive intercept is fitted. This follows from the intercept normal
equation and is not independent evidence of fit quality. Reduced chi-square
and residual structure are the relevant diagnostics.

## Plotting

`plot_hubble_diagram()` displays observed distance moduli and the fitted
FlatLambdaCDM reference plus nuisance offset. `plot_hubble_residuals()` displays
intercept-corrected residuals around zero. Both use logarithmic redshift by
default, return open Matplotlib `Figure` objects, support optional output paths,
and never call `plt.show()`.

The plots do not split on host properties and do not label the nuisance offset
as an H0 or absolute-magnitude estimate.

Plot error bars use a display-only threshold. An explicit positive
`display_error_limit` may be supplied; otherwise the deterministic 95th
percentile of the full uncertainty array is used. Every central point remains
visible, and all calculations and JSON values retain the unmodified
uncertainties. Each plot states the threshold and the number of truncated error
bars. This presentation rule does not filter data or change axis limits
silently.

`plot_distance_uncertainty_distribution()` uses the full uncertainty array on
a logarithmic axis and marks the median, 95th percentile, 99th percentile, and
maximum. It is descriptive: an extreme tail is not labeled as measurement
error.

## Legacy DES adapter and release terminology

The supported `DES-SN5YR_HD+MetaData.csv` is labeled
**Legacy DES-SN5YR HD + metadata catalog** and is associated with the Vincenzi
et al. (2024) release. The current DES-SN5YR repository supersedes the original
cosmology result with DES-Dovekie products. Those products have different
release semantics and are handled by the separate DES-Dovekie adapter
documented in [DES-Dovekie precision-matrix analysis](dovekie_analysis.md).

Neither legacy DES-SN5YR nor DES-Dovekie data are bundled or downloaded by this
project.

## Command-line example

```bash
python scripts/run_hubble_analysis.py \
  /path/to/DES-SN5YR_HD+MetaData.csv \
  --omega-m 0.3 \
  --reference-h0 70
```

The script currently uses only the legacy DES adapter. It writes three figures
and two JSON results under `outputs/`, which remains ignored by Git.

## Release-specific support and limitations

The command-line example above uses the legacy DES-SN5YR adapter. Separate
implemented release pipelines support DES-Dovekie and Pantheon+ with their
release-specific precision/covariance products. Union3 is not implemented.
Their DES-Dovekie/Pantheon+ comparison is descriptive only: it keeps separate
nuisance intercepts and does not construct a joint likelihood, infer H0, or
test survey-tension significance.

The analysis engine evaluates residuals under a fixed reference cosmology; it
does not fit H0 or cosmological parameters, run MCMC, combine BAO or CMB data,
calculate host-mass steps, or make a dark-energy claim.

Full-covariance support for the legacy DES adapter is documented separately in
[Legacy DES-SN5YR covariance analysis](covariance_analysis.md). It preserves
the generic diagonal engine and uses an explicit covariance composition.

The current DES-Dovekie adapter uses the same generic normalized distance
protocol but associates it with HD-ordered complete precision matrices. Its
custom formats and direct precision quadratic are documented in
[DES-Dovekie precision-matrix analysis](dovekie_analysis.md).

Pantheon+SH0ES also maps to the generic distance protocol while retaining all
1,701 released rows and repeated survey/light-curve entries for descriptive
diagnostics. The current residual fit uses 1,624 exact-order non-calibrator
rows (`IS_CALIBRATOR == 0`, zHD 0.00299–2.26137) and matching principal
covariance submatrices. Standard Pantheon+ cosmology uses zHD > 0.01 (1,580
rows); 44 rows at zHD <= 0.01 are included in the current descriptive
non-calibrator result. They add Δχ²=70.3 and shift δ from −0.10900 to −0.11041
mag in the reviewer's comparison. The common cross-catalog interval begins at
z=0.02509 and excludes these 44. The rejected all-row analysis mixed 77
Cepheid calibrators with redshift residuals; a joint likelihood using
`CEPH_DIST` remains future work.
Its order-sensitive `CID::IDSURVEY` identity and covariance treatment are documented in
[Pantheon+SH0ES covariance analysis](pantheonplus_analysis.md).
