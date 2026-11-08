# Scientific methods and terminology

## Observed discrete peak

The observed discrete peak is the smallest apparent magnitude among the actual
observations. The implementation sorts observations by time and selects the
earliest time when the minimum is tied. It does not interpolate or fit the peak,
so noise and observing cadence can shift it away from the physical maximum.

## Linear 15-day decline baseline

The current baseline subtracts the observed peak magnitude from the magnitude
linearly interpolated 15 days later. The Basic mode synthetic sample assumes a
rest-frame B band and redshift zero, so observer-frame and rest-frame intervals
coincide for that specific sample. This calculation is a simple diagnostic, not
a noise-aware fitted estimator. Extrapolation beyond the observations is
rejected.

The public names `calculate_dm15_linear` and `Dm15Result` remain available for
backward compatibility, but their result should be described as the **Linear
15-day decline baseline** unless the scientific conditions for Delta m15(B) have
been established.

## True Delta m15(B)

Delta m15(B) is the difference between the rest-frame B-band magnitude at 15
rest-frame days after B-band maximum and the magnitude at B-band maximum. This
definition requires a well-defined rest frame and B-band maximum; an arbitrary
single-band or observer-frame 15-day difference should not be labeled Delta
m15(B).

## SALT3 scientific fitting

Scientific mode loads sncosmo-compatible multi-band flux photometry and can fit
the pinned `salt3` source version `2.0`. Redshift is supplied externally and
held fixed. The varied parameters are `t0`, `x0`, `x1`, and `c`, in that order:

- `t0` is the reference time of B-band maximum in the SALT3 model.
- `x0` is the model amplitude.
- `x1` describes light-curve shape.
- `c` describes color.

The fit reports chi-square, the number of degrees of freedom (`ndof`), and
reduced chi-square (`chi-square / ndof`). Model covariance can be included in
the objective. The first real fit may download SALT3 model files and required
SDSS bandpasses into the local Astropy/sncosmo cache. Cache data are not copied
into or committed with this repository.

The bundled sncosmo example fit uses `redshift=0.5` only as a demonstration
assumption. It is not an externally measured spectroscopic value. This
single-object SALT3 example does not perform a redshift-distance comparison or
fit cosmological parameters.

## SALT3 residual diagnostics

For each observation used in the fit, the raw residual is observed flux minus
model flux. The observation-only normalized residual divides that value by the
reported flux uncertainty. If SALT3 model covariance is requested, the
total-diagonal normalized residual divides by the square root of the sum of the
observational variance and the model covariance diagonal.

The total-diagonal pull is only a visualization approximation. The SALT3 fit can
use off-diagonal model covariance, and the full-covariance chi-square must not be
reconstructed as a sum of squared diagonal pulls. A residual plot and reduced
chi-square answer complementary questions: reduced chi-square summarizes global
weighted agreement, while residual structure can reveal band-, calibration-, or
phase-dependent mismatch hidden by a single number.

Diagnostic plots use rest-frame phase by default:

```text
phase = (observer-frame time - t0) / (1 + redshift)
```

Smooth curves use each band's registered bandpass and calibration. If a band's
zeropoint changes, observed flux and uncertainty are rescaled to one reference
zeropoint before plotting. Mixed zeropoint systems within a band are rejected
because an arbitrary conversion would not be scientifically safe.

## SALT3 amplitude magnitude

The literature-style Tripp demo uses an x0-only amplitude magnitude:

```text
m_B = -2.5 log10(x0) + C
```

The normalization `C` is derived rather than hard-coded. A model with the same
SALT3 source name/version is evaluated at `z=0`, `t0=0`, `x0=1`, `x1=0`, and
`c=0` through the reference band and magnitude system. The default is Bessell B
AB. This is a project-derived model normalization and is not claimed to match a
specific published survey normalization without explicit validation.

The amplitude magnitude central value is independent of fitted x1 and c. Its
uncertainty is analytic: only `dm_B/dx0` is non-zero, and the x0 variance is
located using the explicit `varied_parameter_names` covariance ordering.

## Full synthetic rest-frame B magnitude

The full synthetic diagnostic uses fitted x0, x1, and c at rest-frame phase
zero. It is useful for spectral-model diagnostics but is not a literature-style
Tripp input. Its t0 derivative is exactly zero because each model is evaluated
at its corresponding t0; only x0, x1, and c are finite-differenced.

Observer-frame B at the fitted redshift is a third, different quantity because
it samples a redshifted rest-frame wavelength range.

## Tripp-like standardization

The educational single-object relation is

```text
mu = m_B - M + alpha*x1 - beta*c
```

The caller supplies externally calibrated `alpha`, `beta`, and `M`; the project
does not fit them or provide scientific defaults. Statistical uncertainty is
propagated with the analytic Jacobian `[0, dm_B/dx0, alpha, -beta]` and the full
SALT3 fitted covariance.

Central-value shape/color double counting is avoided by using x0-only `m_B`;
the full synthetic magnitude is never accepted by the Tripp API. Uncertainty
double counting is avoided by one direct covariance propagation rather than an
independent sum of `m_B_error`, `x1_error`, and `c_error`.

The propagated uncertainty excludes nuisance-parameter calibration uncertainty,
intrinsic scatter, photometric calibration systematics, peculiar velocity,
gravitational lensing, selection bias, and host-galaxy corrections. It is not a
survey-level distance or cosmology result. Demonstration `alpha`, `beta`, and
`M` values are illustrative and their exact calibration provenance is not
asserted.

This Tripp-like single-object calculation does not itself produce a
covariance-aware survey comparison. The Streamlit app has three tabs: Basic
Analysis, Scientific SALT3 Fit, and Cross-Catalog Explorer. The separate
Hubble-analysis workflows compare residuals against a fixed reference
cosmology; the DES-Dovekie/Pantheon+ comparison is descriptive only and does
not construct a joint likelihood, infer H0, or calculate tension significance.
