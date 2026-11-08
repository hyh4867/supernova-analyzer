# Streamlit GUI architecture

## Structure

`app.py` is the presentation entrypoint. It configures the page, renders the
Basic Analysis and Scientific SALT3 Fit tabs, manages session state, and maps
documented backend errors to concise user messages.

Scientific calculations remain in the package modules. The application calls
the public loaders, analysis functions, SALT3 fitter, diagnostics, plotters,
and standardization functions. It does not duplicate their equations.

`supernova_analyzer.app_services` contains Streamlit-independent presentation
helpers. These helpers validate numeric fields, parse optional bounds, create
stable input fingerprints, serialize immutable result objects, build text
summaries, and render Matplotlib figures to PNG bytes. Keeping these operations
outside Streamlit makes them deterministic and directly unit-testable.

## Presentation system

The application uses Streamlit's official theme configuration for a deliberate
light presentation. A restrained blue accent, white primary background, subtle
neutral secondary background, dark text, and the supported sans-serif font
provide consistent contrast without custom CSS. Plot PNGs retain stable opaque
backgrounds for reliable on-screen and downloaded rendering.

The page is content-first: one title, one concise subtitle, and a quiet scope
caption precede the native tabs. Scientific results show reduced chi-square,
amplitude-based m_B, and illustrative mu as the headline row. Detailed fit
parameters, residual diagnostics, and standardization quantities follow in
bordered native containers. Basic results similarly emphasize peak magnitude,
the linear decline baseline, and observation count before secondary values.

Compact UI formatting improves scanning. Basic magnitudes and decline values
use three decimal places, while times remain compact. This display formatting
does not modify calculations or serialized result values; JSON and TXT
downloads keep their established precision.

Scientific central values and statistical uncertainties are rendered on
separate lines rather than as one long metric string. Fit parameters use no
more than two columns per row, and residual summaries are grouped into raw and
normalized pairs. This prevents important numbers from being ellipsized at
normal and narrower widths without custom responsive CSS.

## Analysis workflow

The Basic Analysis tab accepts the bundled synthetic sample or an uploaded CSV
with `time`, `magnitude`, and `magnitude_error`. Analysis starts only after the
user selects the analysis button. The dashboard reports the discrete observed
peak and the Linear 15-day decline baseline, then displays the existing backend
plot.

The Scientific SALT3 Fit tab accepts the bundled sncosmo example photometry or
an uploaded CSV with `time`, `band`, `flux`, `fluxerr`, `zp`, and `zpsys`. The
user supplies a fixed redshift, model-covariance choice, minimum S/N, and
optional initial time or explicit bounds. SALT3 fitting starts only after
`Run SALT3 Analysis` is selected.

After fitting, the dashboard displays fitted parameters, fit statistics,
multi-band curves, normalized residuals, and residual summaries. It calculates
both the x0-only amplitude magnitude used by the illustrative Tripp relation
and the full synthetic rest-frame B spectral diagnostic.

For the live page, the multi-band fit and normalized residuals are separate,
wide, shallow figures. Both consume the same validated photometry, fitted
result, and `Salt3Diagnostics` values through shared plotting helpers. The
combined `plot_salt3_fit()` figure remains unchanged as the standalone
reporting and backward-compatibility download.

The amplitude magnitude appears first and is explicitly identified as the
Tripp-like input. The full synthetic rest-frame B magnitude appears second and
is labeled as a spectral diagnostic that is not used in the Tripp calculation.

## Session state and stale results

Each completed scientific analysis is stored in the user's
`st.session_state`. Its fingerprint includes the selected file content or
bundled sample identity, fixed redshift, model-covariance setting, minimum S/N,
optional initial t0, and explicit bounds.

When any of those inputs change, the previous result is hidden and clearly
marked stale until the user runs SALT3 again. This prevents an old fit from
appearing current.

The illustrative `alpha`, `beta`, and `M` values are intentionally excluded
from the fit fingerprint. They do not affect the fitted SALT3 model, so changing
them recalculates only `calculate_tripp_standardization()` and does not repeat
the expensive fit.

## Error handling

The application boundary catches documented validation, fitting, model
availability, diagnostic, and standardization exceptions. Their English
messages are shown without a traceback. Programmer errors are not hidden by a
global catch-all.

A first real SALT3 run can require internet access if the pinned SALT3 source or
required bandpasses are absent from the local Astropy/sncosmo cache. Cache files
are local runtime resources and are never committed.

The first-use cache notice appears near the run button. Loading the application
or its bundled data does not initiate a model download. During a requested fit,
the progress message covers both model-resource preparation and optimization so
a first-time cache population is not mistaken for an unresponsive interface.

## Downloads

Uploads and generated downloads remain in memory. The Basic workflow provides
a JSON summary, TXT summary, sample CSV, and PNG plot. The Scientific workflow
provides:

- SALT3 fit result JSON
- diagnostics JSON
- amplitude magnitude JSON
- full synthetic diagnostic magnitude JSON
- illustrative Tripp standardization JSON
- combined human-readable TXT summary
- fitted multi-band figure PNG
- normalized residual diagnostics PNG
- combined fitted light-curve plus residual PNG
- bundled scientific sample CSV

Combined TXT summaries and PNG plots are presented as the primary downloads.
Individual JSON files are grouped inside an expander so provenance-rich machine
outputs remain available without dominating the page. The same progressive
disclosure pattern is used for the Basic JSON summary.

Scientific plot bands are differentiated by marker shape in addition to color.
Used and excluded observations also differ by marker fill, preserving meaning
without color alone. Image captions summarize the plotted content, band count,
and reduced chi-square. Native Streamlit tabs, controls, containers, and focus
behavior are retained for keyboard accessibility and responsive stacking.

No interactive workflow requires files in `outputs/`, and the interface never
exposes absolute local paths.

## Scientific limits

The dashboard is an educational single-object tool. It does not infer a
supernova class, fit nuisance calibration parameters, calculate a luminosity
distance, build a Hubble diagram, or estimate cosmological parameters. The
Tripp uncertainty is fit-statistical only, and the plotted total-diagonal pull
is not an exact decomposition of the full-covariance fit chi-square.
