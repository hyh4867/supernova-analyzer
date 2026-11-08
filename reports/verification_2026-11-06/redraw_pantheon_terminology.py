from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

report = Path(__file__).resolve().parent
sys.path.insert(0, str(report))
import run_analysis as pipeline  # noqa: E402

(
    release,
    _full_catalog,
    _selection,
    _full_covariance,
    catalog,
    covariance,
    result,
) = pipeline._load_pantheon()
frame = release.to_pandas()
calibrators = frame["IS_CALIBRATOR"].to_numpy(float) == 1.0
cal_z = frame.loc[calibrators, "zHD"].to_numpy(float)
cal_mu = frame.loc[calibrators, "MU_SH0ES"].to_numpy(float)
cal_err = frame.loc[calibrators, "MU_SH0ES_ERR_DIAG"].to_numpy(float)
z = np.asarray(result.redshift, dtype=float)
mu = np.asarray(catalog.distance_modulus, dtype=float)
residual = np.asarray(result.residual, dtype=float)
reference = np.asarray(result.fitted_reference_distance_modulus, dtype=float)
marginal_error = np.sqrt(np.diag(np.asarray(covariance.values, dtype=float)))
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 21,
})

# Pantheon+ Hubble diagram with release calibrators shown but excluded from fit.
fig, ax = pipeline._figure_axes(
    "Pantheon+ non-calibrator sample — 1,624 fitted rows; 77 calibrators excluded",
    "Redshift (zHD)",
    "Distance modulus μ (mag)",
    figsize=(13, 8),
)
ax.errorbar(
    z,
    mu,
    yerr=marginal_error,
    fmt="o",
    ms=3.3,
    alpha=0.45,
    capsize=0,
    color="#1769aa",
    label="Non-calibrator rows used in fit",
)
order = np.argsort(z)
ax.plot(
    z[order],
    reference[order],
    color="#d24a37",
    linewidth=3,
    label="Flat ΛCDM reference + fitted intercept",
)
ax.errorbar(
    cal_z,
    cal_mu,
    yerr=cal_err,
    fmt="x",
    ms=5,
    alpha=0.75,
    color="#777777",
    label="77 Cepheid calibrators — excluded",
)
ax.legend(loc="best")

# Non-calibrator residuals; calibrators are not part of this panel or fit.
fig_resid, ax_resid = pipeline._figure_axes(
    "Pantheon+ non-calibrator residuals versus redshift",
    "Redshift (zHD)",
    "μ observed − μ reference − δ (mag)",
    figsize=(13, 7),
)
ax_resid.axhline(0.0, color="black", linewidth=1.5)
ax_resid.errorbar(
    z,
    residual,
    yerr=marginal_error,
    fmt="o",
    ms=3,
    alpha=0.35,
    capsize=0,
    color="#1769aa",
)

# Correlation heat map with the requested ±0.2 color range.
values = np.asarray(covariance.values, dtype=float)
correlation = values / np.sqrt(np.outer(np.diag(values), np.diag(values)))
fig_corr, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
image = axes[0].imshow(
    correlation,
    cmap="coolwarm",
    vmin=-0.2,
    vmax=0.2,
    interpolation="nearest",
    aspect="auto",
)
axes[0].set_title("Full non-calibrator correlation\n(1,624 × 1,624)")
axes[0].set_xlabel("Retained row position")
axes[0].set_ylabel("Retained row position")
zoom = min(120, len(correlation))
axes[1].imshow(
    correlation[:zoom, :zoom],
    cmap="coolwarm",
    vmin=-0.2,
    vmax=0.2,
    interpolation="nearest",
    aspect="auto",
)
axes[1].set_title(f"First {zoom} non-calibrator rows (release order)")
axes[1].set_xlabel("Retained row position")
axes[1].set_ylabel("Retained row position")
fig_corr.colorbar(image, ax=axes, label="Correlation coefficient", shrink=0.85)
fig_corr.suptitle("Pantheon+ STAT+SYS covariance correlation matrix")

for fig, stem in (
    (fig, "05a_pantheon_hubble"),
    (fig_resid, "05b_pantheon_residuals"),
    (fig_corr, "05d_pantheon_correlation"),
):
    for extension in ("png", "jpg"):
        options = {"pil_kwargs": {"quality": 94}} if extension == "jpg" else {}
        fig.savefig(
            report / "figures" / f"{stem}.{extension}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(fig)
print({
    "non_calibrator_rows": len(z),
    "calibrators_excluded": int(calibrators.sum()),
    "05d_color_limits": [-0.2, 0.2],
    "formats": ["png", "jpg"],
    "dpi": 200,
})
