from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve

from supernova_analyzer.covariance import catalog_order_fingerprint
from supernova_analyzer.dovekie_io import load_des_dovekie_hd, normalize_des_dovekie_distances
from supernova_analyzer.precision import (
    calculate_hubble_residuals_precision,
    load_packed_precision_matrix,
)

PROJECT = Path(__file__).resolve().parents[2]
REPORT = PROJECT / "reports" / "verification_2026-11-06"
HD_PATH = Path(os.environ["DES_DOVEKIE_HD_PATH"])
P_PATH = Path(os.environ["DES_DOVEKIE_STAT_SYS_PATH"])

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

hd = load_des_dovekie_hd(HD_PATH)
frame = hd.to_pandas()
catalog = normalize_des_dovekie_distances(hd)
precision = load_packed_precision_matrix(
    P_PATH,
    precision_kind="dovekie_stat_plus_systematic_complete_precision",
    catalog_order_fingerprint=catalog_order_fingerprint(catalog),
    release_name=catalog.catalog_release,
    composition_description="Official DES-Dovekie STAT+SYS complete precision.",
)
result = calculate_hubble_residuals_precision(catalog, precision)
P = np.asarray(precision.values, dtype=float)
L = np.linalg.cholesky(P)
C = cho_solve((L, True), np.eye(len(P)), check_finite=False)
marginal_sigma = np.sqrt(np.diag(C))
prob_ia = frame["PROBIA_BEAMS"].to_numpy(float)
muerr = frame["MUERR"].to_numpy(float)
high = prob_ia >= 0.5
low = prob_ia < 0.5
z = np.asarray(result.redshift, dtype=float)
mu = np.asarray(catalog.distance_modulus, dtype=float)
resid = np.asarray(result.residual, dtype=float)
reference = np.asarray(result.fitted_reference_distance_modulus, dtype=float)
low_label = (
    f"P(Ia) < 0.5: n={int(low.sum())}; max MUERR={muerr[low].max():.2f} mag; "
    "bars omitted"
)
high_label = f"P(Ia) ≥ 0.5: n={int(high.sum())}; marginal σ error bars"

# 04a preserves the absolute-modulus Hubble diagram and adds its residual panel;
# the requested −1.2 to 3.2 mag axis is applied to the residual ordinate.
fig, (ax_mu, ax_resid) = plt.subplots(
    2, 1, figsize=(13, 11), sharex=True,
    gridspec_kw={"height_ratios": [2.1, 1.0]}, constrained_layout=True,
)
ax_mu.errorbar(
    z[high], mu[high], yerr=marginal_sigma[high], fmt="o", ms=3,
    alpha=0.42, capsize=1.5, color="#1769aa", label=high_label,
)
ax_mu.scatter(z[low], mu[low], s=21, marker="o", color="#777777", alpha=0.72, label=low_label, zorder=3)
order = np.argsort(z)
ax_mu.plot(z[order], reference[order], color="#d24a37", linewidth=2.6, label="Flat ΛCDM reference + fitted intercept")
ax_mu.set_ylabel("Distance modulus μ (mag)")
ax_mu.set_title("DES-Dovekie Hubble diagram and residuals — STAT+SYS precision fit")
mu_pad = 0.8
ax_mu.set_ylim(float(np.min(mu) - mu_pad), float(np.max(mu) + mu_pad))
ax_mu.grid(alpha=0.22)
ax_mu.legend(loc="best", framealpha=0.94)

ax_resid.errorbar(
    z[high], resid[high], yerr=marginal_sigma[high], fmt="o", ms=3,
    alpha=0.36, capsize=1.5, color="#1769aa", label="P(Ia) ≥ 0.5; marginal σ bars",
)
ax_resid.scatter(z[low], resid[low], s=22, marker="o", color="#777777", alpha=0.75, label=low_label, zorder=3)
ax_resid.axhline(0.0, color="black", linewidth=1.2)
ax_resid.set_xlabel("Redshift zHD (dimensionless)")
ax_resid.set_ylabel("Residual (mag)")
ax_resid.set_ylim(-1.2, 3.2)
ax_resid.grid(alpha=0.22)
ax_resid.legend(loc="upper right", framealpha=0.94)
for ext in ("png", "jpg"):
    out = REPORT / "figures" / f"04a_dovekie_hubble.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", **({"pil_kwargs": {"quality": 93}} if ext == "jpg" else {}))
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 7.6), constrained_layout=True)
ax.errorbar(
    z[high], resid[high], yerr=marginal_sigma[high], fmt="o", ms=3.3,
    alpha=0.39, capsize=1.5, color="#1769aa", label=high_label,
)
ax.scatter(z[low], resid[low], s=25, marker="o", color="#777777", alpha=0.76, label=low_label, zorder=3)
ax.axhline(0.0, color="black", linewidth=1.3, label="zero residual")
ax.set_xlabel("Redshift zHD (dimensionless)")
ax.set_ylabel("μ observed − μ reference − δ (mag)")
ax.set_title("DES-Dovekie residuals versus redshift — STAT+SYS fit")
ax.set_ylim(-1.2, 3.2)
ax.grid(alpha=0.22)
ax.legend(loc="upper right", framealpha=0.94)
for ext in ("png", "jpg"):
    out = REPORT / "figures" / f"04b_dovekie_residuals.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", **({"pil_kwargs": {"quality": 93}} if ext == "jpg" else {}))
plt.close(fig)

print({
    "rows": len(z),
    "prob_ia_ge_0_5": int(high.sum()),
    "prob_ia_lt_0_5": int(low.sum()),
    "max_muerr_low_probability_mag": float(muerr[low].max()),
    "max_marginal_sigma_high_probability_mag": float(marginal_sigma[high].max()),
    "mu_axis_04a": [float(np.min(mu) - mu_pad), float(np.max(mu) + mu_pad)],
    "residual_axis": [-1.2, 3.2],
    "png_jpg_dpi": 200,
})
