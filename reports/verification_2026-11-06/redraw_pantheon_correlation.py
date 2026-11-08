from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os

from supernova_analyzer.pantheonplus_analysis import (
    load_pantheonplus_covariance,
    subset_pantheonplus_covariance,
)
from supernova_analyzer.pantheonplus_io import (
    load_pantheonplus_catalog,
    select_pantheonplus_hubble_flow_subset,
)

report = Path(__file__).resolve().parent
catalog = load_pantheonplus_catalog(os.environ["PANTHEONPLUS_DATA_PATH"])
full = load_pantheonplus_covariance(
    os.environ["PANTHEONPLUS_STAT_SYS_PATH"],
    catalog,
    covariance_kind="pantheonplus_stat_plus_systematic_complete_covariance",
    composition_description="Official Pantheon+ STAT+SYS covariance; report-only redraw.",
)
selection = select_pantheonplus_hubble_flow_subset(catalog)
covariance = subset_pantheonplus_covariance(full, selection)
values = np.asarray(covariance.values, dtype=float)
correlation = values / np.sqrt(np.outer(np.diag(values), np.diag(values)))
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 22,
})
fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
image = None
for ax, block, title in (
    (axes[0], correlation, "Full Hubble-flow correlation\n(1,624 × 1,624)"),
    (axes[1], correlation[:120, :120], "First 120 retained rows (release order)"),
):
    image = ax.imshow(
        block,
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.2,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Retained row position")
    ax.set_ylabel("Retained row position")
fig.colorbar(image, ax=axes, label="Correlation coefficient", shrink=0.85)
fig.suptitle("Pantheon+ STAT+SYS covariance correlation matrix (color limits ±0.2)")
for extension in ("png", "jpg"):
    options = {"pil_kwargs": {"quality": 94}} if extension == "jpg" else {}
    fig.savefig(report / "figures" / f"05d_pantheon_correlation.{extension}", dpi=200, bbox_inches="tight", facecolor="white", **options)
plt.close(fig)
print({"dimension": len(values), "color_limits": [-0.2, 0.2], "dpi": 200})
