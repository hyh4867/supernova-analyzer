"""Independent raw-file GLS recomputation; intentionally imports no package code."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parents[2]
DATA = PROJECT / "data" / "external"


def read_dovekie(path: Path) -> pd.DataFrame:
    names: list[str] | None = None
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if names is None:
        raise ValueError("Dovekie raw SNANA file has no VARNAMES header")
    frame = pd.DataFrame(rows, columns=names)
    for name in frame.columns:
        if name not in {"CID", "IDSURVEY"}:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame


def reference_mu(z_hd: np.ndarray, z_hel: np.ndarray, h0: float = 70.0) -> np.ndarray:
    # Flat LambdaCDM, Omega_m=0.3. Gauss-Legendre integration avoids package imports.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    redshift = np.asarray(z_hd, dtype=float)
    sample_z = redshift[:, None] * (nodes[None, :] + 1.0) / 2.0
    expansion = np.sqrt(0.3 * (1.0 + sample_z) ** 3 + 0.7)
    integral = redshift / 2.0 * np.sum(weights[None, :] / expansion, axis=1)
    c_km_s = 299792.458
    comoving_mpc = (c_km_s / h0) * integral
    luminosity_mpc = (1.0 + np.asarray(z_hel, dtype=float)) * comoving_mpc
    return 5.0 * np.log10(luminosity_mpc) + 25.0


def summarize_precision(frame: pd.DataFrame, packed_path: Path) -> dict[str, float | int]:
    archive = np.load(packed_path, allow_pickle=False)
    dimension = int(np.asarray(archive["nsn"]).reshape(-1)[0])
    packed = np.asarray(archive["cov"], dtype=float)
    upper = np.triu_indices(dimension)
    precision = np.zeros((dimension, dimension), dtype=float)
    precision[upper] = packed
    precision[(upper[1], upper[0])] = packed
    if len(frame) != dimension:
        raise ValueError(f"Dovekie rows {len(frame)} != precision dimension {dimension}")
    z_hd = frame["zHD"].to_numpy(float)
    z_hel = frame["zHEL"].to_numpy(float)
    mu = frame["MU"].to_numpy(float)
    raw = mu - reference_mu(z_hd, z_hel)
    one = np.ones_like(raw)
    denominator = float(one @ precision @ one)
    intercept = float((one @ precision @ raw) / denominator)
    sigma = float(np.sqrt(1.0 / denominator))
    residual = raw - intercept
    chi2 = float(residual @ precision @ residual)
    return {
        "rows": int(len(frame)),
        "intercept_mag": intercept,
        "intercept_sigma_mag": sigma,
        "chi2": chi2,
        "ndof": int(len(frame) - 1),
        "chi2_per_dof": chi2 / (len(frame) - 1),
        "z_min": float(np.min(z_hd)),
        "z_max": float(np.max(z_hd)),
    }


def summarize_pantheon(data_path: Path, covariance_path: Path) -> dict[str, float | int]:
    frame = pd.read_csv(data_path, sep=r"\s+", engine="python")
    mask = frame["IS_CALIBRATOR"].to_numpy(float) == 0.0
    source_positions = np.flatnonzero(mask)
    flow = frame.loc[mask].reset_index(drop=True)
    tokens = covariance_path.read_text(encoding="utf-8").split()
    dimension = int(tokens[0])
    flat = np.fromiter((float(token) for token in tokens[1:]), dtype=float)
    if flat.size != dimension * dimension or len(frame) != dimension:
        raise ValueError(
            f"Pantheon+ data/matrix dimensions disagree: rows={len(frame)}, "
            f"matrix={dimension}, values={flat.size}"
        )
    covariance = flat.reshape((dimension, dimension), order="C")
    covariance = covariance[np.ix_(source_positions, source_positions)]
    z_hd = flow["zHD"].to_numpy(float)
    z_hel = flow["zHEL"].to_numpy(float)
    mu = flow["MU_SH0ES"].to_numpy(float)
    raw = mu - reference_mu(z_hd, z_hel)
    one = np.ones_like(raw)
    solved_raw = np.linalg.solve(covariance, raw)
    solved_one = np.linalg.solve(covariance, one)
    denominator = float(one @ solved_one)
    intercept = float((one @ solved_raw) / denominator)
    sigma = float(np.sqrt(1.0 / denominator))
    residual = raw - intercept
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    return {
        "rows": int(len(flow)),
        "full_rows": int(len(frame)),
        "calibrator_rows": int(np.count_nonzero(~mask)),
        "intercept_mag": intercept,
        "intercept_sigma_mag": sigma,
        "chi2": chi2,
        "ndof": int(len(flow) - 1),
        "chi2_per_dof": chi2 / (len(flow) - 1),
        "z_min": float(np.min(z_hd)),
        "z_max": float(np.max(z_hd)),
        "first_retained_source_positions": source_positions[:10].tolist(),
    }


def main() -> None:
    started = time.perf_counter()
    paths = {
        "des_hd": Path(os.environ["DES_DOVEKIE_HD_PATH"]),
        "des_precision": Path(os.environ["DES_DOVEKIE_STAT_SYS_PATH"]),
        "pantheon_data": Path(os.environ["PANTHEONPLUS_DATA_PATH"]),
        "pantheon_covariance": Path(os.environ["PANTHEONPLUS_STAT_SYS_PATH"]),
    }
    des_frame = read_dovekie(paths["des_hd"])
    independent = {
        "method": "standalone pandas/NumPy; does not import supernova_analyzer",
        "cosmology": "Flat LambdaCDM, H0=70 km s^-1 Mpc^-1, Omega_m=0.3; 128-point Gauss-Legendre integral",
        "des_dovekie": summarize_precision(des_frame, paths["des_precision"]),
        "pantheon_plus": summarize_pantheon(paths["pantheon_data"], paths["pantheon_covariance"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    existing = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    program = existing["steps"]["D_to_I_official_catalogs_retry"]["results"]
    comparisons = {}
    for catalog_key in ("des_dovekie", "pantheon_plus"):
        program_stats = program[catalog_key]["full_statsys"]
        independent_stats = independent[catalog_key]
        comparisons[catalog_key] = {
            field: abs(float(program_stats[program_field]) - float(independent_stats[field]))
            for field, program_field in (
                ("intercept_mag", "intercept_mag"),
                ("intercept_sigma_mag", "intercept_sigma_mag"),
                ("chi2", "chi2"),
                ("chi2_per_dof", "chi2_per_dof"),
            )
        }
    independent["absolute_differences_vs_program"] = comparisons
    independent["status"] = "PASS" if all(
        difference < 1e-8
        for values in comparisons.values()
        for difference in values.values()
    ) else "CHECK"
    (REPORT / "independent_recompute.json").write_text(
        json.dumps(independent, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    existing["independent_recomputation"] = independent
    (REPORT / "summary.json").write_text(
        json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(independent, indent=2))


if __name__ == "__main__":
    main()
