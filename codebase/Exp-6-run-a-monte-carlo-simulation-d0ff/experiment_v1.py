#!/usr/bin/env python3
"""
Experiment 6: Monte Carlo Estimation of π
==========================================
- Phase 1: Single run with 100,000 points, compute running estimate
- Phase 2: Scatter plot colored by hit/miss
- Phase 3: 1,000 independent replications for RMSE scaling
- Phase 4: Log-log regression of RMSE vs N
- Phase 5: Convergence plot with theoretical ±1.96σ envelope

Outputs:
  figures/convergence.png      – running π̂(n) with theoretical envelope
  figures/scatter.png           – quarter-circle hit/miss scatter
  figures/rmse_scaling.png      – log-log RMSE vs N with fitted slope
  figures/loglog_error_single.png – single-run |error| vs n
  results.json                  – all numeric results
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Any

import numpy as np
from numpy.typing import NDArray
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_POINTS: int = 100_000
N_REPLICATIONS: int = 1_000
SEED_SINGLE: int = 42
SEED_REPLICATIONS: int = 2024
TRUE_PI: float = np.pi
P_TRUE: float = TRUE_PI / 4.0  # ≈ 0.7854

# Theoretical RMSE constant: 4*sqrt(p*(1-p))
RMSE_CONST: float = 4.0 * np.sqrt(P_TRUE * (1.0 - P_TRUE))  # ≈ 1.6419

# Log-spaced checkpoints for RMSE measurement
CHECKPOINTS: NDArray[np.int64] = np.unique(
    np.geomspace(10, N_POINTS, num=80).astype(np.int64)
)

os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1: Single run — running estimate
# ---------------------------------------------------------------------------
def single_run(
    n: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Run a single Monte Carlo π estimation.

    Parameters
    ----------
    n : int
        Number of random points.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    x, y : arrays of sampled coordinates
    running_pi : running estimate π̂(k) for k = 1..n
    inside : boolean mask for points inside quarter-circle
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.uniform(0.0, 1.0, size=n)
    inside = x**2 + y**2 <= 1.0
    cumulative_inside = np.cumsum(inside)
    indices = np.arange(1, n + 1, dtype=np.float64)
    running_pi = 4.0 * cumulative_inside / indices
    return x, y, running_pi, inside


print("Phase 1: Single run with N =", N_POINTS)
t0 = time.perf_counter()
x, y, running_pi, inside = single_run(N_POINTS, SEED_SINGLE)
t_single = time.perf_counter() - t0
pi_estimate = running_pi[-1]
abs_error = abs(pi_estimate - TRUE_PI)
fraction_inside = np.sum(inside) / N_POINTS

print(f"  π̂ = {pi_estimate:.6f}  |error| = {abs_error:.6f}  "
      f"fraction inside = {fraction_inside:.5f}  time = {t_single:.4f}s")


# ---------------------------------------------------------------------------
# Phase 2: Scatter plot
# ---------------------------------------------------------------------------
print("Phase 2: Scatter plot")
fig_scatter, ax_scatter = plt.subplots(figsize=(7, 7))
# Subsample for cleaner rendering if needed — 100k is fine with alpha
ax_scatter.scatter(
    x[inside], y[inside],
    s=0.15, alpha=0.3, color="dodgerblue", label="Inside", rasterized=True,
)
ax_scatter.scatter(
    x[~inside], y[~inside],
    s=0.15, alpha=0.3, color="salmon", label="Outside", rasterized=True,
)
# Quarter-circle arc
theta = np.linspace(0, np.pi / 2, 300)
ax_scatter.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5, label="x²+y²=1")
ax_scatter.set_xlim(0, 1)
ax_scatter.set_ylim(0, 1)
ax_scatter.set_aspect("equal")
ax_scatter.set_xlabel("x")
ax_scatter.set_ylabel("y")
ax_scatter.set_title(f"Monte Carlo π: {N_POINTS:,} points  (π̂={pi_estimate:.5f})")
ax_scatter.legend(loc="lower left", markerscale=10, fontsize=9)
fig_scatter.tight_layout()
fig_scatter.savefig("figures/scatter.png", dpi=150)
plt.close(fig_scatter)


# ---------------------------------------------------------------------------
# Phase 5 (moved up): Convergence plot with theoretical envelope
# ---------------------------------------------------------------------------
print("Phase 5: Convergence plot with envelope")
ns = np.arange(1, N_POINTS + 1, dtype=np.float64)
theoretical_sd = RMSE_CONST / np.sqrt(ns)

fig_conv, ax_conv = plt.subplots(figsize=(10, 5))
# Thin the trajectory for plotting speed
thin = max(1, N_POINTS // 5000)
idx = np.arange(0, N_POINTS, thin)

ax_conv.plot(ns[idx], running_pi[idx], lw=0.5, color="steelblue", label="π̂(n)")
ax_conv.axhline(TRUE_PI, color="black", ls="--", lw=1, label=f"π = {TRUE_PI:.5f}")
ax_conv.fill_between(
    ns[idx],
    TRUE_PI - 1.96 * theoretical_sd[idx],
    TRUE_PI + 1.96 * theoretical_sd[idx],
    color="orange", alpha=0.25, label="±1.96 σ(n) envelope",
)
ax_conv.set_xlabel("Number of points (n)")
ax_conv.set_ylabel("Estimate π̂(n)")
ax_conv.set_title("Convergence of Monte Carlo π Estimate")
ax_conv.legend(loc="upper right", fontsize=9)
ax_conv.set_xlim(1, N_POINTS)
fig_conv.tight_layout()
fig_conv.savefig("figures/convergence.png", dpi=150)
plt.close(fig_conv)


# ---------------------------------------------------------------------------
# Phase 1 supplement: Single-run log-log |error|
# ---------------------------------------------------------------------------
print("Phase 1 supplement: log-log |error| for single run")
single_abs_err = np.abs(running_pi - TRUE_PI)
# Use checkpoints to avoid plotting 100k noisy points
cp_errors = single_abs_err[CHECKPOINTS - 1]

fig_ll1, ax_ll1 = plt.subplots(figsize=(8, 5))
ax_ll1.loglog(CHECKPOINTS, cp_errors, "o", ms=3, alpha=0.6, label="|π̂(n)−π| (single run)")
ax_ll1.loglog(
    CHECKPOINTS,
    RMSE_CONST / np.sqrt(CHECKPOINTS.astype(float)),
    "r-", lw=1.5, label=f"Theoretical RMSE = {RMSE_CONST:.4f}/√n",
)
ax_ll1.set_xlabel("n")
ax_ll1.set_ylabel("|π̂(n) − π|")
ax_ll1.set_title("Single-Run Absolute Error (log-log)")
ax_ll1.legend(fontsize=9)
ax_ll1.grid(True, which="both", ls=":", alpha=0.4)
fig_ll1.tight_layout()
fig_ll1.savefig("figures/loglog_error_single.png", dpi=150)
plt.close(fig_ll1)


# ---------------------------------------------------------------------------
# Phase 3 & 4: Multi-run replications → RMSE scaling
# ---------------------------------------------------------------------------
print(f"Phase 3: {N_REPLICATIONS} replications for RMSE at {len(CHECKPOINTS)} checkpoints")
t0 = time.perf_counter()

# Storage: squared errors at each checkpoint, accumulated over replications
sq_errors = np.zeros((len(CHECKPOINTS),), dtype=np.float64)
abs_errors_100k = np.empty(N_REPLICATIONS, dtype=np.float64)

# We also collect final estimates for coverage analysis
final_estimates = np.empty(N_REPLICATIONS, dtype=np.float64)

rng_rep = np.random.default_rng(SEED_REPLICATIONS)

# Process in batches to control memory
BATCH = 100
n_batches = N_REPLICATIONS // BATCH

for b in range(n_batches):
    # Generate batch of random points: shape (BATCH, N_POINTS)
    rx = rng_rep.uniform(0.0, 1.0, size=(BATCH, N_POINTS))
    ry = rng_rep.uniform(0.0, 1.0, size=(BATCH, N_POINTS))
    inside_mask = rx**2 + ry**2 <= 1.0  # (BATCH, N_POINTS)
    cum_inside = np.cumsum(inside_mask.astype(np.float64), axis=1)  # (BATCH, N_POINTS)

    for ci, cp in enumerate(CHECKPOINTS):
        pi_est_cp = 4.0 * cum_inside[:, cp - 1] / cp
        sq_errors[ci] += np.sum((pi_est_cp - TRUE_PI) ** 2)

    # Final estimates
    pi_final = 4.0 * cum_inside[:, -1] / N_POINTS
    start_idx = b * BATCH
    final_estimates[start_idx: start_idx + BATCH] = pi_final
    abs_errors_100k[start_idx: start_idx + BATCH] = np.abs(pi_final - TRUE_PI)

t_rep = time.perf_counter() - t0
print(f"  Replications done in {t_rep:.2f}s")

rmse_empirical = np.sqrt(sq_errors / N_REPLICATIONS)
rmse_theoretical = RMSE_CONST / np.sqrt(CHECKPOINTS.astype(np.float64))


# ---------------------------------------------------------------------------
# Phase 4: Log-log regression
# ---------------------------------------------------------------------------
print("Phase 4: Log-log regression of RMSE vs N")
log_n = np.log(CHECKPOINTS.astype(np.float64))
log_rmse = np.log(rmse_empirical)

slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rmse)
fitted_const = np.exp(intercept)
r_squared = r_value**2

print(f"  Fitted slope = {slope:.4f} ± {std_err:.4f}")
print(f"  Fitted constant = {fitted_const:.4f}  (theoretical = {RMSE_CONST:.4f})")
print(f"  R² = {r_squared:.6f}")


# ---------------------------------------------------------------------------
# Phase 4: RMSE scaling plot
# ---------------------------------------------------------------------------
fig_rmse, ax_rmse = plt.subplots(figsize=(8, 5))
ax_rmse.loglog(CHECKPOINTS, rmse_empirical, "o", ms=4, color="steelblue",
               label=f"Empirical RMSE ({N_REPLICATIONS} runs)")
ax_rmse.loglog(CHECKPOINTS, rmse_theoretical, "r-", lw=1.5,
               label=f"Theoretical: {RMSE_CONST:.4f}/√N")
ax_rmse.loglog(
    CHECKPOINTS,
    fitted_const * CHECKPOINTS.astype(float) ** slope,
    "g--", lw=1.5,
    label=f"Fit: {fitted_const:.4f}·N^({slope:.4f}), R²={r_squared:.5f}",
)
ax_rmse.set_xlabel("N (number of points)")
ax_rmse.set_ylabel("RMSE of π̂")
ax_rmse.set_title("RMSE Scaling: Monte Carlo π Estimation")
ax_rmse.legend(fontsize=9)
ax_rmse.grid(True, which="both", ls=":", alpha=0.4)
fig_rmse.tight_layout()
fig_rmse.savefig("figures/rmse_scaling.png", dpi=150)
plt.close(fig_rmse)


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("HYPOTHESIS TESTING")
print("=" * 60)

# H1a: Point estimate accuracy
# Correct metric: RMSE ≈ 0.00519, E|error| ≈ 0.00414
sigma_theory = RMSE_CONST / np.sqrt(N_POINTS)
expected_abs_error_theory = sigma_theory * np.sqrt(2.0 / np.pi)
empirical_rmse_100k = rmse_empirical[-1] if CHECKPOINTS[-1] == N_POINTS else np.sqrt(
    np.mean(abs_errors_100k**2)
)
empirical_mean_abs_error = np.mean(abs_errors_100k)

print(f"\nH1a – Point estimate accuracy at N={N_POINTS:,}:")
print(f"  Theoretical σ(π̂)           = {sigma_theory:.6f}")
print(f"  Theoretical E|error|        = {expected_abs_error_theory:.6f}")
print(f"  Empirical RMSE              = {empirical_rmse_100k:.6f}")
print(f"  Empirical mean |error|      = {empirical_mean_abs_error:.6f}")
print(f"  RMSE deviation from theory  = {abs(empirical_rmse_100k - sigma_theory)/sigma_theory*100:.2f}%")

# H1 primary: |error| < 0.05 in >95% of runs
frac_within_005 = np.mean(abs_errors_100k < 0.05)
print(f"\n  Fraction |error| < 0.05     = {frac_within_005:.4f}  (require >0.95)")
# Also check tighter bounds
frac_within_001 = np.mean(abs_errors_100k < 0.01)
frac_within_002 = np.mean(abs_errors_100k < 0.02)
print(f"  Fraction |error| < 0.01     = {frac_within_001:.4f}  (theory ≈ 0.946)")
print(f"  Fraction |error| < 0.02     = {frac_within_002:.4f}  (theory ≈ 0.9999)")

# H1b: Convergence slope
print(f"\nH1b – Convergence rate:")
print(f"  Fitted slope                = {slope:.4f} ± {std_err:.4f}")
print(f"  Expected slope              = -0.5000")
print(f"  Within [-0.60, -0.40]?      = {-0.60 <= slope <= -0.40}")
print(f"  R²                          = {r_squared:.6f}")

# H1c: Coverage at checkpoints (corrected interpretation per feasibility)
# For each checkpoint, check what fraction of replications fall within
# ±1.96σ of true π
print(f"\nH1c – Pointwise coverage across replications (corrected):")
# Recompute coverage at selected checkpoints
coverage_cps = [100, 500, 1000, 5000, 10000, 50000, 100000]
coverage_results: Dict[int, float] = {}

# We need to re-run a subset to get per-checkpoint distributions
# Use a fresh but deterministic set
rng_cov = np.random.default_rng(9999)
N_COV_REPS = 1000
for cp in coverage_cps:
    estimates = np.empty(N_COV_REPS)
    for i in range(N_COV_REPS):
        pts_x = rng_cov.uniform(0, 1, size=cp)
        pts_y = rng_cov.uniform(0, 1, size=cp)
        est = 4.0 * np.mean(pts_x**2 + pts_y**2 <= 1.0)
        estimates[i] = est
    sigma_cp = RMSE_CONST / np.sqrt(cp)
    in_band = np.mean(np.abs(estimates - TRUE_PI) < 1.96 * sigma_cp)
    coverage_results[cp] = float(in_band)
    print(f"  n={cp:>6d}: coverage = {in_band:.3f}  (target ≈ 0.95)")

# H1d: Fraction inside
print(f"\nH1d – Fraction inside quarter-circle:")
print(f"  Observed fraction           = {fraction_inside:.5f}")
print(f"  Expected (π/4)              = {P_TRUE:.5f}")
print(f"  |Deviation|                 = {abs(fraction_inside - P_TRUE):.5f}")
print(f"  Within ±0.003?              = {abs(fraction_inside - P_TRUE) < 0.003}")


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OVERALL VERDICT")
print("=" * 60)

h1_primary = frac_within_005 > 0.95
h1a = abs(empirical_rmse_100k - sigma_theory) / sigma_theory < 0.20
h1b = -0.60 <= slope <= -0.40
h1d = abs(fraction_inside - P_TRUE) < 0.003

verdicts = {
    "H1 (primary: |error|<0.05 >95%)": h1_primary,
    "H1a (RMSE within 20% of theory)": h1a,
    "H1b (slope in [-0.60, -0.40])": h1b,
    "H1c (pointwise coverage ~95%)": all(v > 0.90 for v in coverage_results.values()),
    "H1d (fraction ±0.003 of π/4)": h1d,
}

for name, passed in verdicts.items():
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  {name}: {status}")

all_pass = all(verdicts.values())
print(f"\n  → H₁ {'SUPPORTED' if all_pass else 'PARTIALLY SUPPORTED'}")
print(f"  → H₀ {'REJECTED' if all_pass else 'NOT FULLY REJECTED'}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results: Dict[str, Any] = {
    "config": {
        "n_points": N_POINTS,
        "n_replications": N_REPLICATIONS,
        "seed_single": SEED_SINGLE,
        "seed_replications": SEED_REPLICATIONS,
    },
    "single_run": {
        "pi_estimate": float(pi_estimate),
        "absolute_error": float(abs_error),
        "fraction_inside": float(fraction_inside),
        "runtime_seconds": float(t_single),
    },
    "theoretical": {
        "rmse_constant": float(RMSE_CONST),
        "sigma_at_100k": float(sigma_theory),
        "expected_abs_error_at_100k": float(expected_abs_error_theory),
    },
    "replications": {
        "empirical_rmse_100k": float(empirical_rmse_100k),
        "empirical_mean_abs_error": float(empirical_mean_abs_error),
        "rmse_deviation_pct": float(abs(empirical_rmse_100k - sigma_theory) / sigma_theory * 100),
        "frac_error_lt_005": float(frac_within_005),
        "frac_error_lt_001": float(frac_within_001),
        "frac_error_lt_002": float(frac_within_002),
        "runtime_seconds": float(t_rep),
    },
    "loglog_regression": {
        "slope": float(slope),
        "slope_stderr": float(std_err),
        "intercept": float(intercept),
        "fitted_constant": float(fitted_const),
        "r_squared": float(r_squared),
    },
    "coverage_at_checkpoints": {str(k): v for k, v in coverage_results.items()},
    "hypothesis_verdicts": {k: bool(v) for k, v in verdicts.items()},
    "overall": "H1 SUPPORTED" if all_pass else "H1 PARTIALLY SUPPORTED",
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results.json")
print(f"Figures saved to figures/")
print("Done.")