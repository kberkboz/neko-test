"""
Experiment 1: Monte Carlo estimation of π
==========================================
Part A: Single-run demonstration (seed=42, N=100,000)
Part B: Ensemble validation (R=1,000 independent runs, N=100,000)

Tests three hypotheses:
  H1: Coverage at ε=0.02 exceeds 99.8% (descriptive criterion)
  H2: Coverage at ε≈0.01017 falls within acceptance band [0.93, 0.97]
      AND two-sided binomial test consistent with 95%
  H3: Ensemble RMSE log-log slope ∈ [−0.55, −0.45], validated by
      bootstrap CI over runs

Addresses feasibility concerns:
  - H1 uses descriptive threshold (not underpowered binomial test)
  - H2 uses acceptance band + consistency framing (not "fail-to-reject = proof")
  - H3 uses bootstrap over runs for slope CI (not naive OLS stderr on dependent points)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os
import time

# ── Configuration ──────────────────────────────────────────────────────────
N: int = 100_000
R: int = 1_000                # number of independent replications
N_BOOTSTRAP: int = 2_000      # bootstrap resamples for H3 slope CI
SEED_DEMO: int = 42
SEED_ENSEMBLE: int = 12345
TRUE_PI: float = np.pi
P_HIT: float = TRUE_PI / 4.0
SE_CONSTANT: float = 4.0 * np.sqrt(P_HIT * (1.0 - P_HIT))  # ≈ 1.6410
SE_AT_N: float = SE_CONSTANT / np.sqrt(N)                     # ≈ 0.005189
EPSILON_LENIENT: float = 0.02
EPSILON_95: float = 1.96 * SE_AT_N                             # ≈ 0.01017

# Evaluation points for RMSE curve (20 log-spaced from 100 to N)
EVAL_NS: np.ndarray = np.unique(np.geomspace(100, N, num=20).astype(int))

os.makedirs("figures", exist_ok=True)


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(char * 70)
    print(title)
    print(char * 70)


def compute_exact_finite_n_coverage(n: int, epsilon: float) -> float:
    """
    Compute exact coverage P(|π̂_n - π| < epsilon) using binomial CDF.

    π̂_n = 4*K/n where K ~ Binomial(n, π/4).
    |π̂_n - π| < ε  ⟺  (π - ε)/4 < K/n < (π + ε)/4
                   ⟺  n*(π - ε)/4 < K < n*(π + ε)/4

    Parameters
    ----------
    n : int
        Number of samples.
    epsilon : float
        Half-width threshold for |π̂ - π|.

    Returns
    -------
    float
        Exact probability of |π̂_n - π| < epsilon.
    """
    p = TRUE_PI / 4.0
    k_low = n * (TRUE_PI - epsilon) / 4.0
    k_high = n * (TRUE_PI + epsilon) / 4.0
    # K must be integer in (k_low, k_high), i.e., ceil(k_low) to floor(k_high)
    # But we need strict inequality for |error| < epsilon (not ≤)
    # Actually π̂ = 4K/n, |4K/n - π| < ε ⟺ K ∈ (n(π-ε)/4, n(π+ε)/4)
    # Use floor/ceil for integer bounds
    k_lo_int = int(np.ceil(k_low))   # smallest integer K satisfying K > k_low
    k_hi_int = int(np.floor(k_high)) # largest integer K satisfying K < k_high
    # Handle edge: if k_low is exactly integer, exclude it (strict inequality)
    if k_lo_int == k_low:
        k_lo_int += 1
    if k_hi_int == k_high:
        k_hi_int -= 1
    # P(k_lo_int <= K <= k_hi_int)
    if k_hi_int < k_lo_int:
        return 0.0
    coverage = stats.binom.cdf(k_hi_int, n, p) - stats.binom.cdf(k_lo_int - 1, n, p)
    return float(coverage)


def bootstrap_slope_ci(
    estimates_at_eval: np.ndarray,
    eval_ns: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """
    Bootstrap the RMSE log-log slope by resampling over runs (rows).

    Parameters
    ----------
    estimates_at_eval : ndarray of shape (R, n_eval)
        π̂ estimates at each evaluation point for each run.
    eval_ns : ndarray of shape (n_eval,)
        Sample sizes at evaluation points.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level for the CI.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    slope_median : float
        Median of bootstrap slope distribution.
    ci_low : float
        Lower CI bound.
    ci_high : float
        Upper CI bound.
    slopes : ndarray
        All bootstrap slope estimates.
    """
    if rng is None:
        rng = np.random.default_rng(99999)
    r_total = estimates_at_eval.shape[0]
    log_n = np.log10(eval_ns.astype(float))
    slopes = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, r_total, size=r_total)
        resampled = estimates_at_eval[idx, :]
        rmse_b = np.sqrt(np.mean((resampled - TRUE_PI) ** 2, axis=0))
        log_rmse_b = np.log10(rmse_b)
        result = stats.linregress(log_n, log_rmse_b)
        slopes[b] = result.slope

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(slopes, 100 * alpha / 2))
    ci_high = float(np.percentile(slopes, 100 * (1 - alpha / 2)))
    slope_median = float(np.median(slopes))
    return slope_median, ci_low, ci_high, slopes


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print_header("EXPERIMENT 1: MONTE CARLO ESTIMATION OF π")
print(f"  N per run          = {N:,}")
print(f"  Replications R     = {R:,}")
print(f"  Bootstrap samples  = {N_BOOTSTRAP:,}")
print(f"  SE constant        = {SE_CONSTANT:.6f}")
print(f"  SE at N={N:,}    = {SE_AT_N:.6f}")
print(f"  ε (lenient)        = {EPSILON_LENIENT}")
print(f"  ε (95% CI)         = {EPSILON_95:.5f}")
print()

# Exact finite-N coverage computations
exact_cov_lenient = compute_exact_finite_n_coverage(N, EPSILON_LENIENT)
exact_cov_95 = compute_exact_finite_n_coverage(N, EPSILON_95)
print(f"  Exact finite-N coverage at ε=0.02:    {exact_cov_lenient:.6f}")
print(f"  Exact finite-N coverage at ε={EPSILON_95:.5f}: {exact_cov_95:.6f}")
print(f"  Normal approx coverage at ε=0.02:     {2*stats.norm.cdf(EPSILON_LENIENT/SE_AT_N)-1:.6f}")
print(f"  Normal approx coverage at ε={EPSILON_95:.5f}: 0.950000")
print()

# ══════════════════════════════════════════════════════════════════════════
# PART A: Single-run demonstration
# ══════════════════════════════════════════════════════════════════════════
print_header(f"PART A: Single-run demonstration (seed = {SEED_DEMO})", "-")

rng_demo = np.random.default_rng(SEED_DEMO)
x = rng_demo.uniform(0.0, 1.0, size=N)
y = rng_demo.uniform(0.0, 1.0, size=N)
inside = (x ** 2 + y ** 2) <= 1.0

cumulative_hits = np.cumsum(inside)
n_values = np.arange(1, N + 1)
pi_estimates = 4.0 * cumulative_hits / n_values
abs_errors = np.abs(pi_estimates - TRUE_PI)

final_pi: float = float(pi_estimates[-1])
final_error: float = float(abs_errors[-1])
hit_fraction: float = float(cumulative_hits[-1]) / N

print(f"  Final π̂           = {final_pi:.6f}")
print(f"  True π            = {TRUE_PI:.6f}")
print(f"  Absolute error    = {final_error:.6f}")
print(f"  |error| < 0.02   = {final_error < EPSILON_LENIENT}")
print(f"  |error| < ε_95   = {final_error < EPSILON_95}")
print(f"  Hit fraction      = {hit_fraction:.6f}  (expected ≈ {P_HIT:.6f})")
print()

# ── Figure A1: Convergence plot ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(n_values, pi_estimates, linewidth=0.4, color="steelblue", alpha=0.7,
        label=r"Running estimate $\hat{\pi}_n$")
ax.axhline(TRUE_PI, color="crimson", linestyle="--", linewidth=1.2,
           label=f"True π = {TRUE_PI:.5f}")

theoretical_se = SE_CONSTANT / np.sqrt(n_values)
ax.fill_between(n_values,
                TRUE_PI - 2.0 * theoretical_se,
                TRUE_PI + 2.0 * theoretical_se,
                color="crimson", alpha=0.10,
                label=r"$\pm\,2\,\mathrm{SE}$ envelope (theoretical)")

ax.set_xlabel("Number of random points (N)", fontsize=12)
ax.set_ylabel(r"Estimate of $\pi$", fontsize=12)
ax.set_title(
    f"Monte Carlo Estimation of π — Single Run (seed={SEED_DEMO})  |  "
    f"Final π̂ = {final_pi:.5f}  |  Error = {final_error:.5f}",
    fontsize=12, fontweight="bold",
)
ax.set_xscale("log")
ax.set_xlim(1, N)
ax.set_ylim(2.6, 3.7)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/convergence_plot.png", dpi=150)
plt.close(fig)
print("  Saved: figures/convergence_plot.png")

# ── Figure A2: Scatter of first 5,000 points ──────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 7))
n_show: int = 5_000
colors = np.where(inside[:n_show], "steelblue", "salmon")
ax3.scatter(x[:n_show], y[:n_show], c=colors, s=1, alpha=0.6)
theta_arc = np.linspace(0.0, np.pi / 2.0, 200)
ax3.plot(np.cos(theta_arc), np.sin(theta_arc), color="black", linewidth=1.5)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_aspect("equal")
ax3.set_xlabel("x", fontsize=12)
ax3.set_ylabel("y", fontsize=12)
ax3.set_title(f"First {n_show:,} points — Blue = inside, Red = outside", fontsize=13)
fig3.tight_layout()
fig3.savefig("figures/scatter_points.png", dpi=150)
plt.close(fig3)
print("  Saved: figures/scatter_points.png")
print()

# ══════════════════════════════════════════════════════════════════════════
# PART B: Ensemble validation (R = 1,000 runs)
# ══════════════════════════════════════════════════════════════════════════
print_header(f"PART B: Ensemble validation (R = {R:,} independent runs)", "-")

t0 = time.time()

# Pre-allocate storage
final_estimates = np.empty(R)
estimates_at_eval = np.empty((R, len(EVAL_NS)))

# Use SeedSequence for independent, reproducible streams
seed_seq = np.random.SeedSequence(SEED_ENSEMBLE)
child_seeds = seed_seq.spawn(R)

for r in range(R):
    rng_r = np.random.default_rng(child_seeds[r])
    xr = rng_r.uniform(0.0, 1.0, size=N)
    yr = rng_r.uniform(0.0, 1.0, size=N)
    inside_r = (xr ** 2 + yr ** 2) <= 1.0
    cum_hits_r = np.cumsum(inside_r)

    # Extract estimates at evaluation points (EVAL_NS are 1-based sizes)
    estimates_at_eval[r, :] = 4.0 * cum_hits_r[EVAL_NS - 1] / EVAL_NS
    final_estimates[r] = estimates_at_eval[r, -1]

elapsed = time.time() - t0
print(f"  Ensemble completed in {elapsed:.1f} seconds")
print(f"  ({R:,} × {N:,} = {R*N:,.0f} point evaluations)")
print()

final_errors = np.abs(final_estimates - TRUE_PI)

# ── H1: Coverage at ε = 0.02 (descriptive criterion) ──────────────────────
n_success_lenient: int = int(np.sum(final_errors < EPSILON_LENIENT))
coverage_lenient: float = n_success_lenient / R

print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  H1: Coverage at ε = 0.02 (lenient threshold)              │")
print("  └─────────────────────────────────────────────────────────────┘")
print(f"    Observed coverage         = {coverage_lenient:.4f} ({n_success_lenient}/{R})")
print(f"    Exact finite-N coverage   = {exact_cov_lenient:.6f}")
print(f"    Criterion: coverage ≥ 99.8% (descriptive)")
h1_pass: bool = coverage_lenient >= 0.998
print(f"    coverage ≥ 0.998?         = {h1_pass}")
if n_success_lenient < R:
    # Report which runs failed, for diagnostics
    fail_idx = np.where(final_errors >= EPSILON_LENIENT)[0]
    print(f"    Failed runs: {fail_idx.tolist()}")
    print(f"    Their errors: {final_errors[fail_idx].tolist()}")
print(f"    *** H1 {'SUPPORTED ✓' if h1_pass else 'NOT SUPPORTED ✗'} ***")
print()

# ── H2: Coverage at ε ≈ 0.01017 (95% CI half-width) ──────────────────────
n_success_95: int = int(np.sum(final_errors < EPSILON_95))
coverage_95: float = n_success_95 / R

# Two-sided exact binomial test: H0: p = 0.95
h2_binom_result = stats.binomtest(n_success_95, R, 0.95, alternative="two-sided")
h2_pvalue: float = float(h2_binom_result.pvalue)
h2_ci = h2_binom_result.proportion_ci(confidence_level=0.95, method="exact")

# Acceptance band criterion: coverage ∈ [0.93, 0.97]
ACCEPT_LOW: float = 0.93
ACCEPT_HIGH: float = 0.97
in_accept_band: bool = ACCEPT_LOW <= coverage_95 <= ACCEPT_HIGH
ci_contains_95: bool = h2_ci.low <= 0.95 <= h2_ci.high

print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  H2: Coverage at ε ≈ 0.01017 (95% CI half-width)          │")
print("  └─────────────────────────────────────────────────────────────┘")
print(f"    Observed coverage         = {coverage_95:.4f} ({n_success_95}/{R})")
print(f"    Exact finite-N coverage   = {exact_cov_95:.6f}")
print(f"    Binomial test p-value     = {h2_pvalue:.6f}")
print(f"    95% Clopper-Pearson CI    = [{h2_ci.low:.4f}, {h2_ci.high:.4f}]")
print(f"    CI contains 0.95?         = {ci_contains_95}")
print(f"    Acceptance band [{ACCEPT_LOW}, {ACCEPT_HIGH}]")
print(f"    Coverage in band?         = {in_accept_band}")
# H2 passes if BOTH criteria met: acceptance band AND CI contains 0.95
h2_pass: bool = in_accept_band and ci_contains_95
print(f"    *** H2 {'SUPPORTED ✓ (consistent with ~95%)' if h2_pass else 'NOT SUPPORTED ✗'} ***")
print()

# ── H3: Convergence rate (ensemble RMSE slope with bootstrap) ─────────────
# Point estimate via OLS
ensemble_rmse = np.sqrt(np.mean((estimates_at_eval - TRUE_PI) ** 2, axis=0))
log_n_eval = np.log10(EVAL_NS.astype(float))
log_rmse = np.log10(ensemble_rmse)

slope_result = stats.linregress(log_n_eval, log_rmse)
slope_ols: float = float(slope_result.slope)
slope_se_ols: float = float(slope_result.stderr)
intercept_ols: float = float(slope_result.intercept)
r_squared: float = float(slope_result.rvalue ** 2)

# Bootstrap slope CI (resampling over runs)
print("  Computing bootstrap slope CI...")
t1 = time.time()
slope_median, boot_ci_low, boot_ci_high, boot_slopes = bootstrap_slope_ci(
    estimates_at_eval, EVAL_NS, n_bootstrap=N_BOOTSTRAP,
    ci_level=0.95, rng=np.random.default_rng(77777),
)
t_boot = time.time() - t1
print(f"  Bootstrap completed in {t_boot:.1f} seconds")
print()

# Check if theoretical -0.5 is within bootstrap CI
boot_ci_contains_theory: bool = boot_ci_low <= -0.5 <= boot_ci_high
slope_in_band: bool = -0.55 <= slope_ols <= -0.45

print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  H3: Convergence rate (RMSE log-log slope ≈ −0.5)         │")
print("  └─────────────────────────────────────────────────────────────┘")
print(f"    OLS slope                 = {slope_ols:.5f}")
print(f"    OLS stderr (naive)        = {slope_se_ols:.5f}  (unreliable — dependent points)")
print(f"    OLS R²                    = {r_squared:.6f}")
print(f"    Bootstrap median slope    = {slope_median:.5f}")
print(f"    Bootstrap 95% CI          = [{boot_ci_low:.5f}, {boot_ci_high:.5f}]")
print(f"    Bootstrap CI contains -0.5? = {boot_ci_contains_theory}")
print(f"    OLS slope ∈ [-0.55,-0.45]?  = {slope_in_band}")
# H3 passes if slope in band AND bootstrap CI contains -0.5
h3_pass: bool = slope_in_band and boot_ci_contains_theory
print(f"    *** H3 {'SUPPORTED ✓' if h3_pass else 'NOT SUPPORTED ✗'} ***")
print()

# ── Additional ensemble statistics ─────────────────────────────────────────
ensemble_mean: float = float(np.mean(final_estimates))
ensemble_std: float = float(np.std(final_estimates, ddof=1))
mean_bias: float = ensemble_mean - TRUE_PI
se_of_mean: float = ensemble_std / np.sqrt(R)

print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  Ensemble summary statistics                               │")
print("  └─────────────────────────────────────────────────────────────┘")
print(f"    Mean π̂                   = {ensemble_mean:.6f}")
print(f"    Std π̂                    = {ensemble_std:.6f}")
print(f"    Theoretical SE            = {SE_AT_N:.6f}")
print(f"    SE of ensemble mean       = {se_of_mean:.6f}")
print(f"    Bias (mean - π)           = {mean_bias:+.6f}")
print(f"    |bias| < 0.001?           = {abs(mean_bias) < 0.001}")
print(f"    Min π̂                    = {np.min(final_estimates):.6f}")
print(f"    Max π̂                    = {np.max(final_estimates):.6f}")
print(f"    Mean |error|              = {np.mean(final_errors):.6f}")
print(f"    Median |error|            = {np.median(final_errors):.6f}")
print(f"    Max |error|               = {np.max(final_errors):.6f}")
print()

# Shapiro-Wilk test for normality of final estimates (subsample for speed)
rng_shapiro = np.random.default_rng(42)
shapiro_sample = rng_shapiro.choice(final_estimates, size=min(500, R), replace=False)
shapiro_stat, shapiro_pval = stats.shapiro(shapiro_sample)
print(f"    Shapiro-Wilk normality (n={len(shapiro_sample)}): W={shapiro_stat:.5f}, p={shapiro_pval:.4f}")
print(f"    Normal at α=0.05?         = {shapiro_pval >= 0.05}")
print()

# ── Overall verdict ────────────────────────────────────────────────────────
all_pass: bool = h1_pass and h2_pass and h3_pass
print_header("OVERALL RESULT")
print(f"  H1 (lenient coverage ≥ 99.8%):         {'✓ PASS' if h1_pass else '✗ FAIL'}")
print(f"  H2 (95% coverage consistent):           {'✓ PASS' if h2_pass else '✗ FAIL'}")
print(f"  H3 (RMSE slope ≈ −0.5, bootstrap):     {'✓ PASS' if h3_pass else '✗ FAIL'}")
print()
if all_pass:
    print("  CONCLUSION: Results provide strong empirical support for")
    print("  theoretical Monte Carlo predictions (coverage, CLT, O(N^{-1/2}) rate).")
else:
    print("  CONCLUSION: Some hypotheses not supported. See details above.")
print("=" * 70)
print()

# ══════════════════════════════════════════════════════════════════════════
# FIGURES FOR PART B
# ══════════════════════════════════════════════════════════════════════════

# ── Figure B1: Histogram of final estimates ────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 6))

ax4.hist(final_estimates, bins=50, density=True, color="steelblue", alpha=0.7,
         edgecolor="white", linewidth=0.5, label=f"Empirical ({R:,} runs)")

x_norm = np.linspace(
    final_estimates.min() - 3 * ensemble_std,
    final_estimates.max() + 3 * ensemble_std,
    300,
)
y_norm = stats.norm.pdf(x_norm, loc=TRUE_PI, scale=SE_AT_N)
ax4.plot(x_norm, y_norm, color="crimson", linewidth=2,
         label=f"N(π, SE²),  SE={SE_AT_N:.5f}")

ax4.axvline(TRUE_PI, color="black", linestyle="--", linewidth=1,
            label=f"True π = {TRUE_PI:.5f}")

# Mark coverage thresholds
ax4.axvline(TRUE_PI - EPSILON_95, color="orange", linestyle=":", linewidth=1.2)
ax4.axvline(TRUE_PI + EPSILON_95, color="orange", linestyle=":", linewidth=1.2,
            label=f"±ε_95 = ±{EPSILON_95:.5f}")
ax4.axvline(TRUE_PI - EPSILON_LENIENT, color="green", linestyle=":", linewidth=1.2)
ax4.axvline(TRUE_PI + EPSILON_LENIENT, color="green", linestyle=":", linewidth=1.2,
            label=f"±ε_lenient = ±{EPSILON_LENIENT}")

ax4.set_xlabel(r"Final $\hat{\pi}$", fontsize=12)
ax4.set_ylabel("Density", fontsize=12)
ax4.set_title(
    f"Distribution of π̂ across {R:,} runs  |  "
    f"Mean = {ensemble_mean:.5f}  |  Std = {ensemble_std:.5f}",
    fontsize=12, fontweight="bold",
)
ax4.legend(fontsize=10, loc="upper right")
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig("figures/ensemble_histogram.png", dpi=150)
plt.close(fig4)
print("Saved: figures/ensemble_histogram.png")

# ── Figure B2: Ensemble RMSE vs N with bootstrap-validated fit ─────────────
fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.scatter(EVAL_NS, ensemble_rmse, s=50, color="steelblue", zorder=3,
            edgecolors="navy", linewidth=0.5, label="Ensemble RMSE")

# Fitted line (OLS point estimate)
fit_n = np.geomspace(EVAL_NS[0], EVAL_NS[-1], 200)
fit_rmse = 10.0 ** intercept_ols * fit_n ** slope_ols
ax5.plot(fit_n, fit_rmse, color="navy", linewidth=2,
         label=f"OLS fit: slope = {slope_ols:.4f}  "
               f"[bootstrap 95% CI: {boot_ci_low:.4f}, {boot_ci_high:.4f}]")

# Theoretical reference
theo_rmse = SE_CONSTANT / np.sqrt(fit_n)
ax5.plot(fit_n, theo_rmse, color="crimson", linestyle="--", linewidth=1.5,
         label=r"Theoretical $%.4f / \sqrt{N}$ (slope = −0.5)" % SE_CONSTANT)

ax5.set_xscale("log")
ax5.set_yscale("log")
ax5.set_xlabel("Number of random points (N)", fontsize=12)
ax5.set_ylabel("RMSE", fontsize=12)
ax5.set_title(
    f"Ensemble RMSE vs. N  ({R:,} runs)  |  "
    f"Slope = {slope_ols:.4f}  |  R² = {r_squared:.5f}",
    fontsize=12, fontweight="bold",
)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which="both")
fig5.tight_layout()
fig5.savefig("figures/ensemble_rmse.png", dpi=150)
plt.close(fig5)
print("Saved: figures/ensemble_rmse.png")

# ── Figure B3: Bootstrap slope distribution ────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.hist(boot_slopes, bins=60, density=True, color="mediumpurple", alpha=0.7,
         edgecolor="white", linewidth=0.5, label=f"Bootstrap ({N_BOOTSTRAP:,} resamples)")
ax6.axvline(-0.5, color="crimson", linestyle="--", linewidth=1.5,
            label="Theoretical slope = −0.5")
ax6.axvline(slope_ols, color="navy", linestyle="-", linewidth=1.5,
            label=f"OLS point estimate = {slope_ols:.4f}")
ax6.axvline(boot_ci_low, color="gray", linestyle=":", linewidth=1.2,
            label=f"95% CI: [{boot_ci_low:.4f}, {boot_ci_high:.4f}]")
ax6.axvline(boot_ci_high, color="gray", linestyle=":", linewidth=1.2)
ax6.set_xlabel("Log-log slope", fontsize=12)
ax6.set_ylabel("Density", fontsize=12)
ax6.set_title("Bootstrap Distribution of RMSE Log-Log Slope", fontsize=12, fontweight="bold")
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
fig6.tight_layout()
fig6.savefig("figures/bootstrap_slope_distribution.png", dpi=150)
plt.close(fig6)
print("Saved: figures/bootstrap_slope_distribution.png")

print()
print("All figures saved. Experiment complete.")
