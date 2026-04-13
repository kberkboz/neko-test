# Experimental Plan: Exp-6 Monte Carlo π Estimation

## Objective
Estimate π using 100,000 random points via the quarter-circle method and validate convergence properties.

## Phases

### Phase 1 — Single Run (seed=42)
- Draw 100,000 (x,y) pairs from Uniform(0,1)²
- Compute running estimate π̂(n) = 4 × cumsum(inside)/n
- Record final estimate, absolute error, fraction inside

### Phase 2 — Scatter Plot
- Plot all 100,000 points colored by inside/outside quarter-circle
- Overlay unit quarter-circle arc
- Use small markers + transparency for clarity

### Phase 3 — Multi-Run Replications
- 1,000 independent replications (seed=2024, batched ×100)
- At 80 log-spaced checkpoints from n=10 to n=100,000:
  - Accumulate squared errors for RMSE computation
- Store final estimates for distribution analysis

### Phase 4 — Log-Log Regression
- Compute RMSE at each checkpoint
- Regress log(RMSE) on log(N)
- Report: slope, std_err, R², fitted constant
- Compare to theoretical slope = −0.5 and constant ≈ 1.6419

### Phase 5 — Convergence Plot
- Plot running π̂(n) from single run
- Overlay horizontal line at true π
- Overlay ±1.96 × σ(n) theoretical envelope
- σ(n) = 1.6419/√n

### Coverage Analysis (corrected per feasibility review)
- At selected checkpoints {100, 500, 1000, 5000, 10000, 50000, 100000}
- Run 1000 independent estimates
- Check fraction falling within ±1.96σ of true π
- This is a *pointwise cross-replication* coverage check, NOT a pathwise claim

## Falsification Criteria
| Criterion | Threshold |
|-----------|-----------|
| |error| < 0.05 in >95% of 1000 runs | Must hold |
| Log-log slope | Must be in [−0.60, −0.40] |
| Empirical RMSE at 100k | Within 20% of 0.00519 |

## Corrections Applied (from feasibility review)
1. RMSE ≠ expected absolute error; E|X| = σ√(2/π) ≈ 0.00414
2. P(|error|<0.01) ≈ 0.946, not 0.68
3. Envelope coverage is pointwise across replications, not pathwise along one trajectory
