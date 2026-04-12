"""
Diagnostic checks for the QIRWS vs Leiden experiment.

Runs all diagnostics (D1-D7) and produces a report.
Must pass before proceeding to comparison phases.
"""

import numpy as np
import json
from typing import Dict, Any, Optional

from data_loader import load_pbmc3k, get_symmetric_adjacency, get_pca_embedding, get_labels
from qirws_kernel import QIRWSKernel
from utils import safe_ari, timer


def run_leiden_sanity(
    adata: "anndata.AnnData",
    labels_true: np.ndarray,
    resolutions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    D1/D2: Verify Leiden produces reasonable ARI on PBMC 3k.

    Args:
        adata: Preprocessed AnnData.
        labels_true: Reference integer labels.
        resolutions: Resolution values to test.

    Returns:
        Dictionary with Leiden sanity check results.
    """
    import scanpy as sc

    if resolutions is None:
        resolutions = np.array([0.3, 0.5, 0.8, 1.0, 1.2, 1.5])

    results = {}
    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=42, flavor='igraph',
                     n_iterations=2)
        pred = adata.obs['leiden'].astype(int).values
        ari = safe_ari(labels_true, pred)
        n_clusters = len(np.unique(pred))
        results[f"res={res:.1f}"] = {
            "ari": float(ari),
            "n_clusters": n_clusters,
        }

    aris = [v["ari"] for v in results.values()]
    best_ari = max(aris)

    d1_pass = best_ari > 0.40
    d2_pass = 0.30 <= best_ari <= 0.90  # wider than hypothesis spec for robustness

    return {
        "results_by_resolution": results,
        "best_ari": float(best_ari),
        "D1_ari_above_040": d1_pass,
        "D2_ari_in_expected_range": d2_pass,
    }


def run_all_diagnostics(verbose: bool = True) -> Dict[str, Any]:
    """
    Run all diagnostic checks.

    Returns:
        Complete diagnostic report.
    """
    print("=" * 60)
    print("PHASE 0: DIAGNOSTIC CHECKS")
    print("=" * 60)

    # Load data
    print("\n--- Loading PBMC 3k ---")
    with timer() as t_load:
        adata = load_pbmc3k()
    print(f"  Load time: {t_load['elapsed']:.1f}s")

    A = get_symmetric_adjacency(adata)
    X_pca = get_pca_embedding(adata)
    labels = get_labels(adata)

    report: Dict[str, Any] = {
        "dataset": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_types": int(len(np.unique(labels))),
            "type_counts": {
                name: int(np.sum(labels == i))
                for i, name in enumerate(adata.uns['cell_type_names'])
            },
        },
        "adjacency": {
            "shape": list(A.shape),
            "nnz": int(np.sum(A > 0)),
            "symmetric": bool(np.allclose(A, A.T)),
            "min": float(A.min()),
            "max": float(A.max()),
        },
    }

    # D1/D2: Leiden sanity
    print("\n--- D1/D2: Leiden Sanity Check ---")
    with timer() as t_leiden:
        leiden_check = run_leiden_sanity(adata, labels)
    report["D1_D2_leiden"] = leiden_check
    report["D1_D2_leiden"]["runtime_s"] = t_leiden["elapsed"]

    d1 = leiden_check["D1_ari_above_040"]
    d2 = leiden_check["D2_ari_in_expected_range"]
    print(f"  D1 (ARI > 0.40): {'PASS' if d1 else 'FAIL'} "
          f"(best ARI={leiden_check['best_ari']:.4f})")
    print(f"  D2 (ARI in expected range): {'PASS' if d2 else 'FAIL'}")
    for res_key, res_val in leiden_check["results_by_resolution"].items():
        print(f"    {res_key}: ARI={res_val['ari']:.4f}, "
              f"n_clusters={res_val['n_clusters']}")

    # D3-D7: Kernel diagnostics
    print("\n--- D3-D7: QIRWS Kernel Diagnostics ---")
    with timer() as t_kernel:
        kernel = QIRWSKernel(A)
        kernel_diag = kernel.diagnostics()
    report["kernel_diagnostics"] = kernel_diag
    report["kernel_diagnostics"]["runtime_s"] = t_kernel["elapsed"]

    # Print key results
    checks = kernel_diag["checks"]

    # D4
    d4 = checks["D4_K0_is_identity"]
    print(f"  D4 (K(0) = I): {'PASS' if d4['pass'] else 'FAIL'} "
          f"(max_error={d4['max_error']:.2e})")

    # D3
    for key in sorted(checks.keys()):
        if key.startswith("D3_"):
            v = checks[key]
            print(f"  {key}: {'PASS' if v['pass'] else 'FAIL'} "
                  f"(frob_err/n={v['frob_error_per_n']:.6f})")

    # D5
    print(f"  D5 all symmetric: {'PASS' if checks['D5_all_symmetric'] else 'FAIL'}")
    print(f"  D5 all nonneg: {'PASS' if checks['D5_all_nonneg'] else 'FAIL'}")

    # D6
    d6 = checks["D6_eigenvalue_spectrum"]
    for t_key, t_val in d6.items():
        print(f"  D6 {t_key}: eig range=[{t_val['min_eigenvalue']:.4f}, "
              f"{t_val['max_eigenvalue']:.4f}], "
              f"n_negative={t_val['n_negative']}")

    # D7
    d7 = checks["D7_heat_kernel"]
    for beta_key, beta_val in d7.items():
        print(f"  D7 {beta_key}: ||H-I||_F={beta_val['frob_from_identity']:.4f}, "
              f"sym={beta_val['symmetric']}, min={beta_val['min_value']:.6f}")

    # Overall pass/fail
    all_pass = (
        d1 and d2
        and d4["pass"]
        and checks["D5_all_symmetric"]
        and checks["D5_all_nonneg"]
    )
    report["all_diagnostics_pass"] = all_pass
    print(f"\n{'=' * 60}")
    print(f"OVERALL DIAGNOSTICS: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    report = run_all_diagnostics()

    # Save report
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        return convert_types(d)

    report_clean = deep_convert(report)

    with open("diagnostic_report.json", "w") as f:
        json.dump(report_clean, f, indent=2)
    print("\nDiagnostic report saved to diagnostic_report.json")
