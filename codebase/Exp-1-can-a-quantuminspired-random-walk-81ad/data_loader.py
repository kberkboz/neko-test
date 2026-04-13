"""
Data loading and preprocessing for PBMC 3k experiment.

Produces standardized AnnData with PCA embeddings, KNN graph,
and reference cell-type labels.
"""
#test
import numpy as np
import warnings
from typing import Tuple, Optional


def load_pbmc3k(
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> "anndata.AnnData":
    """
    Load and preprocess PBMC 3k dataset.

    Pipeline: filter → normalize → log1p → HVG → PCA → KNN graph.

    Args:
        n_top_genes: Number of highly variable genes.
        n_pcs: Number of principal components.
        n_neighbors: Number of neighbors for KNN graph.
        random_state: Random seed for reproducibility.

    Returns:
        AnnData with .obsm['X_pca'], .obsp['connectivities'],
        .obs['cell_type'] (integer-encoded), .obs['cell_type_str'].
    """
    import scanpy as sc

    # Try processed version first, fall back to raw
    try:
        adata = sc.datasets.pbmc3k_processed()
        # Processed version already has clusters, PCA, etc.
        # We re-do from scratch for reproducibility
        if 'louvain' in adata.obs.columns:
            adata.obs['cell_type_str'] = adata.obs['louvain'].astype(str)
        elif 'leiden' in adata.obs.columns:
            adata.obs['cell_type_str'] = adata.obs['leiden'].astype(str)
        else:
            raise ValueError("No cluster labels found in processed data")

    except Exception:
        # Download raw and process
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # Annotate mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
        )
        adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
        adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()

    # Standard preprocessing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, random_state=random_state)
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            random_state=random_state,
        )

    # Encode cell types as integers
    if 'cell_type_str' not in adata.obs.columns:
        # Look for any label column
        for col in ['louvain', 'leiden', 'cell_type', 'celltype']:
            if col in adata.obs.columns:
                adata.obs['cell_type_str'] = adata.obs[col].astype(str)
                break

    if 'cell_type_str' in adata.obs.columns:
        types = adata.obs['cell_type_str'].values
        unique_types = np.unique(types)
        type_to_int = {t: i for i, t in enumerate(unique_types)}
        adata.obs['cell_type'] = [type_to_int[t] for t in types]
        adata.uns['cell_type_names'] = list(unique_types)
    else:
        raise ValueError("Could not find cell type labels in dataset")

    print(f"PBMC 3k loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Cell types ({len(adata.uns['cell_type_names'])}): "
          f"{adata.uns['cell_type_names']}")
    print(f"PCA: {adata.obsm['X_pca'].shape}")

    return adata


def get_symmetric_adjacency(adata: "anndata.AnnData") -> np.ndarray:
    """
    Extract symmetric adjacency matrix from AnnData KNN graph.

    Args:
        adata: AnnData with .obsp['connectivities'].

    Returns:
        Dense symmetric adjacency matrix (n_cells x n_cells).
    """
    import scipy.sparse as sp

    A = adata.obsp['connectivities']
    if sp.issparse(A):
        A = A.toarray()
    # Symmetrize
    A = (A + A.T) / 2.0
    # Zero diagonal
    np.fill_diagonal(A, 0.0)
    return A


def get_pca_embedding(adata: "anndata.AnnData") -> np.ndarray:
    """Extract PCA embedding."""
    return adata.obsm['X_pca'].copy()


def get_labels(adata: "anndata.AnnData") -> np.ndarray:
    """Extract integer cell-type labels."""
    return np.array(adata.obs['cell_type'].values, dtype=int)


if __name__ == "__main__":
    adata = load_pbmc3k()
    A = get_symmetric_adjacency(adata)
    X_pca = get_pca_embedding(adata)
    labels = get_labels(adata)

    print(f"\nAdjacency: shape={A.shape}, nnz={np.sum(A > 0)}, "
          f"symmetric={np.allclose(A, A.T)}")
    print(f"PCA: shape={X_pca.shape}")
    print(f"Labels: {len(np.unique(labels))} types, "
          f"counts={np.bincount(labels)}")
