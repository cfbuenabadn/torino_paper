#!/usr/bin/env python3
"""
get_fdr_per_gene.py
===================
Compute per-gene, per-tissue empirical FDR for tissue-specific isoform usage.

For each tissue and each shuffle iteration:
  - Randomly permute which samples are "tissue" vs "other" (within ti+oi)
  - Compute per-isoform SSMD across all genes simultaneously (vectorised)
  - Take the max SSMD per gene (the gene-level test statistic under the null)

FDR at threshold t (one-tailed, positive):
  FDR(t) = (mean over k of: fraction of null gene-maxes >= t)
           / (fraction of observed gene-maxes >= t)

Tissues are processed in parallel via ProcessPoolExecutor.

Usage
-----
    python get_fdr_per_gene.py \\
        --species   Human \\
        --el_path   /path/to/EL_corrected.tsv \\
        --meta_path /path/to/metadata.tsv \\
        [--out_dir  ./results] \\
        [--k        100] \\
        [--workers  8] \\
        [--pid_col  pid] \\
        [--gid_col  gid]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tissue lists
# ---------------------------------------------------------------------------

TISSUES = {
    'Human': [
        'Adipose_Subcutaneous', 'Artery_Aorta', 'Brain', 'Cerebellum',
        'Breast_Mammary_Tissue', 'Heart', 'Liver', 'Lung', 'Muscle_Skeletal',
        'Pancreas', 'Skin', 'Testis', 'Whole_Blood',
    ],
    'Cattle': [
        'Blood', 'Brain', 'Colon', 'Heart', 'Liver', 'Lung',
        'Mammary_Gland', 'Muscle', 'Skin', 'Subcutaneous_Adipose', 'Testis',
    ],
}
TISSUES_OTHER = ['Cerebellum', 'Heart', 'Kidney', 'Liver', 'Testis', 'Brain', 'Ovary']


# ---------------------------------------------------------------------------
# Tissue configuration
# ---------------------------------------------------------------------------

def get_tissue_config(species: str, tissue: str, tissues_complete: list) -> dict:
    cardoso_non_brain = ['Heart', 'Kidney', 'Liver', 'Testis', 'Ovary']
    cardoso_non_cereb = ['Heart', 'Kidney', 'Liver', 'Testis', 'Ovary']

    if species == 'Human':
        brain_tissues_human = [
            'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex',
            'Brain_Frontal_Cortex_BA9', 'Brain_Putamen_basal_ganglia',
        ]
        heart_tissues_human = ['Heart_Atrial_Appendage', 'Heart_Left_Ventricle']
        skin_tissues_human  = ['Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg']

        mapping = {
            'Brain': (
                brain_tissues_human,
                [x for x in tissues_complete if 'Brain' not in x and 'Cereb' not in x],
            ),
            'Cerebellum': (
                ['Brain_Cerebellar_Hemisphere'],
                [x for x in tissues_complete if 'Brain' not in x and 'Cereb' not in x],
            ),
            'Heart': (
                heart_tissues_human,
                [x for x in tissues_complete if x not in heart_tissues_human],
            ),
            'Skin': (
                skin_tissues_human,
                [x for x in tissues_complete if x not in skin_tissues_human],
            ),
            'Liver':           (['Liver'],           [x for x in tissues_complete if x != 'Liver']),
            'Testis':          (['Testis'],           [x for x in tissues_complete if x != 'Testis']),
            'Muscle_Skeletal': (['Muscle_Skeletal'],  [x for x in tissues_complete if x != 'Muscle_Skeletal']),
        }

    elif species == 'Cattle':
        if tissue == 'Cerebellum':
            tissue = 'Brain'
        mapping = {
            'Brain':  (['Brain'],   [x for x in tissues_complete if 'Brain' not in x]),
            'Heart':  (['Heart'],   [x for x in tissues_complete if x != 'Heart']),
            'Liver':  (['Liver'],   [x for x in tissues_complete if x != 'Liver']),
            'Testis': (['Testis'],  [x for x in tissues_complete if x != 'Testis']),
            'Muscle': (['Muscle'],  [x for x in tissues_complete if x != 'Muscle']),
        }

    elif species in ['Rat', 'Rabbit']:
        mapping = {
            'Brain':      (['Brain'],                          cardoso_non_brain),
            'Cerebellum': (['Cerebellum', 'Brain,Cerebellum'],
                           [x for x in tissues_complete if 'Brain' not in x and 'Cereb' not in x]),
            'Heart':      (['Heart'],  ['Brain', 'Cerebellum', 'Kidney', 'Liver', 'Testis', 'Ovary']),
            'Liver':      (['Liver'],  ['Brain', 'Cerebellum', 'Kidney', 'Heart', 'Testis', 'Ovary']),
            'Testis':     (['Testis'], ['Brain', 'Cerebellum', 'Kidney', 'Heart', 'Liver', 'Ovary']),
        }

    else:
        mapping = {
            'Brain':      (['Brain'],      cardoso_non_brain),
            'Cerebellum': (['Cerebellum'], cardoso_non_cereb),
            'Heart':      (['Heart'],  ['Brain', 'Cerebellum', 'Kidney', 'Liver', 'Testis', 'Ovary']),
            'Liver':      (['Liver'],  ['Brain', 'Cerebellum', 'Kidney', 'Heart', 'Testis', 'Ovary']),
            'Testis':     (['Testis'], ['Brain', 'Cerebellum', 'Kidney', 'Heart', 'Liver', 'Ovary']),
        }

    if tissue not in mapping:
        tl = [tissue]
        ot = [x for x in tissues_complete if x != tissue]
        return {'tissue_list': tl, 'other_tissues': ot}

    tl, ot = mapping[tissue]
    return {'tissue_list': tl, 'other_tissues': ot}


# ---------------------------------------------------------------------------
# Vectorised SSMD helpers
# ---------------------------------------------------------------------------

def _isoform_stats(mat_cols):
    not_nan = (~np.isnan(mat_cols)).astype(np.float32)
    mat0    = np.where(np.isnan(mat_cols), 0.0, mat_cols)
    count   = not_nan.sum(axis=1)
    mean    = mat0.sum(axis=1) / np.where(count > 0, count, np.nan)
    sq_dev  = np.where(np.isnan(mat_cols), 0.0, (mat_cols - mean[:, None]) ** 2)
    var     = sq_dev.sum(axis=1) / np.where(count > 1, count - 1, np.nan)
    return mean, var


def _vectorised_ssmd(mat, ti, oi):
    tc_mean, tc_var = _isoform_stats(mat[:, ti])
    oc_mean, oc_var = _isoform_stats(mat[:, oi])
    mean_diff = tc_mean - oc_mean
    with np.errstate(invalid='ignore', divide='ignore'):
        ssmd = mean_diff / np.sqrt(tc_var + oc_var)
    return ssmd


def _gene_max_ssmd(mat, ti, oi, gene_starts, n_genes):
    """Per-gene max isoform SSMD (one-tailed: positive only)."""
    ssmd = _vectorised_ssmd(mat, ti, oi)
    ends = np.concatenate([gene_starts[1:], [mat.shape[0]]])
    gene_max = np.empty(n_genes, dtype=np.float32)
    for g in range(n_genes):
        sl    = ssmd[gene_starts[g]:ends[g]]
        valid = sl[~np.isnan(sl)]
        gene_max[g] = valid.max() if len(valid) > 0 else np.nan
    return gene_max


# ---------------------------------------------------------------------------
# Matrix preparation
# ---------------------------------------------------------------------------

def _prepare_matrix(EL, sample_cols, gid_col, pid_col):
    EL = EL.reset_index(drop=True)
    if gid_col not in EL.columns:
        EL[gid_col] = EL[pid_col].str.split('.').str[0]
    EL = EL.sort_values(gid_col)

    gene_labels, gene_idx = np.unique(EL[gid_col].values, return_inverse=True)
    gene_starts    = np.concatenate([[0], np.where(np.diff(gene_idx))[0] + 1])
    isoform_labels = EL[pid_col].values
    mat            = EL[sample_cols].values.astype(np.float32)

    return mat, isoform_labels, gene_labels, gene_starts


# ---------------------------------------------------------------------------
# Per-tissue worker — runs in a subprocess
# ---------------------------------------------------------------------------

def _fdr_for_tissue(tissue, mat, gene_labels, gene_starts, ti, oi, k, seed):
    """
    Designed to be called in a worker process — all args are picklable.

    Returns
    -------
    tissue      : str
    obs         : float32 (n_genes,)
    fdr         : float32 (n_genes,)
    """
    np.random.seed(seed)
    n_genes  = len(gene_labels)
    combined = np.concatenate([ti, oi])
    n_ti     = len(ti)

    obs = _gene_max_ssmd(mat, ti, oi, gene_starts, n_genes)

    # Build null pool: k shuffles × n_genes values, all exchangeable
    null_maxes = np.full((k, n_genes), np.nan, dtype=np.float32)
    for i in range(k):
        perm          = np.random.permutation(len(combined))
        ti_perm       = combined[perm[:n_ti]]
        oi_perm       = combined[perm[n_ti:]]
        null_maxes[i] = _gene_max_ssmd(mat, ti_perm, oi_perm, gene_starts, n_genes)

    null_pool = null_maxes.flatten()
    null_pool = null_pool[~np.isnan(null_pool)]
    n_null    = len(null_pool)

    valid     = ~np.isnan(obs)
    n_obs     = valid.sum()
    obs_valid = obs[valid]

    # Sort descending so prop_obs is monotone and we can break early
    order      = np.argsort(obs_valid)[::-1]
    obs_sorted = obs_valid[order]

    fdr_sorted = np.full(len(obs_sorted), np.nan, dtype=np.float32)
    for rank, t in enumerate(obs_sorted):
        prop_null = (null_pool >= t).sum() / n_null
        prop_obs  = (obs_sorted >= t).sum() / n_obs
        if prop_obs > 0:
            fdr_sorted[rank] = min(prop_null / prop_obs, 1.0)

    # Enforce monotonicity: FDR non-decreasing as score decreases
    for rank in range(1, len(fdr_sorted)):
        if not np.isnan(fdr_sorted[rank]) and not np.isnan(fdr_sorted[rank - 1]):
            fdr_sorted[rank] = max(fdr_sorted[rank], fdr_sorted[rank - 1])

    fdr_gene              = np.full(n_genes, np.nan, dtype=np.float32)
    valid_idx             = np.where(valid)[0]
    fdr_gene[valid_idx[order]] = fdr_sorted

    return tissue, obs, fdr_gene


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_fdr(EL, tissues, species, metadata,
                k=100, fdr_levels=(0.01, 0.05, 0.10),
                n_workers=4, gid_col='gid', pid_col='pid'):

    sample_cols      = list(EL.columns[6:])
    col_pos          = {c: i for i, c in enumerate(sample_cols)}
    tissues_complete = metadata['Tissue'].unique().tolist()

    tissue_indices = {}
    for tissue in tissues:
        cfg = get_tissue_config(species, tissue, tissues_complete)
        ti  = np.array([col_pos[c] for c in
                        metadata.loc[metadata['Tissue'].isin(cfg['tissue_list'])].index
                        if c in col_pos])
        oi  = np.array([col_pos[c] for c in
                        metadata.loc[metadata['Tissue'].isin(cfg['other_tissues'])].index
                        if c in col_pos])
        tissue_indices[tissue] = (ti, oi)

    mat, isoform_labels, gene_labels, gene_starts = _prepare_matrix(
        EL, sample_cols, gid_col, pid_col
    )
    n_genes = len(gene_labels)

    obs_arr = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    fdr_arr = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)

    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for t_idx, tissue in enumerate(tissues):
            ti, oi = tissue_indices[tissue]
            if len(ti) == 0 or len(oi) == 0:
                print(f"  WARNING: skipping {tissue} — no samples found", flush=True)
                continue
            seed = np.random.randint(0, 2**31)
            fut  = executor.submit(
                _fdr_for_tissue,
                tissue, mat, gene_labels, gene_starts, ti, oi, k, seed
            )
            futures[fut] = t_idx

        for fut in tqdm(as_completed(futures), total=len(futures), desc='Tissues'):
            t_idx             = futures[fut]
            tissue, obs, fdr  = fut.result()
            obs_arr[:, t_idx] = obs
            fdr_arr[:, t_idx] = fdr
            print(f"  done: {tissue}", flush=True)

    obs_df = pd.DataFrame(obs_arr, index=gene_labels, columns=tissues)
    fdr_df = pd.DataFrame(fdr_arr, index=gene_labels, columns=tissues)
    obs_df.index.name = gid_col
    fdr_df.index.name = gid_col

    sig_dfs = {level: (fdr_df <= level) for level in fdr_levels}

    return obs_df, fdr_df, sig_dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--species',   required=True)
    p.add_argument('--el_path',   required=True)
    p.add_argument('--meta_path', required=True)
    p.add_argument('--out_dir',   default='.')
    p.add_argument('--k',         default=1000, type=int,
                   help='Permutation iterations per tissue (default: 100).')
    p.add_argument('--workers',   default=4, type=int,
                   help='Parallel worker processes (default: 4).')
    p.add_argument('--pid_col',   default='pid')
    p.add_argument('--gid_col',   default='gid')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sp  = args.species
    pfx = args.out_dir

    tissues = TISSUES.get(sp, TISSUES_OTHER)
    print(f"[{sp}] Tissues: {tissues}", flush=True)

    print(f"[{sp}] Loading EL from {args.el_path}", flush=True)
    EL = pd.read_csv(args.el_path, sep='\t')

    print(f"[{sp}] Loading metadata from {args.meta_path}", flush=True)
    if sp == 'Human':
        metadata = pd.read_csv(args.meta_path, sep='\t')
        metadata = metadata[['entity:sample_id', 'tissue_id']]
        metadata.columns = ['Sample', 'Tissue']
        metadata.index = metadata['Sample']
    elif sp == 'Cattle':
        metadata = pd.read_csv(args.meta_path, sep='\t', index_col=0)
        metadata.columns = ['Tissue', 'counts']
    else:
        metadata = pd.read_csv(args.meta_path, sep='\t', index_col=0)

    metadata = metadata.loc[metadata.index.intersection(EL.columns)]
    print(f"[{sp}] {len(metadata)} samples, "
          f"{metadata['Tissue'].nunique()} tissues", flush=True)

    print(f"[{sp}] Computing empirical FDR "
          f"(k={args.k}, workers={args.workers}) …", flush=True)

    obs_df, fdr_df, sig_dfs = compute_fdr(
        EL, tissues, sp, metadata,
        k=args.k,
        fdr_levels=(0.01, 0.05, 0.10),
        n_workers=args.workers,
        gid_col=args.gid_col,
        pid_col=args.pid_col,
    )

    obs_df.to_csv(f"{pfx}/obs_max_ssmd.tsv.gz", sep='\t')
    fdr_df.to_csv(f"{pfx}/fdr.tsv.gz",          sep='\t')
    for level, sig_df in sig_dfs.items():
        tag = str(level).replace('.', 'p')
        sig_df.to_csv(f"{pfx}/sig_{tag}.tsv.gz", sep='\t')
        n_sig = sig_df.any(axis=1).sum()
        print(f"[{sp}] FDR<={level}: {n_sig} genes significant "
              f"in at least one tissue", flush=True)

    print(f"[{sp}] Done.", flush=True)


if __name__ == '__main__':
    main()