#!/usr/bin/env python3
"""
get_zscore_per_gene.py
======================
Compute per-gene, per-tissue Z-scores by standardising each gene's observed
max-isoform SSMD against the tissue-specific permutation null.

  Z_gene_tissue = (observed_max_SSMD - mean(null_max_SSMD)) / std(null_max_SSMD)

This removes both sample-size effects and tissue-specific background variance,
making Z-scores directly comparable across tissues and species.

Outputs
-------
  obs_max_ssmd.tsv.gz   : gene x tissue observed max SSMD
  null_mean.tsv.gz      : gene x tissue mean of permutation null
  null_std.tsv.gz       : gene x tissue std of permutation null
  zscore.tsv.gz         : gene x tissue Z-scores (main output)

Usage
-----
    python get_zscore_per_gene.py \\
        --species   Human \\
        --el_path   /path/to/EL_corrected.tsv \\
        --meta_path /path/to/metadata.tsv \\
        [--out_dir  ./results] \\
        [--k        100] \\
        [--workers  4] \\
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

def get_tissue_config(species, tissue, tissues_complete):
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
            'Brain':      (brain_tissues_human,
                           [x for x in tissues_complete if 'Brain' not in x and 'Cereb' not in x]),
            'Cerebellum': (['Brain_Cerebellar_Hemisphere'],
                           [x for x in tissues_complete if 'Brain' not in x and 'Cereb' not in x]),
            'Heart':      (heart_tissues_human,
                           [x for x in tissues_complete if x not in heart_tissues_human]),
            'Skin':       (skin_tissues_human,
                           [x for x in tissues_complete if x not in skin_tissues_human]),
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
            'Brain':      (['Brain'], cardoso_non_brain),
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
        return {'tissue_list': [tissue],
                'other_tissues': [x for x in tissues_complete if x != tissue]}
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
    with np.errstate(invalid='ignore', divide='ignore'):
        ssmd = (tc_mean - oc_mean) / np.sqrt(tc_var + oc_var)
    return ssmd


def _gene_max_ssmd(mat, ti, oi, gene_starts, n_genes):
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
# Per-tissue worker
# ---------------------------------------------------------------------------

def _zscore_for_tissue(tissue, mat, gene_labels, gene_starts, ti, oi, k, seed):
    """
    Returns per-gene Z-scores and the null mean/std used to compute them.

    Z = (obs - null_mean) / null_std

    null_mean and null_std are per-gene, so both sample-size effects and
    gene-specific expression variance are absorbed into the normalisation.
    """
    np.random.seed(seed)
    n_genes  = len(gene_labels)
    combined = np.concatenate([ti, oi])
    n_ti     = len(ti)

    obs = _gene_max_ssmd(mat, ti, oi, gene_starts, n_genes)

    # k permutation draws — shape (k, n_genes)
    null_maxes = np.full((k, n_genes), np.nan, dtype=np.float32)
    for i in range(k):
        perm          = np.random.permutation(len(combined))
        ti_perm       = combined[perm[:n_ti]]
        oi_perm       = combined[perm[n_ti:]]
        null_maxes[i] = _gene_max_ssmd(mat, ti_perm, oi_perm, gene_starts, n_genes)

    # Per-gene null statistics
    null_mean = np.nanmean(null_maxes, axis=0)   # (n_genes,)
    null_std  = np.nanstd( null_maxes, axis=0)   # (n_genes,)

    # Z-score: how many null SDs above the null mean is the observed value?
    with np.errstate(invalid='ignore', divide='ignore'):
        zscore = (obs - null_mean) / np.where(null_std > 0, null_std, np.nan)

    return tissue, obs, null_mean, null_std, zscore


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_zscores(EL, tissues, species, metadata,
                    k=100, n_workers=4, gid_col='gid', pid_col='pid'):

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

    obs_arr       = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    null_mean_arr = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    null_std_arr  = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    zscore_arr    = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)

    tissue_to_idx = {t: i for i, t in enumerate(tissues)}

    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for tissue in tissues:
            ti, oi = tissue_indices[tissue]
            if len(ti) == 0 or len(oi) == 0:
                print(f"  WARNING: skipping {tissue} — no samples found", flush=True)
                continue
            seed = np.random.randint(0, 2**31)
            fut  = executor.submit(
                _zscore_for_tissue,
                tissue, mat, gene_labels, gene_starts, ti, oi, k, seed
            )
            futures[fut] = tissue

        for fut in tqdm(as_completed(futures), total=len(futures), desc='Tissues'):
            tissue, obs, null_mean, null_std, zscore = fut.result()
            t_idx = tissue_to_idx[tissue]
            obs_arr      [:, t_idx] = obs
            null_mean_arr[:, t_idx] = null_mean
            null_std_arr [:, t_idx] = null_std
            zscore_arr   [:, t_idx] = zscore
            print(f"  done: {tissue}", flush=True)

    def _df(arr):
        d = pd.DataFrame(arr, index=gene_labels, columns=tissues)
        d.index.name = gid_col
        return d

    return _df(obs_arr), _df(null_mean_arr), _df(null_std_arr), _df(zscore_arr)


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
    p.add_argument('--k',         default=100, type=int,
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

    print(f"[{sp}] Computing Z-scores (k={args.k}, workers={args.workers}) …",
          flush=True)

    obs_df, null_mean_df, null_std_df, zscore_df = compute_zscores(
        EL, tissues, sp, metadata,
        k=args.k, n_workers=args.workers,
        gid_col=args.gid_col, pid_col=args.pid_col,
    )

    obs_df.to_csv(      f"{pfx}/obs_max_ssmd.tsv.gz", sep='\t')
    null_mean_df.to_csv(f"{pfx}/null_mean.tsv.gz",    sep='\t')
    null_std_df.to_csv( f"{pfx}/null_std.tsv.gz",     sep='\t')
    zscore_df.to_csv(   f"{pfx}/zscore.tsv.gz",       sep='\t')

    print(f"[{sp}] obs_max_ssmd → {pfx}/obs_max_ssmd.tsv.gz", flush=True)
    print(f"[{sp}] null_mean    → {pfx}/null_mean.tsv.gz",    flush=True)
    print(f"[{sp}] null_std     → {pfx}/null_std.tsv.gz",     flush=True)
    print(f"[{sp}] zscore       → {pfx}/zscore.tsv.gz",       flush=True)
    print(f"[{sp}] Done.", flush=True)


if __name__ == '__main__':
    main()