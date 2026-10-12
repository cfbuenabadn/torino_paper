#!/usr/bin/env python3
"""
get_storey_fdr_per_gene.py
==========================
Compute per-gene, per-tissue FDR using a Storey-style method that estimates
the null from the left (low-SSMD) portion of the observed max-SSMD distribution.

Rationale
---------
The observed distribution of per-gene max-isoform SSMD is a mixture:
  - A null component (genes with no tissue-specific isoform usage), which
    dominates the left/centre of the distribution
  - A signal component (genuinely tissue-specific genes), which produces
    the right tail

Genes with max-SSMD < threshold_null (default: 1.0) are treated as
predominantly null. A gamma distribution is fit to this left portion,
then extrapolated to estimate null density at higher SSMD values.

FDR(t) = pi0 * P(null >= t) / P(observed >= t)

where pi0 is estimated as the fraction of mass in the null component
(Storey 2002 style, evaluated at lambda = threshold_null).

This approach:
  - Does not require permutation, so is fast
  - Uses the actual data structure as the null reference
  - Is tissue-specific (fit separately per tissue), so sample-size and
    biological background differences across tissues are absorbed
  - Allows cross-tissue comparison of pi0 (fraction of null genes),
    which directly answers "does Brain have more tissue-specific isoform
    usage than Liver?"

Outputs
-------
  obs_max_ssmd.tsv.gz   : gene x tissue observed max SSMD (from SSMD script,
                          or recomputed here if --el_path is provided)
  fdr.tsv.gz            : gene x tissue FDR values
  pi0.tsv               : tissue-level pi0 estimates (fraction null per tissue)
  sig_0p01.tsv.gz       : boolean gene x tissue, FDR <= 0.01
  sig_0p05.tsv.gz       : boolean gene x tissue, FDR <= 0.05
  sig_0p10.tsv.gz       : boolean gene x tissue, FDR <= 0.10

Usage
-----
  From pre-computed max-SSMD table (fast):
    python get_storey_fdr_per_gene.py \\
        --obs_path  /path/to/obs_max_ssmd.tsv.gz \\
        --out_dir   ./results

  Recomputing SSMD from EL (slower, but all-in-one):
    python get_storey_fdr_per_gene.py \\
        --species   Human \\
        --el_path   /path/to/EL_corrected.tsv \\
        --meta_path /path/to/metadata.tsv \\
        --out_dir   ./results
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
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
# Tissue configuration (only needed if recomputing SSMD from EL)
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
# SSMD helpers (only used if recomputing from EL)
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
        return (tc_mean - oc_mean) / np.sqrt(tc_var + oc_var)


def _compute_obs_max_ssmd(EL, tissues, species, metadata,
                          gid_col='gid', pid_col='pid'):
    """Compute per-gene max-isoform SSMD from EL matrix."""
    sample_cols      = list(EL.columns[6:])
    col_pos          = {c: i for i, c in enumerate(sample_cols)}
    tissues_complete = metadata['Tissue'].unique().tolist()

    EL = EL.reset_index(drop=True)
    if gid_col not in EL.columns:
        EL[gid_col] = EL[pid_col].str.split('.').str[0]
    EL = EL.sort_values(gid_col)

    gene_labels, gene_idx = np.unique(EL[gid_col].values, return_inverse=True)
    gene_starts = np.concatenate([[0], np.where(np.diff(gene_idx))[0] + 1])
    ends        = np.concatenate([gene_starts[1:], [len(EL)]])
    n_genes     = len(gene_labels)
    mat         = EL[sample_cols].values.astype(np.float32)

    obs_arr = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)

    for t_idx, tissue in enumerate(tqdm(tissues, desc='Computing SSMD')):
        cfg = get_tissue_config(species, tissue, tissues_complete)
        ti  = np.array([col_pos[c] for c in
                        metadata.loc[metadata['Tissue'].isin(cfg['tissue_list'])].index
                        if c in col_pos])
        oi  = np.array([col_pos[c] for c in
                        metadata.loc[metadata['Tissue'].isin(cfg['other_tissues'])].index
                        if c in col_pos])
        if len(ti) == 0 or len(oi) == 0:
            continue
        ssmd = _vectorised_ssmd(mat, ti, oi)
        for g in range(n_genes):
            sl    = ssmd[gene_starts[g]:ends[g]]
            valid = sl[~np.isnan(sl)]
            obs_arr[g, t_idx] = valid.max() if len(valid) > 0 else np.nan

    df = pd.DataFrame(obs_arr, index=gene_labels, columns=tissues)
    df.index.name = gid_col
    return df


# ---------------------------------------------------------------------------
# Storey-style FDR per tissue
# ---------------------------------------------------------------------------

def _storey_fdr_for_tissue(ssmd_vals, null_threshold=1.0):
    """
    Fit a gamma null to genes with max-SSMD < null_threshold, then compute
    FDR for all genes.

    Parameters
    ----------
    ssmd_vals       : array of per-gene max SSMD for one tissue
    null_threshold  : genes below this are used to fit the null (default 1.0)

    Returns
    -------
    fdr    : array of FDR values aligned with ssmd_vals
    pi0    : estimated fraction of null genes
    params : fitted gamma parameters (a, loc, scale)
    """
    vals  = np.asarray(ssmd_vals, dtype=float)
    valid = ~np.isnan(vals)
    n_obs = valid.sum()

    # Genes below null_threshold are used to fit the null distribution
    null_mask = valid & (vals < null_threshold)
    null_vals = vals[null_mask]

    if null_mask.sum() < 20:
        # Not enough data to fit — return NaNs
        return np.full(len(vals), np.nan), np.nan, None

    # Fit gamma to null portion (positive support, right-skewed — good match
    # for max-SSMD values which are always >= 0)
    try:
        a, loc, scale = stats.gamma.fit(null_vals, floc=0)
    except Exception:
        return np.full(len(vals), np.nan), np.nan, None

    # pi0: Storey estimator at lambda = null_threshold
    # = fraction of all genes that could plausibly be null
    pi0 = min((vals[valid] < null_threshold).sum() / n_obs / 
              stats.gamma.cdf(null_threshold, a, loc=loc, scale=scale), 1.0)

    # FDR(t) = pi0 * P(null >= t) / P(observed >= t)
    fdr = np.full(len(vals), np.nan, dtype=float)
    obs_sorted = np.sort(vals[valid])[::-1]

    for idx in np.where(valid)[0]:
        t        = vals[idx]
        prop_null = stats.gamma.sf(t, a, loc=loc, scale=scale)  # P(null >= t)
        prop_obs  = (obs_sorted >= t).sum() / n_obs              # P(obs >= t)
        if prop_obs > 0:
            fdr[idx] = min(pi0 * prop_null / prop_obs, 1.0)

    # Enforce monotonicity: sort by score descending, take cummax of FDR
    order      = np.argsort(vals[valid])[::-1]
    valid_idx  = np.where(valid)[0]
    fdr_sorted = fdr[valid_idx[order]]
    # cumulative max going from high score to low score
    running_max = np.maximum.accumulate(fdr_sorted)
    fdr[valid_idx[order]] = running_max

    return fdr, pi0, (a, loc, scale)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_storey_fdr(obs_df, null_threshold=1.0, fdr_levels=(0.01, 0.05, 0.10)):
    """
    Apply Storey FDR tissue by tissue.

    Returns
    -------
    fdr_df  : gene x tissue FDR values
    pi0_s   : Series of pi0 per tissue  (biologically interesting:
              low pi0 = tissue has many specific genes)
    sig_dfs : dict of {fdr_level: boolean gene x tissue DataFrame}
    """
    tissues   = obs_df.columns.tolist()
    genes     = obs_df.index
    fdr_arr   = np.full(obs_df.shape, np.nan, dtype=float)
    pi0_vals  = {}

    for t_idx, tissue in enumerate(tqdm(tissues, desc='Storey FDR')):
        ssmd_vals        = obs_df[tissue].values
        fdr, pi0, params = _storey_fdr_for_tissue(ssmd_vals, null_threshold)
        fdr_arr[:, t_idx] = fdr
        pi0_vals[tissue]  = pi0

        n_sig_05 = (fdr <= 0.05).sum()
        print(f"  {tissue}: pi0={pi0:.3f}, "
              f"FDR<=0.05: {n_sig_05}/{(~np.isnan(fdr)).sum()} genes",
              flush=True)

    fdr_df = pd.DataFrame(fdr_arr, index=genes, columns=tissues)
    fdr_df.index.name = obs_df.index.name

    pi0_s = pd.Series(pi0_vals, name='pi0')
    pi0_s.index.name = 'tissue'

    sig_dfs = {level: (fdr_df <= level) for level in fdr_levels}

    return fdr_df, pi0_s, sig_dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Option A: load pre-computed obs_max_ssmd
    p.add_argument('--obs_path',  default=None,
                   help='Pre-computed obs_max_ssmd.tsv.gz from get_zscore_per_gene.py '
                        'or get_fdr_per_gene.py. If provided, --el_path is not needed.')
    # Option B: recompute SSMD from EL
    p.add_argument('--species',   default=None)
    p.add_argument('--el_path',   default=None)
    p.add_argument('--meta_path', default=None)
    p.add_argument('--pid_col',   default='pid')
    p.add_argument('--gid_col',   default='gid')

    p.add_argument('--null_threshold', default=1.0, type=float,
                   help='Max-SSMD below which genes are treated as null '
                        'for fitting (default: 1.0).')
    p.add_argument('--out_dir',   default='.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pfx = args.out_dir

    # ---- load or compute obs_max_ssmd ----
    if args.obs_path:
        print(f"Loading obs_max_ssmd from {args.obs_path}", flush=True)
        obs_df = pd.read_csv(args.obs_path, sep='\t', index_col=0)

    elif args.el_path and args.meta_path and args.species:
        sp      = args.species
        tissues = TISSUES.get(sp, TISSUES_OTHER)

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
        obs_df   = _compute_obs_max_ssmd(
            EL, tissues, sp, metadata,
            gid_col=args.gid_col, pid_col=args.pid_col,
        )
        obs_df.to_csv(f"{pfx}/obs_max_ssmd.tsv.gz", sep='\t')

    else:
        raise ValueError(
            "Provide either --obs_path OR all of --species, --el_path, --meta_path"
        )

    print(f"Obs shape: {obs_df.shape}", flush=True)
    print(f"Null threshold: {args.null_threshold}", flush=True)

    # ---- Storey FDR ----
    fdr_df, pi0_s, sig_dfs = compute_storey_fdr(
        obs_df,
        null_threshold=args.null_threshold,
        fdr_levels=(0.01, 0.05, 0.10),
    )

    fdr_df.to_csv(f"{pfx}/fdr_storey.tsv.gz", sep='\t')
    pi0_s.to_csv( f"{pfx}/pi0.tsv",           sep='\t', header=True)

    print("\nPi0 per tissue (lower = more tissue-specific genes):")
    print(pi0_s.sort_values().to_string())

    for level, sig_df in sig_dfs.items():
        tag   = str(level).replace('.', 'p')
        sig_df.to_csv(f"{pfx}/sig_{tag}.tsv.gz", sep='\t')
        n_sig = sig_df.any(axis=1).sum()
        print(f"FDR<={level}: {n_sig} genes significant in at least one tissue",
              flush=True)

    print("Done.", flush=True)


if __name__ == '__main__':
    main()