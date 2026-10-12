#!/usr/bin/env python3
"""
get_pval_per_gene.py
====================
Fit a per-tissue empirical null N(mu, sigma) to the bulk of isoform SSMDs,
then compute:
 
  Per isoform:
    Z = (SSMD - mu) / sigma      [stored in zscore_isoform.tsv.gz]
 
  Per gene (extreme value of m isoforms):
    p = 1 - Phi(max_Z)^m        [stored in pval.tsv.gz]
    FDR via BH + Storey pi0     [stored in fdr.tsv.gz]
 
  The isoform driving the signal (argmax Z per gene per tissue) is tracked,
  and its median_diff is recorded as the effect size for that gene/tissue call.
 
The null is fitted per tissue to absorb sample-size and biological background
differences. All outputs are directly comparable across tissues and species.
 
Outputs
-------
  null_fit.tsv              : tissue-level null parameters (mu, sigma, pi0)
  zscore_isoform.tsv.gz     : isoform x tissue standardised Z-scores
  obs_max_ssmd.tsv.gz       : gene x tissue max raw SSMD (of argmax isoform)
  obs_max_z.tsv.gz          : gene x tissue max Z-score
  obs_median_diff.tsv.gz    : gene x tissue median diff of the argmax isoform
  argmax_isoform.tsv.gz     : gene x tissue isoform ID driving the call
  n_isoforms.tsv.gz         : gene x tissue number of isoforms tested
  pval.tsv.gz               : gene x tissue p-values
  fdr.tsv.gz                : gene x tissue BH-FDR (Storey-corrected)
  sig_0p01.tsv.gz           : boolean FDR <= 0.01
  sig_0p05.tsv.gz           : boolean FDR <= 0.05
  sig_0p10.tsv.gz           : boolean FDR <= 0.10
 
Usage
-----
    python get_pval_per_gene.py \\
        --ssmd_path        /path/to/ssmd.tsv.gz \\
        --median_diff_path /path/to/median_diff.tsv.gz \\
        [--out_dir         ./results] \\
        [--trim_lo         0.05] \\
        [--trim_hi         0.90] \\
        [--gid_col         gid]
"""
 
import argparse
import os
 
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
 
 
# ---------------------------------------------------------------------------
# Null fitting
# ---------------------------------------------------------------------------
 
def fit_null(ssmd_vals: np.ndarray,
             trim_lo: float = 0.05,
             trim_hi: float = 0.90):
    """
    Fit N(mu, sigma) to the central mass of isoform SSMDs.
    Trims left tail (tissue-depleted) and right tail (true signal).
    Returns (nan, nan) if insufficient data.
    """
    vals = ssmd_vals[~np.isnan(ssmd_vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    lo   = np.quantile(vals, trim_lo)
    hi   = np.quantile(vals, trim_hi)
    core = vals[(vals >= lo) & (vals <= hi)]
    if len(core) < 10:
        return np.nan, np.nan
    mu, sigma = norm.fit(core)
    return float(mu), float(sigma)
 
 
# ---------------------------------------------------------------------------
# Storey pi0
# ---------------------------------------------------------------------------
 
def storey_pi0(pvals: np.ndarray, lambda_: float = 0.5) -> float:
    pvals = pvals[~np.isnan(pvals)]
    if len(pvals) == 0:
        return 1.0
    return float(min((pvals > lambda_).sum() / (len(pvals) * (1 - lambda_)), 1.0))


# ---------------------------------------------------------------------------
# BH FDR with Storey pi0
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray, pi0: float = 1.0) -> np.ndarray:
    n     = len(pvals)
    fdr   = np.full(n, np.nan)
    valid = ~np.isnan(pvals)
    idx   = np.where(valid)[0]
    p     = pvals[idx]
    m     = len(p)
    if m == 0:
        return fdr

    order   = np.argsort(p)
    ranks   = np.argsort(order) + 1
    fdr_raw = np.minimum(pi0 * p * m / ranks, 1.0)

    # monotonicity: step down from largest rank
    fdr_ord = fdr_raw[order]
    for i in range(m - 2, -1, -1):
        fdr_ord[i] = min(fdr_ord[i], fdr_ord[i + 1])
    fdr[idx] = fdr_ord[np.argsort(order)]
    return fdr


# ---------------------------------------------------------------------------
# Per-gene p-value: extreme value of m N(0,1) variables
# ---------------------------------------------------------------------------

def gene_pval(max_z: float, m: int) -> float:
    """
    p = 1 - Phi(max_z)^m

    Uses log space for numerical stability. Returns minimum positive float
    when p underflows to 0 (extremely significant genes).
    """
    if np.isnan(max_z) or m < 1:
        return np.nan
    log_cdf  = norm.logcdf(max_z)
    log_surv = np.log1p(-np.exp(m * log_cdf))
    if np.isnan(log_surv) or np.isinf(log_surv):
        return np.finfo(float).tiny
    return float(np.exp(log_surv))


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all(ssmd_df: pd.DataFrame,
                mdiff_df: pd.DataFrame | None = None,
                gid_col: str = 'gid',
                trim_lo: float = 0.05,
                trim_hi: float = 0.90,
                fdr_levels: tuple = (0.01, 0.05, 0.10)):

    tissues  = [c for c in ssmd_df.columns if c != gid_col]
    gene_ids = ssmd_df[gid_col].values
    genes    = np.unique(gene_ids)
    n_genes  = len(genes)
    gene_pos = {g: i for i, g in enumerate(genes)}

    # ------------------------------------------------------------------ #
    # Step 1: fit null per tissue, compute per-isoform Z-scores
    # ------------------------------------------------------------------ #
    null_records = []
    zscore_arr   = np.full((len(ssmd_df), len(tissues)), np.nan, dtype=np.float32)
    valid_tissues = []

    print("Fitting null distributions …", flush=True)
    for t_idx, tissue in enumerate(tissues):
        vals      = ssmd_df[tissue].values.astype(float)
        mu, sigma = fit_null(vals, trim_lo, trim_hi)
        if np.isnan(mu):
            print(f"  {tissue:30s}: skipping — insufficient data", flush=True)
            continue
        zscore_arr[:, t_idx] = (vals - mu) / sigma
        null_records.append({'tissue': tissue, 'mu': mu, 'sigma': sigma})
        valid_tissues.append(tissue)
        print(f"  {tissue:30s}: mu={mu:.4f}  sigma={sigma:.4f}", flush=True)

    null_df    = pd.DataFrame(null_records).set_index('tissue')
    zscore_iso = pd.DataFrame(zscore_arr, index=ssmd_df.index, columns=tissues)
    zscore_iso.insert(0, gid_col, gene_ids)

    # ------------------------------------------------------------------ #
    # Step 2: per-gene max Z, argmax isoform, p-value, median diff
    # ------------------------------------------------------------------ #
    obs_ssmd_arr  = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    obs_z_arr     = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    obs_mdiff_arr = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    niso_arr      = np.zeros((n_genes, len(tissues)),         dtype=np.int32)
    pval_arr      = np.full((n_genes, len(tissues)), np.nan, dtype=np.float64)
    # store argmax isoform ID as object array
    argmax_arr    = np.full((n_genes, len(tissues)), None,   dtype=object)

    print("\nComputing per-gene p-values …", flush=True)
    for gene, grp in tqdm(ssmd_df.groupby(gid_col), desc='Genes'):
        g_idx = gene_pos[gene]
        for t_idx, tissue in enumerate(tissues):
            if tissue not in null_df.index:
                continue
            raw = grp[tissue].dropna()
            m   = len(raw)
            if m == 0:
                continue

            mu    = null_df.loc[tissue, 'mu']
            sigma = null_df.loc[tissue, 'sigma']
            zvals = (raw.values - mu) / sigma
            am    = np.argmax(zvals)          # argmax within this gene's isoforms
            max_z = zvals[am]
            argmax_pid = raw.index[am]        # isoform ID with max Z

            niso_arr    [g_idx, t_idx] = m
            obs_ssmd_arr[g_idx, t_idx] = raw.values[am]
            obs_z_arr   [g_idx, t_idx] = max_z
            pval_arr    [g_idx, t_idx] = gene_pval(max_z, m)
            argmax_arr  [g_idx, t_idx] = argmax_pid

            # median diff of the argmax isoform
            if mdiff_df is not None and argmax_pid in mdiff_df.index:
                obs_mdiff_arr[g_idx, t_idx] = mdiff_df.loc[argmax_pid, tissue] \
                    if tissue in mdiff_df.columns else np.nan

    # ------------------------------------------------------------------ #
    # Step 3: FDR per tissue
    # ------------------------------------------------------------------ #
    fdr_arr = np.full_like(pval_arr, np.nan)

    print("\nComputing FDR …", flush=True)
    for t_idx, tissue in enumerate(tissues):
        if tissue not in null_df.index:
            continue
        pv  = pval_arr[:, t_idx]
        pi0 = storey_pi0(pv)
        null_df.loc[tissue, 'pi0'] = pi0
        fdr_arr[:, t_idx] = bh_fdr(pv, pi0)

        n_valid = (~np.isnan(pv)).sum()
        for level in fdr_levels:
            n_sig = (fdr_arr[:, t_idx] <= level).sum()
            print(f"  {tissue:30s} FDR<={level:.2f}: "
                  f"{n_sig}/{n_valid} ({100*n_sig/n_valid:.1f}%)",
                  flush=True)

    # ------------------------------------------------------------------ #
    # Assemble DataFrames
    # ------------------------------------------------------------------ #
    def _df(arr, dtype=None):
        d = pd.DataFrame(arr, index=genes, columns=tissues)
        d.index.name = gid_col
        return d

    argmax_df = pd.DataFrame(argmax_arr, index=genes, columns=tissues)
    argmax_df.index.name = gid_col

    return (null_df,
            zscore_iso,
            _df(obs_ssmd_arr),
            _df(obs_z_arr),
            _df(obs_mdiff_arr),
            argmax_df,
            _df(niso_arr),
            _df(pval_arr),
            _df(fdr_arr),
            {level: (_df(fdr_arr) <= level) for level in fdr_levels})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--ssmd_path',        required=True,
                   help='Per-isoform SSMD table (ssmd.tsv.gz).')
    p.add_argument('--median_diff_path', default=None,
                   help='Per-isoform median diff table (median_diff.tsv.gz). '
                        'If provided, records median diff of the argmax isoform.')
    p.add_argument('--out_dir',          default='.')
    p.add_argument('--gid_col',          default='gid')
    p.add_argument('--trim_lo',          default=0.05, type=float)
    p.add_argument('--trim_hi',          default=0.90, type=float)
    p.add_argument('--fdr_levels',       default=[0.01, 0.05, 0.10],
                   nargs='+', type=float)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pfx = args.out_dir

    print(f"Loading SSMD from {args.ssmd_path}", flush=True)
    ssmd_df = pd.read_csv(args.ssmd_path, sep='\t', index_col=0)
    if args.gid_col not in ssmd_df.columns:
        raise ValueError(f"Column '{args.gid_col}' not found. "
                         f"Columns: {ssmd_df.columns.tolist()}")

    mdiff_df = None
    if args.median_diff_path:
        print(f"Loading median diff from {args.median_diff_path}", flush=True)
        mdiff_df = pd.read_csv(args.median_diff_path, sep='\t', index_col=0)
        # drop gid column if present — index is pid
        if args.gid_col in mdiff_df.columns:
            mdiff_df = mdiff_df.drop(columns=[args.gid_col])

    print(f"Shape: {ssmd_df.shape} — "
          f"{ssmd_df[args.gid_col].nunique()} genes, "
          f"{len(ssmd_df.columns)-1} tissues\n", flush=True)

    (null_df, zscore_iso,
     obs_ssmd_df, obs_z_df, obs_mdiff_df,
     argmax_df, niso_df,
     pval_df, fdr_df, sig_dfs) = compute_all(
        ssmd_df,
        mdiff_df=mdiff_df,
        gid_col=args.gid_col,
        trim_lo=args.trim_lo,
        trim_hi=args.trim_hi,
        fdr_levels=tuple(args.fdr_levels),
    )

    null_df.to_csv(      f"{pfx}/null_fit.tsv",              sep='\t')
    zscore_iso.to_csv(   f"{pfx}/zscore_isoform.tsv.gz",     sep='\t')
    obs_ssmd_df.to_csv(  f"{pfx}/obs_max_ssmd.tsv.gz",       sep='\t')
    obs_z_df.to_csv(     f"{pfx}/obs_max_z.tsv.gz",          sep='\t')
    obs_mdiff_df.to_csv( f"{pfx}/obs_median_diff.tsv.gz",    sep='\t')
    argmax_df.to_csv(    f"{pfx}/argmax_isoform.tsv.gz",     sep='\t')
    niso_df.to_csv(      f"{pfx}/n_isoforms.tsv.gz",         sep='\t')
    pval_df.to_csv(      f"{pfx}/pval.tsv.gz",               sep='\t')
    fdr_df.to_csv(       f"{pfx}/fdr.tsv.gz",                sep='\t')

    for level, sig_df in sig_dfs.items():
        tag = str(level).replace('.', 'p')
        sig_df.to_csv(f"{pfx}/sig_{tag}.tsv.gz", sep='\t')

    print(f"\nNull fit per tissue:")
    print(null_df.to_string())
    print(f"\nOutputs written to {pfx}/")
    print("Done.", flush=True)


if __name__ == '__main__':
    main()