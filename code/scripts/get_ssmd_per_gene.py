#!/usr/bin/env python3
"""
run_ssmd_pipeline.py
====================
End-to-end SSMD pipeline:
  1. Write splicing-efficiency-corrected EL to TSV
  2. Compute SSMD and median-diff; save as TSV tables
  3. Compute empirical-FDR thresholds at 0.10, 0.05, 0.01; save as TSV tables

Usage
-----
    python run_ssmd_pipeline.py \\
        --species  Human \\
        --el_path  /path/to/EL.tsv \\
        --meta_path /path/to/metadata.tsv \\
        [--annot_path /path/to/annot.tsv] \\
        [--out_dir   ./results] \\
        [--pid_col   pid] \\
        [--gid_col   gid] \\
        [--k         100] \\
        [--score_type ssmd]

Positional tissue lists (hardcoded per species):
  Human  : Adipose_Subcutaneous, Artery_Aorta, Brain, Cerebellum,
            Breast_Mammary_Tissue, Heart, Liver, Lung, Muscle_Skeletal,
            Pancreas, Skin, Testis, Whole_Blood
  Cattle : Blood, Brain, Colon, Heart, Liver, Lung, Mammary_Gland,
            Muscle, Skin, Subcutaneous_Adipose, Testis
  Other  : Cerebellum, Heart, Kidney, Liver, Testis, Brain, Ovary
"""

import argparse
import os
import sys

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

    else:  # Macaque, Mouse, Chicken, etc.
        mapping = {
            'Brain':      (['Brain'],      cardoso_non_brain),
            'Cerebellum': (['Cerebellum'], cardoso_non_cereb),
            'Heart':      (['Heart'],  ['Brain', 'Cerebellum', 'Kidney', 'Liver', 'Testis', 'Ovary']),
            'Liver':      (['Liver'],  ['Brain', 'Cerebellum', 'Kidney', 'Heart', 'Testis', 'Ovary']),
            'Testis':     (['Testis'], ['Brain', 'Cerebellum', 'Kidney', 'Heart', 'Liver', 'Ovary']),
        }

    if tissue not in mapping:
        # Graceful fallback: treat tissue as its own group vs everything else
        tl = [tissue]
        ot = [x for x in tissues_complete if x != tissue]
        return {'tissue_list': tl, 'other_tissues': ot}

    tl, ot = mapping[tissue]
    return {'tissue_list': tl, 'other_tissues': ot}


# ---------------------------------------------------------------------------
# Vectorised score helpers
# ---------------------------------------------------------------------------

def _isoform_stats(mat_cols):
    not_nan = (~np.isnan(mat_cols)).astype(np.float32)
    mat0    = np.where(np.isnan(mat_cols), 0.0, mat_cols)
    count   = not_nan.sum(axis=1)
    mean    = mat0.sum(axis=1) / np.where(count > 0, count, np.nan)
    sq_dev  = np.where(np.isnan(mat_cols), 0.0, (mat_cols - mean[:, None]) ** 2)
    var     = sq_dev.sum(axis=1) / np.where(count > 1, count - 1, np.nan)
    return mean, var


def _vectorised_ssmd(mat, gene_idx, ti, oi):
    tc_mean, tc_var = _isoform_stats(mat[:, ti])
    oc_mean, oc_var = _isoform_stats(mat[:, oi])
    mean_diff = tc_mean - oc_mean
    with np.errstate(invalid='ignore', divide='ignore'):
        ssmd = mean_diff / np.sqrt(tc_var + oc_var)
    abs_ssmd = np.abs(ssmd)
    boundaries = np.where(np.diff(gene_idx))[0] + 1
    starts     = np.concatenate([[0], boundaries])
    safe = np.where(np.isnan(abs_ssmd), -np.inf, abs_ssmd)
    gene_max = np.maximum.reduceat(safe, starts)
    gene_max[gene_max == -np.inf] = np.nan
    return gene_max


def _vectorised_median_diff(mat, gene_idx, ti, oi, starts):
    n_genes = len(starts)
    ends    = np.concatenate([starts[1:], [mat.shape[0]]])
    result  = np.empty(n_genes, dtype=np.float32)
    for g in range(n_genes):
        sl    = slice(starts[g], ends[g])
        tc    = mat[sl][:, ti]
        oc    = mat[sl][:, oi]
        mdiff = np.abs(np.nanmedian(tc, axis=1) - np.nanmedian(oc, axis=1))
        result[g] = np.nanmean(mdiff)
    return result


# ---------------------------------------------------------------------------
# Splicing-efficiency correction
# ---------------------------------------------------------------------------

def _apply_correction(group, gene, annot, sample_cols):
    select_transcripts = list(
        annot.loc[(annot.gene == gene) & (~annot.efficiency_factor)].transcript.unique()
    )
    if len(select_transcripts) < 2:
        return None
    mask = group['pid'].isin(select_transcripts)
    sub  = group.loc[mask, sample_cols]
    remain_sum = sub.T.dropna().T.sum(axis=0)
    if remain_sum.quantile(0.1) < 0.25:
        return None
    corrected = sub.div(remain_sum, axis=1)
    result = group.loc[mask].copy()
    result[sample_cols] = corrected
    return result


def write_corrected_EL(EL, annot, out_path, gid_col='gid', pid_col='pid', chunksize=500):
    if gid_col not in EL.columns:
        EL = EL.assign(**{gid_col: EL[pid_col].str.split('.').str[0]})

    forced_meta = {pid_col, gid_col}
    sample_cols = [c for c in EL.select_dtypes(include=[np.number]).columns
                   if c not in forced_meta]

    EL_sorted      = EL.sort_values(gid_col)
    out_path       = str(out_path)
    header_written = False
    buf            = []
    buf_rows       = 0

    def _flush(fh, rows, write_header):
        pd.concat(rows, ignore_index=True).to_csv(fh, sep='\t', index=False, header=write_header)

    with open(out_path, 'w') as fh:
        for gene, group in tqdm(EL_sorted.groupby(gid_col, sort=False),
                                desc='Writing corrected EL'):
            corrected = _apply_correction(group, gene, annot, sample_cols)
            out_rows  = corrected if corrected is not None else group
            buf.append(out_rows)
            buf_rows += len(out_rows)
            if buf_rows >= chunksize:
                _flush(fh, buf, not header_written)
                header_written = True
                buf      = []
                buf_rows = 0
        if buf:
            _flush(fh, buf, not header_written)

    return out_path


# ---------------------------------------------------------------------------
# Matrix preparation
# ---------------------------------------------------------------------------

def _prepare_matrix(EL_or_path, sample_cols, gid_col, pid_col):
    if isinstance(EL_or_path, str):
        EL_sorted = pd.read_csv(EL_or_path, sep='\t')
        if gid_col not in EL_sorted.columns:
            EL_sorted[gid_col] = EL_sorted[pid_col].str.split('.').str[0]
        EL_sorted = EL_sorted.sort_values(gid_col)
    else:
        EL = EL_or_path
        if gid_col not in EL.columns:
            EL = EL.assign(**{gid_col: EL[pid_col].str.split('.').str[0]})
        EL_sorted = EL.sort_values(gid_col)

    gene_labels, gene_idx = np.unique(EL_sorted[gid_col].values, return_inverse=True)
    mat    = EL_sorted[sample_cols].values.astype(np.float32)
    starts = np.concatenate([[0], np.where(np.diff(gene_idx))[0] + 1])
    return mat, gene_labels, gene_idx, starts


# ---------------------------------------------------------------------------
# SSMD computation
# ---------------------------------------------------------------------------

def get_species_ssmd(EL_or_path, tissues, species, metadata, gid_col='gid', pid_col='pid'):
    if isinstance(EL_or_path, str):
        header_cols = pd.read_csv(EL_or_path, sep='\t', nrows=0).columns.tolist()
    else:
        header_cols = EL_or_path.columns.tolist()

    meta_cols   = {pid_col, gid_col, 'chrom', 'start', 'end', 'strand', 'score', 'name'}
    sample_cols = [c for c in header_cols if c not in meta_cols and c in metadata.index]
    col_pos     = {c: i for i, c in enumerate(sample_cols)}

    tissues_complete = metadata['Tissue'].unique().tolist()

    tissue_indices = {}
    for tissue in tissues:
        cfg = get_tissue_config(species, tissue, tissues_complete)
        ti  = [col_pos[c] for c in
               metadata.loc[metadata['Tissue'].isin(cfg['tissue_list'])].index
               if c in col_pos]
        oi  = [col_pos[c] for c in
               metadata.loc[metadata['Tissue'].isin(cfg['other_tissues'])].index
               if c in col_pos]
        tissue_indices[tissue] = (ti, oi)

    mat, gene_labels, gene_idx, starts = _prepare_matrix(
        EL_or_path, sample_cols, gid_col, pid_col
    )

    n_genes   = len(gene_labels)
    ssmd_arr  = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)
    mdiff_arr = np.full((n_genes, len(tissues)), np.nan, dtype=np.float32)

    for t_idx, tissue in enumerate(tqdm(tissues, desc=f'{species} SSMD')):
        ti, oi = tissue_indices[tissue]
        if not ti or not oi:
            continue
        ssmd_arr[:, t_idx]  = _vectorised_ssmd(mat, gene_idx, ti, oi)
        mdiff_arr[:, t_idx] = _vectorised_median_diff(mat, gene_idx, ti, oi, starts)

    ssmd_df              = pd.DataFrame(ssmd_arr,  index=gene_labels, columns=tissues)
    mdiff_df             = pd.DataFrame(mdiff_arr, index=gene_labels, columns=tissues)
    ssmd_df.index.name   = 'gene'
    mdiff_df.index.name  = 'gene'

    return {'ssmd': ssmd_df, 'median_diff': mdiff_df, 'sample_cols': sample_cols}


# ---------------------------------------------------------------------------
# Precomputed gene-level stats cache (used by bootstrap)
# ---------------------------------------------------------------------------

def _precompute_gene_stats(mat, gene_idx, starts):
    """
    Reduce raw isoform matrix to gene-level mean and variance once.

    Rather than scanning (n_isoforms x n_samples) on every bootstrap draw,
    we produce (n_genes x n_samples) arrays here using np.add.reduceat.
    Each bootstrap iteration then only needs O(n_genes) work to compute SSMD.

    Returns
    -------
    gene_mean : float32 (n_genes, n_samples)
    gene_var  : float32 (n_genes, n_samples)  — NaN where isoform count < 2
    """
    not_nan  = (~np.isnan(mat)).astype(np.float32)
    mat0     = np.where(np.isnan(mat), 0.0, mat)

    gene_sum   = np.add.reduceat(mat0,    starts, axis=0)
    gene_count = np.add.reduceat(not_nan, starts, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        gene_mean = gene_sum / gene_count

    mean_per_iso = gene_mean[gene_idx]
    sq_dev       = np.where(np.isnan(mat), 0.0, (mat0 - mean_per_iso) ** 2)
    gene_sq_sum  = np.add.reduceat(sq_dev, starts, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        gene_var = gene_sq_sum / np.where(gene_count > 1, gene_count - 1, np.nan)

    return gene_mean.astype(np.float32), gene_var.astype(np.float32)


def _ssmd_from_gene_stats(gene_mean, gene_var, ti, oi):
    """
    Compute per-gene absolute SSMD from cached gene-level stats.
    O(n_genes x |ti+oi|) — no raw matrix scan needed.
    """
    tc_mean = np.nanmean(gene_mean[:, ti], axis=1)
    oc_mean = np.nanmean(gene_mean[:, oi], axis=1)
    tc_var  = np.nanmean(gene_var [:, ti], axis=1)
    oc_var  = np.nanmean(gene_var [:, oi], axis=1)
    mean_diff = tc_mean - oc_mean
    with np.errstate(invalid='ignore', divide='ignore'):
        ssmd = mean_diff / np.sqrt(tc_var + oc_var)
    return np.abs(ssmd)



def filter_EL(EL, q = 0.4):

    EL_cols = EL.columns[6:]
    
    gene_list = []
    for gene, df in EL.groupby('gid'):
        gene_na = df[EL_cols].isna().mean(axis=1).max()
        if gene_na <= q:
            new_na = (df[EL_cols].var(axis=0) < 1e-10).mean()
            if new_na <= q:
                gene_list.append(gene)

    EL = EL.loc[EL.gid.isin(gene_list)]

    return EL


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--species',    required=True,
                   help='Species label, e.g. Human, Cattle, Mouse, Macaque …')
    p.add_argument('--el_path',   required=True,
                   help='Path to raw EL TSV (rows = isoforms, cols = samples).')
    p.add_argument('--meta_path', required=True,
                   help='Path to metadata TSV. Must have a "Tissue" column; '
                        'index = sample IDs (first column).')
    p.add_argument('--annot_path', default=None,
                   help='Path to annotation TSV [gene, transcript, efficiency_factor]. '
                        'If omitted, splicing-efficiency correction is skipped.')
    p.add_argument('--out_dir',   default='.',
                   help='Output directory (default: current directory).')
    p.add_argument('--pid_col',   default='pid',
                   help='Isoform-ID column name (default: pid).')
    p.add_argument('--gid_col',   default='gid',
                   help='Gene-ID column name (default: gid, derived from pid if absent).')
    # p.add_argument('--k',         default=100, type=int,
    #                help='Bootstrap iterations per tissue per FDR level (default: 100).')
    # p.add_argument('--score_type', default='ssmd', choices=['ssmd', 'median_diff'],
    #                help='Score used for threshold estimation (default: ssmd).')
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sp  = args.species
    pfx = args.out_dir

    # ---- tissue list -------------------------------------------------------
    tissues = TISSUES.get(sp, TISSUES_OTHER)
    print(f"[{sp}] Tissues: {tissues}", flush=True)

    # ---- load EL -----------------------------------------------------------
    print(f"[{sp}] Loading EL from {args.el_path}", flush=True)
    EL = pd.read_csv(args.el_path, sep='\t')
    EL.index = EL.pid

    EL = filter_EL(EL)

    # ---- load metadata -----------------------------------------------------
    print(f"[{sp}] Loading metadata from {args.meta_path}", flush=True)

    if sp == 'Human':
        metadata = pd.read_csv(args.meta_path, sep='\t')
        metadata = metadata[['entity:sample_id', 'tissue_id']]
        metadata.columns = ['Sample', 'Tissue']
        metadata.index = metadata.Sample
    elif sp == 'Cattle':
        metadata = pd.read_csv(args.meta_path, sep='\t', index_col=0)
        metadata.columns = ['Tissue', 'counts']
    else:
        metadata = pd.read_csv(args.meta_path, sep='\t', index_col=0)

    metadata = metadata.loc[metadata.index.intersection(EL.columns)]
    tissues_complete = metadata['Tissue'].unique().tolist()

    # ---- step 1: splicing-efficiency correction ----------------------------
    if args.annot_path:
        print(f"[{sp}] Loading annotation from {args.annot_path}", flush=True)
        annot = pd.read_csv(args.annot_path, sep='\t')
        annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])

        efficiency_factor = []
        for idx, row in annot.iterrows():
            appris_ref    = row.appris_ref
            aver          = appris_ref.split('/')
            if 'RI:' in appris_ref:
                ri_factor     = [x for x in aver if x.startswith('RI:')][0]
                ri_numbers    = int(ri_factor.split(':')[1])
                appris_length = int(row.appris_transcript_length)
                if ((ri_numbers <= 2) and (appris_length >= 5)) or \
                   ((ri_numbers <= 1) and (appris_length >= 3)):
                    efficiency_factor.append(False)
                else:
                    efficiency_factor.append(True)
            else:
                efficiency_factor.append(False)

        annot['efficiency_factor'] = efficiency_factor

        corrected_path = f"{pfx}/EL_corrected.tsv"
        print(f"[{sp}] Writing corrected EL → {corrected_path}", flush=True)
        write_corrected_EL(EL, annot, corrected_path,
                           gid_col=args.gid_col, pid_col=args.pid_col)
        el_source = corrected_path
    else:
        print(f"[{sp}] No annotation provided — skipping splicing-efficiency correction.",
              flush=True)
        el_source = EL

    # ---- step 2: SSMD / median-diff ----------------------------------------
    print(f"[{sp}] Computing SSMD and median-diff …", flush=True)
    results = get_species_ssmd(
        el_source, tissues, sp, metadata,
        gid_col=args.gid_col, pid_col=args.pid_col,
    )

    ssmd_path  = f"{pfx}/ssmd.tsv.gz"
    mdiff_path = f"{pfx}/median_diff.tsv.gz"
    results['ssmd'].to_csv(ssmd_path,  sep='\t')
    results['median_diff'].to_csv(mdiff_path, sep='\t')
    print(f"[{sp}] SSMD saved        → {ssmd_path}", flush=True)
    print(f"[{sp}] Median-diff saved → {mdiff_path}", flush=True)

    print(f"[{sp}] Done.", flush=True)


if __name__ == '__main__':
    main()