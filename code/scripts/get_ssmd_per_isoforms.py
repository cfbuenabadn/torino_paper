#!/usr/bin/env python3
"""
get_ssmd_per_isoforms.py
====================
Compute per-isoform SSMD and median-diff from a pre-filtered,
splicing-efficiency-corrected EL matrix.

Usage
-----
    python run_ssmd_pipeline.py \\
        --species   Human \\
        --el_path   /path/to/EL_corrected.tsv \\
        --meta_path /path/to/metadata.tsv \\
        [--out_dir  ./results] \\
        [--pid_col  pid] \\
        [--gid_col  gid]

Tissue lists (hardcoded per species):
  Human  : Adipose_Subcutaneous, Artery_Aorta, Brain, Cerebellum,
            Breast_Mammary_Tissue, Heart, Liver, Lung, Muscle_Skeletal,
            Pancreas, Skin, Testis, Whole_Blood
  Cattle : Blood, Brain, Colon, Heart, Liver, Lung, Mammary_Gland,
            Muscle, Skin, Subcutaneous_Adipose, Testis
  Other  : Cerebellum, Heart, Kidney, Liver, Testis, Brain, Ovary
"""

import argparse
import os

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
            'Brain':              (['CNS'],                [x for x in tissues_complete if x != 'CNS']),
            'Heart':              (['Heart'],              [x for x in tissues_complete if x != 'Heart']),
            'Liver':              (['Liver'],              [x for x in tissues_complete if x != 'Liver']),
            'Testis':             (['Testis'],             [x for x in tissues_complete if x != 'Testis']),
            'Muscle':             (['Muscle'],             [x for x in tissues_complete if x != 'Muscle']),
            'Blood':              (['Blood'],              [x for x in tissues_complete if x != 'Blood']),
            'Colon':              (['Colon'],              [x for x in tissues_complete if x != 'Colon']),
            'Lung':               (['Lung'],               [x for x in tissues_complete if x != 'Lung']),
            'Mammary_Gland':      (['Mammary_Gland'],      [x for x in tissues_complete if x != 'Mammary_Gland']),
            'Skin':               (['Skin'],               [x for x in tissues_complete if x != 'Skin']),
            'Subcutaneous_Adipose': (['Subcutaneous_Adipose'],
                                     [x for x in tissues_complete if x != 'Subcutaneous_Adipose']),
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
        tl = [tissue]
        ot = [x for x in tissues_complete if x != tissue]
        return {'tissue_list': tl, 'other_tissues': ot}

    tl, ot = mapping[tissue]
    return {'tissue_list': tl, 'other_tissues': ot}


# ---------------------------------------------------------------------------
# Vectorised score helpers — per isoform, no gene-level aggregation
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
    """Return signed per-isoform SSMD (n_isoforms,)."""
    tc_mean, tc_var = _isoform_stats(mat[:, ti])
    oc_mean, oc_var = _isoform_stats(mat[:, oi])
    mean_diff = tc_mean - oc_mean
    with np.errstate(invalid='ignore', divide='ignore'):
        ssmd = mean_diff / np.sqrt(tc_var + oc_var)
    return ssmd  # signed; take abs() downstream if needed


def _vectorised_median_diff(mat, ti, oi):
    """Return signed per-isoform median difference (n_isoforms,)."""
    tc_median = np.nanmedian(mat[:, ti], axis=1)
    oc_median = np.nanmedian(mat[:, oi], axis=1)
    return tc_median - oc_median  # signed


# ---------------------------------------------------------------------------
# Matrix preparation
# ---------------------------------------------------------------------------

def _prepare_matrix(EL, sample_cols, gid_col, pid_col):
    if gid_col not in EL.columns:
        EL = EL.copy()
        EL[gid_col] = EL[pid_col].str.split('.').str[0]

    EL = EL.reset_index(drop=True).sort_values([gid_col, pid_col])

    isoform_labels = EL[pid_col].values
    gene_labels    = EL[gid_col].values
    mat            = EL[sample_cols].values.astype(np.float32)

    return mat, isoform_labels, gene_labels


# ---------------------------------------------------------------------------
# SSMD computation — returns per-isoform tables
# ---------------------------------------------------------------------------

def get_species_ssmd(EL, tissues, species, metadata, gid_col='gid', pid_col='pid'):
    # meta_cols   = {pid_col, gid_col, 'chrom', 'start', 'end', 'strand', 'score', 'name'}
    sample_cols = EL.columns[6:] #[c for c in EL.columns if c not in meta_cols and c in metadata.index]
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

    mat, isoform_labels, gene_labels = _prepare_matrix(EL, sample_cols, gid_col, pid_col)

    n_isoforms = len(isoform_labels)
    ssmd_arr   = np.full((n_isoforms, len(tissues)), np.nan, dtype=np.float32)
    mdiff_arr  = np.full((n_isoforms, len(tissues)), np.nan, dtype=np.float32)

    for t_idx, tissue in enumerate(tqdm(tissues, desc=f'{species} SSMD')):
        ti, oi = tissue_indices[tissue]
        if not ti or not oi:
            continue
        ssmd_arr [:, t_idx] = _vectorised_ssmd(mat, ti, oi)
        mdiff_arr[:, t_idx] = _vectorised_median_diff(mat, ti, oi)

    idx = pd.Index(isoform_labels, name=pid_col)

    ssmd_df  = pd.DataFrame(ssmd_arr,  index=idx, columns=tissues)
    mdiff_df = pd.DataFrame(mdiff_arr, index=idx, columns=tissues)

    # keep gene column as the first column for easy groupby downstream
    ssmd_df.insert(0,  gid_col, gene_labels)
    mdiff_df.insert(0, gid_col, gene_labels)

    return {'ssmd': ssmd_df, 'median_diff': mdiff_df}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--species',    required=True,
                   help='Species label, e.g. Human, Cattle, Mouse, Macaque …')
    p.add_argument('--el_path',   required=True,
                   help='Path to pre-filtered, corrected EL TSV (isoforms × samples).')
    p.add_argument('--meta_path', required=True,
                   help='Path to metadata TSV. Must have a "Tissue" column; '
                        'index = sample IDs (first column).')
    p.add_argument('--out_dir',   default='.',
                   help='Output directory (default: current directory).')
    p.add_argument('--pid_col',   default='pid',
                   help='Isoform-ID column name (default: pid).')
    p.add_argument('--gid_col',   default='gid',
                   help='Gene-ID column name (default: gid, derived from pid if absent).')
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
    EL.index = EL.pid

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

    print(f"[{sp}] Computing per-isoform SSMD and median-diff …", flush=True)
    results = get_species_ssmd(
        EL, tissues, sp, metadata,
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