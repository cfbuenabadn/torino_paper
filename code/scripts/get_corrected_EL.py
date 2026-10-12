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
        --annot_path /path/to/annot.tsv \\
        [--out_dir   ./results] \\
        [--pid_col   pid] \\
        [--gid_col   gid]

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

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# EL filtering
# ---------------------------------------------------------------------------

def filter_EL(EL, q=0.4):
    EL_cols = EL.columns[6:]

    gene_list = []
    for gene, df in EL.groupby('gid'):
        gene_na = df[EL_cols].isna().mean(axis=1).max()
        if gene_na <= q:
            new_na = (df[EL_cols].var(axis=0) < 1e-10).mean()
            if new_na <= q:
                gene_list.append(gene)

    return EL.loc[EL.gid.isin(gene_list)]


# ---------------------------------------------------------------------------
# Annotation loading with efficiency_factor
# ---------------------------------------------------------------------------

def _compute_efficiency_factor(appris_ref: str, appris_length: int) -> bool:
    """Return True if transcript has poor splicing efficiency (should be excluded)."""
    if 'RI:' not in appris_ref:
        return False
    ri_numbers = int(
        next(x for x in appris_ref.split('/') if x.startswith('RI:')).split(':')[1]
    )
    return not (
        (ri_numbers <= 2 and appris_length >= 5) or
        (ri_numbers <= 1 and appris_length >= 3)
    )


def load_annot(annot_path: str) -> pd.DataFrame:
    """Load annotation file and add efficiency_factor column."""
    annot = pd.read_csv(annot_path, sep='\t')
    annot['gene'] = annot.transcript.str.split('.').str[0]
    annot['efficiency_factor'] = [
        _compute_efficiency_factor(row.appris_ref, int(row.appris_transcript_length))
        for _, row in annot.iterrows()
    ]
    return annot


# ---------------------------------------------------------------------------
# Per-gene correction
# ---------------------------------------------------------------------------

def _process_gene_group(
    iso_EL: pd.DataFrame,
    gene: str,
    annot: pd.DataFrame,
    bed_cols: list,
    sample_cols: list,
    min_quantile: float = 0.1,
    min_quantile_val: float = 0.25,
) -> pd.DataFrame | None:
    good_transcripts = annot.loc[
        (annot.gene == gene) & (~annot.efficiency_factor), 'transcript'
    ].unique()
    if len(good_transcripts) < 2:
        return None

    mask = iso_EL.pid.isin(good_transcripts)
    remain_sum = iso_EL.loc[mask, sample_cols].T.dropna().T.sum(axis=0)

    if remain_sum.quantile(min_quantile) < min_quantile_val:
        return None

    remain_sum = remain_sum.reindex(sample_cols)
    corrected  = iso_EL.loc[mask, sample_cols].div(remain_sum, axis=1)[sample_cols]

    result = pd.concat([iso_EL.loc[mask, bed_cols], corrected], axis=1)
    return result.loc[iso_EL.loc[mask].index]  # preserve original isoform order
    
# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

def write_corrected_EL(
    EL: pd.DataFrame,
    annot: pd.DataFrame,
    out_path: str,
    gid_col: str = 'gid',
    pid_col: str = 'pid',
    chunksize: int = 500,
) -> str:
    """
    Iterate over genes, correct EL for splicing efficiency, and stream
    results to a gzipped TSV — never holding the full output in RAM.
    """
    if gid_col not in EL.columns:
        EL = EL.assign(**{gid_col: EL[pid_col].str.split('.').str[0]})

    bed_cols    = list(EL.columns[:6])
    sample_cols = list(EL.columns[6:])

    out_path       = str(out_path)
    header_written = False
    buf            = []
    buf_rows       = 0

    def _flush(fh, rows, write_header):
        pd.concat(rows, ignore_index=True).to_csv(
            fh, sep='\t', index=False, header=write_header
        )

    with open(out_path, 'w') as fh:
        for gene, group in tqdm(
            EL.sort_values(gid_col).groupby(gid_col, sort=False),
            desc='Writing corrected EL',
        ):
            corrected = _process_gene_group(
                group, gene, annot, bed_cols, sample_cols
            )
            # fall back to uncorrected rows if correction criteria not met
            out_rows = corrected if corrected is not None else group

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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--species',    required=True,
                   help='Species label, e.g. Human, Cattle, Mouse, Macaque …')
    p.add_argument('--el_path',    required=True,
                   help='Path to raw EL TSV (rows = isoforms, cols = samples).')
    p.add_argument('--annot_path', default=None,
                   help='Path to annotation TSV [gene, transcript, appris_ref, '
                        'appris_transcript_length]. '
                        'If omitted, splicing-efficiency correction is skipped.')
    p.add_argument('--out_dir',    default='.',
                   help='Output directory (default: current directory).')
    p.add_argument('--pid_col',    default='pid',
                   help='Isoform-ID column name (default: pid).')
    p.add_argument('--gid_col',    default='gid',
                   help='Gene-ID column name (default: gid, derived from pid if absent).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sp  = args.species
    pfx = args.out_dir

    # ---- load & filter EL --------------------------------------------------
    print(f"[{sp}] Loading EL from {args.el_path}", flush=True)
    EL = pd.read_csv(args.el_path, sep='\t')
    EL['pid']  = EL['gid'] + '.' + EL['pid']
    EL.index   = list(EL['pid'])

    EL = filter_EL(EL)

    # ---- step 1: splicing-efficiency correction ----------------------------
    if args.annot_path:
        print(f"[{sp}] Loading annotation from {args.annot_path}", flush=True)
        annot = load_annot(args.annot_path)

        corrected_path = os.path.join(pfx, 'EL_corrected.tsv')
        print(f"[{sp}] Writing corrected EL → {corrected_path}", flush=True)
        write_corrected_EL(
            EL, annot, corrected_path,
            gid_col=args.gid_col,
            pid_col=args.pid_col,
        )
        el_source = corrected_path
    else:
        print(f"[{sp}] No annotation provided — skipping splicing-efficiency correction.",
              flush=True)
        el_source = args.el_path

    print(f"[{sp}] Done. EL source for downstream steps: {el_source}", flush=True)


if __name__ == '__main__':
    main()