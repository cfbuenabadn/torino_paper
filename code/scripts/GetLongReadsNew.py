"""
GetLongReadsNew.py
==================
Batch long-read quantification using the unified LongReadTools module.

Produces three output tables per annotation (torino / gencode):
  1. keep      — ambiguous_mode='keep',  no tier-4 rescue
  2. prop      — ambiguous_mode='proportional', no tier-4 rescue
  3. prop_t4   — ambiguous_mode='proportional', with tier-4 rescue

All use default quality filters (min_read_length=200, min_aligned_fraction=0.8,
min_identity=0.9), allow_short_reads=True, and intron_tolerance=200.

Usage:
    python GetLongReadsNew.py <chunk_index> <n_chunks>
"""

import os
import sys
import gzip
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/project2/yangili1/cfbuenabadn/torino_paper/code/scripts')
from LongReadTools import (
    get_gene_coords,
    get_fractions_per_sample_list_lean,
)

# ── Shared parameters ──────────────────────────────────────────────────────
INTRON_TOLERANCE = 200
COMMON_KW = dict(
    allow_short_reads=True,
    intron_tolerance=INTRON_TOLERANCE,
    min_read_length=200,
    min_aligned_fraction=0.8,
    min_identity=0.9,
)

# The three variants: (suffix, extra kwargs)
VARIANTS = [
    #('keep',    dict(ambiguous_mode='keep',         rescue_unmatched=False)),
    #('prop',    dict(ambiguous_mode='proportional',  rescue_unmatched=False)),
    ('prop_t4', dict(ambiguous_mode='proportional',  rescue_unmatched=True,
                     rescue_min_precision=0.8, rescue_min_recall=0.5)),
]

# ── Load annotations ───────────────────────────────────────────────────────
isoform_bed = pd.read_csv(
    'results/torino/Human/corrected_isoforms/isoforms.bed.gz', sep='\t',
    names=['chrom', 'start', 'end', 'gene_id', 'transcript_id',
           'strand', 'factors', 'exon_id', 'notes'])
autosomes = [f'chr{x}' for x in range(1, 23)]
isoform_bed = isoform_bed.loc[isoform_bed.chrom.isin(autosomes)]

gtf_bed = pd.read_csv(
    'resources/annotation/Human/bed/exons.all.bed.gz', sep='\t')
gtf_bed = gtf_bed.loc[gtf_bed.gene_type == 'protein_coding']
gtf_bed.columns = ['chrom', 'start', 'end', 'gene_ensembl', 'gene_id',
                    'strand', 'exon_id', 'transcript_id', 'factors']
gtf_bed['notes'] = ['.'] * gtf_bed.shape[0]

# ── Chunk genes ─────────────────────────────────────────────────────────────
arguments = sys.argv
i = int(arguments[1])
n = int(arguments[2])

genes = sorted(isoform_bed.gene_id.unique())
total_genes = len(genes)
genes_per_slice = int(np.ceil((total_genes / n)))

slice_start = (i - 1) * genes_per_slice
if i == n:
    slice_end = total_genes + 1
else:
    slice_end = i * genes_per_slice

gene_list = genes[slice_start:slice_end]

bam_file_list = ['resources/GTEx/ONT/bam/' + x for x in os.listdir('resources/GTEx/ONT/bam/')
                 if x.endswith('.bam')]

# ── Run ─────────────────────────────────────────────────────────────────────
os.makedirs('results/LongReads/tmp', exist_ok=True)

# Open output files: one per (variant, annotation)
handles = {}
for suffix, _ in VARIANTS:
    for annot_tag, annot_bed in [('torino', isoform_bed), ('gencode', gtf_bed)]:
        key = (suffix, annot_tag)
        path = (f'results/LongReads/tmp/new_lr_{suffix}_{annot_tag}'
                f'.counts.{i}.tab.gz')
        handles[key] = gzip.open(path, 'wb')

header_written = {k: False for k in handles}

for gene in gene_list:
    print(gene)
    for suffix, extra_kw in VARIANTS:
        kw = {**COMMON_KW, **extra_kw}
        for annot_tag, annot_bed in [('torino', isoform_bed),
                                     ('gencode', gtf_bed)]:
            key = (suffix, annot_tag)
            fh = handles[key]

            coords, strand = get_gene_coords(annot_bed, gene)
            isoform_table = get_fractions_per_sample_list_lean(
                annot_bed, gene, bam_file_list, coords, strand, **kw)

            if isoform_table is None:
                continue

            if not header_written[key]:
                hdr = ('\t'.join(isoform_table.columns) + '\n').encode()
                fh.write(hdr)
                header_written[key] = True

            for _, row in isoform_table.astype(str).iterrows():
                fh.write(('\t'.join(list(row)) + '\n').encode())

for fh in handles.values():
    fh.close()
