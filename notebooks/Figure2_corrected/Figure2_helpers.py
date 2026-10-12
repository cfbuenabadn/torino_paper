"""
Figure2_helpers.py — Data loading and processing for Figure 2 (corrected).

This copy restricts every gene-level analysis to `select_genes`, the
higher-confidence subset derived in `analysis/Fig1Numbers.ipynb`: 16,073 of the
16,710 genes the original Figure 2 used. Only panels a, b and c depend on the
gene set; d, e and f are SRSF3/SYNGR1 and both are in the subset, so their
inputs are unchanged.

Contains all the data-loading and data-processing functions. The orchestration
(which function feeds which figure, and where the pickles go) lives in
`prepare_figure2_data.py`, which is a standalone script meant to be run on a
compute node — Figure2.ipynb only unpickles what that script wrote.

Several functions read relative paths, so the CWD must be the analysis/
directory (prepare_figure2_data.py chdirs there for you).
"""
import numpy as np
import pandas as pd
import os
import sys
import re
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

# ── Path setup (assumes CWD = analysis/) ──
sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
sys.path.append('/project2/yangili1/cfbuenabadn/torino_paper/code/scripts')

from torino_isoforms import *
from GetLongReadsTable import *


# ================================================================
# BAM sample lists (shared by UTRs_SRSF3 and long reads figures)
# ================================================================

brain_bams = [
    'GTEx_v9_ONT_data_sequence_data_GTEX-1192X-0011-R10a-SM-4RXXZ.FAK49046.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-145MH-0011-R3b-SM-5CTCM.FAK41439.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-11H98-0011-R11b-SM-4SFLZ.FAK46829.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-QDT8-0011-R10A-SM-2FKJB.FAK49182.bam'
]
muscle_bams = [
    'GTEx_v9_ONT_data_sequence_data_GTEX-13QJ3-0726-SM-7LDHS_rep.FAK93371.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-15RIE-1726-SM-5V7WB.FAK41916.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-17MFQ-1926-SM-6659L.FAK44757.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-1C64N-0326-SM-6ZHT1.FAK53752.bam',
    'GTEx_v9_ONT_data_sequence_data_GTEX-1GN1W-0226-SM-7AGLJ.FAK44655.bam'
]
ms_samples = ['GTEX-1GN1W-0226-SM-7AGLJ', 'GTEX-17MFQ-1926-SM-6659L',
              'GTEX-15RIE-1726-SM-5V7WB', 'GTEX-1C64N-0326-SM-6ZHT1',
              'GTEX-13QJ3-0726-SM-7LDHS']
brain_samples = ['GTEX-1192X-0011-R10a-SM-4RXXZ', 'GTEX-145MH-0011-R3b-SM-5CTCM',
                 'GTEX-QDT8-0011-R10A-SM-2FKJB', 'GTEX-11H98-0011-R11b-SM-4SFLZ']
br_fractions = [f'{x}.fractions' for x in brain_samples]
ms_fractions = [f'{x}.fractions' for x in ms_samples]


# ================================================================
# select_genes: the higher-confidence subset (analysis/Fig1Numbers.ipynb)
# ================================================================

ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
CODE_ROOT = '/project2/yangili1/cfbuenabadn/torino_paper/code'
SELECT_GENES_N = 16073          # the count Fig1Numbers.ipynb records

_SELECT_GENES = None


def load_select_genes():
    """Genes kept for the corrected figure, as `Fig1Numbers.ipynb` defines them.

    Three conditions, intersected: Torino reached K=10 in at least 100 samples
    (`torino_K10_status_Human.csv`), the gene carries a corrected isoform
    annotation, and it does not overlap another gene
    (`config/human.protein_coding.bed.gz`). The result is cached, and its size
    is asserted against the notebook's recorded 16,073 so a change in any of
    the three inputs fails loudly instead of silently shifting every number.
    """
    global _SELECT_GENES
    if _SELECT_GENES is not None:
        return _SELECT_GENES

    torino_status = pd.read_csv(os.path.join(ANALYSIS_DIR,
                                             'torino_K10_status_Human.csv'))
    annot = pd.read_csv(
        f'{CODE_ROOT}/results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz',
        sep='\t')
    annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
    genes_no_overlap = pd.read_csv(f'{CODE_ROOT}/config/human.protein_coding.bed.gz',
                                   sep='\t', index_col=0)

    sel = (pd.Index(torino_status.loc[torino_status.n_samples >= 100].gene)
           .intersection(pd.Index(annot.gene.unique()))
           .intersection(pd.Index(genes_no_overlap.gene_name)))
    if len(sel) != SELECT_GENES_N:
        raise AssertionError(
            f'select_genes has {len(sel)} genes, expected {SELECT_GENES_N}; '
            'one of the three source tables has changed.')
    _SELECT_GENES = sel
    return sel


# ================================================================
# Shared: load annotation table with efficiency_factor
# ================================================================

def load_annot(path='/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz'):
    annot = pd.read_csv(path, sep='\t')
    annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
    efficiency_factor = []
    for idx, row in annot.iterrows():
        appris_ref = row.appris_ref
        aver = appris_ref.split('/')
        if 'RI:' in appris_ref:
            ri_factor = [x for x in aver if x.startswith('RI:')][0]
            ri_numbers = int(ri_factor.split(':')[1])
            appris_length = int(row.appris_transcript_length)
            if ((ri_numbers <= 2) and (appris_length >= 5)) or ((ri_numbers <= 1) and (appris_length >= 3)):
                efficiency_factor.append(False)
            else:
                efficiency_factor.append(True)
        else:
            efficiency_factor.append(False)
    annot['efficiency_factor'] = efficiency_factor
    # Panels a and b both come from here, so the subset is applied once.
    #
    # reset_index matters: `compute_upset_data` combines this frame's
    # efficiency_factor with the `is_annot` mask built on the *merged* frame
    # from load_appris_data, and pandas aligns the two on index. merge()
    # always returns a fresh RangeIndex, so subsetting here with .loc and
    # keeping the gappy index would union the two into 98,704 rows and inflate
    # the upset counts. Contiguous on both sides keeps them row-for-row, which
    # is what the original (unsubset) code relied on.
    return annot.loc[annot.gene.isin(load_select_genes())].reset_index(drop=True)


# ================================================================
# Group 1: Appris figures
# (torino_gene_annot_pct, torino_annot_transcripts, upset_isoforms)
# ================================================================

def load_appris_data():
    """Load annotation + expression data for Appris figures."""
    annot = load_annot()

    gtex_samples = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/config/gtex_samples.tsv', sep='\t')
    gtex_tissues = sorted(gtex_samples.tissue_id.unique())

    median_expression = pd.read_csv(
        '/project2/mstephens/cfbuenabadn/gtex-stm/code/gtex_tables/'
        'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz',
        sep='\t', skiprows=2)

    def transform_string(s):
        s = s.replace(" - ", "_").replace(" ", "_")
        s = re.sub(r"\(", "", s)
        s = re.sub(r"\)", "", s)
        return s

    median_expression.columns = [transform_string(s) for s in median_expression.columns]
    median_expression['gene'] = median_expression.Name.apply(lambda x: x.split('.')[0])
    mexp = median_expression.loc[median_expression[gtex_tissues].max(axis=1) >= 10].gene

    selected_genes = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/config/human.protein_coding.bed.gz', sep='\t')
    with open('/project2/mstephens/cfbuenabadn/gtex-stm/data/whole_genes.txt') as fh:
        whole_genes = [x.rstrip() for x in fh.readlines()]
    selected_genes['length'] = selected_genes.end - selected_genes.start
    annot = annot.merge(selected_genes[['gene_name', 'gene_id', 'length']],
                        left_on='gene', right_on='gene_name')
    return annot, mexp, whole_genes


def compute_annot_masks(annot, mexp, whole_genes):
    """Compute boolean masks for annotation filtering."""
    is_expressed = annot.gene_id.isin(mexp)
    is_whole = annot.gene_id.isin(whole_genes)
    is_appris = annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x))
    is_other_annot = annot.gencode_ref != '.'
    is_annot = is_appris | is_other_annot
    is_super_appris = annot.appris_ref.apply(
        lambda x: ('ref_transcript' in x) or ('intron_chain' in x) or
                   ('superset' in x) or ('subset' in x))
    is_super_annot = is_super_appris | is_other_annot
    appris_len_1 = annot.groupby('gene').appris_transcript_length.mean().quantile(0.1)
    appris_len_9 = annot.groupby('gene').appris_transcript_length.mean().quantile(0.9)
    is_len = ((annot.appris_transcript_length >= appris_len_1) &
              (annot.appris_transcript_length <= appris_len_9))
    return is_expressed, is_whole, is_annot, is_super_annot, is_len


def compute_gene_pct(annot, is_expressed, is_whole, is_annot, is_super_annot, is_len):
    """Compute gene-level annotation percentages for torino_gene_annot_pct."""
    total_col = [
        len(annot.loc[is_annot].gene.unique()) / len(annot.gene.unique()),
        len(annot.loc[is_expressed & is_annot].gene.unique()) / len(annot.loc[is_expressed].gene.unique()),
        len(annot.loc[is_len & is_annot].gene.unique()) / len(annot.loc[is_len].gene.unique()),
        len(annot.loc[is_len & is_expressed & is_annot].gene.unique()) / len(annot.loc[is_len & is_expressed].gene.unique()),
        len(annot.loc[is_super_annot].gene.unique()) / len(annot.gene.unique()),
        len(annot.loc[is_len & is_expressed & is_super_annot].gene.unique()) / len(annot.loc[is_len & is_expressed].gene.unique()),
    ]
    whole_col = [
        len(annot.loc[is_annot & is_whole].gene.unique()) / len(annot.loc[is_whole].gene.unique()),
        len(annot.loc[is_expressed & is_annot & is_whole].gene.unique()) / len(annot.loc[is_expressed & is_whole].gene.unique()),
        len(annot.loc[is_len & is_annot & is_whole].gene.unique()) / len(annot.loc[is_len & is_whole].gene.unique()),
        len(annot.loc[is_len & is_expressed & is_annot & is_whole].gene.unique()) / len(annot.loc[is_len & is_expressed & is_whole].gene.unique()),
        len(annot.loc[is_super_annot & is_whole].gene.unique()) / len(annot.loc[is_whole].gene.unique()),
        len(annot.loc[is_len & is_expressed & is_super_annot & is_whole].gene.unique()) / len(annot.loc[is_len & is_expressed & is_whole].gene.unique()),
    ]
    return total_col, whole_col


def compute_gene_counts(annot, is_expressed, is_whole, is_annot, is_super_annot, is_len):
    """Numerator and denominator gene counts behind compute_gene_pct.

    Same six filters, same order; the published panel prints these as
    "12,155 / 16,710" above the percentage, and 100*num/den reproduces
    `gene_pct_df` exactly.

    Returns
    -------
    num_df, den_df : pd.DataFrame with columns 'All genes' / 'No overlap genes'
    """
    def n(mask=None):
        sub = annot if mask is None else annot.loc[mask]
        return len(sub.gene.unique())

    total_num = [n(is_annot), n(is_expressed & is_annot), n(is_len & is_annot),
                 n(is_len & is_expressed & is_annot), n(is_super_annot),
                 n(is_len & is_expressed & is_super_annot)]
    total_den = [n(), n(is_expressed), n(is_len), n(is_len & is_expressed),
                 n(), n(is_len & is_expressed)]
    whole_num = [n(is_annot & is_whole), n(is_expressed & is_annot & is_whole),
                 n(is_len & is_annot & is_whole),
                 n(is_len & is_expressed & is_annot & is_whole),
                 n(is_super_annot & is_whole),
                 n(is_len & is_expressed & is_super_annot & is_whole)]
    whole_den = [n(is_whole), n(is_expressed & is_whole), n(is_len & is_whole),
                 n(is_len & is_expressed & is_whole), n(is_whole),
                 n(is_len & is_expressed & is_whole)]
    num_df = pd.DataFrame({'All genes': total_num, 'No overlap genes': whole_num})
    den_df = pd.DataFrame({'All genes': total_den, 'No overlap genes': whole_den})
    return num_df, den_df


def compute_transcript_denoms(annot, is_expressed, is_whole, is_annot,
                              is_super_annot, is_len):
    """Denominators for compute_transcript_counts, row for row.

    The gene-level counts have had these all along; the transcript-level ones
    only carried the two unfiltered totals, so the filtered rows could not be
    turned into percentages. Same six filters, same order.
    """
    def n(mask=None):
        sub = annot if mask is None else annot.loc[mask]
        return len(sub.transcript.unique())

    total_den = [n(), n(is_expressed), n(is_len), n(is_len & is_expressed),
                 n(), n(is_len & is_expressed)]
    whole_den = [n(is_whole), n(is_expressed & is_whole), n(is_len & is_whole),
                 n(is_len & is_expressed & is_whole), n(is_whole),
                 n(is_len & is_expressed & is_whole)]
    return pd.DataFrame({'All genes': total_den, 'No overlap genes': whole_den})


def appris_length_bounds(annot):
    """The 10th/90th percentile Appris exon counts behind the `is_len` filter.

    The published row label ("Appris exons 2 - 23") is these two numbers, so
    read them from the data rather than hard-coding them.
    """
    per_gene = annot.groupby('gene').appris_transcript_length.mean()
    return per_gene.quantile(0.1), per_gene.quantile(0.9)


def compute_transcript_counts(annot, is_expressed, is_whole, is_annot, is_super_annot, is_len):
    """Compute transcript-level annotation counts for torino_annot_transcripts."""
    total_col = [
        len(annot.loc[is_annot].transcript.unique()),
        len(annot.loc[is_expressed & is_annot].transcript.unique()),
        len(annot.loc[is_len & is_annot].transcript.unique()),
        len(annot.loc[is_len & is_expressed & is_annot].transcript.unique()),
        len(annot.loc[is_super_annot].transcript.unique()),
        len(annot.loc[is_len & is_expressed & is_super_annot].transcript.unique()),
    ]
    whole_col = [
        len(annot.loc[is_annot & is_whole].transcript.unique()),
        len(annot.loc[is_expressed & is_annot & is_whole].transcript.unique()),
        len(annot.loc[is_len & is_annot & is_whole].transcript.unique()),
        len(annot.loc[is_len & is_expressed & is_annot & is_whole].transcript.unique()),
        len(annot.loc[is_super_annot & is_whole].transcript.unique()),
        len(annot.loc[is_len & is_expressed & is_super_annot & is_whole].transcript.unique()),
    ]
    return total_col, whole_col


def compute_upset_data(annot, is_annot):
    """Compute upset plot data for upset_isoforms."""
    annot2 = annot.loc[(~annot.efficiency_factor) | (annot.efficiency_factor & is_annot)]
    IR_iso = annot2.appris_ref.apply(lambda x: 'RI:' in x)
    altSpl_iso = annot2.appris_ref.apply(lambda x: ('CE' in x) or ('SE' in x) or ('MXE' in x))
    UTR_iso = annot2.appris_ref.apply(lambda x: ('UTR3' in x))
    annot_iso = (annot2.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x))
                 | (annot2.gencode_ref != '.'))
    df = pd.DataFrame({
        'Annotated': annot_iso,
        'Retained Intron': IR_iso,
        'alt. Splicing': altSpl_iso,
        "alt. 3'UTR": UTR_iso,
    })
    return df


def compute_upset_data_extended(annot, is_annot):
    """Upset categories widened to cover every isoform, not just 56,659.

    The four original categories have no row for a difference at the
    transcript ends, so 12,446 isoforms fell out of the plot entirely. Here:

    * alt_5exon / alt_3exon  -> "alt. terminal exon" (a new category)
    * alt.ss.1 .. alt.ss.5   -> folded into "alt. splicing"
    * UTR5                   -> "alt. 5'UTR" (a new category)
    * whatever is left       -> "uncategorized"

    Every row now lands in at least one category, so the >= 1 filter that
    `compute_upset_data` relies on keeps all of them.
    """
    annot2 = annot.loc[(~annot.efficiency_factor) | (annot.efficiency_factor & is_annot)]
    ref = annot2.appris_ref.astype(str)
    df = pd.DataFrame({
        'Annotated': (ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x))
                      | (annot2.gencode_ref != '.')),
        'Retained Intron': ref.apply(lambda x: 'RI:' in x),
        'alt. Splicing': ref.apply(
            lambda x: ('CE' in x) or ('SE' in x) or ('MXE' in x) or ('alt.ss' in x)),
        "alt. 3'UTR": ref.apply(lambda x: 'UTR3' in x),
        'alt. Terminal exon': ref.apply(
            lambda x: ('alt_5exon' in x) or ('alt_3exon' in x)),
        "alt. 5'UTR": ref.apply(lambda x: 'UTR5' in x),
    })
    df['Uncategorized'] = ~df.any(axis=1)
    return df


# ================================================================
# Group 2: gencode_v_torino_introns
# ================================================================

def load_intron_matrices():
    """Load precomputed intron heatmap matrices, restricted to select_genes.

    Both matrices are indexed by gene name, so panel c is subset the same way
    as a and b rather than being recomputed.
    """
    gencode_X = pd.read_csv('gencode_introns_X.tab.gz', sep='\t', index_col=0)
    torino_X = pd.read_csv('torino_introns_X.tab.gz', sep='\t', index_col=0)
    sel = load_select_genes()
    return (gencode_X.loc[gencode_X.index.isin(sel)],
            torino_X.loc[torino_X.index.isin(sel)])


# ================================================================
# Group 3: UTRs_SRSF3
# ================================================================

def get_coverage(bam, chrom, start, end, normalize=None):
    """Compute coverage over a genomic interval from a pysam AlignmentFile."""
    cov = np.zeros(end - start, dtype=np.int32)
    for read in bam.fetch(chrom, start, end):
        if read.is_unmapped:
            continue
        ref_pos = read.reference_start
        for op, length in read.cigartuples:
            if op in (0, 7, 8, 2):
                seg_start = max(ref_pos, start)
                seg_end = min(ref_pos + length, end)
                if seg_start < seg_end:
                    cov[seg_start - start: seg_end - start] += 1
                ref_pos += length
            elif op in [3]:
                ref_pos += length
    positions = np.arange(start, end)
    if normalize is not None:
        if normalize == 'max':
            cov = cov / np.max(cov)
        elif normalize == 'mean':
            cov = cov / np.mean(cov)
    return cov, positions


def get_bam_coverage(bamf, chrom, start, end, bam_dir='/project2/yangili1/cfbuenabadn/torino_paper/code/resources/GTEx/ONT/bam/'):
    """Open BAM, compute normalized coverage, close."""
    import pysam
    samfile = pysam.AlignmentFile(bam_dir + bamf, "rb")
    cov, pos = get_coverage(samfile, chrom, start, end, 'max')
    samfile.close()
    return cov, pos


def load_utrs_srsf3_data():
    """Load data needed for UTRs_SRSF3 figure."""
    annot = load_annot()
    human_iso = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
        names=['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand',
               'factors', 'exon', 'notes'])
    human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

    gencode_iso = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz', sep='\t',
        names=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id',
               'strand', 'exon', 'score', 'basic', 'ensembl', 'MANE', 'appris', 'annot'])

    annot_ref = annot.loc[
        annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x))
        | (annot.gencode_ref != '.')]
    ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]

    gencode_iso_plot_A = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/Human/bed/exons.all.bed.gz', sep='\t')
    gencode_iso_plot_A['chrom'] = gencode_iso_plot_A['#chrom']
    gencode_iso_plot_A['gene_id'] = gencode_iso_plot_A.gene_name

    gencode_iso_plot_B = gencode_iso.loc[
        (~gencode_iso.appris.isna()) & gencode_iso.gene_name.isin(annot.gene)].copy()
    gencode_iso_plot_B['gene_id'] = gencode_iso_plot_B.gene_name

    return annot, human_iso, gencode_iso, ref_chain_iso, gencode_iso_plot_A, gencode_iso_plot_B


# ================================================================
# Group 4: Long reads figures
# ================================================================

def load_long_reads_data():
    """Load data needed by the long reads figures."""
    isoform_bed = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
        names=['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand',
               'factors', 'exon_id', 'notes'])

    gtf_bed = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/Human/bed/exons.all.bed.gz', sep='\t')
    gtf_bed['chrom'] = gtf_bed['#chrom']
    gtf_bed['gene_id'] = gtf_bed.gene_name

    lr_table_strict = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/LongReads/long_read_torino_strict.counts.tab.gz', sep='\t')
    lr_table_strict.index = lr_table_strict.gene + '.' + lr_table_strict.isoform

    lr_table_lax = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/LongReads/long_read_torino.counts.tab.gz', sep='\t')
    lr_table_lax.index = lr_table_lax.gene + '.' + lr_table_lax.isoform

    gencode_lr = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/LongReads/long_read_gencode_strict.counts.tab.gz', sep='\t')
    gencode_lr.index = gencode_lr.gene + '.' + gencode_lr.isoform

    return isoform_bed, gtf_bed, lr_table_strict, lr_table_lax, gencode_lr


def compute_correlation_data(human_EL, human_metadata, annot, human_ssmd,
                             human_isoform_ssmd, on_el_done=None):
    """Compute correlation table for GTEx.LongRead_vs_ShorRead.Correlation.

    Parameters
    ----------
    human_EL, human_metadata, annot, human_ssmd, human_isoform_ssmd
        From load_human.py (%run in notebook).
    on_el_done : callable, optional
        Called once the full EL matrix is no longer needed — only the matched
        isoforms x brain/muscle samples corner is used from there on. Lets the
        caller drop its own reference to the 1.5 GB frame before the
        (long) correlation loop.

    Returns
    -------
    corr_df : pd.DataFrame
        Ready for sns.clustermap.
    n_brain_sr, n_muscle_sr : int
        Counts of brain/muscle short-read samples (for col_colors).
    """
    import gc

    from scipy.stats import pearsonr
    from tqdm import tqdm

    lr_table_lax = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/LongReads/long_read_torino.counts.tab.gz', sep='\t')
    lr_table_lax.index = lr_table_lax.gene + '.' + lr_table_lax.isoform

    gtex_samples_df = pd.read_csv(
        '/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv', sep='\t')
    brain_sr_samples = list(
        pd.Index(gtex_samples_df.loc[
            gtex_samples_df.tissue_id == 'Brain_Frontal_Cortex_BA9', 'entity:sample_id'
        ]).intersection(human_EL.columns))
    muscle_sr_samples = list(
        pd.Index(gtex_samples_df.loc[
            gtex_samples_df.tissue_id == 'Muscle_Skeletal', 'entity:sample_id'
        ]).intersection(human_EL.columns))

    all_samples_counts = [f'{sample}.counts' for sample in ms_samples + brain_samples]
    all_samples_psi = [f'{sample}.fractions' for sample in ms_samples + brain_samples]

    lr_ = lr_table_lax.loc[
        pd.Index(annot.loc[~annot.efficiency_factor].transcript).intersection(lr_table_lax.index)]

    ssmd_genes = (human_ssmd['Human.Muscle_Skeletal'].sort_values()[-1000:].dropna().index
                  .union(human_ssmd['Human.Brain'].sort_values()[-1000:].dropna().index))
    lr_genes = lr_.loc[
        lr_.isoform != 'unmatched', ['gene'] + all_samples_counts
    ].groupby('gene').sum().sum(axis=1)
    lr_genes = lr_genes.loc[lr_genes >= 200].index
    lr_psi = lr_.loc[
        ~lr_.isoform.isin(['unmatched', 'ambiguous']), ['gene'] + all_samples_psi
    ].groupby('gene').sum().mean(axis=1)
    lr_genes_psi = lr_psi.loc[lr_psi >= 0.5].index.intersection(ssmd_genes)

    gene_matches = lr_.loc[
        lr_.gene.isin(lr_genes) & lr_.gene.isin(lr_genes_psi)
    ].index.intersection(human_EL.index)

    brain_min = pd.Index(human_isoform_ssmd['Brain'].sort_values('ssmd')[:500].pid)
    brain_max = pd.Index(human_isoform_ssmd['Brain'].sort_values('ssmd')[-500:].pid)
    muscle_min = pd.Index(human_isoform_ssmd['Muscle_Skeletal'].sort_values('ssmd')[:500].pid)
    muscle_max = pd.Index(human_isoform_ssmd['Muscle_Skeletal'].sort_values('ssmd')[-500:].pid)
    ssmd_iso = muscle_max.union(muscle_min).union(brain_max).union(brain_min)
    gene_matches = (lr_.loc[lr_.gene.isin(lr_genes) & lr_.gene.isin(lr_genes_psi)]
                    .index.intersection(human_EL.index).intersection(ssmd_iso))

    # From here the loop only ever reads EL.loc[gene_matches, one_sr_sample],
    # so keep that corner and let the caller free the rest.
    human_EL = human_EL.loc[gene_matches, brain_sr_samples + muscle_sr_samples].copy()
    if on_el_done is not None:
        on_el_done()
    gc.collect()

    def get_arrays_lr(lr_table, EL, gm, lr_sample, sr_sample):
        ms_table = lr_table.loc[gm, ['gene', lr_sample]].groupby('gene').sum()
        ms_table.columns = ['total_psi']
        ms_table_ = ms_table.merge(lr_table.loc[gm, ['gene', lr_sample]],
                                   left_index=True, right_on='gene')
        ms_table_['adj_psi'] = ms_table_[lr_sample] / ms_table_.total_psi
        ms_table_ = ms_table_.dropna()
        EL_ = pd.DataFrame(EL.loc[gm, sr_sample].dropna())
        final_match = EL_.index.intersection(ms_table_.index)
        x = np.array(ms_table_.loc[final_match, 'adj_psi'])
        y = np.array(EL_.loc[final_match, sr_sample])
        return x, y

    corr_table = []
    for lr_sample_ in brain_samples + ms_samples:
        lr_sample = lr_sample_ + '.fractions'
        corr_row = []
        for sr_sample in tqdm(brain_sr_samples + muscle_sr_samples, leave=True, position=0):
            x, y = get_arrays_lr(lr_, human_EL, gene_matches, lr_sample, sr_sample)
            corr_row.append(pearsonr(x, y)[0])
        corr_table.append(corr_row)

    corr_df = pd.DataFrame(corr_table)
    n_brain_sr = len(brain_sr_samples)
    n_muscle_sr = len(muscle_sr_samples)
    return corr_df, n_brain_sr, n_muscle_sr


def load_torino_rds(gene, species='Human'):
    """Read a torino factorization .rds into python via rpy2.

    The .rds files are keyed by gene *symbol* (e.g. 'SRSF3'), not by Ensembl id.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    readRDS = ro.r['readRDS']
    if species in ['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']:
        path = f'/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/factorization/{species}/{gene}.rds'
    else:
        path = f'/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/factorization_filtered/{species}/{gene}.rds'
    df = readRDS(path)
    with (ro.default_converter + pandas2ri.converter).context():
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(df)
    return pd_from_r_df


def get_rds_coords(rds):
    """Return the list of 'chrom:pos' coordinates covered by a torino .rds.

    Current .rds files store these at the top level as `coords_all`; older ones
    kept them under `ebpmf_5`.
    """
    try:
        return list(rds['coords_all'])
    except KeyError:
        return list(rds['ebpmf_5']['coords'])


def compute_srsf3_gencode_b_data(gene_symbol='SRSF3', gene_id='ENSG00000112081'):
    """Pre-compute data for SRSF3.gencode Version B (rpy2 + gencode.v44).

    Returns
    -------
    gencode_exons_v44 : pd.DataFrame
        gencode.v44 exons for `gene_id` only, with `gene_id`/`transcript_id`
        columns added (that is what the plotting code indexes on).
    xlim : tuple of (int, int)
        Gene coordinate range from the torino factorization, +/- 1kb.
    """
    gencode_exons_v44 = pd.read_csv(
        '/project2/mstephens/cfbuenabadn/gtex-stm/code/Annotations/'
        'gencode.v44.primary_assembly.exons.sorted.bed.gz', sep='\t',
        names = ['chrom', 'start', 'end', 'gene', 'transcript', 'strand', 'exon', 'score', 'gencode_annot',
                              'ensembl', 'mane', 'appris', 'annotation'])
    gencode_exons_v44 = gencode_exons_v44.loc[gencode_exons_v44.gene == gene_id].copy()
    gencode_exons_v44['gene_id'] = gencode_exons_v44['gene']
    gencode_exons_v44['transcript_id'] = gencode_exons_v44['transcript']

    rds = load_torino_rds(gene_symbol)
    coords = get_rds_coords(rds)
    xlim1 = int(coords[0].split(':')[1]) - 1000
    xlim2 = int(coords[-1].split(':')[1]) + 1000
    xlim = (xlim1, xlim2)

    return gencode_exons_v44, xlim


def compute_utrs_coverage():
    """Pre-compute BAM coverage arrays for UTRs_SRSF3.

    Returns
    -------
    coverage_data : list of dict
        Each dict: {'cov': np.array, 'pos': np.array, 'tissue': str}.
    start, end : int
        Genomic coordinates.
    """
    chrom = "chr6"
    start = 36601962 - 500
    end = 36605911 + 200

    coverage_data = []
    for bbam in brain_bams:
        cov, pos = get_bam_coverage(bbam, chrom, start, end)
        coverage_data.append({'cov': cov, 'pos': pos, 'tissue': 'brain'})
    for mbam in muscle_bams:
        cov, pos = get_bam_coverage(mbam, chrom, start, end)
        coverage_data.append({'cov': cov, 'pos': pos, 'tissue': 'muscle'})

    return coverage_data, start, end


def compute_long_reads_isoforms(isoform_bed, annot, gene, bams_br, bams_ms,
                                get_long_reads_multi_samples_func):
    """Pre-compute long reads isoform dicts for a gene.

    Parameters
    ----------
    get_long_reads_multi_samples_func : callable
        get_long_reads_multi_samples from LongReadPlots.py.

    Returns
    -------
    long_reads_bed_br, isoform_dict_br, long_reads_bed_ms, isoform_dict_ms
    """
    long_reads_bed_br, isoform_dict_br = get_long_reads_multi_samples_func(
        isoform_bed, annot, gene, bams_br)
    long_reads_bed_ms, isoform_dict_ms = get_long_reads_multi_samples_func(
        isoform_bed, annot, gene, bams_ms)
    return long_reads_bed_br, isoform_dict_br, long_reads_bed_ms, isoform_dict_ms


# ================================================================
# Minimal loader: replaces %run load_human.py
# ================================================================

# The EL matrix is ~93k rows x ~2016 sample columns (1.5 GB). Every full-width
# slice of it — `EL.loc[:, some_samples]` — copies most of that, and the SSMD
# functions below take four such slices per tissue. Reducing row-wise (mean,
# var, median along axis=1) one row block at a time gives bit-identical results
# for a bounded ~130 MB working set, so that is how they are written.
_CHUNK_ROWS = 8000


def _row_chunks(n, chunk=_CHUNK_ROWS):
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def _row_stats(EL, cols, want_median=False, chunk=_CHUNK_ROWS):
    """Row-wise mean / variance / median over `cols`, in row blocks.

    Equivalent to EL.loc[:, cols].mean(axis=1) etc. — these are per-row
    reductions, so chunking rows cannot change the answer — but without
    materialising a full-width copy of EL.
    """
    means, varis, medians = [], [], []
    for a, b in _row_chunks(len(EL), chunk):
        block = EL.iloc[a:b].loc[:, cols]
        means.append(block.mean(axis=1))
        varis.append(block.var(axis=1))
        if want_median:
            medians.append(block.median(axis=1))
    return (pd.concat(means), pd.concat(varis),
            pd.concat(medians) if want_median else None)


def _filter_EL(EL, q=0.2, chunk=_CHUNK_ROWS):
    """Filter EL matrix by NA rate and variance (from load_human.py).

    Vectorised: the original looped over all ~15k genes in python and took
    ~26 minutes. groupby does the same two reductions in one cython pass.
    """
    EL_cols = list(EL.columns[6:])

    # Per-row NA fraction, then the worst row in each gene.
    na_parts = []
    for a, b in _row_chunks(len(EL), chunk):
        na_parts.append(EL.iloc[a:b].loc[:, EL_cols].isna().mean(axis=1))
    gene_na = pd.concat(na_parts).groupby(EL.gid.to_numpy()).max()

    # Fraction of sample columns that are ~constant across a gene's isoforms.
    # groupby.var matches Series.var: ddof=1, NA-skipping, NaN for 1-row genes.
    gene_var = EL.groupby('gid')[EL_cols].var()
    frac_constant = (gene_var < 1e-10).mean(axis=1)

    keep = gene_na.index[(gene_na <= q) & (frac_constant.reindex(gene_na.index) <= q)]
    return EL.loc[EL.gid.isin(keep)]


def _get_tissue_ssmd(EL, sample_tissue, other_tissues):
    """Gene-level tissue SSMD (from load_human.py)."""
    mean_t, var_t, _ = _row_stats(EL, sample_tissue)
    mean_o, var_o, _ = _row_stats(EL, other_tissues)
    mean_diff = mean_t - mean_o
    sqrt_var = np.sqrt(var_t + var_o)
    ssmd = (mean_diff / sqrt_var).reset_index()
    ssmd.columns = ['pid', 'ssmd']
    ssmd['gene'] = ssmd.pid.apply(lambda x: x.split('.')[0])
    ssmd['abs_ssmd'] = ssmd.ssmd.abs()
    ssmd = ssmd.groupby('gene').abs_ssmd.mean()
    return ssmd


def _get_tissue_ssmd_isoforms(EL, tissue_list, other_tissues, metadata):
    """Isoform-level tissue SSMD (from load_human.py)."""
    tissue_samples = metadata.loc[metadata.Tissue.isin(tissue_list)].index.intersection(EL.columns)
    other_tissues_samples = metadata.loc[metadata.Tissue.isin(other_tissues)].index.intersection(EL.columns)
    mean_t, var_t, median_t = _row_stats(EL, tissue_samples, want_median=True)
    mean_o, var_o, median_o = _row_stats(EL, other_tissues_samples, want_median=True)
    mean_diff = mean_t - mean_o
    sqrt_var = np.sqrt(var_t + var_o)
    ssmd = (mean_diff / sqrt_var).reset_index()
    median_diff = median_t - median_o
    ssmd.columns = ['pid', 'ssmd']
    ssmd['median_diff'] = list(median_diff)
    ssmd['gene'] = ssmd.pid.apply(lambda x: x.split('.')[0])
    ssmd['abs_ssmd'] = ssmd.ssmd.abs()
    return ssmd


def load_human_EL():
    """Load and filter the human isoform loadings matrix (from load_human.py)."""
    human_EL = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/tables/Human/isoforms_filtered.merged_isoforms.EL.bed.gz',
        sep='\t')
    human_EL['pid'] = human_EL.gid + '.' + human_EL.pid
    human_EL.index = human_EL.pid
    human_EL = _filter_EL(human_EL)
    return human_EL


def load_annot_lr():
    """Load the `annot` table as load_human.py defines it.

    This is `annotated.isoforms.tab.gz` — NOT the same file as
    :func:`load_annot` (`corrected_annotated.isoforms.tab.gz`), which the
    Appris figures use. The long reads figures need this one.
    """
    annot = pd.read_csv(
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/Human/isoforms/annotated.isoforms.tab.gz', sep='\t')
    annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
    efficiency_factor = []
    for idx, row in annot.iterrows():
        appris_ref = row.appris_ref
        if 'RI:' in appris_ref:
            aver = appris_ref.split('/')
            ri_factor = [x for x in aver if x.startswith('RI:')][0]
            ri_numbers = int(ri_factor.split(':')[1])
            appris_length = int(row.appris_transcript_length)
            if ((ri_numbers <= 2) and (appris_length >= 5)) or \
               ((ri_numbers <= 1) and (appris_length >= 3)):
                efficiency_factor.append(False)
            else:
                efficiency_factor.append(True)
        else:
            efficiency_factor.append(False)
    annot['efficiency_factor'] = efficiency_factor
    return annot


def load_human_metadata():
    """Sample -> tissue table (from load_human.py)."""
    metadata = pd.read_csv(
        '/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv', sep='\t')
    human_metadata = metadata[['entity:sample_id', 'tissue_id']]
    human_metadata.columns = ['Sample', 'Tissue']
    human_metadata.index = human_metadata.Sample
    return human_metadata


ALL_SSMD_TISSUES = [
    'Adipose_Subcutaneous', 'Artery_Aorta', 'Brain', 'Cerebellum',
    'Breast_Mammary_Tissue', 'Heart', 'Liver', 'Lung',
    'Muscle_Skeletal', 'Pancreas', 'Skin', 'Testis', 'Whole_Blood',
]


def compute_human_ssmd(human_EL, human_metadata, tissues=None):
    """Gene-level tissue SSMD (from load_human.py).

    Each tissue's column is computed independently — the `other` group is
    derived from the tissues *present in the data*, not from `tissues` — so
    asking for a subset gives exactly the same numbers for those columns.
    Figure 2 only ever reads Human.Brain and Human.Muscle_Skeletal, and each
    tissue costs a full pass over the EL matrix, so the caller passes those two.
    """
    human_tissues = sorted(human_metadata.loc[human_EL.columns[6:]].Tissue.unique())
    human_tissues_ = ALL_SSMD_TISSUES if tissues is None else list(tissues)
    species_ssmd = []
    cols = []
    for tissue in human_tissues_:
        if tissue == 'Heart':
            tissue_list = ['Heart_Atrial_Appendage', 'Heart_Left_Ventricle']
            other = [x for x in human_tissues if x not in tissue_list]
        elif tissue == 'Skin':
            tissue_list = ['Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg']
            other = [x for x in human_tissues if x not in tissue_list]
        elif tissue == 'Cerebellum':
            tissue_list = ['Brain_Cerebellar_Hemisphere']
            other = [x for x in human_tissues if x not in tissue_list and 'Brain' not in x]
        elif tissue == 'Brain':
            tissue_list = ['Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex',
                           'Brain_Frontal_Cortex_BA9', 'Brain_Putamen_basal_ganglia']
            other = [x for x in human_tissues if x not in tissue_list and 'Cereb' not in x]
        else:
            tissue_list = [tissue]
            other = [x for x in human_tissues if x != tissue]
        tissue_samples = human_metadata.loc[
            human_metadata.Tissue.isin(tissue_list)].index.intersection(human_EL.columns)
        other_samples = human_metadata.loc[
            human_metadata.Tissue.isin(other)].index.intersection(human_EL.columns)
        tissue_ssmd = _get_tissue_ssmd(human_EL, tissue_samples, other_samples)
        species_ssmd.append(tissue_ssmd)
        cols.append(tissue)
    human_ssmd = pd.DataFrame(species_ssmd).T
    # Rename columns to match load_human.py format
    human_ssmd.columns = ['Human.' + c for c in cols]
    return human_ssmd


def compute_human_isoform_ssmd(human_EL, human_metadata):
    """Isoform-level SSMD for Brain and Muscle_Skeletal (the two Figure 2 uses)."""
    human_tissues = sorted(human_metadata.loc[human_EL.columns[6:]].Tissue.unique())
    brain_tissues_list = ['Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex',
                          'Brain_Frontal_Cortex_BA9', 'Brain_Putamen_basal_ganglia']
    return {
        'Brain': _get_tissue_ssmd_isoforms(
            human_EL, brain_tissues_list,
            [x for x in human_tissues if 'Brain' not in x], human_metadata),
        'Muscle_Skeletal': _get_tissue_ssmd_isoforms(
            human_EL, ['Muscle_Skeletal'],
            [x for x in human_tissues if x != 'Muscle_Skeletal'], human_metadata),
    }


def load_human_minimal():
    """Load only the variables from load_human.py that Figure 2 needs.

    Replaces `%run load_human.py` to avoid loading ~20 unused module-scope
    variables (human_median_diff, human_ssmd_cereb, 10 unused isoform SSMD
    tissues, etc.) that cause memory crashes.

    Convenience wrapper around the individual loaders; prepare_figure2_data.py
    calls those directly so that each step only pays for what it needs.

    Returns
    -------
    dict with keys: human_EL, human_metadata, human_ssmd,
                    human_isoform_ssmd, annot
    """
    human_EL = load_human_EL()
    human_metadata = load_human_metadata()
    return {
        'human_EL': human_EL,
        'human_metadata': human_metadata,
        'human_ssmd': compute_human_ssmd(human_EL, human_metadata),
        'human_isoform_ssmd': compute_human_isoform_ssmd(human_EL, human_metadata),
        'annot': load_annot_lr(),
    }

