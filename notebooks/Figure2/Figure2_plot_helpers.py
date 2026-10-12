"""
Figure2_plot_helpers.py — Plotting functions ONLY for Figure 2.

Every function receives fully-processed, plot-ready data and produces a
matplotlib/seaborn figure.  No file I/O, no heavy computation.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ================================================================
# torino_gene_annot_pct
# Source: analysis/Appris_1.ipynb (Cell 107)
# ================================================================

def plot_torino_gene_annot_pct(gene_pct_df):
    """Heatmap of gene-level annotation percentages.

    Parameters
    ----------
    gene_pct_df : pd.DataFrame
        DataFrame with columns 'All genes' and 'No overlap genes',
        values in [0, 100].
    """
    fig, ax = plt.subplots(figsize=(4, 6))
    hm = sns.heatmap(
        gene_pct_df,
        cmap='viridis', vmin=50, vmax=100, annot=True, fmt=".1f",
        cbar_kws=dict(orientation='horizontal', shrink=0.75, pad=0.08), ax=ax
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', rotation=0)
    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=3))
    cbar = hm.collections[0].colorbar
    cbar.set_label("Percentage of tested genes", labelpad=10)
    return fig


# ================================================================
# torino_annot_transcripts
# Source: analysis/Appris_1.ipynb (Cell 116)
# ================================================================

def plot_torino_annot_transcripts(tx_count_df):
    """Heatmap of transcript-level annotation counts.

    Parameters
    ----------
    tx_count_df : pd.DataFrame
        DataFrame with columns 'All genes' and 'No overlap genes',
        integer values.
    """
    fig, ax = plt.subplots(figsize=(4, 6))
    hm = sns.heatmap(
        tx_count_df,
        cmap='Blues', vmin=0, annot=True, fmt="d",
        cbar_kws=dict(orientation='horizontal', shrink=0.75, pad=0.08), ax=ax
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', rotation=0)
    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=3))
    cbar = hm.collections[0].colorbar
    cbar.set_label("Percentage of tested genes", labelpad=10)
    return fig


# ================================================================
# upset_isoforms
# Source: analysis/Appris_1.ipynb (Cell 122)
# ================================================================

def plot_upset_isoforms(df_upset):
    """UpSet plot of isoform categories.

    Parameters
    ----------
    df_upset : pd.DataFrame
        Boolean DataFrame with columns like 'Annotated', 'Retained Intron',
        'alt. Splicing', "alt. 3'UTR".  Already filtered to rows with >= 1 True.
    """
    from upsetplot import from_indicators, UpSet

    upset_data = from_indicators(df_upset.columns, data=df_upset)
    UpSet(upset_data).plot()
    fig = plt.gcf()
    return fig


# ================================================================
# gencode_v_torino_introns
# Source: analysis/Retained_Introns_figure.ipynb (Cell 230)
# ================================================================

def plot_gencode_v_torino_introns(gencode_X, torino_X):
    """Side-by-side intron-position heatmaps.

    Parameters
    ----------
    gencode_X, torino_X : pd.DataFrame
        Sorted intron matrices with columns bp_1..bp_200 and a 'position_i' column.
    """
    bp_ = [f'bp_{str(i)}' for i in range(1, 201)]

    fig, ax = plt.subplots(ncols=2, figsize=(6, 9))
    hm = sns.heatmap(gencode_X.sort_values('position_i')[bp_].astype(int) == 1,
                     cmap='Greys', cbar=False, ax=ax[0])
    for im in hm.get_images():
        im.set_rasterized(True)

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    sns.heatmap(torino_X.sort_values('position_i')[bp_].astype(int) == 1,
                cmap='Greys', cbar=False, ax=ax[1])
    ax[1].set_yticks([])

    ax[0].spines['bottom'].set_visible(True)
    ax[1].spines['bottom'].set_visible(True)
    ax[0].set_xticks([0, 201], ['TSS', 'TES'], rotation=0)
    ax[1].set_xticks([0, 201], ['TSS', 'TES'], rotation=0)

    ax[0].set_xlabel('gene body')
    ax[1].set_xlabel('gene body')
    ax[0].set_ylabel('genes')
    ax[1].set_ylabel('')
    ax[0].set_title('Gencode')
    ax[1].set_title('Torino')
    return fig


# ================================================================
# SRSF3.gencode  --  Version A
# Source: analysis/Fig2_LongReads.ipynb (Cell 188)
# ================================================================

def plot_srsf3_gencode_a(gtf_bed, plot_isoform_annotations_func):
    """SRSF3 gencode annotations — Version A (black, exons.all.bed.gz).

    Parameters
    ----------
    gtf_bed : pd.DataFrame
        Exon annotation table (from exons.all.bed.gz) with gene_id column.
    plot_isoform_annotations_func : callable
        The plot_isoform_annotations function from LongReadPlots.py.
    """
    plot_isoform_annotations_func(
        gtf_bed, 'ENSG00000112081', colores=['black'] * 200,
        iso_order=list(gtf_bed.loc[gtf_bed.gene_id == 'ENSG00000112081'].transcript_id.unique()),
        figsize=(20, 5))
    fig = plt.gcf()
    return fig


# ================================================================
# SRSF3.gencode  --  Version B
# Source: gtex-stm/analysis/Figure_1A_scratch.ipynb (Cell 40)
# ================================================================

def plot_srsf3_gencode_b(gencode_exons_v44, xlim):
    """SRSF3 gencode annotations — Version B (navy, gencode.v44).

    Parameters
    ----------
    gencode_exons_v44 : pd.DataFrame
        Exon annotation from gencode.v44.
    xlim : tuple
        (xlim1, xlim2) computed from RDS.
    """
    from torino_plots import plot_gene_isoforms as _pgf, plot_isoform as _pi

    def plot_isoform_annotations_v1A(annotation_exons, gene, colores=None, start=None, end=None,
                                      figsize=None, lwidth=5, iso_order=None, axes=None, xlim=None):
        gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]
        if iso_order is None:
            isoforms = sorted(gene_exons.transcript_id.unique())
        else:
            isoforms = [gene + '.' + x for x in iso_order]
        isoform_dict = {}
        for i, iso in enumerate(isoforms):
            isoform_name = f'isoform_{str(i+1)}'
            df = gene_exons.loc[gene_exons.transcript_id == iso].copy()
            df['transcript_id'] = f'{gene}.{isoform_name}'
            isoform_dict.update({isoform_name: {'df': df}})
        chrom = list(annotation_exons.chrom)[0]
        if start is None:
            start = str(np.min([int(list(gene_exons.start)[0]),
                                int(list(gene_exons.start)[0])]) - 1000)
        if end is None:
            end = str(np.max([int(list(gene_exons.end)[-1]),
                              int(list(gene_exons.end)[-1])]) + 1000)
        coords = [f'{chrom}:{start}', f'{chrom}:{end}']
        _pgf(isoform_dict, coords, color_list=colores, figsize=figsize,
             lwidth=lwidth, axes=axes, xlim=xlim)

    fig, ax = plt.subplots(nrows=9, figsize=(4, 1.6),
                           gridspec_kw={'height_ratios': ([1.5] * 8) + [0.5],
                                        'wspace': 0.3, 'hspace': 0.3})

    colores_gencode = ['navy'] * 300
    plot_isoform_annotations_v1A(gencode_exons_v44, 'ENSG00000112081',
                                  colores=colores_gencode,
                                  axes=ax[:8], xlim=xlim, lwidth=2.5)

    ax[8].spines['top'].set_visible(False)
    ax[8].spines['right'].set_visible(False)
    ax[8].spines['left'].set_visible(False)
    ax[8].set_yticks([])
    ax[8].set_xlim(xlim)
    ax[8].set_xticks([36595000, 36600000, 36605000])
    ax[8].set_xticklabels(['36595000', '36600000', '36605000'])
    ax[8].tick_params(labelsize=10)
    ax[8].set_xlabel('Chromosome 6', fontsize=10)
    return fig


# ================================================================
# UTRs_SRSF3
# Source: analysis/UTR3.ipynb (Cell 316)
# ================================================================

def plot_utrs_srsf3(coverage_data, ref_chain_iso_srsf3, gencode_iso_plot_relevant,
                    relevant_gencode_transcripts, plot_isoform_annotations_func,
                    start, end):
    """UTRs_SRSF3 — coverage + isoform annotation.

    Parameters
    ----------
    coverage_data : list of dict
        Each dict has keys 'cov', 'pos', 'tissue' ('brain' or 'muscle').
    ref_chain_iso_srsf3 : pd.DataFrame
        Filtered ref_chain_iso for SRSF3 (excluding isoform_4).
    gencode_iso_plot_relevant : pd.DataFrame
        Gencode isoforms for relevant transcripts.
    relevant_gencode_transcripts : array-like
        Transcript IDs for the relevant gencode isoforms.
    plot_isoform_annotations_func : callable
        plot_isoform_annotations from torino_plots.py.
    start, end : int
        Genomic coordinates for x-axis.
    """
    brain_color = '#EEEE00'
    muscle_color = '#7A67EE'

    fig, axes = plt.subplots(nrows=7, figsize=(6, 4),
                             gridspec_kw={'height_ratios': [10, 1, 1, 1, 1, 1, 1]})

    for item in coverage_data:
        if item['tissue'] == 'brain':
            axes[0].plot(item['pos'], item['cov'], c=brain_color, linewidth=3)
        else:
            axes[0].plot(item['pos'], item['cov'], c=muscle_color, alpha=0.4, linewidth=3)

    # Arrows and text annotations
    axes[0].arrow(x=36602300, y=1.1, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='tab:blue', ec='tab:blue', linewidth=1.5, label='Torino')
    axes[0].text(36602300 - 50, 1.15, '21-50%')
    axes[0].arrow(x=36602776, y=1, dx=0, dy=-1e-10, head_width=160,
                  head_length=0.1, fc='tab:blue', ec='tab:red', linewidth=1.2)
    axes[0].text(36602776 - 50, 1.05, '31-55%')

    axes[0].arrow(x=36603379, y=0.35, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='tab:blue', ec='tab:blue', linewidth=1.5)
    axes[0].text(36603379 - 50, 0.4, '1-15%')

    axes[0].arrow(x=36602776 + 150, y=0.5, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='none', ec='tab:gray', linewidth=1.5)
    axes[0].text(36602776 + 100, 0.55, '4-17%')

    axes[0].arrow(x=36603379 + 1100, y=0.15, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='none', ec='tab:gray', linewidth=1.5)
    axes[0].text(36603379 + 1050 - 50, 0.2, '0-2%')

    axes[0].arrow(x=36605600, y=0.15, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='tab:red', ec='none', label='Gencode')
    axes[0].text(36605600 - 150, 0.2, '<1%')

    axes[0].arrow(x=36605911, y=0.15, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='tab:red', ec='none')
    axes[0].text(36605911 - 50, 0.2, '<1%')

    axes[0].arrow(x=36602040, y=1.1, dx=0, dy=-1e-10, head_width=120,
                  head_length=0.075, fc='none', ec='tab:red', linewidth=1.5)
    axes[0].text(36602040 - 350, 1.15, '0-10%')

    axes[0].set_xlim((start, end))
    plot_isoform_annotations_func(
        pd.concat((ref_chain_iso_srsf3, gencode_iso_plot_relevant)),
        'SRSF3',
        colores=['tab:red', 'tab:red', 'tab:red', 'navy', 'navy', 'navy'],
        axes=axes[1:], xlim=(start, end))

    axes[0].set_xticks([])
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Relative coverage')
    return fig


# ================================================================
# GTEx.LongRead_vs_ShorRead.Correlation
# Source: analysis/Fig2_LongReads.ipynb (Cell 194)
# ================================================================

def plot_correlation(corr_df):
    """Clustermap of long-read vs short-read correlations.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Rows = LR samples (4 brain + 5 muscle), columns = SR samples
        (brain_sr + muscle_sr).
    """
    n_brain_sr = len(corr_df.columns) // 2  # approximate split
    # Use actual split based on row/col structure
    g = sns.clustermap(
        corr_df, cmap='coolwarm', col_cluster=False, row_cluster=False,
        col_colors=['#EEEE00'] * (len(corr_df.columns) // 2) + ['#7A67EE'] * (len(corr_df.columns) - len(corr_df.columns) // 2),
        row_colors=['#EEEE00'] * 4 + ['#7A67EE'] * 5,
        vmin=-0.3, vmax=0.3)

    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_cbar.set_ylabel("Pearson's r", fontsize=12)
    g.ax_heatmap.collections[0].set_rasterized(True)
    return g


def plot_correlation_with_counts(corr_df, n_brain_sr, n_muscle_sr):
    """Clustermap of long-read vs short-read correlations with explicit SR counts.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Rows = LR samples (4 brain + 5 muscle), columns = SR samples.
    n_brain_sr : int
        Number of brain short-read samples.
    n_muscle_sr : int
        Number of muscle short-read samples.
    """
    g = sns.clustermap(
        corr_df, cmap='coolwarm', col_cluster=False, row_cluster=False,
        col_colors=['#EEEE00'] * n_brain_sr + ['#7A67EE'] * n_muscle_sr,
        row_colors=['#EEEE00'] * 4 + ['#7A67EE'] * 5,
        vmin=-0.3, vmax=0.3)

    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_cbar.set_ylabel("Pearson's r", fontsize=12)
    g.ax_heatmap.collections[0].set_rasterized(True)
    return g


# ================================================================
# SRSF3.long_reads  --  Version A (all isoforms)
# Source: analysis/Fig2_LongReads_Definitive.ipynb (Cell 15)
# ================================================================

def plot_srsf3_long_reads_a(isoform_bed, long_reads_bed_br, long_reads_bed_ms,
                            isoform_dict_br, isoform_dict_ms,
                            plot_comparison_func):
    """SRSF3 long reads — Version A (all isoforms).

    Parameters
    ----------
    isoform_bed : pd.DataFrame
        Full isoform bed.
    long_reads_bed_br, long_reads_bed_ms : pd.DataFrame
        Long reads bed for brain / muscle.
    isoform_dict_br, isoform_dict_ms : dict
        Isoform dicts for brain / muscle.
    plot_comparison_func : callable
        plot_gene_long_read_transcripts_comparison_single_ax from LongReadPlots.py.
    """
    plot_comparison_func(
        isoform_bed, 'SRSF3', long_reads_bed_br, long_reads_bed_ms,
        isoform_dict_br, isoform_dict_ms,
        list(isoform_dict_br.keys()), sns.color_palette('tab10'),
        '#EEEE00', '#7A67EE', pos=True,
        yticks=([0, 0.1, 0.2, 0.3, 0.4], ['0', '10', '20', '30', '40']))
    fig = plt.gcf()
    return fig


# ================================================================
# SRSF3.long_reads  --  Version B (isoforms 2-6)
# Source: analysis/LongReads2.ipynb (Cell 30)
# ================================================================

def plot_srsf3_long_reads_b(isoform_bed, long_reads_bed_br, long_reads_bed_ms,
                            isoform_dict_br, isoform_dict_ms,
                            plot_comparison_func):
    """SRSF3 long reads — Version B (isoforms 2-6 only).

    Parameters
    ----------
    Same as Version A.
    """
    plot_comparison_func(
        isoform_bed, 'SRSF3', long_reads_bed_br, long_reads_bed_ms,
        isoform_dict_br, isoform_dict_ms,
        [f'isoform_{str(x)}' for x in range(2, 7)], sns.color_palette('tab10'),
        '#EEEE00', '#7A67EE', pos=True,
        yticks=([0, 0.1, 0.2, 0.3, 0.4], ['0', '10', '20', '30', '40']))
    fig = plt.gcf()
    return fig


# ================================================================
# SRSF3.long_reads_loadings
# Source: analysis/Fig2_LongReads_Definitive.ipynb (Cell 19)
# NOTE: plot_loadings_vs_longreads both computes and plots (deeply
# interleaved). Kept here; will recompute on pickle-only re-runs.
# ================================================================

def plot_srsf3_long_reads_loadings(human_EL, lr_table_str, annot, human_metadata,
                                   plot_loadings_func):
    """SRSF3 loadings vs long reads.

    Parameters
    ----------
    human_EL : pd.DataFrame
        Expression-level matrix.
    lr_table_str : pd.DataFrame
        Strict long-read table.
    annot : pd.DataFrame
        Annotation table.
    human_metadata : pd.DataFrame
        Sample -> tissue table; selects the BA9 and muscle short-read columns.
    plot_loadings_func : callable
        plot_loadings_vs_longreads from LongReadPlots.py.
    """
    pr = plot_loadings_func(human_EL, lr_table_str, human_metadata, 'SRSF3',
                            filter_efficiency=True, annot=annot)
    print(pr)
    fig = plt.gcf()
    return fig


# ================================================================
# SYNGR1.long_reads  --  Version A (tab10, 0-100%)
# Source: analysis/Fig2_LongReads_Definitive.ipynb (Cell 27)
# ================================================================

def plot_syngr1_long_reads_a(isoform_bed, long_reads_bed_br, long_reads_bed_ms,
                             isoform_dict_br, isoform_dict_ms,
                             plot_comparison_func):
    """SYNGR1 long reads — Version A (tab10, 0-100%)."""
    plot_comparison_func(
        isoform_bed, 'SYNGR1', long_reads_bed_br, long_reads_bed_ms,
        isoform_dict_br, isoform_dict_ms,
        list(isoform_dict_br.keys()), sns.color_palette('tab10'),
        '#EEEE00', '#7A67EE', pos=True,
        yticks=([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100']))
    fig = plt.gcf()
    return fig


# ================================================================
# SYNGR1.long_reads  --  Version B (tab10, 0-40%)
# Source: analysis/LongReads2.ipynb (Cell 33)
# ================================================================

def plot_syngr1_long_reads_b(isoform_bed, long_reads_bed_br, long_reads_bed_ms,
                             isoform_dict_br, isoform_dict_ms,
                             plot_comparison_func):
    """SYNGR1 long reads — Version B (tab10, 0-40%)."""
    plot_comparison_func(
        isoform_bed, 'SYNGR1', long_reads_bed_br, long_reads_bed_ms,
        isoform_dict_br, isoform_dict_ms,
        list(isoform_dict_br.keys()), sns.color_palette('tab10'),
        '#EEEE00', '#7A67EE', pos=True,
        yticks=([0, 0.1, 0.2, 0.3, 0.4], ['0', '10', '20', '30', '40']))
    fig = plt.gcf()
    return fig


# ================================================================
# SYNGR1.long_reads  --  Version C (navy, 0-100%)
# Source: analysis/Fig2_LongReads.ipynb (Cell 14)
# ================================================================

def plot_syngr1_long_reads_c(isoform_bed, long_reads_bed_br, long_reads_bed_ms,
                             isoform_dict_br, isoform_dict_ms,
                             plot_comparison_func):
    """SYNGR1 long reads — Version C (navy, 0-100%)."""
    plot_comparison_func(
        isoform_bed, 'SYNGR1', long_reads_bed_br, long_reads_bed_ms,
        isoform_dict_br, isoform_dict_ms,
        list(isoform_dict_br.keys()),
        ['navy'] * len(list(isoform_dict_br.keys())),
        '#EEEE00', '#7A67EE', pos=True,
        yticks=([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100']))
    fig = plt.gcf()
    return fig


# ================================================================
# SYNGR1.long_reads_loadings
# Source: analysis/Fig2_LongReads_Definitive.ipynb (Cell 35)
# NOTE: Same as SRSF3 — plot_loadings_vs_longreads is interleaved.
# ================================================================

def plot_syngr1_long_reads_loadings(human_EL, lr_table_str, annot, human_metadata,
                                    plot_loadings_func):
    """SYNGR1 loadings vs long reads."""
    pr = plot_loadings_func(human_EL, lr_table_str, human_metadata, 'SYNGR1',
                            filter_efficiency=True, annot=annot)
    print(pr)
    fig = plt.gcf()
    return fig


# ================================================================
# Isoform annotation plotting (from load_human.py)
# Used by plot_srsf3_gencode_a
# ================================================================

def plot_isoform(isoform_df, ax, color, xlim1, xlim2, lwidth):
    is_first = True
    for idx, row in isoform_df.iterrows():
        start = int(row.start)
        end = int(row.end)
        if is_first:
            first = end
            is_first = False
        ax.fill_between([start, end], [0, 0], [1, 1], color=color, zorder=2)
    ax.plot([first, start], [0.5, 0.5], c=color, linewidth=lwidth)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['bottom', 'top', 'right', 'left']].set_visible(False)
    ax.set_xlim([xlim1, xlim2])


def plot_gene_isoforms(isoforms_dict, coordinates, color_list=None, axes=None,
                       figsize=None, lwidth=5, xlim=None):
    if xlim is None:
        xlim1 = int(coordinates[0].split(':')[1])
        xlim2 = int(coordinates[-1].split(':')[1])
    else:
        xlim1 = xlim[0]
        xlim2 = xlim[1]

    if color_list is None:
        color_list = sns.color_palette("tab10")

    K = len(isoforms_dict)

    if figsize is None:
        figsize = (20, 3)

    if axes is None:
        fig, axes = plt.subplots(K, 1, figsize=figsize)

    if K == 1:
        isoform_df = isoforms_dict['isoform_1']['df']
        ax = axes
        color = color_list[0]
        plot_isoform(isoform_df, ax, color, xlim[0], xlim[1], lwidth=lwidth)
    else:
        for i in range(K):
            isoform_df = isoforms_dict[f'isoform_{str(i+1)}']['df']
            ax = axes[i]
            color = color_list[i]
            plot_isoform(isoform_df, ax, color, xlim[0], xlim[1], lwidth=lwidth)


def plot_isoform_annotations(annotation_exons, gene, colores=None, start=None,
                             end=None, figsize=None, lwidth=5, iso_order=None,
                             axes=None, xlim=None):
    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]

    if iso_order is None:
        isoforms = sorted(gene_exons.transcript_id.unique())
    else:
        isoforms = [gene + '.' + x for x in iso_order]

    isoform_dict = {}
    for i, iso in enumerate(isoforms):
        isoform_name = f'isoform_{str(i+1)}'
        df = gene_exons.loc[gene_exons.transcript_id == iso].copy()
        df['transcript_id'] = f'{gene}.{isoform_name}'
        isoform_dict.update({isoform_name: {'df': df}})

    chrom = list(annotation_exons.chrom)[0]
    if start is None:
        start = str(np.min([int(list(gene_exons.start)[0]),
                            int(list(gene_exons.start)[0])]) - 1000)
    if end is None:
        end = str(np.max([int(list(gene_exons.end)[-1]),
                          int(list(gene_exons.end)[-1])]) + 1000)

    coords = [f'{chrom}:{start}', f'{chrom}:{end}']
    plot_gene_isoforms(isoform_dict, coords, color_list=colores,
                       figsize=figsize, lwidth=lwidth, axes=axes, xlim=xlim)
