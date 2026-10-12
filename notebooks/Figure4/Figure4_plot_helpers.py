"""
Figure4_plot_helpers.py — plotting functions for Figure 4.

Every function here receives fully processed, plot-ready data from
Figure4_helpers.run_all() and does nothing but draw.  No file access, no
computation.

The isoform-structure drawing primitives live in analysis/GeneticsPlots.py and
are imported lazily (that module pulls in sklearn/tqdm/scipy at import time).
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'


def _genetics_plots():
    import sys
    if ANALYSIS_DIR not in sys.path:
        sys.path.append(ANALYSIS_DIR)
    import GeneticsPlots
    return GeneticsPlots


# ================================================================
# Panel a — QTLs
# ================================================================

def plot_qtl_barplot(qtl_barplot):
    """Factor-QTL counts per GTEx tissue, sorted descending."""
    qtls_sorted       = qtl_barplot['qtls_sorted']
    tissues_sorted    = qtl_barplot['tissues_sorted']
    col_colors_sorted = qtl_barplot['col_colors_sorted']

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(18), qtls_sorted, color=col_colors_sorted)
    ax.set_xlabel('Tissue')
    ax.set_ylabel('Factor QTLs')
    for i, tissue in enumerate(tissues_sorted):
        tissue_text = tissue + ' (' + str(qtls_sorted[i]) + ')'
        ax.text(i - 0.15, 100, tissue_text, rotation=90, size=8)
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    plt.plot([-0.5, 17.5], [np.median(qtls_sorted), np.median(qtls_sorted)], 'r--')
    plt.text(17, 3250, f'Median:\n{np.median(qtls_sorted)}')


# ================================================================
# Panel b — upsetplot_coloc
# ================================================================

def plot_upset_coloc(upset_data):
    """UpSet plot of RNA-processing classes across colocalised GWAS loci."""
    from upsetplot import UpSet

    upset = UpSet(upset_data, sort_categories_by=None)
    colores = ['#55B8A9', '#CCCC47', '#8D89C4', '#E8402D']
    upset.style_subsets(present="Intron Retention",  facecolor=colores[0])
    upset.style_subsets(present="Alt. PolyA",        facecolor=colores[1])
    upset.style_subsets(present="Cassette Exon",     facecolor=colores[2])
    upset.style_subsets(present="MXE",               facecolor=colores[2])
    upset.style_subsets(present="Other",             facecolor=colores[2])
    upset.style_subsets(present="Efficiency factor", facecolor="tab:gray")

    axes = upset.plot()

    # Annotate the intersection bar chart (top axes)
    ax_bars = axes['intersections']
    for bar in ax_bars.patches:
        height = bar.get_height()
        if height > 0:
            ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha='center', va='bottom', fontsize=8
            )


# ================================================================
# Isoform structure panels (*.gencode, *.torino)
# ================================================================

def plot_annotation_panel(panel, figsize=(20, 5)):
    """Exon-structure tracks for every transcript of one gene."""
    GeneticsPlots = _genetics_plots()
    GeneticsPlots.plot_isoform_annotations(
        panel['bed'], panel['gene'], colores=['black'] * 200,
        iso_order=panel['iso_order'], figsize=figsize)


# ================================================================
# Genotype coverage metaplots
# ================================================================

def plot_coverage_panel(panel, xlim, figwidth=12, linewidth=3, ylim=None):
    """Genotype-stratified coverage over the locus, with isoform tracks below."""
    GeneticsPlots = _genetics_plots()
    GeneticsPlots.plot_genotype_coverage(
        panel['coverage_dir'], panel['iso'], panel['gene'],
        xlim=xlim, figwidth=figwidth, linewidth=linewidth, ylim=ylim)


# ================================================================
# Genotype boxplots
# ================================================================

def plot_qtl_boxplot(boxplot_df, isoform, ylabel, xlabel, color):
    """Normalised isoform loadings split by genotype at the lead SNP."""
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.boxplot(data=boxplot_df, x='genotype', y=isoform, width=.5, fliersize=0,
                linewidth=2, patch_artist=True, ax=ax, hue='genotype',
                boxprops=dict(edgecolor=color, facecolor='white'),
                whiskerprops=dict(color=color, linewidth=2),
                capprops=dict(color=color, linewidth=2),
                medianprops=dict(color=color, linewidth=2))
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    sns.stripplot(data=boxplot_df, x='genotype', y=isoform, c=color, alpha=0.4,
                  rasterized=True)
    ax.set_xlabel(xlabel)


# ================================================================
# GWAS / QTL colocalisation scatters
# ================================================================

def plot_coloc_scatter(merged_df, lead_var, rsid, xlabel, ylabel,
                       label_offset, rasterized=False):
    """-log10 GWAS p-values against -log10 factor-QTL p-values."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(np.array(-np.log10(merged_df.pval_x)),
               np.array(-np.log10(merged_df.pval_y)),
               facecolor='none', edgecolor='black', s=10, rasterized=rasterized)
    # The original cells passed 1-element arrays as the scatter/text position.
    # Current matplotlib rejects an array as a Text position at draw time
    # ("only 0-dimensional arrays can be converted to Python scalars"), so the
    # lead-SNP coordinates are taken as scalars here. Same point, same label.
    lead = merged_df.loc[merged_df.var_id == lead_var]
    lead_x = float(-np.log10(lead.pval_x.iloc[0]))
    lead_y = float(-np.log10(lead.pval_y.iloc[0]))
    ax.scatter(lead_x, lead_y, c='red')
    ax.text(lead_x + label_offset[0], lead_y + label_offset[1], rsid, c='red')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)


# ================================================================
# Torino factor tracks
# ================================================================

def factor_lm(factor, strand=None):
    """Flatten the 5'->3' coverage trend out of a factor (Figure_4.ipynb cell 167)."""
    from scipy.stats import linregress

    x = np.arange(1, len(factor) + 1)
    y = np.array(factor)
    y /= np.quantile(y, 0.99)

    y_ = y[y > 0.25]
    x_ = x[y > 0.25]

    slope, intercept, _, _, _ = linregress(x_, y_)

    if strand is None:
        y_model = x * slope + intercept
        y_corrected = y / np.maximum(y_model, 0.1)
        y_corrected = np.where(y > 0.1, y_corrected, y)
    else:
        if ((((strand == 'plus') or (strand == '+')) and slope > 0) or
                (((strand == 'minus') or (strand == '-')) and slope < 0)):
            y_model = x * slope + intercept
            y_corrected = y / np.maximum(y_model, 0.1)
            y_corrected = np.where(y > 0.1, y_corrected, y)
        else:
            y_corrected = factor

    return y_corrected


def plot_factor_tracks(EF, gene, K, title=None, fill=True, smooth=False, colores=None,
                       q=0.99, figsize=None, ax=None, xlim=None, plot_coords=None,
                       lwidth=None, mute_y_axis=False, strand='+'):
    """Stacked factor tracks (Figure_4.ipynb cell 167)."""
    EF = np.minimum(EF / EF.quantile(q, axis=0), 1)

    if colores is None:
        if K <= 5:
            colores = ['tab:blue', 'tab:orange', 'tab:green', 'goldenrod', 'tab:red']
        else:
            colores = sns.color_palette("tab10")

    if K <= 3:
        S = K
    else:
        S = K * 0.8

    if figsize is None:
        figsize = (15, S)

    if ax is None:
        fig, ax = plt.subplots(nrows=K, figsize=figsize,
                               gridspec_kw={'height_ratios': [2] * K,
                                            'wspace': 0.3, 'hspace': 0.3})

    start = int(EF.index[0].split(':')[1])
    end = int(EF.index[-1].split(':')[1])
    length = EF.shape[0]

    coords = np.linspace(start, end, num=length)

    for i in range(K):
        factor = f'factor{i+1}'
        scaled_y = EF[factor]
        if strand == '-':
            scaled_y = scaled_y[::-1]
        if fill:
            ax[i].fill_between(coords, np.zeros(len(coords)), scaled_y,
                               color=colores[i], alpha=0.3)
        ax[i].plot(coords, scaled_y, c=colores[i], alpha=0.9, linewidth=lwidth)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if (i < K - 1) or (not plot_coords):
            ax[i].set_xticks([])
            ax[i].spines['bottom'].set_visible(False)

        if mute_y_axis:
            ax[i].spines['left'].set_visible(False)
            ax[i].set_yticks([])

        ax[i].set_xlim([xlim[0], xlim[1]])

    if title:
        fig.suptitle(title, fontsize=12, x=0.5, y=1.1)


def plot_factors_wrapper(EF_smooth, coords, gene, strand='+', colores=None, figsize=None,
                         lm=True, factor_list=None, ax=None, xlim=None,
                         plot_coords=False, lwidth=3, mute_y_axis=True):
    """
    Detrend and draw the smoothed torino factors (Figure_4.ipynb cell 167).

    The notebook redefines this on top of the GeneticsPlots.py version; the
    notebook version — which takes the factorisation directly rather than an
    `ebpmf` key — is the one that was live when the figures were saved.  Here it
    takes the already-extracted `EF_smooth` matrix and `coords` instead of the
    rpy2 RDS handle.
    """
    EF = pd.DataFrame(EF_smooth[1:-1, :])
    coords = list(coords)[1:-1]
    EF.index = coords

    if strand == '-':
        EF = EF.loc[EF.index[::-1]]

    if xlim is None:
        xlim1 = int(coords[0].split(':')[1]) - 1000
        xlim2 = int(coords[-1].split(':')[1]) + 1000
        xlim = (xlim1, xlim2)

    EF.columns = [f'factor{str(i+1)}' for i in range(EF.shape[1])]

    if factor_list is not None:
        EF = EF[factor_list]
        EF = pd.DataFrame(EF)
        K = EF.shape[1]
        EF.columns = [f'factor{str(i)}' for i in range(1, K + 1)]

    for factor in EF.columns:
        if lm:
            y = factor_lm(EF[factor], strand)
        else:
            y = np.array(EF[factor])
        EF[factor] = y

    plot_factor_tracks(EF / EF.quantile(0.99, axis=0), gene, EF.shape[1],
                       colores=colores, figsize=figsize, ax=ax, xlim=xlim,
                       plot_coords=plot_coords, mute_y_axis=mute_y_axis,
                       lwidth=lwidth, strand=strand)


def plot_factors_with_annot(panel, colores=sns.color_palette('tab10'), linewidth=5,
                            figwidth=12, ylim=None):
    """
    Torino factor tracks with the isoform structures underneath
    (Figure_4.ipynb cell 170).
    """
    GeneticsPlots = _genetics_plots()

    iso = panel['iso']
    gene = panel['gene']
    w = panel['w']
    factor_list = panel['factor_list']
    xlim = panel['xlim']

    iso_gene = iso.loc[iso.gene_id == gene]
    strand = iso_gene.strand.iloc[0]

    coords = [int(x.split(':')[1]) for x in panel['coords_all']]
    start = coords[0] - w
    end = coords[-1] + w

    K = len(iso_gene.transcript_id.unique())

    if xlim is None:
        xlim = (start, end)

    if factor_list is None:
        factor_list = [f'factor{str(x+1)}' for x in range(10)]

    F = len(factor_list)
    figsize = (figwidth, (F + K) / 1.5)

    fig, axes = plt.subplots(F + K, 1, figsize=figsize, height_ratios=[4] * F + [1] * K,
                             gridspec_kw={'wspace': 0, 'hspace': 0.05})

    plot_factors_wrapper(panel['EF_smooth'], panel['coords'], gene,
                         factor_list=factor_list, xlim=xlim, mute_y_axis=True,
                         ax=axes[:F], strand=strand)

    GeneticsPlots.plot_isoform_annotations(iso, gene, colores=['midnightblue'] * 50,
                                           axes=axes[F:], xlim=xlim)
