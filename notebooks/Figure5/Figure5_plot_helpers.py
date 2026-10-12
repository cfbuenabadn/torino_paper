"""
Figure5_plot_helpers.py — plotting functions for Figure 5 (Alzheimer's disease).

Every function receives fully processed, plot-ready data from
Figure5_helpers.run_all() and does nothing but draw. No file access, no
computation.

The isoform / coverage drawing primitives live in analysis/GeneticsPlots.py and
are imported lazily (that module pulls in sklearn/tqdm/scipy at import time).
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'


def _genetics_plots():
    import sys
    if ANALYSIS_DIR not in sys.path:
        sys.path.append(ANALYSIS_DIR)
    import GeneticsPlots
    return GeneticsPlots


# ================================================================
# Gene example panels — coverage / loadings boxplot / factor tracks
# ================================================================

def plot_coverage_panel(panel, figwidth=12, linewidth=3):
    """
    ENSG*.{tissue}.coverage — genotype-stratified coverage with isoform tracks.

    Source: Figure5_Genetics_Plots.ipynb (cells 23, 33, 83, 94, 121, 144).
    """
    GeneticsPlots = _genetics_plots()
    GeneticsPlots.plot_genotype_coverage(
        panel['coverage_dir'], panel['iso'], panel['gene'],
        figwidth=figwidth, linewidth=linewidth)


def plot_loading_boxplot(df, factor, ylabel, xlabel, color=None, palette=None):
    """
    ENSG*.boxplot / .boxplot_tab — factor loadings split by genotype.

    Two styles occur in the source notebook and both are kept:
      * `palette=["darkred", "orangered", "gold"]` with black edges (the
        '.boxplot' figures)
      * a single `color` outline on white (the '.boxplot_tab' figures)

    Source: Figure5_Genetics_Plots.ipynb (cells 24, 48, 49, 91, 95, 113,
    127, 128, 129, 145).
    """
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    if palette is not None:
        sns.boxplot(data=df, x='genotype', y=factor, width=.5, fliersize=0,
                    linewidth=2, patch_artist=True, ax=ax, hue='genotype',
                    boxprops=dict(edgecolor='black'), palette=palette)
        strip_c = 'black'
    else:
        sns.boxplot(data=df, x='genotype', y=factor, width=.5, fliersize=0,
                    linewidth=2, patch_artist=True, ax=ax, hue='genotype',
                    boxprops=dict(edgecolor=color, facecolor='white'),
                    whiskerprops=dict(color=color, linewidth=2),
                    capprops=dict(color=color, linewidth=2),
                    medianprops=dict(color=color, linewidth=2))
        strip_c = color
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    sns.stripplot(data=df, x='genotype', y=factor, c=strip_c, alpha=0.4,
                  rasterized=True)
    if xlabel is not None:          # one source cell sets no x label
        ax.set_xlabel(xlabel)


def factor_lm(factor, strand=None):
    """Flatten the 5'->3' coverage trend out of a factor (cell 130)."""
    from scipy.stats import linregress
    x = np.arange(1, len(factor) + 1)
    y = np.array(factor)
    y /= np.quantile(y, 0.99)
    y_, x_ = y[y > 0.25], x[y > 0.25]
    slope, intercept, _, _, _ = linregress(x_, y_)
    if strand is None or ((((strand == 'plus') or (strand == '+')) and slope > 0) or
                          (((strand == 'minus') or (strand == '-')) and slope < 0)):
        y_model = x * slope + intercept
        y_corrected = y / np.maximum(y_model, 0.1)
        return np.where(y > 0.1, y_corrected, y)
    return factor


def plot_factor_tracks(EF, gene, K, title=None, fill=True, colores=None, q=0.99,
                       figsize=None, ax=None, xlim=None, plot_coords=None,
                       lwidth=None, mute_y_axis=False, strand='+'):
    """Stacked factor tracks (cell 130)."""
    EF = np.minimum(EF / EF.quantile(q, axis=0), 1)
    if colores is None:
        colores = (['tab:blue', 'tab:orange', 'tab:green', 'goldenrod', 'tab:red']
                   if K <= 5 else sns.color_palette("tab10"))
    if figsize is None:
        figsize = (15, K if K <= 3 else K * 0.8)
    if ax is None:
        fig, ax = plt.subplots(nrows=K, figsize=figsize,
                               gridspec_kw={'height_ratios': [2] * K,
                                            'wspace': 0.3, 'hspace': 0.3})
    start = int(EF.index[0].split(':')[1])
    end = int(EF.index[-1].split(':')[1])
    coords = np.linspace(start, end, num=EF.shape[0])
    for i in range(K):
        scaled_y = EF[f'factor{i+1}']
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


def plot_factors_wrapper(EF_smooth, coords, gene, strand='+', colores=None,
                         figsize=None, lm=True, factor_list=None, ax=None,
                         xlim=None, plot_coords=False, lwidth=3, mute_y_axis=True):
    """Detrend and draw the smoothed factors (cell 130)."""
    EF = pd.DataFrame(EF_smooth[1:-1, :])
    coords = list(coords)
    EF.index = coords
    if strand == '-':
        EF = EF.loc[EF.index[::-1]]
    if xlim is None:
        xlim = (int(coords[0].split(':')[1]) - 1000,
                int(coords[-1].split(':')[1]) + 1000)
    EF.columns = [f'factor{str(i+1)}' for i in range(EF.shape[1])]
    if factor_list is not None:
        EF = pd.DataFrame(EF[factor_list])
        EF.columns = [f'factor{str(i)}' for i in range(1, EF.shape[1] + 1)]
    for factor in EF.columns:
        EF[factor] = factor_lm(EF[factor], strand) if lm else np.array(EF[factor])
    plot_factor_tracks(EF / EF.quantile(0.99, axis=0), gene, EF.shape[1],
                       colores=colores, figsize=figsize, ax=ax, xlim=xlim,
                       plot_coords=plot_coords, mute_y_axis=mute_y_axis,
                       lwidth=lwidth, strand=strand)


def plot_factor_panel(panel, colores=None, w=1000, figwidth=12, xlim=None,
                      plot_annot=True, label_factors=True, lm=True):
    """
    ENSG*.factors — factor tracks with the isoform structures below.

    Source: Figure5_Genetics_Plots.ipynb (cells 134, 86; function from cell 130).
    This is a different function from the same-named one in Figure 4: it reads
    the snmf_5 factorisation and indexes EF_smooth by `filtered_coords`.
    """
    if colores is None:
        colores = sns.color_palette('tab10')
    iso, gene = panel['iso'], panel['gene']
    strand = panel['strand']
    factor_list = panel['factor_list']
    K = len(iso.transcript_id.unique())

    if xlim is None:
        xc = panel['xlim_coords']
        xlim = (xc[0] - w, xc[-1] + w)
    if factor_list is None:
        factor_list = [f'factor{str(x+1)}' for x in range(5)]

    F = len(factor_list)
    fk, hr = (F + K, [4] * F + [1] * K) if plot_annot else (F, [1] * F)
    fig, axes = plt.subplots(fk, 1, figsize=(figwidth, fk / 1.5), height_ratios=hr,
                             gridspec_kw={'wspace': 0, 'hspace': 0.05})

    plot_factors_wrapper(panel['EF_smooth'], panel['coords'], gene,
                         factor_list=factor_list, xlim=xlim, mute_y_axis=True,
                         lm=lm, ax=axes[:F], strand=strand, colores=colores)

    if label_factors:
        for i, factor in enumerate(factor_list):
            xend = xlim[0] + ((xlim[1] - xlim[0]) * 0.85)
            axes[i].text(xend, axes[i].get_ylim()[1] * 0.7, factor, c=colores[i])

    if plot_annot:
        _genetics_plots().plot_isoform_annotations(
            iso, gene, colores=['midnightblue'] * 50, axes=axes[F:], xlim=xlim)


# ================================================================
# beta direction / counts summaries
# ================================================================


def plot_beta_percentage(stats):
    """gao_k3/beta_percentage_allIR — AD_K3_RetainedIntrons.ipynb (cell 68)."""
    total_ratio = stats['total_ratio']
    si_pos_ratio = stats['si_pos_ratio']
    ir_pos_ratio = stats['ir_pos_ratio']
    fig, ax = plt.subplots(figsize=(1.5, 3))
    plt.bar(range(1, 4), [total_ratio, si_pos_ratio, ir_pos_ratio],
            color=['#d9d9d9', 'tab:blue', 'tab:red'])
    plt.plot([2, 2], [0.55, 1.01], c='gray')
    plt.plot([2, 3 - 0.07], [1.01, 1.01], c='gray')
    plt.plot([3 - 0.07, 3 - 0.07], [1.01, ir_pos_ratio + 0.01], c='gray')
    plt.text(2.225, 1.02, '***', size=8)
    plt.plot([1, 1], [total_ratio + 0.01, 1.1], c='gray')
    plt.plot([1, 3.07], [1.1, 1.1], c='gray')
    plt.plot([3.07, 3.07], [1.1, ir_pos_ratio + 0.01], c='gray')
    plt.text(1.9, 1.11, '**', size=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim((0.25, 3.75))
    ax.set_ylabel('Percentage of factors\nwith beta > 0', size=9)
    ax.set_xticks([])
    ax.set_xlabel('Significant\nfactors', size=9)
    ax.text(0.85, 0.05, 'No IR genes (88.3%)', rotation=90)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.text(1.85, 0.05, 'No IR (29.6%)', rotation=90)
    ax.text(2.85, 0.05, 'IR (98.2%)', rotation=90)
    ax.spines['left'].set_bounds(0, 1)


def plot_beta_direction(stats):
    """beta_direction — Fig5_AD_figures.ipynb (cell 73)."""
    p_no_ir, p_ir = stats['positive_sig_no_ir'], stats['positive_sig_ir']
    fig, ax = plt.subplots(figsize=(1, 3))
    plt.bar(range(1, 3), [p_no_ir, p_ir], color=['#d9d9d9', '#8dd3c7'])
    plt.plot([1, 1], [p_no_ir + 0.02, 1.1], c='gray')
    plt.plot([1, 2], [1.1, 1.1], c='gray')
    plt.plot([2, 2], [1.1, p_ir + 0.02], c='gray')
    plt.text(0.45, 1.15, 'pval: 3.3e-26', size=8)
    ax.spines[['top', 'right']].set_visible(False)
    plt.plot([0.5, 2.5], [0.5, 0.5], 'r--')
    plt.text(2.8, 0.48, '50%', c='red')
    ax.set_xlim((0.25, 2.75))
    ax.set_ylabel('Percentage beta > 0\non top factor', size=9)
    ax.set_xticks([])
    ax.set_xlabel('Significant\nfactors', size=9)
    ax.text(0.85, 0.05, 'Other (2728)', rotation=90)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.text(1.85, 0.05, 'Intron retention (2215)', rotation=90)
    ax.spines['left'].set_bounds(0, 1)



# ================================================================
# GSEA dotplot / KNOP1 braak boxplot
# ================================================================

GSEA_YLABELS = ['HP absent speech', 'GOCC presynapse', 'GOCC axon',
                'GOCC somatodendritic\ncompartment', 'GOCC neuron to\nneuron synapse',
                'GOCC denditric tree', 'GOCC postsynapse',
                'GOCC postsynaptic\nspecialization', 'GOCC neuron projection',
                'GOCC synapse']


def plot_gsea_dotplot(res2d, xticks=None):
    """
    GSEA_AD — gene-set enrichment of the significant AD genes.

    `xticks` reproduces the extra axis styling that only the Fig5_ad/ version
    carries (Fig5_AD_figures.ipynb cell 75); the gao_k3/ version
    (AD_K3_RetainedIntrons.ipynb cell 74) omits it.
    """
    from gseapy import dotplot
    dp = dotplot(res2d, title='Top AD affected genes', cmap='viridis', size=8,
                 figsize=(3, 5))
    dp.set_yticklabels(GSEA_YLABELS, size=8)
    dp.set_title('Significant AD genes', size=12)
    dp.set_xlabel('Combined score', size=12)
    if xticks is not None:
        dp.set_xticks(xticks)
        dp.set_xticklabels([str(x) for x in xticks], size=8)
    return dp


def plot_knop1_braak(mdf):
    """gao_k3/KNOP1_braak_boxplot — AD_K3_RetainedIntrons.ipynb (cell 90)."""
    fig, ax = plt.subplots(figsize=(3, 4))
    sns.boxplot(x='braaksc', y='ENSG00000103550.factor_2', data=mdf,
                hue='condition', ax=ax, palette=['tab:blue', 'tab:red'])
    ax.legend(title='condition', frameon=False, loc='center left',
              bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('KNOP1 factor 2 loadings')
    ax.set_xlabel('Braak stage')
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])
    ax.spines[['top', 'right']].set_visible(False)


# ================================================================
# Intron-retention gene-body heatmaps
# ================================================================
#
# The source notebooks carry seven near-identical ~150-line copies of this loop
# differing only in the intron score column, colour scale, label palette and
# output name. Only the figure below is still requested:
#
#   gao_k3/IR_only_progression_labeled.PCC.{i}
#                                     AD_K3_RetainedIntrons.ipynb cell 119
#
# `gencode_beds` is precomputed once in Figure5_helpers.run_all(); the originals
# re-ran a pybedtools merge for every gene of every figure.

# colour-scale bounds used by the progression figures (select_color in the source)
BOUNDS_WIDE = [-1.5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 1.5]


def select_color(z, colores, bounds=BOUNDS_WIDE):
    """Bin a z score onto the 10-colour vlag scale."""
    lo = bounds[1:5]        # negative breakpoints, most negative first
    hi = bounds[9:5:-1]     # positive breakpoints, most positive first
    if z <= lo[0]:
        return colores[0]
    elif z <= lo[1]:
        return colores[1]
    elif z <= lo[2]:
        return colores[2]
    elif z <= lo[3]:
        return colores[3]
    elif z <= 0:
        return colores[4]
    elif z >= hi[0]:
        return colores[9]
    elif z >= hi[1]:
        return colores[8]
    elif z >= hi[2]:
        return colores[7]
    elif z >= hi[3]:
        return colores[6]
    elif z > 0.0:
        return colores[5]
    return 'gray'


def add_vertical_colorbar(ax, colores, bounds=BOUNDS_WIDE):
    cmap = ListedColormap(colores)
    norm = BoundaryNorm(bounds, cmap.N)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        orientation='vertical', boundaries=bounds, pad=0.1)
    cbar.set_label('Z score')


def place_labels(order_df, gene_label, min_gap=1.5, n_iter=200):
    """
    1D force-directed label placement.

    Starts each label at its true row position, then iteratively resolves
    overlaps by pushing overlapping pairs apart. `gene_label` maps gene -> colour
    for the genes that get a label (absent genes are unlabelled).
    """
    entries, pos = [], 0
    for _, row in order_df.sort_values('position').iterrows():
        if gene_label.get(row.gene) is not None:
            entries.append((pos, row.gene))
        pos -= 1
    if not entries:
        return {}

    true_pos = np.array([e[0] for e in entries], dtype=float)
    label_y = true_pos.copy()
    genes = [e[1] for e in entries]

    for _ in range(n_iter):
        changed = False
        order = np.argsort(-label_y)
        for k in range(len(order) - 1):
            a, b = order[k], order[k + 1]
            overlap = min_gap - (label_y[a] - label_y[b])
            if overlap > 0:
                label_y[a] += overlap / 2
                label_y[b] -= overlap / 2
                changed = True
        if not changed:
            break
    return {gene: label_y[i] for i, gene in enumerate(genes)}


def plot_ir_heatmap(panel, score_col, title=None, bounds=BOUNDS_WIDE,
                    colour_ticks=False, figsize=(2, 4)):
    """
    Draw one gene-body IR heatmap.

    panel        : dict from Figure5_helpers with order_df, introns_bed,
                   gencode_beds, gene_names, gene_label, placements
    score_col    : column of introns_bed holding the z score to colour by
    title        : axis title (the braak-stage figures set one, IR_labeled does not)
    colour_ticks : colour the TSS/TES ticks with the gene's label colour
                   (only IR_progression_labeled.AC does this, cell 123)
    """
    order_df = panel['order_df']
    introns_bed = panel['introns_bed']
    gencode_beds = panel['gencode_beds']
    gene_names = panel['gene_names']
    gene_label = panel['gene_label']
    placements = panel['placements']

    colores = sns.color_palette("vlag", 10)
    fig, axes = plt.subplots(ncols=2, figsize=figsize,
                             gridspec_kw={'width_ratios': [0.25, 1], 'wspace': 0.05})
    pos = 0
    for _, row in order_df.sort_values('position').iterrows():
        gene, strand = row.gene, row.strand
        gencode_bed = gencode_beds.get(gene)
        if gencode_bed is None:
            continue
        start_ = gencode_bed.end.iloc[0]
        end_ = gencode_bed.start.iloc[-1]

        col = gene_label.get(gene)
        tick_col = (col if col is not None else 'black') if colour_ticks else 'black'
        axes[1].plot([-0.02, 0], [pos, pos], c=tick_col, linewidth=0.5)
        axes[1].plot([1, 1.02], [pos, pos], c=tick_col, linewidth=0.5)

        axes[0].spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        if col is not None:
            label_y = placements[gene]
            axes[0].plot([-0.4, -0.05], [label_y, pos], c=col, linewidth=0.75)
            axes[0].text(-0.43, label_y, gene_names[gene], c=col, size=8,
                         ha='right', va='center')

        if gencode_bed.shape[0] >= 3:
            for _, exon in gencode_bed.iloc[1:-1].iterrows():
                ex_start = (exon.start - start_) / (end_ - start_)
                ex_end = (exon.end - start_) / (end_ - start_)
                if strand == '-':
                    ex_start, ex_end = 1 - ex_start, 1 - ex_end
                axes[1].plot([ex_start, ex_end], [pos, pos], c='black',
                             linewidth=0.5, rasterized=True)

        for _, intron in introns_bed.loc[introns_bed.gene == gene].iterrows():
            int_start = (intron.start - start_) / (end_ - start_)
            int_end = (intron.end - start_) / (end_ - start_)
            if strand == '-':
                int_start, int_end = 1 - int_start, 1 - int_end
            axes[1].plot([int_start, int_end], [pos, pos], linewidth=0.5,
                         c=select_color(intron[score_col], colores, bounds),
                         rasterized=True)
        pos -= 1

    axes[1].set_yticks([])
    if title is not None:
        axes[1].set_title(title, size=9)
    axes[1].spines[['top', 'left', 'right']].set_visible(False)
    axes[1].set_xticks((-0.02, 1.02))
    axes[1].set_xticklabels(['TSS', 'TES'], size=8, rotation=45)
    axes[1].spines['bottom'].set_bounds(-0.02, 1.02)
    axes[0].set_ylim(axes[1].get_ylim())

    axes[1].spines[['top', 'bottom', 'right', 'left']].set_visible(False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    add_vertical_colorbar(axes[1], colores, bounds)



# ================================================================
# PKFP coverage / TWAS summary / 3' UTR heatmap
# ================================================================

def plot_pkfp_counts(panel):
    """
    gao_k3/PKFP_{PCC,AC,DLPFC}_counts — mean normalised coverage in AD vs
    control, with the two PKFP isoform structures below.

    Source: Figure5_examples.ipynb (cells 56, 57, 59) — three copies differing
    only in `tissue_id` and the output name.

    Note the axis title is hard-coded to 'dorsolateral prefrontal cortex' in all
    three source cells, including the PCC and AC ones; it is preserved verbatim.
    """
    GeneticsPlots = _genetics_plots()
    coords_, df1, df2 = panel['coords_'], panel['df1'], panel['df2']

    fig, axes = plt.subplots(nrows=4, figsize=(10, 2.5),
                             gridspec_kw={'height_ratios': [6, 1, 1, 0.1],
                                          'hspace': 0.3})
    ax = axes[0]
    ax.plot(coords_, np.array(df1.rolling(150,).mean().T.iloc[0]),
            color='tab:red', label='Alzheimer disease')
    ax.plot(coords_, np.array(df2.rolling(150,).mean().T.iloc[0]),
            color='tab:gray', label='Control')
    ax.legend(frameon=False, bbox_to_anchor=(1, 1.1))

    GeneticsPlots.plot_isoform_annotations(
        panel['snmf_exons'], 'ENSG00000067057',
        colores=['navy', 'navy'], axes=axes[1:-1], lwidth=4)

    xlim = (3066503 - 4000, 3137800 + 3700)
    for a in (axes[-3], axes[-2], axes[-1]):
        a.set_xlim(list(xlim))

    xticks = [3080000, 3100000, 3120000, 3140000]
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([str(x) for x in xticks])
    axes[-1].spines[['top', 'right', 'left']].set_visible(False)
    axes[-1].set_yticks([])
    axes[-1].set_xlabel('chrom 10')
    axes[0].set_xticks([])
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean normalized counts')
    axes[0].set_title('PKFP, dorsolateral prefrontal cortex')
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([str(int(x)) for x in xticks])
    axes[0].set_ylim((0, 0.00012))


def plot_twas_summary(panel):
    """
    TWAS_summary — per-gene TWAS p-values, coloured by colocalisation and
    shaped by Mendelian-randomisation support.

    Source: Figure5_summary_plots.ipynb (cell 41).
    """
    import matplotlib.lines as mlines
    df_significant = panel['df_significant']
    coloc_phenos = panel['coloc_phenotypes']
    mr_phenos = panel['mr_phenotypes']
    gene_names = panel['gene_names']

    df_sig_gp = df_significant.groupby('molecular_id')
    fig, ax = plt.subplots(figsize=(4, 6))
    i = 0
    glist, ilist = [], []
    for gene, pval in df_significant.groupby('molecular_id').twas_pval.min().sort_values().items():
        gene_df = df_sig_gp.get_group(gene)
        ax.plot([0, -np.log10(pval)], [i, i], c='tab:gray', zorder=1)
        pvals_ = gene_df.loc[gene_df.passes_cutoff].groupby('phenotype').twas_pval.min()
        for pheno, pval_ in pvals_.items():
            c = 'tab:red' if pheno in coloc_phenos else 'tab:blue'
            marker = '^' if pheno in mr_phenos else 'o'
            ax.scatter([-np.log10(pval_)], [i], c=c, zorder=2, marker=marker, alpha=0.75)
        glist.append(gene_names[gene])
        ilist.append(i)
        i -= 1

    ax.set_yticks(ilist)
    ax.set_yticklabels(glist)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.set_xlabel(r'$-log_{10}(p)$')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim([i, 1])

    color_handles = [
        mlines.Line2D([], [], color='tab:red', marker='o', linestyle='None',
                      markersize=6, label='Colocalized'),
        mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None',
                      markersize=6, label='Not colocalized'),
    ]
    legend1 = ax.legend(handles=color_handles, title='Colocalization',
                        loc='lower right', fontsize=7, title_fontsize=7,
                        frameon=False)
    ax.add_artist(legend1)

    shape_handles = [
        mlines.Line2D([], [], color='k', marker='^', linestyle='None',
                      markersize=6, label='Significant'),
        mlines.Line2D([], [], color='k', marker='o', linestyle='None',
                      markersize=6, label='Not significant'),
    ]
    ax.legend(handles=shape_handles, title='Mendelian Randomization',
              loc='center right', fontsize=7, title_fontsize=7, frameon=False)
    plt.tight_layout()


UTR3_PALETTE = {
    'ctrl_AC': '#4e9af1', 'ctrl_DLPC': '#1a5fb4', 'ctrl_PCC': '#0d3b8c',
    'AD_AC': '#f16a4e', 'AD_DLPC': '#c0392b', 'AD_PCC': '#7b0e0e',
}


def plot_utr3_changes(panel):
    """
    UTR3_changes — clustermap of 3' UTR isoform usage across condition x tissue.

    Source: AD_UTR.ipynb (cell 129 calling build_el_matrix_and_plot, cell 106).
    The matrix construction lives in Figure5_helpers; only the drawing is here.
    """
    import matplotlib.patches as mpatches
    el_means = panel['el_means']
    col_colors = [UTR3_PALETTE[key] for key in el_means.columns]
    vmax = np.nanpercentile(np.abs(el_means.values), 95)

    g = sns.clustermap(el_means, col_colors=col_colors, row_cluster=False,
                       col_cluster=False, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       xticklabels=True, yticklabels=True, figsize=(6, 6),
                       cbar_kws={'label': 'z-score (mean)'}, rasterized=True)

    legend_patches = [mpatches.Patch(color=col, label=label)
                      for label, col in UTR3_PALETTE.items()]
    g.ax_heatmap.legend(handles=legend_patches, bbox_to_anchor=(1.25, 1),
                        loc='upper left', borderaxespad=False, frameon=False,
                        title='Condition x Tissue')

    positions, labels = panel['label_positions'], panel['label_names']
    if positions:
        g.ax_heatmap.set_yticks([p + 0.5 for p in positions])
        g.ax_heatmap.set_yticklabels(labels, fontsize=7)
    else:
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=7)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
    return g
