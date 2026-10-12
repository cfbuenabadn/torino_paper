"""
Figure3_plot_helpers.py — plotting functions for Figure 3.

Every function here receives fully processed, plot-ready data from
Figure3_helpers.run_all() and does nothing but draw.  No file access, no
computation.
"""

import colorsys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# ================================================================
# Palettes
# ================================================================

# Tissue colours used by the *_top_quantiles heatmaps (analysis/species_heatmap.py)
TISSUE_COLORS = {
    'Brain':                 '#EEEE00',
    'Cerebellum':            '#EEEE00',
    'Heart':                 '#B452CD',
    'Liver':                 '#CDB79E',
    'Testis':                '#A6A6A6',
    'Muscle':                '#7A67EE',
    'Muscle_Skeletal':       '#7A67EE',
    'Breast_Mammary_Tissue': '#00CDCD',
    'Mammary_Gland':         '#00CDCD',
    'Whole_Blood':           '#FF00FF',
    'Blood':                 '#FF00FF',
    'Lung':                  '#9ACD32',
    'Skin':                  '#3A5FCD',
    'Adipose_Subcutaneous':  '#FFA54F',
    'Subcutaneous_Adipose':  '#FFA54F',
    'Kidney':                '#FFB6C1',
    'Ovary':                 '#8B1C62',
    'Colon':                 '#CD9B1D',
}
_DEFAULT_COLOR = '#CCCCCC'

# Tissue colours used by the significant-SSMD barplot (analysis/SSMD.ipynb)
BARPLOT_TISSUE_COLORS = {
    'Adipose_Subcutaneous':  '#FFA54F',
    'Artery_Aorta':          '#8B1C62',
    'Brain':                 '#EEEE00',
    'Cerebellum':            '#EEEE00',
    'Breast_Mammary_Tissue': '#00CDCD',
    'Mammary_Gland':         '#00CDCD',
    'Heart':                 '#7A378B',
    'Liver':                 '#CDB79E',
    'Lung':                  '#9ACD32',
    'Muscle_Skeletal':       '#7A67EE',
    'Pancreas':              '#CD9B1D',
    'Skin':                  '#3A5FCD',
    'Testis':                '#A6A6A6',
    'Whole_Blood':           '#FF00FF',
    'Blood':                 '#FF00FF',
    'Ovary':                 '#FFB6C1',
    'Kidney':                '#CDB79E',
    'Muscle':                '#7A67EE',
    'Colon':                 '#CDB79E',
}

# Coverage-track colours for the example gene panels (plot_isoforms_lazy.py)
COVERAGE_COLORS = {
    'brain':      '#EEEE00',
    'cerebellum': '#EEEE00',
    'heart':      '#B452CD',
    'liver':      '#CDB79E',
    'testis':     '#A6A6A6',
    'adipose':    '#FFA54F',
    'artery':     '#8B1C62',
    'mammary':    '#00CDCD',
    'lung':       '#9ACD32',
    'muscle':     '#7A67EE',
    'pancreas':   '#CD9B1D',
    'skin':       '#3A5FCD',
    'wblood':     '#FF00FF',
}

COVERAGE_LABELS = {
    'brain':      'Brain',
    'cerebellum': 'Cerebellum',
    'heart':      'Heart',
    'liver':      'Liver',
    'testis':     'Testis',
    'adipose':    'Adipose Subcutaneous',
    'artery':     'Artery - Aorta',
    'mammary':    'Mammary Gland',
    'lung':       'Lung',
    'muscle':     'Muscle - Skeletal',
    'pancreas':   'Pancreas',
    'skin':       'Skin',
    'wblood':     'Whole Blood',
}


def _darken(hex_or_named, factor=0.75):
    """Return a darkened version of a matplotlib colour."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_or_named)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, max(0, l * factor), s)
    return (r2, g2, b2)


# ================================================================
# Human_significant_SSMD
# ================================================================

def plot_significant_ssmd_barplot(df_sorted, figsize=(8, 4), label_y=200,
                                  luminance_weights=(0.399, 0.587, 0.114)):
    """
    Number of genes with a significant tissue-specific isoform, per tissue.
    Source: analysis/SSMD.ipynb (Cell 26).
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (tissue, row) in enumerate(df_sorted.iterrows()):
        n = int(row['n_sig'])
        hex_color = BARPLOT_TISSUE_COLORS.get(tissue, '#888888')

        r, g, b = tuple(int(hex_color.lstrip('#')[j:j + 2], 16) for j in (0, 2, 4))
        wr, wg, wb = luminance_weights
        luminance = (wr * r + wg * g + wb * b) / 255
        label_color = 'black' if luminance > 0.5 else 'white'

        ax.bar(i, n, color=hex_color)
        ax.text(i, label_y, f'{n:,}', ha='center', va='bottom',
                rotation=90, fontsize=12, color=label_color)

    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([t.replace('_', ' ') for t in df_sorted.index],
                       rotation=45, ha='right')
    ax.set_ylabel('Number of significant genes')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()


# ================================================================
# human / mouse isoforms_cumulative_and_percentage_fdr
# ================================================================

CUMULATIVE_TISSUE_ROWS = ['Brain only', 'Brain & Cerebellum only', 'Cerebellum only',
                          'Heart only', 'Liver only', 'Testis only', 'Multiple tissues']
CUMULATIVE_COLORS = ['#EEEE00', '#CCCC00', '#EEEE00', '#B452CD',
                     '#CDB79E', '#A6A6A6', 'tab:red']
CUMULATIVE_LABELS = ['Brain', 'Brain & Cerebellum', 'Cerebellum', 'Heart',
                     'Liver', 'Testis', 'Multiple tissues']

# Hard-coded totals per conservation step, from the source notebook
HUMAN_TOTALS = np.array([150, 150 + 109, 150 + 109 + 977, 150 + 109 + 977 + 4600])
MOUSE_TOTALS = np.array([149, 149 + 107, 149 + 107 + 168, 149 + 107 + 168 + 1079,
                         149 + 107 + 168 + 1079 + 2138])

HUMAN_SPECIES_COLS  = ['All animals', 'Mammals', 'Primates', 'Human only']
HUMAN_SPECIES_TICKS = ['All species', 'Mammals', 'Primates', 'Human']
MOUSE_SPECIES_COLS  = ['All animals', 'Mammals', 'Gilres', 'Rodents', 'Mouse only']
MOUSE_SPECIES_TICKS = ['All species', 'Mammals', 'Gilres', 'Rodents', 'Mouse']


def plot_cumulative_and_percentage(df_table, species_categories, xticklabels,
                                   totals, figsize=(6, 3)):
    """
    Two-panel stacked area plot: absolute gene counts (left) and the same data
    as a percentage of the per-step total (right).

    Source: analysis/Fig3_SSMD_species.ipynb (Cell 60 for human, Cell 67 for
    mouse) — one function with the species axis parameterised.
    """
    xs = list(range(len(species_categories)))
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.1})

    # --- Left: absolute counts ---
    base = np.zeros(len(species_categories), dtype=int)
    for i, tissue in enumerate(CUMULATIVE_TISSUE_ROWS):
        species_y, n = [], 0
        for species in species_categories:
            n += df_table.loc[tissue, species]
            species_y.append(n)
        species_y = np.array(species_y) + base
        axes[0].fill_between(xs, base, species_y, color=CUMULATIVE_COLORS[i])
        base = species_y

    for x, total in zip(xs, totals):
        axes[0].text(x, total, str(total), ha='center', va='bottom', fontsize=8)

    axes[0].set_xticks(xs, xticklabels, rotation=45, ha='right')
    axes[0].set_ylabel("Genes with\ntissue-specific isoforms")
    axes[0].spines[['top', 'right']].set_visible(False)

    # --- Right: normalised ---
    base = np.zeros(len(species_categories), dtype=int)
    for i, tissue in enumerate(CUMULATIVE_TISSUE_ROWS):
        species_y, n = [], 0
        for species in species_categories:
            n += df_table.loc[tissue, species]
            species_y.append(n)
        species_y = np.array(species_y) + base
        axes[1].fill_between(xs, base / totals, species_y / totals,
                             color=CUMULATIVE_COLORS[i], label=CUMULATIVE_LABELS[i])
        base = species_y

    axes[1].set_xticks(xs, xticklabels, rotation=45, ha='right')
    axes[1].yaxis.set_label_position('right')
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    axes[1].set_ylabel("Percentage of genes with\ntissue-specific isoforms")
    axes[1].spines[['top', 'left']].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.35, 1),
                   loc='upper left', frameon=False)

    plt.tight_layout()


def plot_human_cumulative(df_table, figsize=(6, 3)):
    """`human_isoforms_cumulative_and_percentage_fdr`."""
    plot_cumulative_and_percentage(
        df_table, HUMAN_SPECIES_COLS, HUMAN_SPECIES_TICKS, HUMAN_TOTALS, figsize)


def plot_mouse_cumulative(df_table_mouse, figsize=(6, 3)):
    """`mouse_isoforms_cumulative_and_percentage_fdr`."""
    plot_cumulative_and_percentage(
        df_table_mouse, MOUSE_SPECIES_COLS, MOUSE_SPECIES_TICKS, MOUSE_TOTALS, figsize)


# ================================================================
# human_<species>_top_quantiles
# ================================================================

def plot_top_quantiles_heatmap(hm, use_enrichment=True, vmax=3.0, vmin=1.0,
                               annotate_counts=True):
    """
    Clustermap of tissue-specificity conservation between two species; colour is
    hypergeometric fold-enrichment, cell text the raw count of matching
    orthologous gene pairs.

    Source: analysis/species_heatmap.py `plot_species_heatmap`, called from
    analysis/SSMD.ipynb (Cells 153-158).
    """
    match_df  = hm['match_df']
    enrich_df = hm['enrich_df']

    row_colors = [TISSUE_COLORS.get(t, _DEFAULT_COLOR) for t in hm['tissue_list1']]
    col_colors = [TISSUE_COLORS.get(t, _DEFAULT_COLOR) for t in hm['tissue_list2']]
    plot_data = enrich_df if use_enrichment else match_df.astype(float)

    g = sns.clustermap(
        plot_data,
        cmap        = 'Blues',
        row_colors  = row_colors,
        col_colors  = col_colors,
        row_cluster = False,
        col_cluster = False,
        annot       = False,
        figsize     = hm['figsize'],
        vmax        = vmax,
        vmin        = vmin if use_enrichment else None,
        cbar_pos    = (0.02, 0.75, 0.03, 0.18),
    )

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                 rotation=45, ha='right')
    g.ax_heatmap.set_ylabel(hm['sp1'])
    g.ax_heatmap.set_xlabel(hm['sp2'])

    if annotate_counts:
        for i in range(match_df.shape[0]):
            for j in range(match_df.shape[1]):
                count = int(match_df.iloc[i, j])
                enr   = enrich_df.iloc[i, j]
                tcolor = ('white'
                          if use_enrichment and not np.isnan(enr) and enr >= vmax * 0.75
                          else 'black')
                g.ax_heatmap.text(j + 0.5, i + 0.5, str(count),
                                  ha='center', va='center', color=tcolor, fontsize=8)

    g.fig.suptitle(hm['title'], y=1.01, fontsize=10)
    return g


# ================================================================
# Example gene panels — isoform structure primitives
# ================================================================

def plot_isoform(isoform_df, ax, color, xlim1, xlim2, lwidth, strand='-'):
    is_first = True
    for _, row in isoform_df.iterrows():
        start = int(row.start)
        end   = int(row.end)
        if strand == '-':
            end_  = xlim2 - start + xlim1
            start = xlim2 - end + xlim1
            end   = end_
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
                       figsize=None, lwidth=5, xlim=None, iso_order=None,
                       strand='+'):
    if xlim is None:
        xlim1 = int(coordinates[0].split(':')[1])
        xlim2 = int(coordinates[-1].split(':')[1])
    else:
        xlim1, xlim2 = xlim

    if color_list is None:
        color_list = sns.color_palette('tab10')

    K = len(isoforms_dict)
    if figsize is None:
        figsize = (20, 3)
    if axes is None:
        _, axes = plt.subplots(K, 1, figsize=figsize)

    if K == 1:
        isoform_df = isoforms_dict['isoform_1']['df']
        plot_isoform(isoform_df, axes, color_list[0], xlim1, xlim2,
                     lwidth=lwidth, strand=strand)
    else:
        order = iso_order if iso_order is not None else list(isoforms_dict.keys())
        for i, iso in enumerate(order):
            try:
                plot_isoform(isoforms_dict[iso]['df'], axes[i],
                             color_list[i], xlim1, xlim2,
                             lwidth=lwidth, strand=strand)
            except Exception:
                print(iso)
                print(isoforms_dict[iso]['df'])


def plot_isoform_annotations(annotation_exons, gene, colores=None, start=None,
                             end=None, figsize=None, lwidth=5, iso_order=None,
                             axes=None, xlim=None):
    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]
    strand = gene_exons.strand.iloc[0]

    isoforms = iso_order if iso_order is not None \
        else sorted(gene_exons.transcript_id.unique())

    isoform_dict = {}
    for iso in isoforms:
        df = gene_exons.loc[gene_exons.transcript_id == iso].copy()
        df['transcript_id'] = iso
        isoform_dict[iso] = {'df': df}

    chrom = annotation_exons.chrom.iloc[0]
    if start is None:
        start = str(gene_exons.start.min() - 1000)
    if end is None:
        end = str(gene_exons.end.max() + 1000)

    plot_gene_isoforms(isoform_dict, [f'{chrom}:{start}', f'{chrom}:{end}'],
                       color_list=colores, figsize=figsize, lwidth=lwidth,
                       axes=axes, xlim=xlim, iso_order=iso_order, strand=strand)


# ================================================================
# Example gene panels — the two saved figures per gene/species
# ================================================================

def plot_gene_structure(panel, figsize=(8, 3), gene=None):
    """
    `<GENE>.<Species>.<Tissue>.gene` — per-tissue coverage tracks stacked above
    the isoform exon structures.

    Source: analysis/plot_isoforms_lazy.py `plot_isoforms_species`, called via
    `plot_isoforms_and_loadings_lazy` in analysis/Fig3_examples_final.ipynb.

    Passing `gene` recolours the isoforms to the hand-picked pair in
    `Figure3_helpers.ISOFORM_PICKS`: every isoform is still drawn, in the same
    order, but only the chosen two are red and navy.
    """
    from Figure3_helpers import picked_colors_in_place
    colores = (picked_colors_in_place(panel, gene) if gene
               else panel['colores'])
    tissues_n = panel['tissues_n']
    total_iso = len(panel['iso_order'])
    figsize = (figsize[0], figsize[1] + 2)

    fig, ax = plt.subplots(
        nrows=total_iso + tissues_n,
        figsize=figsize,
        gridspec_kw={'height_ratios': [3] * tissues_n + [1.5] * total_iso,
                     'wspace': 0.3, 'hspace': 0.5},
    )

    x, y, xlim = panel['x'], panel['y'], panel['xlim']
    for i, tk in enumerate(panel['track_order']):
        ax[i].plot(x, y[tk], c=COVERAGE_COLORS[tk], rasterized=True)
        ax[i].fill_between(x, 0, y[tk], color=COVERAGE_COLORS[tk],
                           alpha=0.5, rasterized=True)
        ax[i].set_xlim(xlim)
        ax[i].set_xticks([])
        ax[i].spines[['top', 'bottom', 'right']].set_visible(False)
        ax[i].text(1.01, 0.5, COVERAGE_LABELS.get(tk, tk),
                   transform=ax[i].transAxes, va='center', ha='left',
                   fontsize=12, color=_darken(COVERAGE_COLORS[tk]))

    plot_isoform_annotations(
        panel['iso'], panel['gene'], colores=colores,
        axes=ax[tissues_n:], xlim=xlim, lwidth=2.5,
        iso_order=panel['iso_order'],
    )

    ax[0].set_title(panel['gene'], style='italic')


def plot_gene_loadings(panel, gene=None):
    """
    `<GENE>.<Species>.<Tissue>.loadings` — median (+10-90 %) isoform loadings
    per tissue, with a tissue colour bar underneath.

    'Brain,Cerebellum' is folded into Cerebellum, matching the coverage tracks —
    left separate it gives one more tissue than the per-species colour list has
    entries, which shifts every colour after it and drops the last tissue.

    Passing `gene` orders the rows by the hand-picked pair in
    `Figure3_helpers.ISOFORM_PICKS`: chosen tissue-specific isoform, chosen
    non-specific isoform, then the rest.

    Source: analysis/plot_isoforms_lazy.py `plot_factor_loadings`.
    """
    from Figure3_helpers import merged_tissue_ids, picked_isoforms
    EL = panel['el_annot']
    ids = merged_tissue_ids(panel)
    samples = panel['meta'].loc[ids.index].copy()
    samples['tissue_id'] = ids
    tissues = sorted(samples.tissue_id.unique())
    if gene:
        factors, color_list = picked_isoforms(panel, gene)
    else:
        factors, color_list = panel['iso_order'], panel['colores']
    tissue_colors = panel['tissue_colors'][:len(tissues)]

    K = len(factors)
    N = len(tissues)
    n_ax = min(K, 3)   # at most 3 loading panels

    fig, axes = plt.subplots(
        n_ax + 1, 1, figsize=(1.5 * (N / 5), 0.8 * n_ax + 0.5),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2},
        height_ratios=[1] * n_ax + [0.3],
    )

    pos = 0
    pos_list = []
    max_load_ceil = 0

    for tissue in tissues:
        pos_list.append(pos)
        for i, factor in enumerate(factors):
            if i >= 2:
                k_range = (np.arange(K - 2) / (2 * (K - 2))) - 0.25
                ax_i = min(i, n_ax - 1)          # clamp to last loading panel
                plot_pos = pos + k_range[i - 2]
                max_load = np.max([EL.loc[:, factor].max(), max_load_ceil])
                max_load_ceil = max_load
            else:
                ax_i = i
                plot_pos = pos
                max_load = EL.loc[:, factor].max()

            color = color_list[i]
            tissue_samples = samples.loc[samples.tissue_id == tissue].index
            loadings = list(EL.loc[tissue_samples, factor].dropna())

            med_load = np.median(loadings)
            q9 = np.quantile(loadings, 0.9)
            q1 = np.quantile(loadings, 0.1)

            axes[ax_i].scatter([plot_pos], [med_load], c=color, s=30)
            axes[ax_i].plot([plot_pos, plot_pos], [q1, q9], c=color)
            axes[ax_i].spines[['top', 'right']].set_visible(False)
            axes[ax_i].set_ylim([-0.1, max_load + 0.1])
            axes[ax_i].set_xticks([])

        pos += 1

    axes[-1].bar(pos_list, [1] * len(tissues), color=tissue_colors)
    axes[-1].spines[['top', 'bottom', 'right', 'left']].set_visible(False)
    axes[-1].set_yticks([])
    axes[-1].set_xticks(pos_list, [t.replace('_', ' ') for t in tissues], rotation=90)
    axes[-1].set_xlim(axes[0].get_xlim())
