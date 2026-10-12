import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_factor_tracks(EF, gene, K, title=None, fill=True, smooth=False, colores=None, q=0.99, figsize=None, ax=None, xlim=None,
                       plot_coords=None, lwidth=None, mute_y_axis=False):
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
        fig, ax = plt.subplots(nrows=K, figsize=figsize, gridspec_kw={'height_ratios': [2] * K,
                                                                       'wspace': 0.3, 'hspace': 0.3})
    start = int(EF.index[0].split(':')[1])
    end = int(EF.index[-1].split(':')[1])
    length = EF.shape[0]
    coords = np.linspace(start, end, num=length)
    for i in range(K):
        factor = f'factor{i + 1}'
        scaled_y = EF[factor]
        if fill:
            ax[i].fill_between(coords, np.zeros(len(coords)), scaled_y, color=colores[i], alpha=0.3)
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


def plot_factors_wrapper(EF, gene, K, colores=None, figsize=None, ax=None, xlim=None,
                         plot_coords=False, lwidth=3, mute_y_axis=False):
    """Plot pre-processed factor tracks. EF should already be lm-corrected and normalized."""
    plot_factor_tracks(EF / EF.quantile(0.99, axis=0), gene, K, colores=colores, figsize=figsize, ax=ax,
                       xlim=xlim, plot_coords=plot_coords, mute_y_axis=mute_y_axis, lwidth=lwidth)


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


def plot_gene_isoforms(isoforms_dict, coordinates, color_list=None, axes=None, figsize=None, lwidth=5, xlim=None):
    if xlim is None:
        xlim1 = int(coordinates[0].split(':')[1])
        xlim2 = int(coordinates[-1].split(':')[1])
        xlim = (xlim1, xlim2)
    else:
        xlim1 = xlim[0]
        xlim2 = xlim[1]
        xlim = (xlim1, xlim2)
    if color_list is None:
        color_list = sns.color_palette("tab10")
    K = len(isoforms_dict)
    if figsize is None:
        figsize = (20, 3)
    if axes is None:
        fig, axes = plt.subplots(K, 1, figsize=figsize)
    for i in range(K):
        isoform_df = isoforms_dict[f'isoform_{str(i + 1)}']['df']
        ax = axes[i]
        color = color_list[i]
        plot_isoform(isoform_df, ax, color, xlim[0], xlim[1], lwidth=lwidth)


def plot_isoform_annotations(annotation_exons, gene, colores=None, start=None, end=None, figsize=None, lwidth=5, iso_order=None, axes=None, xlim=None):
    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]
    if iso_order is None:
        isoforms = sorted(gene_exons.transcript_id.unique())
    else:
        isoforms = [gene + '.' + x for x in iso_order]
    isoform_dict = {}
    for i, iso in enumerate(isoforms):
        isoform_name = f'isoform_{str(i + 1)}'
        df = gene_exons.loc[gene_exons.transcript_id == iso].copy()
        df['transcript_id'] = f'{gene}.{isoform_name}'
        isoform_dict.update({isoform_name: {'df': df}})
    chrom = list(annotation_exons.chrom)[0]
    if start is None:
        start = str(np.min([int(list(gene_exons.start)[0]), int(list(gene_exons.start)[0])]) - 1000)
    if end is None:
        end = str(np.max([int(list(gene_exons.end)[-1]), int(list(gene_exons.end)[-1])]) + 1000)
    coords = [f'{chrom}:{start}', f'{chrom}:{end}']
    plot_gene_isoforms(isoform_dict, coords, color_list=colores, figsize=figsize, lwidth=lwidth, axes=axes, xlim=xlim)


def plot_bar_ax(loadings, color, ax, label='label', sort_factor=False):
    x = range(len(loadings))
    if sort_factor:
        loadings = sorted(loadings)
    ax.bar(x, loadings, width=4, color=color, alpha=0.7, rasterized=True)
    if sort_factor:
        median = np.median(loadings)
        middle_point = len(loadings) / 2
        ax.plot([0, len(loadings)], [median, median], linestyle='--', c='black', linewidth=2)
        ax.scatter([middle_point], [median], marker="D", c='black', s=10)


def factor_barplot(EL, samples, color_list, label_list, tissue_colors, sort_factor=True, iso_order=None, figsize=None):
    samples = samples.loc[EL.index]
    tissues = sorted(samples.Tissue.unique())
    if iso_order is None:
        factors = EL.columns
    else:
        factors = iso_order
    K = len(factors)
    N = len(tissues)
    if figsize is None:
        figsize = (N / 3, K / 1.5)
    fig, axes = plt.subplots(K + 1, N + 1, figsize=figsize, gridspec_kw={'wspace': 0.1, 'hspace': 0.2},
                             width_ratios=([1] + ([4] * N)), height_ratios=[1] * K + [0.25])
    for i, factor in enumerate(factors):
        color = color_list[i]
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 0].spines['right'].set_visible(False)
        axes[i, 0].spines['bottom'].set_visible(False)
        axes[i, 0].set_xticks([])
        factor_max = EL[factor].max()
        axes[i, 0].set_ylim([0, factor_max])
        for j, tissue in enumerate(tissues):
            if i == (len(factors) - 1):
                label = label_list[j]
            else:
                label = ''
            tissue_samples = samples.loc[samples.Tissue == tissue].index
            loadings = list(EL.loc[tissue_samples, factor].dropna())
            ax = axes[i, j + 1]
            plot_bar_ax(loadings, color, ax, label=label, sort_factor=sort_factor)
            ax.set_ylim([0, factor_max])
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.margins(0, 0)
            ax_ = axes[K, j + 1]
            ax_.barh([1], [1], color=tissue_colors[j])
            ax_.spines['top'].set_visible(False)
            ax_.spines['bottom'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
            ax_.set_xticks([0.5], [tissue], rotation=90)
            ax_.set_yticks([])
            axes[K, 0].axis('off')
            axes[K, 0].margins(0, 0)


def factor_barplot_h(EL, samples, color_list, label_list, sort_factor=False, figsize=None, ylim=None, one_row=False,
                     tissue_colors=None, print_labels=False):
    samples = samples.loc[EL.index]
    tissues = samples.tissue_id.unique()
    factors = EL.columns
    K = len(factors)
    N = len(tissues)
    if figsize is None:
        figsize = (N / 3, K / 1.2)
    if one_row:
        fig, axes = plt.subplots(N + 1, 2, figsize=figsize, gridspec_kw={'wspace': 0.05, 'hspace': 0.1},
                                 height_ratios=(([4] * N) + [1]), width_ratios=[1.2, 10])
        for j, tissue in enumerate(tissues):
            label = label_list[j]
            tissue_samples = samples.loc[samples.tissue_id == tissue].index
            EL_sorted = EL.loc[tissue_samples].sort_values(factors[0])
            ax = axes[j, 1]
            ax_ = axes[j, 0]
            ax_.barh([1], [1], color=tissue_colors[j])
            ax_.axis('off')
            ax_.margins(0, 0)
            ax_.spines['top'].set_visible(False)
            ax_.spines['bottom'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
            ax_.set_xticks([])
            ax_.set_yticks([])
            if j == 0:
                ax_ = axes[0, 1]
                ax_.axis('off')
                ax_.margins(0, 0)
                ax_.spines['top'].set_visible(False)
                ax_.spines['bottom'].set_visible(False)
                ax_.spines['right'].set_visible(False)
                ax_.spines['left'].set_visible(False)
                ax_.set_xticks([])
                ax_.set_yticks([])
            loadings = np.array(EL_sorted[factors[0]])
            median = np.median(loadings)
            middle_point = len(loadings) / 2
            for i, loading in enumerate(loadings):
                ax.plot([0, loading], [i, i], c=color_list[0])
                cum_sum = loading
                for j in range(1, len(factors)):
                    loading_ = EL_sorted.iloc[i, j]
                    if j == (len(factors) - 1):
                        top_bar = 1
                    else:
                        top_bar = cum_sum + loading_
                    ax.plot([cum_sum, top_bar], [i, i], c=color_list[j])
                    cum_sum += loading_
            ax.plot([median, median], [len(loadings), 0], linestyle='--', c='black', linewidth=2)
            ax.scatter([median], [middle_point], marker="D", c='black', s=10, zorder=len(loadings) + 2)
            axes[N, 1].spines['top'].set_visible(False)
            axes[N, 1].spines['right'].set_visible(False)
            axes[N, 1].spines['left'].set_visible(False)
            axes[N, 1].set_yticks([])
            axes[N, 1].tick_params(labelsize=10, rotation=90)
            if print_labels:
                ax.set_xlabel(label, rotation=45)
            if ylim is None:
                axes[N, 0].set_xlim([0, 1])
                ax.set_xlim([0, 1])
            else:
                axes[N, 0].set_xlim(ylim)
                ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax_ = axes[N, 0]
            ax_.axis('off')
            ax_.margins(0, 0)
            ax_.spines['top'].set_visible(False)
            ax_.spines['bottom'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
            ax_.set_xticks([])
            ax_.set_yticks([])
    else:
        fig, axes = plt.subplots(K, N + 1, figsize=figsize, gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                                 width_ratios=([1] + ([4] * N)))
        for i, factor in enumerate(factors):
            color = color_list[i]
            axes[i, 0].spines['top'].set_visible(False)
            axes[i, 0].spines['right'].set_visible(False)
            axes[i, 0].spines['bottom'].set_visible(False)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_ylabel(factor)
            factor_max = EL[factor].max()
            axes[i, 0].set_ylim([0, factor_max])
            for j, tissue in enumerate(tissues):
                if i == (len(factors) - 1):
                    label = label_list[j]
                else:
                    label = ''
                tissue_samples = samples.loc[samples.tissue_id == tissue].index
                loadings = list(EL.loc[tissue_samples, factor])
                ax = axes[i, j + 1]
                plot_barh_ax(loadings, color, ax, label=label, sort_factor=sort_factor)
                ax.set_ylim([0, factor_max])
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.margins(0, 0)
                ax.set_xlabel(label, rotation=45)


def plot_counts_heatmap(data, tissues_display, colors_list_display):
    """Plot the SRSF3 counts heatmap with tissue color bar."""
    cticks = np.log10(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 201, 301, 401, 501, 601, 701, 801, 901]))
    ctags = ['0', '', '', '', '', '', '', '', '', '', '10', '', '', '', '', '', '', '', '', '100', '', '', '', '', '', '', '', '']

    fig, axs = plt.subplots(1, 3, figsize=(8, 4), width_ratios=[0.1, 1, 40])
    plt.subplots_adjust(wspace=0.01)

    g = sns.heatmap(np.log10(data + 1), ax=axs[2], cmap='binary', cbar=True, cbar_kws={'label': 'read counts'}, rasterized=True)

    cbar = g.collections[0].colorbar
    cbar.set_ticks(cticks)
    cbar.set_ticklabels(ctags)

    for i, tick in enumerate(cbar.ax.yaxis.get_major_ticks()):
        tick_text = cbar.ax.yaxis.get_ticklabels()[i].get_text()
        if tick_text in ['0', '10', '100']:
            tick.set_pad(10)
            tick.tick1line.set_markersize(10)
            tick.tick2line.set_markersize(10)
        else:
            tick.tick1line.set_markersize(3)
            tick.tick2line.set_markersize(3)

    cbar.ax.tick_params(labelsize=10)

    colors_list = colors_list_display

    x = np.array(range(1, len(tissues_display) + 1)[::-1]) * 100
    x_ = []
    c_ = []
    for i, pos in enumerate(x):
        x_.append(pos)
        x_.append(pos - 95)
        c_.append(colors_list[i])
        c_.append('white')

    axs[1].bar([1] * len(tissues_display) * 2, x_, color=c_)

    pos_tag = np.array(range(1, len(tissues_display) + 1)[::-1]) * 100
    for i, tissue in enumerate(tissues_display):
        axs[1].text(0, pos_tag[i] - 60, tissue + ' -', ha='right', fontsize=8)

    axs[1].axis('off')
    axs[1].margins(0, 0)

    plt.subplots_adjust(hspace=0.01)

    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_xlabel('SRSF3 basepairs')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
