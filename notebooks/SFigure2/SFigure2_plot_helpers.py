"""
SFigure2_plot_helpers.py -- plotting functions only

Assembled by the Figure code cleaner from the notebooks in
torino_paper/analysis/. Every line of analysis code is taken verbatim from the
cell it came from; only the plumbing (function wrappers, path constants) is new.

Each function takes fully plot-ready data and draws one panel. The original
savefig call is kept, commented out, with its original path.
"""
import os
import pickle
import sys

ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'

sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
sys.path.append('/project2/yangili1/cfbuenabadn/torino_paper/code/scripts')
sys.path.append(ANALYSIS_DIR)   # plot_ssmd_results and friends live here

# Every path in the extracted cells is relative to analysis/, as in the source
# notebooks -- and some of the modules imported below (load_fig3_metadata, a
# converted %run) read their data at import time, before run_all can chdir.
os.chdir(ANALYSIS_DIR)

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata, norm as sp_norm
from sklearn.preprocessing import scale as sk_scale
from tqdm import tqdm
from pybedtools import BedTool
import gzip
import sys
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from scipy.special import logit
from scipy.stats import kruskal, zscore, norm, normaltest, ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale
import importlib
import plot_ssmd_results
from plot_ssmd_results import SSMDResults
import itertools
from plot_isoforms_lazy import *   # was: %run plot_isoforms_lazy.py
from load_fig3_metadata import *   # was: %run load_fig3_metadata.py
import os
from torino_plots import *
from torino_plots import *   # was: %run ../../torino/scripts/torino_plots.py
import tabix
from scipy.stats import linregress
import pybedtools
from scipy import stats
import bisect
import pysam
import pysam  # pip install pysam
from scipy.stats import zscore

# Shared function definitions (several source cells define plotting and
# data helpers together, so they all live in SFigure2_helpers).
from SFigure2_helpers import *  # noqa: F401,F403


def plot_torino_annot_transcripts(total_col, whole_col):
    """torino_annot_transcripts -- from Appris_1.ipynb, cell 116."""
    fig, ax = plt.subplots(figsize=(4, 6))

    hm = sns.heatmap(
        pd.DataFrame({
            'All genes': total_col,
            'No overlap genes': whole_col
        }),
        cmap='Blues',
        vmin=0,
        # vmax=100,
        annot=True,
        fmt="d",
        cbar_kws=dict(
            orientation='horizontal',
            shrink=0.75,
            pad=0.08
        ),
        ax=ax
    )

    # ---- Column names on top ----
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', rotation=0)

    # White boxes behind column labels
    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=3))

    # ---- Colorbar label ----
    cbar = hm.collections[0].colorbar
    cbar.set_label("Percentage of tested genes", labelpad=10)

    # plt.savefig('../code/results/figures/revision/fig2/torino_annot_transcripts.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_annot_transcripts.png', bbox_inches='tight', dpi=500)


def plot_torino_isoform_recovery_by_length(x_list, y_list):
    """torino_isoform_recovery_by_length -- from Appris_1.ipynb, cell 6."""
    fig, ax = plt.subplots(nrows=2, height_ratios=[3,1], gridspec_kw={'hspace': 0.1})
    ax[0].scatter(range(1, 31), x_list)
    ax[0].plot(range(1, 31), x_list)
    ax[0].plot([2, 2], [0, 1], linestyle = '--', c='gray')
    ax[0].plot([23.5, 23.5], [0, 1], linestyle = '--', c='gray')
    ax[0].text(0.9, 0.1, 'Length decile 1', rotation=90)
    ax[0].text(24, 0.1, 'Length decile 10', rotation=90)
    ax[0].set_xticks([])
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    ax[0].set_ylabel('Percentage of non-overlapping genes\nwith at least one recovered isoform')
    ax[0].spines[['top', 'right']].set_visible(False)
    ax[0].set_title('Genes with Gencode isoform recovered by Torino')

    xlim = ax[0].get_xlim()
    ax[0].set_ylim([0,1])

    ax[1].bar(range(1, 31), y_list, color='gray')
    ax[1].set_ylabel('Genes')
    ax[1].set_xlabel("Exons in gene's APPRIS principal transcript")
    ax[1].spines[['top', 'right']].set_visible(False)
    ax[1].set_xticks([1, 5, 10, 15, 20, 25, 30], ['1', '5', '10', '15', '20', '25', '+30'])
    ax[1].set_xlim(xlim)
    ax[1].text(23, 200, 'Total: 6,218')

    # plt.savefig('../code/results/figures/revision/fig2/torino_isoform_recovery_by_length.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_isoform_recovery_by_length.png', bbox_inches='tight', dpi=500)


def plot_torino_isoform_recovery_by_expression(x_list, y_list):
    """torino_isoform_recovery_by_expression -- from Appris_1.ipynb, cell 66."""
    fig, ax = plt.subplots(nrows=2, height_ratios=[3,1], gridspec_kw={'hspace': 0.1})
    ax[0].scatter(range(1, 21), x_list)
    ax[0].plot(range(1, 21), x_list)
    # ax[0].plot([2, 2], [0, 1], linestyle = '--', c='gray')
    # ax[0].plot([23.5, 23.5], [0, 1], linestyle = '--', c='gray')
    # ax[0].text(0.9, 0.1, 'Length decile 1', rotation=90)
    # ax[0].text(24, 0.1, 'Length decile 10', rotation=90)
    ax[0].set_xticks([])
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    ax[0].set_ylabel('Percentage of non-overlapping genes\nwith at least one recovered isoform')
    ax[0].spines[['top', 'right']].set_visible(False)
    ax[0].set_title('Genes with Gencode isoform recovered by Torino')

    xlim = ax[0].get_xlim()
    ax[0].set_ylim([0,1])

    ax[1].bar(range(1, 21), [np.log10(x+1) for x in y_list], color='gray')
    ax[1].set_ylabel('Median TPM\nacross tissues')
    ax[1].set_xlabel("Median gene expression binned rank")
    ax[1].spines[['top', 'right']].set_visible(False)
    ax[1].set_xticks([1, 6, 11, 16, 20], ['20', '15', '10', '5', '1'])
    ax[1].set_xlim(xlim)
    # ax[1].text(1, 1, 'Total: 6,218')
    ax[1].set_yticks([0, np.log10(2), np.log10(11), np.log10(101)], ['0', '1', '10', '100'])


    # plt.savefig('../code/results/figures/revision/fig2/torino_isoform_recovery_by_expression.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_isoform_recovery_by_expression.png', bbox_inches='tight', dpi=500)


def plot_torino_missing_junctions(annot_summary):
    """torino_missing_junctions -- from Appris_1.ipynb, cell 61."""
    fig, ax = plt.subplots()
    ax.bar(range(1, 21),[len(annot_summary.loc[(annot_summary.mexp_rank == x) & (~annot_summary.annot_match)]) for x in range(1, 21)],
          color='lightsteelblue', label='Missing isoform')
    ax.bar(range(1, 21),[(~annot_summary.loc[(annot_summary.mexp_rank == x) & (~annot_summary.annot_match)].obs_juncs).sum() for x in range(1, 21)],
           label='Missing junctions')

    ax.legend(frameon=False)
    ax.set_ylabel('Genes')
    ax.set_xlabel("Median gene expression binned rank")
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([1, 6, 11, 16, 20], ['20', '15', '10', '5', '1'])

    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.png', bbox_inches='tight', dpi=500)


def plot_torino_missing_junctions_v2(annot_summary):
    """torino_missing_junctions_v2 -- from Appris_1.ipynb, cell 67."""
    fig, ax = plt.subplots()
    ax.bar(range(1, 21),[len(annot_summary.loc[(annot_summary.mexp_rank2 == x) & (~annot_summary.annot_match)]) for x in range(1, 21)],
          color='lightsteelblue', label='Missing isoform')
    ax.bar(range(1, 21),[(~annot_summary.loc[(annot_summary.mexp_rank2 == x) & (~annot_summary.annot_match)].obs_juncs).sum() for x in range(1, 21)],
           label='Missing junctions')

    ax.legend(frameon=False)
    ax.set_ylabel('Genes')
    ax.set_xlabel("Median gene expression binned rank")
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([1, 6, 11, 16, 20], ['20', '15', '10', '5', '1'])

    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.png', bbox_inches='tight', dpi=500)


def plot_PCA_rank20(qqnorm, tissue_labels):
    """PCA_rank20 -- from Appris_1.ipynb, cell 34."""
    pca = plot_pca_scatter(qqnorm, tissue_labels, gtex_colors_cereb, figsize=(7,4), 
                           save_path='../code/results/figures/Fig1_torino/PCA_rank_bottom_decile')
    #'../code/results/figures/Fig4_animals/enrichment_heatmap.human_v_mouse.final.pdf'


def plot_heatmap_rank20(qqnorm, tissue_labels):
    """heatmap_rank20 -- from Appris_1.ipynb, cell 37."""
    plot_clustermap(qqnorm, tissue_labels, gtex_colors_cereb, figsize=(6, 5), save_path = '../code/results/figures/Fig1_torino/heatmap_rank_bottom_decile')


def plot_PCA_rank1(qqnorm, tissue_labels):
    """PCA_rank1 -- from Appris_1.ipynb, cell 41."""
    pca = plot_pca_scatter(qqnorm, tissue_labels, gtex_colors_cereb,figsize=(7,4), 
                           save_path='../code/results/figures/Fig1_torino/PCA_rank_top_decile')
    #'../code/results/figures/Fig4_animals/enrichment_heatmap.human_v_mouse.final.pdf'


def plot_heatmap_rank1(qqnorm, tissue_labels):
    """heatmap_rank1 -- from Appris_1.ipynb, cell 43."""
    plot_clustermap(qqnorm, tissue_labels, gtex_colors_cereb, figsize=(6, 5), 
                    save_path = '../code/results/figures/Fig1_torino/heatmap_rank_top_decile')


def plot_torino_missing_junctions_CGB8_factors(rds):
    """torino_missing_junctions_CGB8_factors -- from CoveragePlots.ipynb, cell 5."""
    plot_factors_wrapper(rds['rds']['torino_K10'], 'CGB8', figsize=(6, 4), mute_y_axis=True, strand = 'minus', lwidth=1.5, lm=False)

    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.CGB8_factors.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.CGB8_factors.png', bbox_inches='tight', dpi=500)


def plot_torino_missing_junctions_CGB8_iso(gencode_iso):
    """torino_missing_junctions_CGB8_iso -- from CoveragePlots.ipynb, cell 6."""
    plot_isoform_annotations(gencode_iso, 'CGB8', colores = ['black']*50, figsize=(20, 5))
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.CGB8_iso.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.CGB8_iso.png', bbox_inches='tight', dpi=500)


def plot_torino_missing_junctions_CGB8_torino_iso(human_iso):
    """torino_missing_junctions_CGB8_torino_iso -- from CoveragePlots.ipynb, cell 7."""
    plot_isoform_annotations(human_iso, 'CGB8', colores = ['midnightblue']*10)
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.CGB8_torino_iso.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/fig2/torino_missing_junctions.CGB8_torino_iso.png', bbox_inches='tight', dpi=500)


def plot_polyA_barplot():
    """polyA_barplot -- from UTR3.ipynb, cell 65."""
    fig, ax = plt.subplots(figsize=(3,3))

    ax.bar([0, 1, 3, 4], [9587, 10905, 15411, 5959, ], color=['tab:blue', 'tab:blue', 'salmon', 'salmon'])
    ax.bar([0, 1, 3, 4], [9155, 6973, 13736, 3516], color=['navy', 'navy', 'tab:red', 'tab:red'])
    ax.text(-0.1, 800, '9155 / 9587', c='white', rotation=90)
    ax.text(0.9, 800, '6973 / 10905', c='white', rotation=90)
    ax.text(2.9, 800, '13736 / 15411', c='white', rotation=90)
    ax.text(3.9, 1000, '3516 / 5959', c='white', rotation=90)
    ax.text(3.9, 6300, '9', c='black', rotation=90)

    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(['Single polyA', 'Multiple polyA', 'Single polyA', 'Multiple polyA'], rotation=90 )
    ax.set_ylabel('counts')
    ax.spines[['top', 'right']].set_visible(False)

    # plt.savefig('../code/results/figures/revision/supp_fig/polyA/barplot.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/polyA/barplot.png', bbox_inches='tight', dpi=500)


def plot_polyA_heatmap(brain_tissues, cereb_tissues, color_list, scaled_all):
    """polyA_heatmap -- from UTR3.ipynb, cell 135."""
    # fig, ax = plt.subplots(figsize=(8,8))
    g = sns.clustermap(scaled_all.loc[scaled_all[brain_tissues + cereb_tissues].mean(axis=1).sort_values().index[::-1], 
        scaled_all.mean(axis=0).sort_values().index[::-1]], col_colors=color_list, col_cluster=False, row_cluster=False, 
                   cmap='bwr', figsize=(6,8), rasterized=True)

    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel("3' UTR across 2898 genes with\n> 0.05 loadings across tissues")
    g.ax_cbar.set_ylabel('Scaled loadings', fontsize=10)

    # plt.savefig('../code/results/figures/revision/supp_fig/polyA/heatmap.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/polyA/heatmap.png', bbox_inches='tight', dpi=500)


def plot_UTRs_cumulative(gx, gy, tgx, tgy, tx, ty):
    """UTRs_cumulative -- from UTR3.ipynb, cell 288."""
    # plt.plot(tgx_9, tgy_9)
    # plt.plot(tgx_1, tgy_1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(tgy, tgx, c='tab:purple', linewidth=3, label='Torino & Gencode')
    # ax.plot(tgy, tgx, c='tab:blue', linewidth=2)
    ax.plot(ty, tx, c='tab:blue', linewidth=3, label = 'Torino only')
    ax.plot(gy, gx, c='tab:red', linewidth=3, label='Gencode only')
    ax.set_xlabel('Percentage of polyA sites')
    ax.set_ylabel('Cumulative site usage percentage')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_title('Long-read mapping to polyA site')
    ax.legend(frameon=False)

    # plt.savefig(f'../code/results/figures/revision/fig2/UTRs_cumulative.pdf', bbox_inches='tight')
    # plt.savefig(f'../code/results/figures/revision/fig2/UTRs_cumulative.png', bbox_inches='tight', dpi=500)


def plot_cassette_exons_brain_cortex(lista):
    """cassette_exons_brain_cortex -- from CassetteExonsDeNovo.ipynb, cell 76."""
    fig, axes = plot_cassette_exon_metaplots(
        'Brain_Cortex',
        lista,
        color='#FFD700',
        figsize=(4, 4),
    )
    fig.suptitle('Brain - Cortex', fontsize=12, y=1.01)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.brain_cortex.png', bbox_inches='tight', dpi=500)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.brain_cortex.pdf', bbox_inches='tight')


def plot_cassette_exons_heart_atrial_appendage(lista):
    """cassette_exons_heart_atrial_appendage -- from CassetteExonsDeNovo.ipynb, cell 77."""
    fig, axes = plot_cassette_exon_metaplots(
        'Heart_Atrial_Appendage',
        lista,
        color='#B452CD',
        figsize=(4, 4),
    )
    fig.suptitle('Heart - Atrial Appendage', fontsize=12, y=1.01)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.heart_atrial_appendage.png', 
    # bbox_inches='tight', dpi=500)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.heart_atrial_appendage.pdf', bbox_inches='tight')


def plot_cassette_exons_liver(lista):
    """cassette_exons_liver -- from CassetteExonsDeNovo.ipynb, cell 78."""
    fig, axes = plot_cassette_exon_metaplots(
        'Liver',
        lista,
        color='#CDB79E',
        figsize=(4, 4),
    )
    fig.suptitle('Liver', fontsize=12, y=1.01)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.liver.png', 
    # bbox_inches='tight', dpi=500)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.liver.pdf', bbox_inches='tight')


def plot_cassette_exons_median_loadings(mean_loadings):
    """cassette_exons_median_loadings -- from CassetteExonsDeNovo.ipynb, cell 94."""
    labels = ['Gencode & VastDB', 'Gencode', 'VastDB', 'Overlap', 'No Overlap']

    fig, ax = plt.subplots(figsize=(4,4))
    for i, x in enumerate(mean_loadings):
        plot_cumulative(x, labels[i], ax, 5-i)

    ax.legend(frameon=False, title='Torino Cassette Exons')
    ax.set_ylabel('Cumulative percentage')
    ax.set_xlabel('Mean isoform loadings')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])

    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.median_loadings.png', 
    # bbox_inches='tight', dpi=500)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.median_loadings.pdf', bbox_inches='tight')


def plot_cassette_exons_torino_v_gencode_brain_cortex(lista_de_listas):
    """cassette_exons_torino_v_gencode_brain_cortex -- from CassetteExonsDeNovo.ipynb, cell 128."""
    fig, axes = plot_cassette_exon_metaplots_comparison(
        'Brain_Cortex',
        lista_de_listas,
        list_labels=None,
        q=0.5,
        color='gray',
        figsize=(10, 3),
        annot_color = 'tab:orange',
        annot_name = 'Gencode',
        annotation = 'gencode'
    )

    fig.suptitle('Cassette exons annotated by Torino and Gencode', fontsize=12, y=1.01)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.torino_v_gencode.brain_cortex.png', 
    # bbox_inches='tight', dpi=500)
    # plt.savefig('../code/results/figures/revision/supp_fig/cassette_exons.torino_v_gencode.brain_cortex.pdf', bbox_inches='tight')


def plot_VastDB_cassette_exons_brain(found_CE, found_SE):
    """VastDB_cassette_exons_brain -- from Brain_Cassette_SSMD.ipynb, cell 39."""
    y, x = cdf(np.array(found_SE.loc[found_SE.found_exon & found_SE.found_skip].dropna().top_brain))
    y_skip, x_skip = cdf(np.array(found_SE.loc[found_SE.found_exon & found_SE.found_skip].dropna().bottom_skip))


    fig, ax = plt.subplots(ncols = 2, figsize=(7,3))
    ax[0].plot(x, y, c='tab:red')
    ax[0].plot(x_skip, y_skip, c='tab:blue')
    ax[0].plot([0, 0], [0, 1], linestyle='--', c='tab:grey')

    y, x = cdf(np.array(found_CE.loc[found_CE.found_exon & found_CE.found_skip].dropna().top_brain))
    y_skip, x_skip = cdf(np.array(found_CE.loc[found_CE.found_exon & found_CE.found_skip].dropna().bottom_skip))

    ax[1].plot(x, y, c='tab:red', label='Cassette exon')
    ax[1].plot(x_skip, y_skip, c='tab:blue', label='Skipped exon')
    ax[1].plot([0, 0], [0, 1], linestyle='--', c='tab:grey')

    ax[1].set_yticks([])

    ax[1].legend(frameon=False, title='Brain isoform')

    ax[0].set_title('VastDB Skipped Exons')
    ax[1].set_title('VastDB Casette Exons')

    ax[0].set_ylabel('Cumulative distribution')
    ax[0].set_xlabel('Torino isoform SSMD in Brain')
    ax[1].set_xlabel('Torino isoform SSMD in Brain')

    fig.subplots_adjust(wspace=0.1)

    # plt.savefig('../code/results/figures/revision/supp_fig/VastDB.cassette_exons.brain.png', bbox_inches='tight', dpi=500)
    # plt.savefig('../code/results/figures/revision/supp_fig/VastDB.cassette_exons.brain.pdf', bbox_inches='tight')


def plot_intron_metaplot_quintiles(appris_X, appris_introns, gencode_X, gencode_bed, torino_X, torino_all_bed, torino_bed):
    """intron_metaplot_quintiles -- from Retained_Introns_figure.ipynb, cell 68."""
    # plot_intron_metaplot(torino_X)
    plot_intron_metaplot_comparison(torino_X, gencode_X, appris_X, torino_bed, gencode_bed, 
                                    BedTool.from_dataframe(appris_introns).intersect(torino_all_bed, v = True), torino_all_bed)

    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_metaplot_quintiles.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_metaplot_quintiles.png', bbox_inches='tight', dpi=500)


def plot_intron_metaplot_quintiles_vastdb(appris_X, appris_introns, torino_all_bed, vastdb_introns):
    """intron_metaplot_quintiles_vastdb -- from Retained_Introns_figure.ipynb, cell 70."""
    # BedTool.from_dataframe(vastdb_introns).intersect(
    #     BedTool.from_dataframe(appris_introns), F=1, f=1
    # ).intersect(torino_all_bed, v = True).to_dataframe()


    plot_intron_metaplot_comparison(appris_X, appris_X, appris_X, 
                                    BedTool.from_dataframe(appris_introns).intersect(torino_all_bed, F=1,f=1),
                                    BedTool.from_dataframe(appris_introns).intersect(BedTool.from_dataframe(vastdb_introns), F=1, f=1), 
                                    BedTool.from_dataframe(appris_introns).intersect(BedTool.from_dataframe(vastdb_introns), v = True), 
                                    torino_all_bed, alt_color = 'tab:green', alt_name = 'VastDB')

    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_metaplot_quintiles.vastdb.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_metaplot_quintiles.vastdb.png', bbox_inches='tight', dpi=500)


def plot_intron_metaplot_Liver(appris_X, appris_introns, gencode_X, gencode_bed, torino_X, torino_all_bed, torino_bed):
    """intron_metaplot_Liver -- from Retained_Introns_figure.ipynb, cell 79."""
    plot_intron_single_metaplot_comparison(torino_X, gencode_X, appris_X, torino_bed, gencode_bed, 
                                    BedTool.from_dataframe(appris_introns).intersect(torino_all_bed, v = True), torino_all_bed)

    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_metaplot.Liver.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_metaplot.Liver.png', bbox_inches='tight', dpi=500)


def plot_intron_barplot():
    """intron_barplot -- from Retained_Introns_figure.ipynb, cell 137."""
    fig, ax = plt.subplots(figsize=(1.5, 3))

    ax.bar(['Appris introns', 'VastDB RI','Torino (all)', 'Torino RI', 'Gencode RI'], [174635, 158506, 115176, 25305, 9131],
           color = ['tab:gray', 'tab:orange', 'navy', 'tab:blue', 'tab:red'])
    ax.set_xticks([])

    ax.text(-0.2, 4000, 'Appris intron (174,635)', rotation=90, c='white')

    ax.text(0.8, 4000, 'VastDB (158,506)', rotation=90, c='white')
    ax.text(1.8, 4000, 'Torino - all (115,176)', rotation=90, c='white')
    # ax.text(1.8, 4000, 'Torino RI', rotation=90, c='white')
    ax.text(2.8, 30000, 'Torino RI (25,305)', rotation=90, c='tab:blue')
    ax.text(3.8, 15000, 'Gencode RI (9,131)', rotation=90, c='tab:red')
    ax.set_yticks([0, 50000, 100000, 150000])#, ['0', '5e5', '1e6', '15e6'])

    ax.set_yticklabels(['0', '5e5', '1e6', '1.5e6'])
    ax.set_ylabel('Total introns')
    ax.spines[['right', 'top']].set_visible(False)


    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_barplot.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_barplot.png', bbox_inches='tight', dpi=500)

    # plt.savefig('../code/results/figures/Fig2_isoforms/intron_barplot.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/Fig2_isoforms/intron_barplot.png', bbox_inches='tight', dpi=500)


def plot_intron_cdf_Liver(df):
    """intron_cdf_Liver -- from Retained_Introns_figure.ipynb, cell 144."""
    fig, ax = plt.subplots(figsize=(3,3))
    colores = ['tab:purple', #'tab:cyan', 
               'tab:blue', 'tab:red', 'tab:gray'][::-1]


    labels_write = ['Torino & Gencode', 'Torino', 'Gencode', 'Random intron'][::-1]
    for i, label in enumerate(['snmf_and_gencode', #'snmf_overlap', 
                               'snmf_only', 'gencode_only', 'no_intron_retention'][::-1]):
        df_ = df.loc[df.label == label]
        c = colores[i]
        ax.plot(np.array(df_.ir_percent.sort_values()), np.linspace(0, 1, len(df_)), c=c, label=labels_write[i], linewidth=3)

    # ax.plot([np.sqrt(0.1), np.sqrt(0.1)], [0, 1], 'r--')

    ax.set_xticks([0, np.sqrt(0.1),  np.sqrt(0.5), np.sqrt(1)], ['0',  '0.1','0.5', '1'], rotation=0)
    ax.set_ylabel('cumulative\ndistribution')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('intron coverage')

    ax.legend(frameon=False, bbox_to_anchor=(1.05, 0.5))

    # plt.show()
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_cdf.Liver.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/intron_cdf.Liver.png', bbox_inches='tight', dpi=500)
    # plt.savefig('../code/scratch_figures/figure2/intron_cdf.pdf', bbox_inches='tight')
    # plt.savefig('../code/scratch_figures/figure2/intron_cdf.png', bbox_inches='tight', dpi=500)


def plot_PCA_all(human_EL, human_metadata):
    """PCA_all -- from Retained_Introns_figure.ipynb, cell 174."""
    # All rows
    pca = plot_pca_scatter_EL(human_EL, human_metadata, gtex_colors2)

    # Subset of transcripts
    # pca = plot_pca_scatter_EL(human_EL, human_metadata, gtex_colors2,
    #                           subindex=IR_transcripts.transcript.tolist(),
    #                           title="PCA – IR transcripts")

    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/PCA.all.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/PCA.all.png', bbox_inches='tight', dpi=500)


def plot_PCA_introns(IR_transcripts, human_EL, human_metadata):
    """PCA_introns -- from Retained_Introns_figure.ipynb, cell 175."""
    pca = plot_pca_scatter_EL(human_EL.loc[pd.Index(IR_transcripts.transcript.tolist()).intersection(human_EL.index)], human_metadata, gtex_colors2)

    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/PCA.introns.pdf', bbox_inches='tight')
    # plt.savefig('../code/results/figures/revision/supp_fig/intron_retention/PCA.introns.png', bbox_inches='tight', dpi=500)
