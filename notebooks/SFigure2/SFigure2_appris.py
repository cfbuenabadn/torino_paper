"""SFigure2 annotation-recovery panels, from the results already on disk.

From analysis/Appris_1.ipynb. Five panels over the corrected isoform
annotation: how much of the reference Torino recovers, by transcript length and
by expression rank, and how many junctions it misses.

The four PCA/heatmap panels of that notebook are not here yet -- they need the
EL table and so want ~8 GB, like the PCA panels in the intron-retention module.

Which definition each panel used is settled by the execution counts: the
notebook rebinds `annot_summary`, `x_list`, `y_list` and `mexp_rank` several
times over, and the panels ran at In[7], In[130], In[397], In[410] and In[411].

Nothing here writes; the guard in SFigure2_intron_retention is active.
"""

import os
import sys

from SFigure2_intron_retention import (      # noqa: F401  -- shared setup
    ANALYSIS_DIR, np, pd, plt, sns, BedTool, bedtools_safe, _cached, _panel,
    free, _subset, _tag, gtex_colors2,
)

import re
from tqdm import tqdm

# appris_juncs_obs calls these; the notebook had them from its star-imports
from torino_isoforms import get_intron_chain, run_tabix_on_junc
from pybedtools import BedTool


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_expression():
    """GTEx median TPM and the tissue list -- notebook cell 2 (In[3])."""
    def build():
        gtex_samples = pd.read_csv('../code/config/gtex_samples.tsv', sep='\t')
        gtex_tissues = sorted(gtex_samples.tissue_id.unique())
        median_expression  = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/gtex_tables/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz',
                                        sep='\t', skiprows =2)

        import re 
        def transform_string(s):
            s = s.replace(" - ", "_")  # Remove spaces
            s = s.replace(" ", "_")  # Remove spaces
            s = re.sub(r"\(", "", s)  # Replace '(' with '_'
            s = re.sub(r"\)", "", s)  # Remove ')'
            return s

        median_expression.columns = [transform_string(s) for s in median_expression.columns]
        median_expression['gene'] = median_expression.Name.apply(lambda x: x.split('.')[0])
        return median_expression, gtex_tissues
    return _cached('expression', build)


def load_whole_genes():
    """Genes whose body is fully covered -- notebook cell 3 (In[4])."""
    def build():
        with open('/project2/mstephens/cfbuenabadn/gtex-stm/data/whole_genes.txt') as fh:
            return [x.rstrip() for x in fh.readlines()]
    return _cached('whole_genes', build)


def load_annot_full():
    """The annotation with gene_id, length, mexp_rank and annot_match.

    Cells 1, 3, 7 and 48 (In[2], In[4], In[8], In[242]).
    """
    def build():
        median_expression, gtex_tissues = load_expression()
        annot = pd.read_csv('../code/results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz', sep='\t')
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

        selected_genes = pd.read_csv('../code/config/human.protein_coding.bed.gz',
                                     sep='\t')
        selected_genes['length'] = selected_genes.end - selected_genes.start
        annot = annot.merge(selected_genes[['gene_name', 'gene_id', 'length']],
                            left_on='gene', right_on='gene_name')

        mexp_tab = median_expression.loc[
            median_expression.gene.isin(annot.gene_id), gtex_tissues].copy()
        mexp_tab.index = median_expression.loc[
            median_expression.gene.isin(annot.gene_id)].gene
        mexp_rank = pd.qcut(mexp_tab.mean(axis=1), 20, labels=range(1, 21)).reset_index()
        mexp_rank.columns = ['gene_id', 'mexp_rank']
        # LEFT, not inner: an inner merge drops genes absent from the GTEx
        # median-TPM table -- 48 of 16,073, 6 of them non-overlapping -- which
        # silently moved the non-overlapping gene count to 6,180 when Figure 2A
        # reports 6,186 from the same annotation. mexp_rank is NaN for those
        # genes, and every use of it filters on a specific rank, so they drop
        # out of the ranked panels on their own without changing the totals.
        annot = annot.merge(mexp_rank, on='gene_id', how='left')

        annot['annot_match'] = (
            annot.appris_ref.apply(lambda x: ('intron_chain' in x)
                                   or ('ref_transcript' in x))
            | (annot.gencode_ref != '.'))
        return _subset(annot, annot.gene)
    return _cached(f'annot_full:{_tag()}', build)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
@_panel
def panel_torino_isoform_recovery_by_length():
    """Notebook cell 6 (In[7]); x_list from In[5], y_list from In[6]."""
    annot = load_annot_full()
    whole_genes = load_whole_genes()
    x_list = [len(annot.loc[(annot.appris_transcript_length == x) & annot.gene_id.isin(whole_genes) & ((annot.appris_ref.apply(
        lambda x: ('ref_transcript' in x) or ('intron_chain' in x))) | (annot.gencode_ref != '.'))].gene.unique())/len(
        annot.loc[(annot.appris_transcript_length == x) & annot.gene_id.isin(whole_genes) ].gene.unique()
                                                                                        ) for x in range(1, 30)] + [
        len(annot.loc[(annot.appris_transcript_length >= 30) & annot.gene_id.isin(whole_genes) & ((annot.appris_ref.apply(
        lambda x: ('ref_transcript' in x) or ('intron_chain' in x))) | (annot.gencode_ref != '.'))].gene.unique())/len(
        annot.loc[(annot.appris_transcript_length >= 30) & annot.gene_id.isin(whole_genes)].gene.unique())
    ]
    y_list = [len(
        annot.loc[(annot.appris_transcript_length == x) & annot.gene_id.isin(whole_genes)].gene.unique()
                                                                                        ) for x in range(1, 30)] + [len(
        annot.loc[(annot.appris_transcript_length >= 30) & annot.gene_id.isin(whole_genes)].gene.unique())
    ]
    # Total and decile boundaries computed from the data. They used to be the
    # literals 'Total: 6,218', x=2 and x=23.5, carried over from an uncorrected
    # run: 6,218 is the all-genes count (corrected is 6,180), and 23.5 matches
    # no decile of this distribution at all -- the 90th percentile is 19 exons.
    #
    # Decile boundaries over the pool of ALL tested genes -- overlapping and
    # non-overlapping -- which is the population Figure 2A's is_len filter uses,
    # so the two figures quote one set of bounds (3 and 23). The bars themselves
    # are the non-overlapping genes, hence the two different denominators: the
    # deciles describe the whole tested pool, the Total counts what is plotted.
    #
    # appris_transcript_length is constant within a gene (checked: 0 of 16,073
    # vary), so the per-gene statistic does not matter; only the population
    # does. Restricting the deciles to non-overlapping genes would give 3-19,
    # which is why these are labelled with the pool they come from.
    per_gene_all = annot.groupby('gene').appris_transcript_length.max()
    d1_edge = float(per_gene_all.quantile(0.10))
    d10_edge = float(per_gene_all.quantile(0.90))

    per_gene = annot.loc[annot.gene_id.isin(whole_genes)].groupby(
        'gene').appris_transcript_length.max()
    n_genes = int(per_gene.size)
    n_all = int(per_gene_all.size)

    fig, ax = plt.subplots(nrows=2, height_ratios=[3,1], gridspec_kw={'hspace': 0.1})
    ax[0].scatter(range(1, 31), x_list)
    ax[0].plot(range(1, 31), x_list)
    ax[0].plot([d1_edge + 0.5, d1_edge + 0.5], [0, 1], linestyle='--', c='gray')
    ax[0].plot([d10_edge - 0.5, d10_edge - 0.5], [0, 1], linestyle='--', c='gray')
    ax[0].text(d1_edge - 0.7, 0.08, f'Length decile 1\n($\\leq$ {d1_edge:.0f} exons)',
               rotation=90, fontsize=7, va='bottom')
    ax[0].text(d10_edge + 0.2, 0.08, f'Length decile 10\n($\\geq$ {d10_edge:.0f} exons)',
               rotation=90, fontsize=7, va='bottom')
    ax[0].text(0.5, -0.02, f'Length deciles over all {n_all:,} tested genes '
               f'(overlapping and non-overlapping)', transform=ax[0].transAxes,
               ha='center', va='top', fontsize=6.5, color='0.35')
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
    ax[1].text(0.97, 0.9, f'Total: {n_genes:,}', transform=ax[1].transAxes,
               ha='right', va='top', fontsize=8)

    return plt.gcf()


@_panel
def panel_torino_missing_junctions():
    """Notebook cell 61 (In[397]); annot_summary with obs_juncs from In[336]."""
    annot_summary = load_annot_summary()
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

    return plt.gcf()


def _annot_summary_v2():
    """annot_summary as it stood at In[403], obs_juncs included."""
    return load_annot_summary(with_mexp_rank2=True)


@_panel
def panel_torino_missing_junctions_v2():
    """Notebook cell 67 (In[411]); annot_summary as rebuilt at In[403]."""
    annot_summary, mexp_tab, mexp_rank = _annot_summary_v2()
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

    return plt.gcf()


def wilson_ci(k, n, conf=0.90):
    """Wilson score interval for a binomial proportion.

    Used for the recovery-by-expression panel: each point is "of the genes in
    this expression bin, the fraction with a recovered isoform", so the spread
    is binomial, not a spread of measurements -- there is nothing to take a
    standard deviation of. Wilson rather than the normal approximation because
    the proportions here run above 0.9, where the normal interval overshoots 1.
    """
    from scipy.stats import norm as _norm
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = _norm.ppf(0.5 + conf / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


@_panel
def panel_torino_isoform_recovery_by_expression():
    """Notebook cell 66 (In[410]); x_list/y_list from cell 65 (In[408])."""
    annot_summary, mexp_tab, mexp_rank = _annot_summary_v2()
    y_list = [mexp_tab.loc[mexp_rank.loc[mexp_rank.mexp_rank2 == x].gene_id].median(axis=1).median() for x in range(1, 21)]
    x_list = [len(annot_summary.loc[(annot_summary.mexp_rank2 == x) & (annot_summary.annot_match)].index.unique())/len(
        annot_summary.loc[(annot_summary.mexp_rank2 == x)].index.unique()) for x in range(1, 21)]
    # 90% Wilson interval on each bin's recovery fraction. k and n are the
    # same counts the fraction is built from, recomputed here so the band
    # cannot disagree with the point it surrounds.
    k_list = [len(annot_summary.loc[(annot_summary.mexp_rank2 == x) & (annot_summary.annot_match)].index.unique()) for x in range(1, 21)]
    n_list = [len(annot_summary.loc[(annot_summary.mexp_rank2 == x)].index.unique()) for x in range(1, 21)]
    lo, hi = wilson_ci(k_list, n_list, conf=0.90)

    fig, ax = plt.subplots(nrows=2, height_ratios=[3,1], gridspec_kw={'hspace': 0.1})
    ax[0].fill_between(range(1, 21), lo, hi, alpha=0.25, color='tab:blue',
                       linewidth=0, label='90% CI')
    ax[0].scatter(range(1, 21), x_list)
    ax[0].plot(range(1, 21), x_list)
    ax[0].legend(frameon=False, fontsize=8, loc='lower right')
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


    return plt.gcf()


@_panel
def panel_torino_annot_transcripts():
    """Notebook cell 116 (In[130]); the columns from cell 115 (In[129]).

    The is_* masks are cell 105 (In[125]).
    """
    annot = load_annot_full()
    whole_genes = load_whole_genes()
    median_expression, gtex_tissues = load_expression()
    mexp = median_expression.loc[
        median_expression[gtex_tissues].max(axis=1) >= 10].gene
    is_expressed = annot.gene_id.isin(mexp)
    is_whole = annot.gene_id.isin(whole_genes)
    is_appris = annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x))
    is_other_annot = annot.gencode_ref != '.'
    is_annot = is_appris | is_other_annot
    is_super_appris = annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x) or ('superset' in x) or ('subset' in x))
    is_super_annot = is_super_appris | is_other_annot

    appris_len_1 = annot.groupby('gene').appris_transcript_length.mean().quantile(0.1)
    appris_len_9 = annot.groupby('gene').appris_transcript_length.mean().quantile(0.9)

    is_len = (annot.appris_transcript_length >= appris_len_1) & (annot.appris_transcript_length <= appris_len_9)
    total_col = [
        len(annot.loc[is_annot].transcript.unique()),#/len(annot.gene.unique()),
        len(annot.loc[is_expressed & is_annot].transcript.unique()),#/len(annot.loc[is_expressed].gene.unique()),
        len(annot.loc[is_len & is_annot].transcript.unique()),#/len(annot.loc[is_len].gene.unique()),
        len(annot.loc[is_len & is_expressed & is_annot].transcript.unique()),#/len(annot.loc[is_len & is_expressed].gene.unique()),
        len(annot.loc[is_super_annot].transcript.unique()),#/len(annot.gene.unique()),
            len(annot.loc[is_len & is_expressed & is_super_annot].transcript.unique()),#/len(annot.loc[is_len & is_expressed].gene.unique()),
    ]

    whole_col = [
        len(annot.loc[is_annot & is_whole].transcript.unique()),#/len(annot.loc[is_whole].gene.unique()),
        len(annot.loc[is_expressed & is_annot & is_whole].transcript.unique()),#/len(annot.loc[is_expressed & is_whole].gene.unique()),
        len(annot.loc[is_len & is_annot & is_whole].transcript.unique()),#/len(annot.loc[is_len & is_whole].gene.unique()),
        len(annot.loc[is_len & is_expressed & is_annot & is_whole].transcript.unique()),#/len(annot.loc[is_len & is_expressed & is_whole].gene.unique()),
        len(annot.loc[is_super_annot & is_whole].transcript.unique()),#/len(annot.loc[is_whole].gene.unique()),
            len(annot.loc[is_len & is_expressed & is_super_annot & is_whole].transcript.unique()),#/len(annot.loc[is_len & is_expressed& is_whole].gene.unique()),
    ]
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

    return plt.gcf()


PANELS = {
    'torino_isoform_recovery_by_length': panel_torino_isoform_recovery_by_length,
    'torino_isoform_recovery_by_expression': panel_torino_isoform_recovery_by_expression,
    'torino_missing_junctions': panel_torino_missing_junctions,
    'torino_missing_junctions_v2': panel_torino_missing_junctions_v2,
    'torino_annot_transcripts': panel_torino_annot_transcripts,
}


# --------------------------------------------------------------------------
# Whether the Appris intron chain is observed, per gene
#
# This is the expensive part of the figure: appris_juncs_obs tabix-queries the
# junction file once per gene without an annotation match, a few thousand times.
# Reckon on an hour. Nothing caches it to disk, so it has to be computed.
# --------------------------------------------------------------------------
# --- get_junctions_bed
# Copied verbatim from Appris_1.ipynb cell 47 (In[249]).
def get_junctions_bed(coords, gene_id, junc_thres=None):
    """
    Function: run_tabix
    
    Description:
    This function retrieves junction coordinates from a specified genomic region 
    using Tabix, and filters the junctions for a specific gene ID. It then formats 
    the retrieved junctions into a pandas DataFrame and converts them to a 
    BedTool object.
    
    Input:
    - coords: A list of genomic coordinates in the format 'chromosome:start-end'.
    - junc_file: A file path to a Tabix-indexed file containing junction information.
    - gene_id: A string representing the gene ID to filter the junctions.
    
    Output:
    - junctions_bed: A BedTool object containing junction coordinates for the specified gene ID, 
                     or an empty DataFrame if no junctions are found.
    """
    junc_file = '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'
    junctions_bed = run_tabix_on_junc(junc_file, coords)

    

    
    if junctions_bed.shape[0] == 0:
        junctions_bed = BedTool.from_dataframe(junctions_bed)
        return junctions_bed

    junctions_bed[3] = [x.split('.')[0] for x in junctions_bed[3]]

    junctions_bed = junctions_bed.loc[junctions_bed[3] == gene_id]
    junctions_bed_counts = junctions_bed.loc[junctions_bed[3] == gene_id, junctions_bed.columns[4:]].astype(int).quantile(0.999,axis=1)

    if junc_thres is None:
        junc_thres = junctions_bed_counts.max()/200

    
    # Filter junctions for the specified gene ID
    junctions_bed = junctions_bed.loc[:, range(4)]
    junctions_bed.columns = ['chrom', 'start', 'end', 'gene']

    junctions_bed['junc_counts'] = list(junctions_bed_counts)

    

    # print(junctions_bed)


    # print(junctions_bed_counts)

    

    junctions_bed = junctions_bed[junctions_bed_counts >= junc_thres]

    # Convert start and end positions to integers
    junctions_bed.start = junctions_bed.start.astype(int)
    junctions_bed.end = junctions_bed.end.astype(int)

    ##### Make sure juncs are within gene body
    coords_start = int(coords[0].split(':')[1])
    coords_end = int(coords[-1].split(':')[1])
    junctions_bed = junctions_bed.loc[(junctions_bed.start >= coords_start) & (junctions_bed.end <= coords_end)]
    
    # Return empty DataFrame if no junctions are found for the gene ID
    if junctions_bed.shape[0] == 0:
        junctions_bed = BedTool.from_dataframe(junctions_bed)
        return junctions_bed
    
    


    
    
    # Generate unique junction names
    junctions_bed['junc_names'] = ['junction_' + str(i) for i in range(1, junctions_bed.shape[0]+1)]
    
    # Rearrange DataFrame columns
    junctions_bed = junctions_bed[['chrom', 'start', 'end', 'junc_names', 'junc_counts']]
    
    # Convert DataFrame to BedTool object
    junctions_bed = BedTool.from_dataframe(junctions_bed)
    
    return junctions_bed


# --- appris_juncs_obs
# Copied verbatim from Appris_1.ipynb cell 53 (In[287]).
# get_intron_chain and juncs2introns come from torino_isoforms, already on the
# path; the notebook had them from its own star-imports.
def appris_juncs_obs(bed_appris, gene):
    gene_df = bed_appris.loc[bed_appris.gene_name == gene]
    gene_id = gene_df.iloc[0].ensembl_id
    chrom = gene_df.iloc[0].chrom
    start = str(gene_df.start.min())
    end = str(gene_df.start.max())

    for transcript, transcript_df in gene_df.groupby('transcript_id'):
        appris_juncs = get_intron_chain(transcript_df)
        juncs_df = get_junctions_bed([f'{chrom}:{start}', f'{chrom}:{end}'], gene_id).to_dataframe()
        if len(juncs_df) == 0:
            return False
        obs_juncs = juncs2introns(get_junctions_bed([f'{chrom}:{start}', f'{chrom}:{end}'], gene_id).to_dataframe())

        if all([x in obs_juncs for x in appris_juncs]):
            return True

    return False

    # return all([x in obs_juncs for x in appris_juncs])

def juncs2introns(junc_bed):
    return list(zip(junc_bed.start - 1, junc_bed.end +1))


def load_bed_appris():
    """Appris principal exons with Ensembl ids -- cell 52 (In[239])."""
    def build():
        gencode_appris_iso = pd.read_csv('../code/resources/annotation/Human/bed/exons.apris_principal.bed.gz', sep='\t')
        gencode_appris_iso['chrom'] = gencode_appris_iso['#chrom']
        gencode_appris_iso['gene_id'] = gencode_appris_iso.gene_name

        bed_table = pd.read_csv('../code/config/human.protein_coding.bed.gz', sep='\t')
        bed_table = bed_table[['gene_name', 'gene_id']].copy()
        bed_table.columns = ['gene_name', 'ensembl_id']
        bed_appris = gencode_appris_iso.merge(bed_table, on = 'gene_name')
        return bed_appris
    return _cached('bed_appris', build)


def load_annot_summary(with_mexp_rank2=False):
    """Per-gene summary with obs_juncs -- cells 56, 57, 58 (In[330]-In[336]).

    with_mexp_rank2 continues to In[402]/In[403], which is the state the
    `_v2` panel and the expression-recovery panel ran against.
    """
    def build():
        annot = load_annot_full()
        whole_genes = load_whole_genes()
        bed_appris = load_bed_appris()
        annot_summary = annot.loc[annot.gene_id.isin(whole_genes)].groupby('gene')[['annot_match', 'appris_transcript_length', 'mexp_rank']].max()

        obs_all_juncs = []
        for gene, row in tqdm(annot_summary.iterrows()):
            if row.annot_match:
                obs_all_juncs.append(True)
            else:
                obs_all_juncs.append(appris_juncs_obs(bed_appris, gene))
        annot_summary['obs_juncs'] = obs_all_juncs
        return annot_summary
    summary = _cached(f'annot_summary:{_tag()}', build)
    if not with_mexp_rank2:
        return summary

    def build2():
        annot = load_annot_full()
        median_expression, gtex_tissues = load_expression()
        annot_summary = summary.copy()
        annot_summary = annot_summary.merge(annot[['gene_name', 'gene_id']].drop_duplicates(), left_index = True, right_on = 'gene_name')
        mexp_tab = median_expression.loc[median_expression.gene.isin(annot_summary.gene_id), gtex_tissues].copy()
        mexp_tab.index = median_expression.loc[median_expression.gene.isin(annot_summary.gene_id)].gene
        mexp_rank = pd.qcut(mexp_tab.mean(axis=1), 20, labels=range(1, 21)).reset_index()
        mexp_rank.columns = ['gene_id', 'mexp_rank2']

        annot_summary = annot_summary.merge(mexp_rank, on = 'gene_id')
        return annot_summary, mexp_tab, mexp_rank
    return _cached(f'annot_summary2:{_tag()}', build2)


# --------------------------------------------------------------------------
# PCA and clustermap of isoform usage, by expression decile
#
# The notebook runs the same four plots over two gene sets -- mexp_rank <= 1
# and mexp_rank >= 20 -- reusing the names df_imp / qqnorm for both, so which
# plot used which set cannot be read off the execution counts: cells 21 and 41
# both report In[23], which is impossible in one session and means the notebook
# was edited after running. The save paths say bottom_decile for cells 34/37 and
# top_decile for 41/43, while the panel names say rank20 / rank1. RANK_OF_PANEL
# below records the pairing that reproduces the published PNGs; flip it if a
# future annotation changes which is which.
#
# Heavy: prepare_data imputes the EL matrix, so these want ~8 GB.
# --------------------------------------------------------------------------
# --- the GTEx pipeline: prepare_data, normalize_qqnorm, plot_pca_scatter,
# --- plot_clustermap and friends.
# Copied from Appris_1.ipynb cell 19 (In[18]); definitions only.
import numpy as np


import pandas as pd


import matplotlib.patches as mpatches


import matplotlib.pyplot as plt


import seaborn as sns


from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold, cross_val_score


from sklearn.preprocessing import LabelEncoder


# Pipeline defaults, from Appris_1.ipynb cell 19 (In[18]). These are the
# defaults of prepare_data / plot_pca_scatter / run_rf_classifier below, so
# they must exist at import time.
MAX_MISSING_FEATURE = 0.50
MAX_MISSING_SAMPLE = 0.50
N_PCA_COMPONENTS = 50
RF_N_ESTIMATORS = 300
RF_CV_FOLDS = 5
RANDOM_STATE = 42


def _hex_to_rgb01(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def build_tissue_color_map(gtex_colors: dict) -> dict:
    """Return {tissue_name: (r, g, b)} in matplotlib [0,1] float format."""
    return {t: _hex_to_rgb01(v["tissue_color_hex"]) for t, v in gtex_colors.items()}


def get_tissue_labels(metadata: pd.DataFrame, sample_cols) -> pd.Series:
    """Return a Series {sample_id: tissue} restricted to sample_cols."""
    sample_map = dict(zip(metadata["entity:sample_id"], metadata["tissue_site_detail"]))
    common = [s for s in sample_cols if s in sample_map]
    return pd.Series({s: sample_map[s] for s in common}, name="tissue")


def prepare_data(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    max_missing_feature: float = MAX_MISSING_FEATURE,
    max_missing_sample:  float = MAX_MISSING_SAMPLE,
) -> tuple:
    """
    1. Propagate NAs at the gid-group level: for each (gid, sample) pair,
       if any row in that gid has NA for that sample, set ALL rows in that
       gid to NA for that sample. This ensures we never impute partial groups,
       fully preserving the zero-sum constraint.
    2. Filter rows  (features) with > max_missing_feature fraction of NAs.
    3. Filter columns (samples) with > max_missing_sample  fraction of NAs.
    4. Restrict to samples present in metadata; attach tissue labels.
    5. Impute remaining NAs with the column (sample) median.

    Returns
    -------
    df_imp        : DataFrame with meta cols + sample cols, fully imputed.
    tissue_labels : Series {sample_id: tissue}, aligned to df_imp sample cols.
    """
    gid_col     = df.columns[4]
    sample_cols = list(df.columns[6:])
    meta_cols   = list(df.columns[:6])

    # ── 1. group-level NA propagation ────────────────────────────────────────
    expr = df[sample_cols].copy()
    for _, idx in df.groupby(gid_col).groups.items():
        group_expr = expr.loc[idx]
        na_cols = group_expr.columns[group_expr.isna().any(axis=0)]
        if len(na_cols):
            expr.loc[idx, na_cols] = np.nan

    df = df.copy()
    df[sample_cols] = expr

    # ── 2. filter features (rows) ─────────────────────────────────────────────
    feat_na   = expr.isna().mean(axis=1)
    keep_rows = feat_na[feat_na <= max_missing_feature].index
    n_dropped_rows = len(df) - len(keep_rows)
    df = df.loc[keep_rows].reset_index(drop=True)
    print(f"[filter]  rows  dropped: {n_dropped_rows:,}  "
          f"(>{max_missing_feature*100:.0f}% NA)  ->  {len(df):,} remain")

    # ── 3. filter samples (columns) ───────────────────────────────────────────
    expr       = df[sample_cols]
    sample_na  = expr.isna().mean(axis=0)
    keep_cols  = sample_na[sample_na <= max_missing_sample].index.tolist()
    n_dropped_cols = len(sample_cols) - len(keep_cols)
    df = df[meta_cols + keep_cols].copy()
    print(f"[filter]  cols  dropped: {n_dropped_cols:,}  "
          f"(>{max_missing_sample*100:.0f}% NA)  ->  {len(keep_cols):,} remain")

    # ── 4. intersect with metadata ────────────────────────────────────────────
    tissue_labels = get_tissue_labels(metadata, keep_cols)
    common = tissue_labels.index.tolist()
    df = df[meta_cols + common].copy()
    print(f"[align]   samples with tissue label: {len(common):,}")

    # ── 5. impute with column median ──────────────────────────────────────────
    expr        = df[common]
    col_medians = expr.median(axis=0)
    df[common]  = expr.fillna(col_medians)
    remaining   = df[common].isna().sum().sum()
    print(f"[impute]  column-median imputation done  |  remaining NAs: {remaining}")

    return df, tissue_labels


def plot_pca_scatter(
    df_imp: pd.DataFrame,
    tissue_labels: pd.Series,
    gtex_colors: dict,
    n_components: int = N_PCA_COMPONENTS,
    save_path: str = None,
    figsize=(7,4)
) -> PCA:
    """
    PCA on (samples x features), scatter of PC1 vs PC2, dots coloured by tissue.
    Returns the fitted PCA object.
    """
    tissue_color_map = build_tissue_color_map(gtex_colors)
    sample_cols = tissue_labels.index.tolist()

    X = df_imp[sample_cols].values.T          # (n_samples, n_features)
    y = tissue_labels.values

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X)

    var = pca.explained_variance_ratio_
    print(f"[PCA]  PC1+PC2 variance explained: {(var[0]+var[1])*100:.1f}%")

    tissues = np.unique(y)
    fig, ax = plt.subplots(figsize=figsize)

    for tissue in tissues:
        mask  = y == tissue
        color = tissue_color_map.get(tissue, (0.5, 0.5, 0.5))
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], s=18, alpha=0.75, linewidths=0, label=tissue, rasterized=True)

    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)", fontsize=12)
    ax.set_title("PCA – samples coloured by tissue", fontsize=14, fontweight="bold")
    ax.text(0.99, 0.01, f'n = {X.shape[0]:,} samples', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(color=tissue_color_map.get(t, (0.5, 0.5, 0.5)), label=t)
        for t in tissues
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=False)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:

        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=500)
        
        # plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot]  PCA scatter saved -> {save_path}")
        plt.close()
        return pca


def run_rf_classifier(
    df_imp: pd.DataFrame,
    tissue_labels: pd.Series,
    n_components: int = N_PCA_COMPONENTS,
    n_estimators: int = RF_N_ESTIMATORS,
    cv_folds: int     = RF_CV_FOLDS,
) -> dict:
    """
    PCA -> Random Forest tissue classifier with stratified CV.

    Returns dict with keys: rf, le, X_pca, y_enc, y_tissue, pca, scores
    """
    sample_cols = tissue_labels.index.tolist()
    X           = df_imp[sample_cols].values.T    # (n_samples, n_features)
    y_tissue    = tissue_labels.values

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_tissue)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(rf, X_pca, y_enc, cv=cv, scoring="accuracy")

    print(f"[RF]  {cv_folds}-fold CV accuracy: "
          + "  ".join(f"{s:.3f}" for s in scores))
    print(f"[RF]  mean +/- std: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"[RF]  {len(le.classes_)} classes: {list(le.classes_)}")

    rf.fit(X_pca, y_enc)
    return dict(rf=rf, le=le, X_pca=X_pca, y_enc=y_enc,
                y_tissue=y_tissue, pca=pca, scores=scores)


def plot_clustermap(
    df_imp: pd.DataFrame,
    tissue_labels: pd.Series,
    gtex_colors: dict,
    n_top_features: int = 2000,
    figsize: tuple = (18, 14),
    save_path: str = None,
):
    """
    Seaborn clustermap (no row/col dendrograms) with tissue colour bar.
    Uses the top n_top_features rows by inter-quartile range.
    """
    tissue_color_map = build_tissue_color_map(gtex_colors)
    sample_cols      = tissue_labels.index.tolist()
    expr             = df_imp[sample_cols]

    if n_top_features is not None and n_top_features < len(expr):
        iqr  = expr.quantile(0.75, axis=1) - expr.quantile(0.25, axis=1)
        expr = expr.loc[iqr.nlargest(n_top_features).index]
        print(f"[clustermap]  using top {n_top_features} features by IQR")

    col_colors = tissue_labels.loc[sample_cols].map(
        lambda t: tissue_color_map.get(t, (0.5, 0.5, 0.5))
    )

    g = sns.clustermap(
        expr,
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=False,
        cmap="RdBu_r",
        center=0,
        robust=True,
        figsize=figsize,
        dendrogram_ratio=(0, 0),
        colors_ratio=0.015,
        cbar_pos=None,          # avoid tight_layout conflict with hidden dendrograms
    )

    g.ax_row_dendrogram.set_visible(False)

    # Suppress the column dendrogram
    g.ax_col_dendrogram.set_visible(False)

    # place colorbar just outside the top-right corner of the heatmap
    g.figure.canvas.draw()                          # force layout so bbox is populated
    hm_pos = g.ax_heatmap.get_position()            # (x0, y0, width, height) in fig fraction
    cbar_ax = g.figure.add_axes([
        hm_pos.x1 + 0.01,                          # just to the right of the heatmap
        hm_pos.y1 - 0.15,                          # near the top
        0.015,                                      # narrow bar
        0.12,                                       # height
    ])
    plt.colorbar(g.ax_heatmap.collections[0], cax=cbar_ax)
    cbar_ax.tick_params(labelsize=7)

    g.ax_heatmap.set_xlabel("Samples", fontsize=11)
    g.ax_heatmap.set_ylabel("Features", fontsize=11)
    g.figure.suptitle(
        f"Expression heatmap  (top {n_top_features} features by IQR)",
        y=1.01, fontsize=13, fontweight="bold",
    )


    g.ax_heatmap.set_rasterized(True)

    handles = [
        mpatches.Patch(color=tissue_color_map.get(t, (0.5, 0.5, 0.5)), label=t)
        for t in sorted(tissue_labels.unique())
    ]
    g.figure.legend(handles=handles, loc="lower center",
                    ncol=min(3, len(handles)), fontsize=7, frameon=False,
                    bbox_to_anchor=(0.5, -0.3), title="Tissue", title_fontsize=8)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=500)
        # plt.savefig(save_path, dpi=150, bbox_inches=None)
        print(f"[plot]  clustermap saved -> {save_path}")
        plt.close()


RANK_OF_PANEL = {'rank1': 'top', 'rank20': 'bottom'}
_RANK_QUERY = {'top': 'mexp_rank <= 1', 'bottom': 'mexp_rank >= 20'}


def load_qqnorm(rank):
    """df_imp / tissue_labels / qqnorm for one expression decile.

    rank='top'    -> mexp_rank <= 1   (notebook cell 21)
    rank='bottom' -> mexp_rank >= 20  (notebook cell 38)
    Efficiency factors are excluded either way, as the notebook does.
    """
    def build():
        annot = load_annot_full()
        human_EL = pd.read_csv('../code/results/torino/tables/Human/'
                               'isoforms_filtered.merged_isoforms.EL.bed.gz', sep='\t')
        human_EL['pid'] = human_EL.gid + '.' + human_EL.pid
        human_EL.index = human_EL.pid
        metadata = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv',
                               sep='\t')
        genes = annot.query(f'({_RANK_QUERY[rank]}) and (not efficiency_factor)').gene.unique()
        df_imp, tissue_labels = prepare_data(
            human_EL.loc[human_EL.gid.isin(genes)], metadata,
            max_missing_feature=0.1)
        return normalize_qqnorm(df_imp), tissue_labels
    return _cached(f'qqnorm:{rank}:{_tag()}', build)


def _decile_panel(panel, kind):
    qqnorm, tissue_labels = load_qqnorm(RANK_OF_PANEL[panel])
    if kind == 'pca':
        return plot_pca_scatter(qqnorm, tissue_labels, gtex_colors_cereb,
                                figsize=(7, 4))
    return plot_clustermap(qqnorm, tissue_labels, gtex_colors_cereb,
                           figsize=(6, 5))


@_panel
def panel_PCA_rank1():
    """Notebook cell 41. PCA of isoform usage, one expression decile."""
    _decile_panel('rank1', 'pca')
    return plt.gcf()


@_panel
def panel_heatmap_rank1():
    """Notebook cell 43. Clustermap of the same."""
    g = _decile_panel('rank1', 'heatmap')
    return getattr(g, 'figure', plt.gcf())


@_panel
def panel_PCA_rank20():
    """Notebook cell 34. PCA of isoform usage, the other decile."""
    _decile_panel('rank20', 'pca')
    return plt.gcf()


@_panel
def panel_heatmap_rank20():
    """Notebook cell 37. Clustermap of the same."""
    g = _decile_panel('rank20', 'heatmap')
    return getattr(g, 'figure', plt.gcf())


PANELS.update({
    'PCA_rank1': panel_PCA_rank1,
    'heatmap_rank1': panel_heatmap_rank1,
    'PCA_rank20': panel_PCA_rank20,
    'heatmap_rank20': panel_heatmap_rank20,
})


# --- normalize_qqnorm
# Copied from Appris_1.ipynb cell 20 (In[19]); definition only.
def normalize_qqnorm(
    df_imp: pd.DataFrame,
    min_quantile_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Apply the qqnorm normalization pipeline to the imputed data.

    Steps (mirroring qqnorm_multi_tissue from the reference code):
      1. Row-wise z-score scaling  (scale across samples for each feature).
      2. Column-wise qqnorm        (rank-based normal quantile transform
                                    applied to each sample independently).
      3. Filter features:
           - drop rows that are constant across all samples (fully
             median-imputed rows with no real variation).
           - drop rows whose 90th percentile is below min_quantile_threshold.

    Parameters
    ----------
    df_imp                 : fully imputed DataFrame from prepare_data()
    min_quantile_threshold : 90th-percentile minimum in original scale

    Returns
    -------
    DataFrame with meta cols + sample cols, qqnorm-normalised and filtered.
    """
    from scipy.stats import rankdata, norm as sp_norm
    from sklearn.preprocessing import scale as sk_scale

    def _qqnorm(x):
        n = len(x)
        a = 3.0 / 8.0 if n <= 10 else 0.5
        return sp_norm.ppf((rankdata(x) - a) / (n + 1.0 - 2.0 * a))

    meta_cols   = list(df_imp.columns[:6])
    sample_cols = list(df_imp.columns[6:])
    expr        = df_imp[sample_cols].copy()

    # ── 1. row-wise z-score ───────────────────────────────────────────────────
    scaled = pd.DataFrame(
        sk_scale(expr, axis=1),
        index=expr.index,
        columns=expr.columns,
    )

    # ── 2. column-wise qqnorm ─────────────────────────────────────────────────
    qqnorm_arr = np.array([_qqnorm(scaled[s].values) for s in sample_cols]).T
    df_qqnorm  = pd.DataFrame(qqnorm_arr, index=expr.index, columns=sample_cols)

    # ── 3. feature filtering ──────────────────────────────────────────────────
    is_constant = expr.apply(lambda r: r.nunique() == 1, axis=1)
    q90         = expr.quantile(0.90, axis=1)
    keep        = (~is_constant) & (q90 >= min_quantile_threshold)
    print(f"[qqnorm]  features dropped by filter: {(~keep).sum():,}  "
          f"->  {keep.sum():,} remain")

    # ── 4. re-attach meta cols ────────────────────────────────────────────────
    result = pd.concat([df_imp.loc[keep, meta_cols], df_qqnorm.loc[keep]], axis=1)
    print(f"[qqnorm]  output shape: {result.shape}")
    return result



# --- gtex_colors_cereb
# Used by all four decile panels (Appris_1 cells 34/37/41/43) but defined in no
# cell of that notebook and in no .py -- it was left in the kernel from an
# earlier session. It is reconstructible exactly, not guessed: the appris
# pipeline looks colours up by `tissue_site_detail` (get_tissue_labels), while
# gtex_colors2 is keyed by `tissue_id`, and sample.tsv maps the two one-to-one
# across all 54 tissues. So this is gtex_colors2 re-keyed, same colours.
#
# If the keying were wrong, build_tissue_color_map would miss every tissue and
# plot_pca_scatter's .get(..., grey) fallback would draw the whole figure grey
# -- which the comparison against the published PNG would catch at once.
def _build_gtex_colors_cereb():
    meta = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv',
                       sep='\t')[['tissue_id', 'tissue_site_detail']]
    detail_of = dict(meta.drop_duplicates().values)
    out = {detail_of[k]: v for k, v in gtex_colors2.items() if k in detail_of}
    missing = [k for k in gtex_colors2 if k not in detail_of]
    if missing:
        print(f'[colors] {len(missing)} tissue(s) in gtex_colors2 absent from '
              f'sample.tsv, left out: {", ".join(sorted(missing))}')
    return out


gtex_colors_cereb = _build_gtex_colors_cereb()


# ==========================================================================
# Missing GENCODE isoforms and missing APPRIS junctions, by expression
#
# This is NOT a replication of cells 61/67. Those compute the right thing the
# wrong way: appris_juncs_obs() calls get_junctions_bed() twice per transcript
# per gene -- the same call, line 437 and line 440 -- and each call spawns a
# tabix subprocess and builds a BedTool with a temp file. Across the gene set
# that is tens of thousands of subprocesses, which is where the ~83 minutes
# went. The underlying question is a handful of dataframe operations.
#
# Done directly instead:
#
#   1. Pool: non-overlapping genes (whole_genes), intersected with GENE_SUBSET
#      so the corrected notebook gets the corrected pool.
#   2. Missing a GENCODE-annotated transcript in Torino: annot_match is False
#      for every isoform of the gene.
#   3. Expression: median TPM across the 18 tissues Torino tested
#      (code/config/gtex_samples.tsv), from the GTEx v8 median-TPM gct.
#   4. Missing an APPRIS principal junction: none of the gene's APPRIS
#      transcripts has its full intron chain present in the junction data.
#
# The junction thresholding is the notebook's: per junction, the 0.999 quantile
# across samples; per gene, keep junctions at or above max(q999)/200. That
# needs the count columns, which is why the file is summarised once (one
# streaming pass, ~20 min) into figure_data/junction_q999.pkl and read from
# there afterwards.
#
# One deliberate difference: the notebook fetched a tabix region spanning
# gene_df.start.min() to gene_df.start.max() and *then* filtered by gene id,
# which silently drops junctions of the gene lying outside that span. Filtering
# by gene id alone keeps them. Noted because it can only move numbers up.
# ==========================================================================
JUNC_FILE = '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'
JUNC_Q999_PKL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'figure_data', 'junction_q999.pkl')
TORINO_SAMPLES_TSV = '../code/config/gtex_samples.tsv'


def build_junction_q999(force=False, chunksize=2000, out_path=None):
    """Summarise junctions.tab.gz to one 0.999-quantile per junction.

    3.5 GB and ~17,400 sample columns, so it is streamed in chunks and reduced
    to chrom/start/end/gene/q999 (357,747 rows). Refuses to redo the pass if
    the pickle exists; force=True to rebuild. Run on a compute node.
    """
    import pickle
    out_path = out_path or JUNC_Q999_PKL
    if os.path.exists(out_path) and not force:
        print(f'{out_path} already exists -- nothing to do. force=True to rebuild.')
        return out_path

    parts = []
    for i, ch in enumerate(pd.read_csv(JUNC_FILE, sep='\t', chunksize=chunksize,
                                       dtype={0: str})):
        vals = ch.iloc[:, 4:].to_numpy(dtype=np.float32)
        out = ch.iloc[:, :4].copy()
        out.columns = ['chrom', 'start', 'end', 'gene']
        out['q999'] = np.quantile(vals, 0.999, axis=1)
        parts.append(out)
        if i % 20 == 0:
            print(f'  chunk {i}  ({(i + 1) * chunksize:,} junctions)', flush=True)
    juncs = pd.concat(parts, ignore_index=True)
    juncs['gene'] = juncs.gene.astype(str).str.split('.').str[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(juncs, fh)
    print(f'wrote {out_path}  ({len(juncs):,} junctions)')
    return out_path


def load_junction_q999():
    """The summarised junction table."""
    def build():
        import pickle
        if not os.path.exists(JUNC_Q999_PKL):
            raise FileNotFoundError(
                f'{JUNC_Q999_PKL} does not exist. It summarises the 3.5 GB '
                f'junction table and takes ~20 min, so it is not built on '
                f'demand. Run build_junction_q999() on a compute node once.')
        with open(JUNC_Q999_PKL, 'rb') as fh:
            return pickle.load(fh)
    return _cached('junction_q999', build)


def _observed_introns_by_gene():
    """{ensembl gene id: set of (intron_start, intron_end)} after thresholding.

    Threshold is the notebook's: per gene, max(q999)/200.
    """
    juncs = load_junction_q999()
    thres = juncs.groupby('gene').q999.transform('max') / 200
    keep = juncs.loc[juncs.q999 >= thres]
    # juncs2introns: (start - 1, end + 1)
    out = {}
    for gene, df in keep.groupby('gene', sort=False):
        out[gene] = set(zip(df.start.astype(int) - 1, df.end.astype(int) + 1))
    return out


def missing_junctions_numbers(nbins=20):
    """Per expression bin: genes missing a GENCODE isoform, and of those, how
    many are also missing an APPRIS principal junction.

    Returns (DataFrame indexed by bin, dict of totals).
    """
    def build():
        annot = load_annot_full()
        whole = set(load_whole_genes())
        median_expression, _ = load_expression()
        bed_appris = load_bed_appris()
        obs_by_gene = _observed_introns_by_gene()

        torino_tissues = sorted(
            pd.read_csv(TORINO_SAMPLES_TSV, sep='\t').tissue_id.unique())

        # --- the pool: non-overlapping genes, through GENE_SUBSET
        pool = annot.loc[annot.gene_id.isin(whole)]
        pool = _subset(pool, pool.gene)

        # --- expression: median TPM across the 18 Torino tissues
        missing_cols = [t for t in torino_tissues
                        if t not in median_expression.columns]
        if missing_cols:
            raise KeyError(f'tissue columns absent from the gct: {missing_cols}')
        mexp = median_expression[['gene'] + torino_tissues].copy()
        mexp['median_tpm'] = mexp[torino_tissues].median(axis=1)

        # Attach expression at the row level, then collapse to ONE ROW PER GENE.
        # Grouping by ['gene', 'gene_id'] instead gave 6,191 rows for 6,186
        # genes -- a handful of genes carry more than one gene_id, and each
        # extra row was counted again, which also pushed the missing-isoform
        # total from 746 to 747. The gene is the unit here, so group on it
        # alone; median_tpm takes whichever gene_id the gct has, and a LEFT
        # merge keeps genes the gct lacks instead of dropping them silently.
        pool = pool.merge(mexp[['gene', 'median_tpm']], left_on='gene_id',
                          right_on='gene', how='left', suffixes=('', '_gct'))
        per_gene = pool.groupby('gene').agg(
            annot_match=('annot_match', 'max'),
            median_tpm=('median_tpm', 'max')).reset_index()
        per_gene['bin'] = pd.qcut(per_gene.median_tpm, nbins,
                                  labels=range(1, nbins + 1), duplicates='drop')

        # --- APPRIS junctions, only for the genes that need the check
        need = per_gene.loc[~per_gene.annot_match.astype(bool)]
        appris = bed_appris.loc[bed_appris.gene_name.isin(set(need.gene))]
        appris_by_gene = dict(list(appris.groupby('gene_name', sort=False)))
        # gene -> one gene_id, for the junction lookup
        gid_of = (annot.drop_duplicates('gene').set_index('gene').gene_id
                  .to_dict())

        obs_juncs = {}
        for gene in need.gene:
            gene_id = gid_of.get(gene)
            gdf = appris_by_gene.get(gene)
            observed = obs_by_gene.get(str(gene_id).split('.')[0])
            if gdf is None or not observed:
                obs_juncs[gene] = False
                continue
            found = False
            for _, tdf in gdf.groupby('transcript_id', sort=False):
                chain = get_intron_chain(tdf)
                if chain and all(x in observed for x in chain):
                    found = True
                    break
            obs_juncs[gene] = found
        per_gene['obs_juncs'] = per_gene.gene.map(obs_juncs)

        miss = per_gene.loc[~per_gene.annot_match.astype(bool)]
        tab = pd.DataFrame({
            'genes': per_gene.groupby('bin', observed=False).size(),
            'missing_isoform': miss.groupby('bin', observed=False).size(),
            'missing_junctions': miss.loc[~miss.obs_juncs.astype(bool)]
                                     .groupby('bin', observed=False).size(),
            'median_tpm': per_gene.groupby('bin', observed=False).median_tpm.median(),
        }).fillna(0)
        for c in ('genes', 'missing_isoform', 'missing_junctions'):
            tab[c] = tab[c].astype(int)

        totals = {
            'pool genes': int(len(per_gene)),
            'missing a GENCODE isoform': int(len(miss)),
            'also missing an APPRIS junction': int((~miss.obs_juncs.astype(bool)).sum()),
            'APPRIS junctions observed': int(miss.obs_juncs.astype(bool).sum()),
            'tissues': len(torino_tissues),
            'bins': int(tab.index.size),
        }
        return tab, totals
    return _cached(f'missing_junctions:{_tag()}:{nbins}', build)


@_panel
def panel_missing_junctions_by_expression(nbins=20):
    """Genes lacking a GENCODE isoform in Torino, by expression bin."""
    tab, totals = missing_junctions_numbers(nbins)
    x = list(range(1, len(tab) + 1))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x, tab.missing_isoform, color='lightsteelblue',
           label=f'Missing isoform (n = {totals["missing a GENCODE isoform"]:,})')
    ax.bar(x, tab.missing_junctions,
           label=f'Missing junctions (n = '
                 f'{totals["also missing an APPRIS junction"]:,})')
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel('Genes')
    ax.set_xlabel('Median gene expression, binned rank (low to high)')
    ax.spines[['top', 'right']].set_visible(False)
    # Mark whole bins only -- a tick between two bins (17.5) labels nothing.
    ticks = [t for t in (5, 10, 15, 20) if t <= len(tab)]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_title(f'{totals["pool genes"]:,} non-overlapping genes, '
                 f'median TPM over {totals["tissues"]} tissues', fontsize=9)
    return fig


PANELS['missing_junctions_by_expression'] = panel_missing_junctions_by_expression
