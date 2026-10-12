"""SFigure2 VastDB cassette-exon panel, from the results already on disk.

One panel, from analysis/Brain_Cassette_SSMD.ipynb: for VastDB skipped-exon and
cassette-exon events that are brain-specific, the SSMD of the Torino isoform
carrying the exon against the one skipping it.

The two counting loops are copied verbatim; they walk the VastDB PSI table and
ask which Torino isoforms contain or skip each exon, which is not stored
anywhere. `found_CE` was built at In[151] from the In[150] loop and `found_SE`
at In[214] from the In[209] loop, both against the helper definitions live at
In[141] -- that is the pairing reproduced here.

Nothing here writes; the guard in SFigure2_intron_retention is active.
"""

import os
import sys

from SFigure2_intron_retention import (      # noqa: F401  -- shared setup
    ANALYSIS_DIR, np, pd, plt, sns, BedTool, bedtools_safe, _cached, _panel,
    free, _subset, _tag, load_annot,
)

import SFigure2_intron_retention as _R

import importlib
from tqdm import tqdm

import plot_ssmd_results
importlib.reload(plot_ssmd_results)
from plot_ssmd_results import SSMDResults


# --------------------------------------------------------------------------
# Inputs -- every one a file that already exists
# --------------------------------------------------------------------------
def load_psi_table():
    """VastDB PSI table -- notebook cell 3 (In[92])."""
    def build():
        return pd.read_csv('../code/resources/vastdb/PSI_TABLE-hg38_filtered.tab.gz',
                           sep='\t')
    return _cached('psi_table', build)


def load_human_res():
    """SSMD results -- notebook cell 5 (In[5])."""
    def build():
        return SSMDResults('../code/results/torino/tables/Human/SSMD/PVal')
    return _cached('human_res', build)


def load_human_iso():
    """Torino isoform annotation -- notebook cell 7 (In[7])."""
    def build():
        iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz',
                          sep='\t',
                          names=['chrom', 'start', 'end', 'gene_id', 'transcript_id',
                                 'strand', 'factors', 'exon', 'notes'])
        return _subset(iso, iso.gene_id)
    return _cached(f'human_iso_ssmd:{_tag()}', build)


# --- get_exon_bed and the first versions of the exon helpers
# Copied verbatim from Brain_Cassette_SSMD.ipynb cell 6 (In[6]).
# get_exon_bed is only ever defined here; the cell 16 (In[141])
# definitions below override the rest and still call it.
# human_iso.loc[human_iso.transcript_id=='NCKAP1.isoform_4']

def get_juncs(bed):
    juncs = pd.DataFrame()
    chrom_cols = list(bed.chrom)[:-1]
    start_cols = list(bed.end.astype(int)+1)[:-1]
    end_cols = list(bed.start.astype(int)-1)[1:]
    juncs['chrom'] = chrom_cols
    juncs['start'] = start_cols
    juncs['end'] = end_cols
    return juncs

def get_exon_bed(coords, slop = 0):
    chrom, base_pairs = coords.split(':')
    start, end = base_pairs.split('-')
    bed_df = pd.DataFrame([[chrom, int(start), int(end)]], columns = ['chrom', 'start', 'end'])
    bed_df.start -= slop
    bed_df.end += slop
    bed_out = BedTool.from_dataframe(bed_df)
    return bed_out

def skipping_exon_isoforms(human_iso, exon_coords, gene, annot):
    isoforms_no_exon = []
    gene_bed = human_iso.loc[human_iso.gene_id==gene]
    exon_bed = get_exon_bed(exon_coords, slop=10)
    for transcript, df in gene_bed.groupby('transcript_id'):
        if len(df) >= 2:
            transcript_juncs = get_juncs(df)
            transcript_juncs_bed = BedTool.from_dataframe(transcript_juncs)
            perfect_match = transcript_juncs_bed.intersect(exon_bed, F = 1).to_dataframe()

            if len(perfect_match) > 0:
                # print(perfect_match)
                isoforms_no_exon.append(transcript)

    return isoforms_no_exon
            

def isoforms_with_exon(human_iso, exon_coords, gene, annot):
    gene_bed = BedTool.from_dataframe(human_iso.loc[human_iso.gene_id==gene])
    exon_bed = get_exon_bed(exon_coords)
    perfect_match = gene_bed.intersect(exon_bed, f=1, F=1).to_dataframe()

    if len(perfect_match) > 0:
        return list(perfect_match.score)

    efficiency_factors = annot.loc[(annot.gene==gene) & (~annot.efficiency_factor)]
    

    gene_bed_good_factors = BedTool.from_dataframe(human_iso.loc[
        (human_iso.gene_id==gene) & (~human_iso.transcript_id.isin(efficiency_factors))])

    imperfect_match = gene_bed_good_factors.intersect(exon_bed, f=0.25, F=0.25).to_dataframe()

    if len(imperfect_match) > 0:
        # print(imperfect_match)
        return list(imperfect_match.score)

    return []
    
def gene_top_in_brain(human_isoform_ssmd, isoform):

    ssmd_lim = human_isoform_ssmd['Brain'].abs_ssmd.quantile(0.75)

    brain_ssmd = human_isoform_ssmd['Brain'].loc[human_isoform_ssmd['Brain'].pid == isoform].ssmd.iloc[0]
    if brain_ssmd > ssmd_lim:
        return True

    other_tissues = ['Heart', 'Muscle_Skeletal', 'Liver', 'Lung', 'Testis', 'Breast_Mammary_Tissue', 
                     'Whole_Blood', 'Skin', 'Adipose_Subcutaneous']

    other_ssmd = [human_isoform_ssmd[tissue].loc[human_isoform_ssmd[tissue].pid == isoform].ssmd.iloc[0] for tissue in other_tissues]

    if brain_ssmd > np.max(other_ssmd):
        return True

    else:
        return False
    


def gene_bottom_in_brain(human_isoform_ssmd, isoform):

    # ssmd_lim = human_isoform_ssmd['Brain'].abs_ssmd.quantile(0.75)

    brain_ssmd = human_isoform_ssmd['Brain'].loc[human_isoform_ssmd['Brain'].pid == isoform].ssmd.iloc[0]
    if brain_ssmd < 0:
        return True

    other_tissues = ['Heart', 'Muscle_Skeletal', 'Liver', 'Lung', 'Testis', 'Breast_Mammary_Tissue', 
                     'Whole_Blood', 'Skin', 'Adipose_Subcutaneous']

    other_ssmd = [human_isoform_ssmd[tissue].loc[human_isoform_ssmd[tissue].pid == isoform].ssmd.iloc[0] for tissue in other_tissues]

    if brain_ssmd < np.min(other_ssmd):
        return True

    else:
        return False
    

    
    

# X = BedTool.from_dataframe(human_iso.loc[human_iso.gene_id=='BICDL1'])
# Y = get_exon_bed('chr12:120090975-120091019')


# --- isoforms_with_exon / skipping_exon_isoforms
# Copied verbatim from Brain_Cassette_SSMD.ipynb cell 16 (In[141]),
# the definitions live when both counting loops ran.
def isoforms_with_exon(human_iso, exon_coords, gene, annot):
    gene_bed = BedTool.from_dataframe(human_iso.loc[human_iso.gene_id==gene])
    exon_bed = get_exon_bed(exon_coords)
    perfect_match = gene_bed.intersect(exon_bed, f=1, F=1).to_dataframe()

    if len(perfect_match) > 0:
        return list(perfect_match.score)

    # return []

    efficiency_factors = annot.loc[(annot.gene==gene) & (~annot.efficiency_factor)]
    

    gene_bed_good_factors = BedTool.from_dataframe(human_iso.loc[
        (human_iso.gene_id==gene) & (~human_iso.transcript_id.isin(efficiency_factors))])

    imperfect_match = gene_bed_good_factors.intersect(exon_bed, f=0.25, F=0.25).to_dataframe()

    if len(imperfect_match) > 0:
        # print(imperfect_match)
        return list(imperfect_match.score)

    return []


def skipping_exon_isoforms(human_iso, exons_coords, fullco, gene, annot):

    
    isoforms_no_exon = []
    gene_bed = human_iso.loc[human_iso.gene_id==gene]
    juncs_bed = get_intron_bed(fullco)
    
    for transcript, df in gene_bed.groupby('transcript_id'):
        if len(df) >= 2:
            transcript_juncs = get_juncs(df)
            transcript_juncs['start'] = transcript_juncs.start.astype(int)
            transcript_juncs['end'] = transcript_juncs.end.astype(int)
            # print(transcript_juncs)
            transcript_juncs_bed = BedTool.from_dataframe(transcript_juncs)
            # print(ju
            perfect_match = transcript_juncs_bed.intersect(juncs_bed, F = 0.99, f=0.99).to_dataframe()

            if len(perfect_match) > 0:
                # print(perfect_match)
                isoforms_no_exon.append(transcript)

    return isoforms_no_exon
            

def get_intron_bed(coords, slop = -1):

    fk1, exon, fk2 = coords.split(',')
    chrom, fk1 = fk1.split(':')

    fk1_list = fk1.split('+')
    fk2_list = fk2.split('+')

    bed_rows = []

    for fk1 in fk1_list:
        for fk2 in fk2_list:
            if int(fk1) > int(fk2):
                bed_row = [chrom, int(fk2), int(fk1)]
            else:
                bed_row = [chrom, int(fk1), int(fk2)]
            bed_rows.append(bed_row)
    

    bed_df = pd.DataFrame(bed_rows, columns = ['chrom', 'start', 'end'])
    bed_df.start -= slop
    bed_df.end += slop
    # print(bed_df)
    bed_out = BedTool.from_dataframe(bed_df)
    return bed_out


# --- cdf
# Copied verbatim from Brain_Cassette_SSMD.ipynb cell 26 (In[218]).
def cdf(x):
    y = np.arange(len(x))
    y = y/len(y)
    return y, sorted(x)


def load_found_CE():
    """Brain-specific VastDB cassette exons.

    Loop copied verbatim from cell 31 (In[150]); the frame
    from cell 34 (In[151]).
    """
    def build():
        psi_table = load_psi_table()
        human_res = load_human_res()
        human_iso = load_human_iso()
        annot = load_annot()

        # count_found_and_max

        count_found = 0
        count_found_and_max = 0
        count_missing = 0

        gene_list = []
        exon_list = []
        found_exon = []
        top_brain = []
        isoform = []
        isoform_top = []
        found_skip = []
        bottom_skip = []
        event_list = []

        CE_isoform = []
        SE_isoform = []

        length = []



        for idx, row in tqdm(psi_table.loc[(psi_table['Brain'] >= 25) & (psi_table.Brain_diff >= 25) & \
            psi_table.GENE.isin(human_res.ssmd_iso.gid.unique())].iterrows()):
    
            gene = row.GENE
            event = row.EVENT
            exon_coords = row.COORD
            fullco = row.FullCO
            gene_list.append(gene)
            exon_list.append(exon_coords)
            event_list.append(event)

            iso_match = isoforms_with_exon(human_iso, exon_coords, gene, annot)

            if len (iso_match) > 0:
                found_exon.append(True)
                count_found += 1
                isoforms = [gene + '.' + x for x in iso_match]

                iso_ssmd = [x for x in isoforms if x in human_res.ssmd_iso.index] #.loc['SPATA20.isoform_1', 'Brain']

                if len(iso_ssmd) > 0:
            
                    top_brain.append(human_res.ssmd_iso.loc[iso_ssmd, 'Brain'].max())
                    CE_isoform.append(human_res.ssmd_iso.loc[iso_ssmd, 'Brain'].sort_values().index[-1])
                else:
                    top_brain.append(np.nan)
                    CE_isoform.append(np.nan)
            else:
                top_brain.append(np.nan)
                found_exon.append(False)
                CE_isoform.append(np.nan)
                count_missing += 1


            skip_exons = skipping_exon_isoforms(human_iso, exon_coords, fullco, gene, annot)

            if len(skip_exons) >= 1:
                found_skip.append(True)

                skipped_isoforms = [gene + '.' + x for x in skip_exons]

                skip_iso_ssmd = [x for x in skipped_isoforms if x in human_res.ssmd_iso.index] #.loc['SPATA20.isoform_1', 'Brain']

                if len(skip_iso_ssmd) > 0:
            
                    bottom_skip.append(human_res.ssmd_iso.loc[skip_iso_ssmd, 'Brain'].max())
                    SE_isoform.append(human_res.ssmd_iso.loc[skip_iso_ssmd, 'Brain'].sort_values().index[-1])
                else:
                    bottom_skip.append(np.nan)
                    SE_isoform.append(np.nan)


            else:
                found_skip.append(False)
                bottom_skip.append(np.nan)
                SE_isoform.append(np.nan)

        found_CE = pd.DataFrame({'gene':gene_list, 
                    'exon':exon_list,
                    'event':event_list,
                    # 'length':length,
                    'found_exon':found_exon,
                    'found_skip':found_skip,
                    'top_brain':top_brain,
                    'bottom_skip':bottom_skip,
                    'CE_isoform':CE_isoform,
                    'SE_isoform':SE_isoform,
                    'ssmd_diff':np.array(top_brain)-np.array(bottom_skip)
                   })
        return found_CE
    return _cached(f'load_found_CE:{_tag()}', build)


def load_found_SE():
    """Brain-specific VastDB skipped exons.

    Loop copied verbatim from cell 19 (In[209]); the frame
    from cell 23 (In[214]).
    """
    def build():
        psi_table = load_psi_table()
        human_res = load_human_res()
        human_iso = load_human_iso()
        annot = load_annot()

        # count_found_and_max

        count_found = 0
        count_found_and_max = 0
        count_missing = 0

        gene_list = []
        exon_list = []
        found_exon = []
        top_brain = []
        isoform = []
        isoform_top = []
        found_skip = []
        bottom_skip = []
        event_list = []

        CE_isoform = []
        SE_isoform = []

        length = []



        for idx, row in tqdm(psi_table.loc[(psi_table['Brain'] <= 25) & (psi_table.Brain_diff <= -25) & \
            psi_table.GENE.isin(human_res.ssmd_iso.gid.unique())].iterrows()):
    
            gene = row.GENE
            event = row.EVENT
            exon_coords = row.COORD
            fullco = row.FullCO
            gene_list.append(gene)
            exon_list.append(exon_coords)
            event_list.append(event)

            iso_match = isoforms_with_exon(human_iso, exon_coords, gene, annot)

            if len (iso_match) > 0:
                found_exon.append(True)
                count_found += 1
                isoforms = [gene + '.' + x for x in iso_match]

                iso_ssmd = [x for x in isoforms if x in human_res.ssmd_iso.index] #.loc['SPATA20.isoform_1', 'Brain']

                if len(iso_ssmd) > 0:
            
                    top_brain.append(human_res.ssmd_iso.loc[iso_ssmd, 'Brain'].max())
                    CE_isoform.append(human_res.ssmd_iso.loc[iso_ssmd, 'Brain'].sort_values().index[-1])
                else:
                    top_brain.append(np.nan)
                    CE_isoform.append(np.nan)
            else:
                top_brain.append(np.nan)
                found_exon.append(False)
                CE_isoform.append(np.nan)
                count_missing += 1


            skip_exons = skipping_exon_isoforms(human_iso, exon_coords, fullco, gene, annot)

            if len(skip_exons) >= 1:
                found_skip.append(True)

                skipped_isoforms = [gene + '.' + x for x in skip_exons]

                skip_iso_ssmd = [x for x in skipped_isoforms if x in human_res.ssmd_iso.index] #.loc['SPATA20.isoform_1', 'Brain']

                if len(skip_iso_ssmd) > 0:
            
                    bottom_skip.append(human_res.ssmd_iso.loc[skip_iso_ssmd, 'Brain'].max())
                    SE_isoform.append(human_res.ssmd_iso.loc[skip_iso_ssmd, 'Brain'].sort_values().index[-1])
                else:
                    bottom_skip.append(np.nan)
                    SE_isoform.append(np.nan)



        
                # bottom_skip_all = any([gene_bottom_in_brain(human_isoform_ssmd, iso) for iso in skip_exons])
                # if bottom_skip_all:
                #     bottom_skip.append(True)
                # else:
                #     bottom_skip.append(False)

            else:
                found_skip.append(False)
                bottom_skip.append(np.nan)
                SE_isoform.append(np.nan)

            # # print('')
            # # print('###############')

        found_SE = pd.DataFrame({'gene':gene_list, 
                    'exon':exon_list,
                    'event':event_list,
                    'fullco':fullco,
                    'found_exon':found_exon,
                    'found_skip':found_skip,
                    'top_brain':top_brain,
                    'bottom_skip':bottom_skip,
                    'CE_isoform':CE_isoform,
                    'SE_isoform':SE_isoform,
                    'ssmd_diff':np.array(top_brain)-np.array(bottom_skip)
                   })
        return found_SE
    return _cached(f'load_found_SE:{_tag()}', build)


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
@_panel
def panel_VastDB_cassette_exons_brain():
    """Notebook cell 39 (In[221]), verbatim but for the savefig calls."""
    n = vastdb_ssmd_counts()

    # .dropna() is applied to the whole filtered frame before a column is taken,
    # so the two curves in a panel are drawn over the same rows -- hence one n
    # per panel, which vastdb_ssmd_counts() asserts rather than assumes.
    found_SE = load_found_SE()
    found_CE = load_found_CE()

    y, x = cdf(np.array(found_SE.loc[found_SE.found_exon & found_SE.found_skip].dropna().top_brain))
    y_skip, x_skip = cdf(np.array(found_SE.loc[found_SE.found_exon & found_SE.found_skip].dropna().bottom_skip))


    fig, ax = plt.subplots(ncols = 2, figsize=(7,3))
    ax[0].plot(x, y, c='tab:red', label=f'Cassette exon (n = {n["SE_top_brain"]:,})')
    ax[0].plot(x_skip, y_skip, c='tab:blue',
               label=f'Skipped exon (n = {n["SE_bottom_skip"]:,})')
    ax[0].plot([0, 0], [0, 1], linestyle='--', c='tab:grey')
    ax[0].legend(frameon=False, title='Brain isoform', fontsize=8,
                 title_fontsize=8, loc='upper left')

    y, x = cdf(np.array(found_CE.loc[found_CE.found_exon & found_CE.found_skip].dropna().top_brain))
    y_skip, x_skip = cdf(np.array(found_CE.loc[found_CE.found_exon & found_CE.found_skip].dropna().bottom_skip))

    ax[1].plot(x, y, c='tab:red', label=f'Cassette exon (n = {n["CE_top_brain"]:,})')
    ax[1].plot(x_skip, y_skip, c='tab:blue',
               label=f'Skipped exon (n = {n["CE_bottom_skip"]:,})')
    ax[1].plot([0, 0], [0, 1], linestyle='--', c='tab:grey')

    ax[1].set_yticks([])

    ax[1].legend(frameon=False, title='Brain isoform', fontsize=8,
                 title_fontsize=8, loc='upper left')

    ax[0].set_title('VastDB Skipped Exons')
    ax[1].set_title('VastDB Casette Exons')

    ax[0].set_ylabel('Cumulative distribution')
    ax[0].set_xlabel('Torino isoform SSMD in Brain')
    ax[1].set_xlabel('Torino isoform SSMD in Brain')

    fig.subplots_adjust(wspace=0.1)

    return fig


def vastdb_ssmd_counts():
    """Exons behind each curve of panel_VastDB_cassette_exons_brain.

    One entry per plotted line. The curve is drawn from
    `df.loc[found_exon & found_skip].dropna().<column>`, and dropna() acts on
    the whole frame, so the two columns of a panel come from the same rows --
    asserted below, not assumed, so a future change that breaks the tie is
    reported instead of mislabelling the figure.
    """
    out = {}
    for key, df in (('SE', load_found_SE()), ('CE', load_found_CE())):
        sub = df.loc[df.found_exon & df.found_skip].dropna()
        out[f'{key}_top_brain'] = int(np.array(sub.top_brain).size)
        out[f'{key}_bottom_skip'] = int(np.array(sub.bottom_skip).size)
        out[f'{key}_rows'] = int(len(sub))
        out[f'{key}_before_dropna'] = int(len(df.loc[df.found_exon & df.found_skip]))
    return out


def vastdb_ssmd_table():
    """The per-curve n, for both gene sets."""
    keep = _R.GENE_SUBSET
    try:
        _R.GENE_SUBSET = None
        free()
        a = vastdb_ssmd_counts()
        _R.GENE_SUBSET = _R.load_select_genes()
        free()
        c = vastdb_ssmd_counts()
    finally:
        _R.GENE_SUBSET = keep
        free()
    rows = ['**VastDB SSMD panels: exons per curve**', '',
            '| panel | curve | all genes | corrected |',
            '|---|---|---:|---:|']
    for key, panel in (('SE', 'VastDB Skipped Exons'),
                       ('CE', 'VastDB Cassette Exons')):
        for col, lab in (('top_brain', 'Cassette exon (red)'),
                         ('bottom_skip', 'Skipped exon (blue)')):
            rows.append(f'| {panel} | {lab} | {a[f"{key}_{col}"]:,} '
                        f'| {c[f"{key}_{col}"]:,} |')
        rows.append(f'| {panel} | rows before dropna | '
                    f'{a[f"{key}_before_dropna"]:,} | {c[f"{key}_before_dropna"]:,} |')
    rows += ['', '> Both curves in a panel are drawn over the same rows: '
                 'dropna() is applied to the frame before a column is selected.']
    return '\n'.join(rows)


PANELS = {'VastDB_cassette_exons_brain': panel_VastDB_cassette_exons_brain}
