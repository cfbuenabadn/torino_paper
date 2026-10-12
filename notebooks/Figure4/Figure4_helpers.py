"""
Figure4_helpers.py — Data loading, processing, and orchestration for Figure 4.

Contains all data-loading and data-processing functions, plus a run_all()
function that produces the final plot-ready variables and saves them as
pickles in figure_data/.

Must be run from the analysis/ directory (or have CWD set appropriately), so
that the relative '../code/...' paths resolve.  GeneticsPlots.prepare_QTL_metaplot
also hard-codes '../code/resources/count_matrices_filtered/...', so the analysis/
CWD is mandatory for the coverage panels.

Heavy/specialised imports (rpy2, vcf, tabix, tqdm, scipy, upsetplot, and the
analysis/*.py modules that pull them in) are done *inside* the functions that
need them, so that `from Figure4_helpers import run_all` works in a bare pandas
environment.
"""

import gzip
import os
import pickle

import numpy as np
import pandas as pd


# ================================================================
# Paths / constants
# ================================================================

ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
TORINO_SCRIPTS = '/project2/yangili1/cfbuenabadn/torino/scripts'

PI1_TABLE   = '../code/results/pi1/pi1_tables/Pi1Grouped.txt'
QTL_PERM    = '../code/results/QTLs/GTEx/{tissue}/GroupedPermutationPass.FDR_Added.txt.gz'
QTL_QQNORM  = '../code/results/QTLs/GTEx/{tissue}/qqnorm.sorted.bed.gz'
QTL_NOMINAL = '../code/results/QTLs/GTEx/{tissue}/ForGWASColoc.NominalPass.txt.tabix.gz'

HYPRCOLOC   = '../code/hyprcoloc/Results/ForGWASColoc/GWAS_GTEx_01/results.txt.gz'
HUMAN_ANNOT = '../code/results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz'
ISOFORM_BED = '../code/results/torino/Human/isoforms/isoforms.bed.gz'
GTF_BED     = '../code/resources/annotation/Human/bed/exons.all.bed.gz'
RDS_PATH    = '../code/results/torino/factorization_filtered/Human/{gene}.rds'

GTEX_VCF = ('../code/resources/GTEx/genotype/GTEx_Analysis_2017-06-05_v8_WGS_VCF_files_'
            'GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.'
            'SHAPEIT2_phased.vcf.gz')

GWAS_SUMSTATS = '/project2/mstephens/cfbuenabadn/gtex-stm/code/gwas/hg38_summary_stats/{trait}.bed.gz'

# --- ITGA1 / coronary artery disease -----------------------------------------
ITGA1_GENE_ID = 'ENSG00000213949'
ITGA1_TISSUE  = 'Liver'
ITGA1_VAR     = 'chr5_52897294_A_G_b38'
ITGA1_XLIM    = (52893675 - 10000, 52897528 + 10000)

# --- ZSWIM7 / Parkinson's disease --------------------------------------------
ZSWIM7_GENE_ID = 'ENSG00000214941'
ZSWIM7_TISSUE  = 'Brain_Anterior_cingulate_cortex_BA24'
ZSWIM7_VAR     = 'chr17_15976596_C_T_b38'
ZSWIM7_XLIM    = (15999710 - 10000, 15999710 + 1000)


def _genetics_plots():
    """Import the analysis/GeneticsPlots.py helpers lazily (pulls sklearn/tqdm/scipy)."""
    import sys
    if ANALYSIS_DIR not in sys.path:
        sys.path.append(ANALYSIS_DIR)
    import GeneticsPlots
    return GeneticsPlots


# ================================================================
# Panel a — QTLs (factor-QTL counts per tissue)
# ================================================================

def compute_qtl_barplot():
    """
    Count factor QTLs (q <= 0.1) per GTEx tissue and sort descending.

    Returns the plot-ready dict consumed by plot_qtl_barplot().
    Source: analysis/Figure_4.ipynb cells 1, 10, 14.
    """
    import sys
    from tqdm import tqdm
    if TORINO_SCRIPTS not in sys.path:
        sys.path.append(TORINO_SCRIPTS)
    from torino_plots import gtex_colors2

    pi1_tables = pd.read_csv(
        PI1_TABLE, sep='\t',
        names=['ref_tissue', 'query_tissue', 'pi1_1e1', 'pi1_5e2', 'pi1_1e2', 'pi1_1e4'],
    )
    tissues = pi1_tables.ref_tissue.unique()

    qtls_list = []
    col_colors = []
    for tissue in tqdm(tissues):
        qtls = pd.read_csv(QTL_PERM.format(tissue=tissue), sep=' ', index_col=0)
        qtls_list.append((qtls.q <= 0.1).sum())
        col_colors.append(gtex_colors2[tissue]['tissue_color_hex'])

    # Sort by QTL count, descending (cell 14)
    order = np.argsort(qtls_list)[::-1]
    return {
        'qtls_sorted':       [qtls_list[i] for i in order],
        'tissues_sorted':    [tissues[i] for i in order],
        'col_colors_sorted': [col_colors[i] for i in order],
    }


# ================================================================
# Panel b — upsetplot_coloc (RNA-processing classes of colocalised loci)
# ================================================================

def load_isoform_annotation():
    """
    Load the corrected isoform annotation and flag 'efficiency factor' isoforms.

    Source: analysis/Figure_4.ipynb cell 19 (identical code repeated in cell 105).
    """
    annot = pd.read_csv(HUMAN_ANNOT, sep='\t')
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
    return annot


def load_hyprcoloc_results():
    """
    Load the hyprcoloc output and add the GWAS_trait / coloc / Dropped_GWAS_trait
    columns, then restrict to traits with >= 10 testable GWAS loci.

    Source: analysis/Figure_4.ipynb cells 19, 20, 27-30, 34.

    Note: cells 27-28 build the full `trait_annotations` table only to read off
    `counter` for the >= 10 threshold, so `counter` is computed directly here.
    The resulting trait set (`list_67`) is identical; it is only ever used via
    `.isin(...)`, so its order does not matter.
    """
    hyprcoloc_results = pd.read_csv(HYPRCOLOC, sep='\t')

    hyprcoloc_results['GWAS_trait'] = \
        hyprcoloc_results.GWASLeadSnpChrom_Pos_RefAllele_AltAllele_rsID_trait.apply(
            lambda x: '_'.join(x.split('_')[2:]))

    trait_coloc = []
    for idx, row in hyprcoloc_results.iterrows():
        trait = row.GWASLeadSnpChrom_Pos_RefAllele_AltAllele_rsID_trait
        try:
            trait_coloc.append(trait in row.ColocalizedTraits)
        except:
            trait_coloc.append(False)
    hyprcoloc_results['coloc'] = trait_coloc

    dropped_trait = []
    for idx, row in hyprcoloc_results.iterrows():
        if str(row.DroppedTrait) == 'nan':
            dropped_trait.append(False)
        else:
            dropped_trait.append(row.GWAS_trait in row.DroppedTrait)
    hyprcoloc_results['Dropped_GWAS_trait'] = dropped_trait

    # counter = number of distinct, non-dropped GWAS loci per trait (cell 27)
    counter = hyprcoloc_results.loc[~hyprcoloc_results.Dropped_GWAS_trait].groupby(
        'GWAS_trait').GWASLeadSnpChrom_Pos_RefAllele_AltAllele_rsID_trait.nunique()
    list_67 = counter.loc[counter >= 10].index

    return hyprcoloc_results.loc[hyprcoloc_results.GWAS_trait.isin(list_67)].copy()


def build_coloc_indicators(hyprcoloc_results, annot, fallback=False):
    """
    Assign every colocalised GWAS locus to one or more RNA-processing classes and
    return the boolean indicator matrix.

    Source: analysis/Figure_4.ipynb cells 41, 44.

    `fallback` controls the `ref == 'ref_transcript'` branch below, which
    re-classifies a locus from the gene's *other* isoforms when its own yield no
    class. It is off by default: it rests on `ref` being whatever the preceding
    loop happened to leave behind, and it makes this panel disagree with
    SFigure4, which has no such branch. With it on, 61 loci move out of 'Other'
    into real classes -- 11 of them into Intron Retention alone, which is the
    291-versus-280 difference between the two figures.
    """
    trait_dir_lists = {'RI': [],
                       "3' UTR": [],
                       'Other': [],
                       'MXE': [],
                       'CE': [],
                       'Efficiency factor': []}

    for idx, row in hyprcoloc_results.loc[hyprcoloc_results.coloc].iterrows():
        trait = row.GWASLeadSnpChrom_Pos_RefAllele_AltAllele_rsID_trait

        traits = row.ColocalizedTraits.split(', ')
        isoforms = sorted(set([y.split(';')[1] for y in traits if 'isoform_' in y]))

        if len(isoforms) > 0:
            if (all(annot.loc[annot.transcript.isin(isoforms), 'efficiency_factor'])):
                trait_dir_lists['Efficiency factor'].append(trait)
                continue
            else:
                mark = False
                appris_refs = annot.loc[annot.transcript.isin(isoforms) & (~annot.efficiency_factor), 'appris_ref']
                for ref in appris_refs:
                    if 'RI' in ref:
                        trait_dir_lists['RI'].append(trait)
                        mark = True
                    if ('CE' in ref) or ('SE' in ref):
                        trait_dir_lists['CE'].append(trait)
                        mark = True
                    if ('MXE' in ref):
                        trait_dir_lists['MXE'].append(trait)
                        mark = True
                    if ('UTR3' in ref):
                        trait_dir_lists["3' UTR"].append(trait)
                        mark = True
                if not mark:
                    if fallback and ref == 'ref_transcript':
                        gene = isoforms[0].split('.')[0]
                        gene_iso = list(annot.loc[(annot.gene == gene) & (~annot.efficiency_factor) &
                                                  (~annot.transcript.isin(isoforms))].transcript)

                        appris_refs = annot.loc[annot.transcript.isin(gene_iso) & (~annot.efficiency_factor), 'appris_ref']
                        for ref in appris_refs:
                            if 'RI' in ref:
                                trait_dir_lists['RI'].append(trait)
                                mark = True
                            if ('CE' in ref) or ('SE' in ref):
                                trait_dir_lists['CE'].append(trait)
                                mark = True
                            if ('MXE' in ref):
                                trait_dir_lists['MXE'].append(trait)
                                mark = True
                            if ('UTR3' in ref):
                                trait_dir_lists["3' UTR"].append(trait)
                                mark = True

                    if not mark:
                        trait_dir_lists['Other'].append(trait)

    all_items = sorted({item for lst in trait_dir_lists.values() for item in lst})
    return pd.DataFrame(
        {tag: [item in lst for item in all_items] for tag, lst in trait_dir_lists.items()},
        index=all_items,
    )


def build_upset_data(df):
    """
    Turn the indicator matrix into the upsetplot input, renamed and level-ordered
    so that 'Efficiency factor' sits last (far right).

    Source: analysis/Figure_4.ipynb cell 51.
    """
    from upsetplot import from_indicators

    cols = ['RI', "3' UTR", 'CE', 'MXE', 'Other', 'Efficiency factor']
    df_ = df[cols].fillna(False).astype(bool).copy()
    cols_ = ['Intron Retention', "Alt. PolyA", 'Cassette Exon', 'MXE', 'Other', 'Efficiency factor']
    df_.columns = cols_
    upset_data = from_indicators(cols_, data=df_)

    current_order = list(upset_data.index.names)
    current_order.remove('Efficiency factor')
    new_order = current_order + ['Efficiency factor']
    return upset_data.reorder_levels(new_order)


# ================================================================
# Example loci — shared loading
# ================================================================

def load_beds():
    """
    Load the torino isoform bed and the GENCODE exon bed.

    Source: analysis/Figure_4.ipynb cell 105.
    """
    isoform_bed = pd.read_csv(
        ISOFORM_BED, sep='\t',
        names=['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand',
               'factors', 'exon_id', 'notes'],
    )
    gtf_bed = pd.read_csv(GTF_BED, sep='\t')
    gtf_bed['chrom'] = gtf_bed['#chrom']
    return isoform_bed, gtf_bed


def open_gtex_vcf():
    """Open the GTEx phased genotype VCF (cell 104)."""
    import vcf
    return vcf.Reader(open(GTEX_VCF, 'br'))


def prepare_annotation_panel(bed, gene):
    """
    Plot-ready payload for an isoform-structure panel: the exon rows for one gene
    plus the transcript order used in the original call.

    plot_isoform_annotations() reads `chrom` off the first row of whatever frame
    it is handed, but only ever uses it to build 'chrom:pos' strings that are
    immediately split back to the integer position, so restricting the frame to a
    single gene here is safe.

    Source: analysis/Figure_4.ipynb cells 123, 140, 141.
    """
    gene_bed = bed.loc[bed.gene_id == gene].copy()
    return {'bed': gene_bed, 'gene': gene,
            'iso_order': list(gene_bed.transcript_id.unique())}


# ================================================================
# Example loci — genotype coverage metaplots
# ================================================================

def prepare_coverage_panel(gene, tissue, var_id, gtex_vcf, isoform_bed,
                           transcripts, mean_counts=False, log_counts=True):
    """
    Build the genotype-stratified coverage payload for one locus.

    Source: analysis/Figure_4.ipynb cells 108-110 (ITGA1) and 136-137 (ZSWIM7).
    """
    GeneticsPlots = _genetics_plots()
    coverage_dir = GeneticsPlots.prepare_QTL_metaplot(
        gene, tissue, var_id, gtex_vcf, mean_counts=mean_counts, log_counts=log_counts)
    return {'coverage_dir': coverage_dir, 'gene': gene,
            'iso': isoform_bed.loc[isoform_bed.transcript_id.isin(transcripts)].copy()}


# ================================================================
# Example loci — genotype boxplots
# ================================================================

def get_boxplot_input(EL, var_id, gtex_vcf, isoform):
    """
    Split normalised isoform loadings by genotype at `var_id`.

    Source: analysis/Figure_4.ipynb cell 130.
    """
    GeneticsPlots = _genetics_plots()

    rec_ = GeneticsPlots.get_snp_record(gtex_vcf, var_id)
    ref = rec_.REF
    alt = str(rec_.ALT[0])

    hom_refs = pd.Index([x.sample for x in rec_.get_hom_refs()]).intersection(EL.columns)
    hets     = pd.Index([x.sample for x in rec_.get_hets()]).intersection(EL.columns)
    hom_alts = pd.Index([x.sample for x in rec_.get_hom_alts()]).intersection(EL.columns)

    boxplot_df = pd.DataFrame()
    aa_norm = list(EL.loc[isoform, hom_refs.intersection(EL.columns)])
    ag_norm = list(EL.loc[isoform, hets.intersection(EL.columns)])
    gg_norm = list(EL.loc[isoform, hom_alts.intersection(EL.columns)])

    norm_list = aa_norm + ag_norm + gg_norm

    genotype  = [f'{ref}/{ref}'] * len(hom_refs.intersection(EL.columns))
    genotype += [f'{ref}/{alt}'] * len(hets.intersection(EL.columns))
    genotype += [f'{alt}/{alt}'] * len(hom_alts.intersection(EL.columns))

    boxplot_df['genotype'] = genotype
    boxplot_df[isoform] = norm_list

    return boxplot_df


def prepare_nominal_stat(tissue, var_id, isoform):
    """
    The factor-QTL nominal-pass statistic for one isoform-variant pair.

    Reads the same ForGWASColoc.NominalPass tabix the coloc scatters use. Its
    `phe_id` is '{gene}.{isoform}:{gwas_locus}', so one isoform-variant pair has
    a record per GWAS locus tested in that window -- they carry the same
    association, and this checks that before returning it rather than taking the
    first one blindly.

    Returns beta (slope), se, pval, and how many records agreed.
    """
    import tabix
    chrom, pos = var_id.split('_')[0], int(var_id.split('_')[1])
    tb = tabix.open(QTL_NOMINAL.format(tissue=tissue))
    hits = [r for r in tb.query(chrom, pos - 1, pos + 1)
            if r[7] == var_id and r[0].split(':')[0] == isoform]
    if not hits:
        raise LookupError(f'no nominal record for {isoform} / {var_id} in {tissue}')
    stats = {(r[11], r[13], r[14]) for r in hits}
    if len(stats) > 1:
        print(f'  [nominal] {isoform} {var_id}: {len(stats)} differing records, '
              f'using the first')
    pval, slope, se = hits[0][11], hits[0][13], hits[0][14]
    return {'beta': float(slope), 'se': float(se), 'pval': float(pval),
            'n_records': len(hits), 'isoform': isoform, 'var_id': var_id,
            'tissue': tissue}


def prepare_boxplot_panel(tissue, var_id, isoform, gtex_vcf):
    """Load the tissue qqnorm bed and split isoform loadings by genotype (cells 106, 128, 138, 142)."""
    qqnorm = pd.read_csv(QTL_QQNORM.format(tissue=tissue), sep='\t', index_col=3)
    return get_boxplot_input(qqnorm, var_id, gtex_vcf, isoform)


# ================================================================
# Example loci — GWAS / QTL colocalisation scatters
# ================================================================

def prepare_coloc_panel(gwas_trait, chrom, gwas_pos, tissue, qtl_key, gwas_key,
                        window=500000, swap_alleles=False):
    """
    Merge GWAS and factor-QTL nominal p-values in a window around the lead SNP.

    Source: analysis/Figure_4.ipynb cells 111-118 (ITGA1) and 145-148 (ZSWIM7).

    `swap_alleles` reproduces the differing variant-ID reconstruction in the two
    original cells: the ITGA1 cell writes the GWAS allele fields in the order
    (field 3, field 2), the ZSWIM7 cell writes them as (field 2, field 3).
    """
    import tabix

    # --- GWAS summary statistics ---
    tb_ = tabix.open(GWAS_SUMSTATS.format(trait=gwas_trait))
    records_ = tb_.query(chrom, gwas_pos - window, gwas_pos + window)

    pos_list, pval_list, var_list = [], [], []
    for record in records_:
        pos = int(record[1])
        pval = float(record[3])
        pos_list.append(pos)
        pval_list.append(pval)

        var_parts = record[4].split(':')
        if swap_alleles:
            var_ = '_'.join([var_parts[0], str(pos), var_parts[3].upper(), var_parts[2].upper()])
        else:
            var_ = '_'.join([var_parts[0], str(pos), var_parts[2].upper(), var_parts[3].upper()])
        var_list.append(var_ + '_b38')

    gwas_df = pd.DataFrame()
    gwas_df['pos'] = pos_list
    gwas_df['pval'] = pval_list
    gwas_df['var_id'] = var_list

    # --- factor-QTL nominal pass ---
    tb_ = tabix.open(QTL_NOMINAL.format(tissue=tissue))
    records_ = tb_.query(chrom, gwas_pos - window, gwas_pos + window)

    pos_list, pval_list, beta_list, se_list, var_list = [], [], [], [], []
    for record in records_:
        if (qtl_key in record[0]) and (gwas_key in record[0]):
            pos_list.append(int(record[10]))
            pval_list.append(float(record[11]))
            beta_list.append(float(record[13]))
            se_list.append(float(record[14]))
            var_list.append(record[7])

    qtl_df = pd.DataFrame()
    qtl_df['pos'] = pos_list
    qtl_df['pval'] = pval_list
    qtl_df['beta'] = beta_list
    qtl_df['se'] = se_list
    qtl_df['var_id'] = var_list

    return gwas_df.merge(qtl_df, left_on='var_id', right_on='var_id')


# ================================================================
# Example loci — torino factor tracks
# ================================================================

def load_ebpmf_gene(gene, species='Human'):
    """
    Read the per-gene torino factorisation RDS through rpy2.

    Source: analysis/Figure_4.ipynb cell 170.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    readRDS = ro.r['readRDS']
    df = readRDS(RDS_PATH.format(gene=gene) if species == 'Human'
                 else f'../code/results/torino/factorization/{species}/{gene}.rds')
    with (ro.default_converter + pandas2ri.converter).context():
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(df)
    return pd_from_r_df


def prepare_factors_panel(gene, isoform_bed, transcripts, factor_list, xlim, w=1000):
    """
    Extract the plot-ready pieces of the torino_K10 factorisation for one gene.

    The RDS handle itself is an rpy2 object and cannot be pickled, so only the
    smoothed factor matrix and coordinate vectors are kept.

    Source: analysis/Figure_4.ipynb cells 170-172.
    """
    rds = load_ebpmf_gene(gene)
    rds_k10 = rds['torino_K10']
    return {
        'EF_smooth':   np.asarray(rds_k10['EF_smooth']),
        'coords':      list(rds_k10['coords']),
        'coords_all':  list(rds['coords_all']),
        'gene':        gene,
        'iso':         isoform_bed.loc[isoform_bed.transcript_id.isin(transcripts)].copy(),
        'factor_list': factor_list,
        'xlim':        xlim,
        'w':           w,
    }


# ================================================================
# Orchestrator
# ================================================================

def run_all(data_dir='../notebooks/Figure4/figure_data'):
    """
    Run the full Figure 4 data pipeline, pickle every plot-ready variable into
    `data_dir`, and return them as a dict.
    """
    os.makedirs(data_dir, exist_ok=True)

    def _dump(name, obj):
        with open(os.path.join(data_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)

    # --- QTLs --------------------------------------------------------------
    print('[Figure4] factor-QTL counts per tissue ...')
    qtl_barplot = compute_qtl_barplot()
    _dump('qtl_barplot', qtl_barplot)

    # --- upsetplot_coloc ---------------------------------------------------
    print('[Figure4] hyprcoloc results and isoform annotation ...')
    annot = load_isoform_annotation()
    hyprcoloc_results = load_hyprcoloc_results()

    print('[Figure4] RNA-processing classes of colocalised loci ...')
    coloc_indicators = build_coloc_indicators(hyprcoloc_results, annot)
    upset_data = build_upset_data(coloc_indicators)
    _dump('upset_data', upset_data)

    # --- example loci ------------------------------------------------------
    print('[Figure4] isoform / GENCODE annotations ...')
    isoform_bed, gtf_bed = load_beds()

    itga1_gencode  = prepare_annotation_panel(gtf_bed, ITGA1_GENE_ID)
    zswim7_gencode = prepare_annotation_panel(gtf_bed, ZSWIM7_GENE_ID)
    zswim7_torino  = prepare_annotation_panel(isoform_bed, 'ZSWIM7')
    _dump('itga1_gencode', itga1_gencode)
    _dump('zswim7_gencode', zswim7_gencode)
    _dump('zswim7_torino', zswim7_torino)

    print('[Figure4] torino factor tracks ...')
    itga1_factors = prepare_factors_panel(
        'ITGA1', isoform_bed, ['isoform_4', 'isoform_5'],
        factor_list=['factor3', 'factor4'], xlim=ITGA1_XLIM)
    zswim7_factors = prepare_factors_panel(
        'ZSWIM7', isoform_bed, ['isoform_4', 'isoform_5'],
        factor_list=['factor3', 'factor8', 'factor9'], xlim=ZSWIM7_XLIM, w=0)
    _dump('itga1_factors', itga1_factors)
    _dump('zswim7_factors', zswim7_factors)

    print('[Figure4] genotype coverage metaplots ...')
    gtex_vcf = open_gtex_vcf()
    itga1_coverage = prepare_coverage_panel(
        'ITGA1', ITGA1_TISSUE, ITGA1_VAR, gtex_vcf, isoform_bed,
        ['isoform_4', 'isoform_5', 'isoform_7'], mean_counts=True, log_counts=False)
    itga1_coverage_alt = prepare_coverage_panel(
        'ITGA1', ITGA1_TISSUE, ITGA1_VAR, gtex_vcf, isoform_bed,
        ['isoform_4', 'isoform_5', 'isoform_7'], log_counts=False)
    zswim7_coverage = prepare_coverage_panel(
        'ZSWIM7', ZSWIM7_TISSUE, ZSWIM7_VAR, gtex_vcf, isoform_bed,
        ['isoform_1', 'isoform_2', 'isoform_3'], mean_counts=True, log_counts=False)
    _dump('itga1_coverage', itga1_coverage)
    _dump('itga1_coverage_alt', itga1_coverage_alt)
    _dump('zswim7_coverage', zswim7_coverage)

    print('[Figure4] genotype boxplots ...')
    itga1_boxplot = prepare_boxplot_panel(
        ITGA1_TISSUE, ITGA1_VAR, 'ITGA1.isoform_4', gtex_vcf)
    zswim7_boxplot = prepare_boxplot_panel(
        ZSWIM7_TISSUE, ZSWIM7_VAR, 'ZSWIM7.isoform_2', gtex_vcf)
    _dump('itga1_boxplot', itga1_boxplot)
    _dump('zswim7_boxplot', zswim7_boxplot)

    print('[Figure4] QTL nominal-pass statistics for the boxplots ...')
    _dump('itga1_nominal', prepare_nominal_stat(
        ITGA1_TISSUE, ITGA1_VAR, 'ITGA1.isoform_4'))
    _dump('zswim7_nominal', prepare_nominal_stat(
        ZSWIM7_TISSUE, ZSWIM7_VAR, 'ZSWIM7.isoform_2'))

    print('[Figure4] GWAS / QTL colocalisation scatters ...')
    itga1_coloc = prepare_coloc_panel(
        'coronary_artery_disease', 'chr5', 52897294, ITGA1_TISSUE,
        qtl_key='ITGA1.isoform_4',
        gwas_key='chr5_52897294_N_N_coronary_artery_disease',
        swap_alleles=True)
    zswim7_coloc = prepare_coloc_panel(
        'parkinson_disease', 'chr17', 16107606, ZSWIM7_TISSUE,
        qtl_key='ZSWIM7.isoform_2',
        gwas_key='chr17_16107606_N_N_parkinson_disease',
        swap_alleles=False)
    _dump('itga1_coloc', itga1_coloc)
    _dump('zswim7_coloc', zswim7_coloc)

    return {
        'qtl_barplot':        qtl_barplot,
        'upset_data':         upset_data,
        'itga1_gencode':      itga1_gencode,
        'itga1_factors':      itga1_factors,
        'itga1_coverage':     itga1_coverage,
        'itga1_coverage_alt': itga1_coverage_alt,
        'itga1_boxplot':      itga1_boxplot,
        'itga1_coloc':        itga1_coloc,
        'zswim7_gencode':     zswim7_gencode,
        'zswim7_torino':      zswim7_torino,
        'zswim7_factors':     zswim7_factors,
        'zswim7_coverage':    zswim7_coverage,
        'zswim7_boxplot':     zswim7_boxplot,
        'zswim7_coloc':       zswim7_coloc,
    }
