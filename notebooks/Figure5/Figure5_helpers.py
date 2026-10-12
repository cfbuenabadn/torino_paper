"""
Figure5_helpers.py — data loading, processing and orchestration for Figure 5
(Alzheimer's disease).

Contains all data-loading and data-processing functions, plus a run_all() that
produces the final plot-ready variables and saves them as pickles in
figure_data/.

Must be run from the analysis/ directory (or with CWD set there), so that the
relative '/project2/yangili1/cfbuenabadn/torino_paper/code/...' paths resolve.

Heavy/specialised imports (rpy2, pybedtools, gseapy, tqdm, sklearn, and the
analysis/*.py modules that pull them in) are done *inside* the functions that
need them, so `from Figure5_helpers import run_all` works in a bare pandas
environment.

The figures come from seven exploratory notebooks; each function names its
source. Two output directories are covered:
    /project2/yangili1/cfbuenabadn/torino_paper/code/results/figures/Fig5_ad/
    /project2/yangili1/cfbuenabadn/torino_paper/code/results/figures/Fig5_ad/gao_k3/
"""

import gzip
import os
import pickle

import numpy as np
import pandas as pd

# ================================================================
# Paths
# ================================================================

ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
FIG5_DIR_DEFAULT = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure5'
GTEX_STM_data = '/project2/mstephens/cfbuenabadn/gtex-stm'
GTEX_STM = '/project2/mstephens/cfbuenabadn/gtex-stm/code'
TORINO_CODE = '/project2/yangili1/cfbuenabadn/torino_paper/code'

# --- gao (snmf5 / snmf3) universe, used by the Fig5_ad/ figures -------------
GAO_ISOFORMS = f'{GTEX_STM}/results/gao/isoforms/isoforms.bed.gz'
EL_SNMF3 = f'{GTEX_STM}/ebpmf_models/gao_models/tables/snmf3_EL.tab.gz'
EL_SNMF5 = f'{GTEX_STM}/ebpmf_models/gao_models/tables/snmf5_EL.tab.gz'
METADATA = f'{GTEX_STM}/gao_data/metadata/merged_metadata.tab.gz'
GENES_BED = f'{GTEX_STM_data}/data/all_genes.protein_coding.bed.gz'
GENES_DIR = f'{GTEX_STM_data}/data/genes.tab.gz'
GENCODE_EXONS = f'{GTEX_STM}/Annotations/gencode.v44.primary_assembly.exons.sorted.bed.gz'
COUNTS_DIR = f'{GTEX_STM}/gao_data/counts'
GMT = f'{GTEX_STM}/c5.all.v2023.1.Hs.symbols.gmt'

# --- gao_k3 universe, used by the gao_k3/ figures ---------------------------
K3_LR_OUTPUT = '/project2/yangili1/cfbuenabadn/torino_paper/code/results/gao_k3/tables/LR.output.tab.gz'
K3_IR_ONLY_PLOTS = f'{TORINO_CODE}/results/gao_k3/tables/AD_IR_only_plots.bed.gz'
# The relaxed bed (rules/AD.smk get_introns_for_ad_plot_IR_only_relaxed): a
# torino exon only has to fall within the body of the gene, not within the span
# of the APPRIS-principal transcript. 3,225 genes against the strict 2,940 --
# MRPL43 is the one of the 18 K=3 colocalisation genes it brings back.
K3_IR_ONLY_PLOTS_RELAXED = (f'{TORINO_CODE}/results/gao_k3/tables/'
                            'AD_IR_only_plots.relaxed.bed.gz')
# ... and the same bed restricted to features significant in the logistic
# regression at FDR <= 0.1. Without that cut the gene set is an annotation
# filter rather than a statistical one: 3,908 genes carry an IR-only factor,
# 1,082 of them reach FDR 0.1.
K3_IR_ONLY_PLOTS_RELAXED_SIG = (f'{TORINO_CODE}/results/gao_k3/tables/'
                                'AD_IR_only_plots.relaxed.fdr0p1.bed.gz')
K3_ISOFORMS = f'{TORINO_CODE}/results/gao_k3/isoforms/isoforms.bed.gz'
# per-tissue K=3 genotype boxplots, quantile-normalised within each tissue
K3_BOX_RDS = f'{TORINO_CODE}/k3_figure_202609-v2/boxplot_rds_qqnorm'

# The gao_k3 genetics tables behind TWAS_summary_K3. These are the K=3 rerun of
# the TWAS / MR / colocalisation analysis, in the newer file formats: a TWAS
# `.twas.summary.tsv.gz` rather than a `.twas.bed.gz`, an MR
# `.mr_result.summary.tsv.gz`, and a colocboost summary rather than the
# `_colocalization.txt` table. The column names the plot needs are unchanged.
K3_GENETICS = f'{TORINO_CODE}/results/gao_k3/sQTL_factor_202609'
K3_TWAS_FILE = f'{K3_GENETICS}/twas/ROSMAP_sQTL_factor.twas.summary.tsv.gz'
K3_MR_FILE = f'{K3_GENETICS}/MR/ROSMAP_sQTL_factor.mr_result.summary.tsv.gz'
K3_COLOC_FILE = (f'{TORINO_CODE}/results/gao_k3/genetics/'
                 'AD_UofCsQTL_colocboost_summary.tsv')

# --- AD gene lists / genome-wide summary tables -----------------------------
COLOC_GENES = '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/AD_data/AD_genes_FunGen_AD_GVC_xQTL_20250325.tsv'
RU_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/code/genome_wide'
COLOC_FILE = f'{RU_DIR}/ROSMAP_sQTL_factor.GenomeWide_colocalization.txt.gz'
TWAS_FILE = f'{RU_DIR}/ROSMAP_sQTL_factor.GenomeWide.twas.bed.gz'
MR_FILE = f'{RU_DIR}/ROSMAP_sQTL_factor.GenomeWide.mr.bed.gz'

# The AD-gene subset. TWAS_summary is built from these, not from the genome-wide
# tables above: Figure5_summary_plots rebuilds coloc_df / df_significant / mr_ad
# from them in cells 32, 34 and 39, after the genome-wide load in cell 2.
AD_GENES_DIR = ('/project2/yangili1/cfbuenabadn/torino_paper/code/results/'
                '202509_UofC_splicing_factor_analtsis_ADgenes')
COLOC_AD_FILE = f'{AD_GENES_DIR}/ROSMAP_sQTL_factor.ADgenes_colocalization.txt.gz'
TWAS_AD_FILE = f'{AD_GENES_DIR}/ROSMAP_sQTL_factor.ADgenes.twas.bed.gz'
MR_AD_FILE = f'{AD_GENES_DIR}/ROSMAP_sQTL_factor.ADgenes.mr.bed.gz'

# --- per-gene RDS panels (Figure5_Genetics_Plots.ipynb) ---------------------
COV_RDS = f'{TORINO_CODE}/ad_genetics/UofC_plots/coverage_matrices_rds'
COV_RDS_OLD = f'{TORINO_CODE}/UofC_plots/coverage_matrices_rds'
BOX_RDS = f'{TORINO_CODE}/ad_genetics/UofC_plots/boxplot_rds'
GAO_RDS_ALL = f'{GTEX_STM}/ebpmf_models/gao_models/RDS_all'



# ================================================================
# RDS readers (rpy2)
# ================================================================

def _readRDS():
    import rpy2.robjects as ro
    return ro.r['readRDS']


def read_dataframe_rds(path):
    """Read an R data.frame into pandas. Figure5_Genetics_Plots.ipynb (cell 6)."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    df = _readRDS()(path)
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(df)


def rds_to_coverage_dir(coverage_rds):
    """
    Convert a coverage_matrices RDS into the dict GeneticsPlots expects.

    Source: Figure5_Genetics_Plots.ipynb (cell 8).
    """
    coords = list(coverage_rds[5])
    return {'var': coverage_rds[0][0],
            'var_pos': coverage_rds[1][0],
            'first_position': coords[0],
            'ref_allele': coverage_rds[3][0],
            'alt_allele': coverage_rds[4][0],
            'coords': coords,
            'ref_coverage': list(coverage_rds[6]),
            'het_coverage': list(coverage_rds[7]),
            'alt_coverage': list(coverage_rds[8])}


def load_gao_isoform_bed(path=GAO_ISOFORMS):
    """The gao isoform bed used by every gene example panel (cell 11 onwards)."""
    iso = pd.read_csv(path, sep='\t',
                      names=['chrom', 'start', 'end', 'gene_id', 'isoform',
                             'strand', 'factors', 'exons', 'notes'])
    iso['transcript_id'] = iso.gene_id + '.' + iso.isoform
    return iso


# ================================================================
# Gene example panels (Figure5_Genetics_Plots.ipynb)
# ================================================================
#
# Every panel is one of three shapes:
#   coverage : readRDS(coverage_matrices) -> rds_to_coverage_dir -> plot
#   boxplot  : read_dataframe_rds(boxplot) -> seaborn boxplot + stripplot
#   factors  : plot_factors_with_annot(iso, gene, ...)
#
# The variant id, tissue and RDS root differ per panel, so they are tabulated
# rather than repeated. `rds_root` distinguishes the two coverage directories
# the notebook draws from.

COVERAGE_PANELS = {
    # name                              gene              variant                        tissue  root
    'ENSG00000103550.AC.coverage':    ('ENSG00000103550', 'chr16:19708463:CTCA:C', 'AC',  COV_RDS_OLD),
    'ENSG00000122873.AC.coverage':    ('ENSG00000122873', 'chr10:58286894:G:A',    'AC',  COV_RDS_OLD),
}

BOXPLOT_PANELS = {
    # name: (gene, variant, tissue, factor, style, ylabel, xlabel)
    # style None -> the red/orange/gold palette; otherwise a single outline colour.
    # Labels are transcribed from the producing cell.
    'ENSG00000103550.AC.boxplot_tab': ('ENSG00000103550', 'chr16:19708463:CTCA:C', 'AC', 'factor_2', 'tab:orange',
                                       'KNOP1 isoform 2\nnormalized loadings\n(AC)', 'SNP chr16:19708463'),
    'ENSG00000122873.AC.boxplot':     ('ENSG00000122873', 'chr10:58286894:G:A', 'AC', 'factor_4', 'tab:red',
                                       'CISD1 isoform 2\nnormalized loadings\n(AC)', 'SNP chr10:58286894'),
}

FACTOR_PANELS = {
    # name                            gene              factor_list             colores
    'ENSG00000103550.factors': ('ENSG00000103550', ['factor1', 'factor2'], None),
    'ENSG00000122873.factors': ('ENSG00000122873', ['factor4', 'factor5'], ['tab:red', 'tab:purple']),
}


def prepare_coverage_panel(gene, variant, tissue, rds_root, iso):
    """Coverage payload for one gene example panel."""
    rds = _readRDS()(f'{rds_root}/{gene}.{variant}.{tissue}.coverage_matrices.rds')
    return {'coverage_dir': rds_to_coverage_dir(rds), 'gene': gene, 'iso': iso}


def prepare_boxplot_panel(gene, variant, tissue):
    """Genotype boxplot payload for one gene example panel."""
    return read_dataframe_rds(f'{BOX_RDS}/{gene}.{variant}.{tissue}.boxplot.rds')


def load_ebpmf_gene(gene):
    """
    Read the per-gene snmf factorisation RDS.

    Source: Figure5_Genetics_Plots.ipynb (cell 130). Note this reads
    `gao_models/RDS_all`, a different root from the torino factorisations that
    Figure 4 uses.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    df = ro.r['readRDS'](f'{GAO_RDS_ALL}/{gene}.rds')
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(df)


def prepare_factor_panel(gene, iso, factor_list, k=5):
    """
    Plot-ready payload for a factor-track panel.

    The rpy2 handle cannot be pickled, so only the smoothed factor matrix and
    the two coordinate vectors are kept. Source: Figure5_Genetics_Plots.ipynb
    (cell 130) — `filtered_coords` indexes EF_smooth while `coords` sets the
    default x limits; they are different vectors and both are needed.

    `k` picks the model inside the RDS. Pass the matching isoform bed with it:
    gao for k=5, gao_k3 for k=3. The two beds give the same xlim but different
    gene extents, and the isoform mirroring is anchored to the factorisation,
    so a mismatch silently shifts the exon models.
    """
    rds = load_ebpmf_gene(gene)
    ebpmf = rds[f'snmf_{k}']['ebpmf']
    iso_gene = iso.loc[iso.gene_id == gene]
    return {'EF_smooth': np.asarray(ebpmf['EF_smooth']),
            'coords': list(rds['filtered_coords']),
            'xlim_coords': [int(x.split(':')[1]) for x in list(rds['coords'])],
            'gene': gene, 'strand': iso_gene.strand.iloc[0],
            'iso': iso_gene.copy(), 'factor_list': factor_list, 'k': k}


def prepare_by_ad_pooled_boxplot(gene, variant, tissues=('AC', 'PCC', 'DLPFC'),
                                 groups=('AD', 'nonAD'), root=None):
    """
    The pooled boxplot frame split by AD status: the *_by_AD trees keep the same
    filenames under an extra AD/ and nonAD/ level. Adds `tissue` and `group`.
    """
    root = root or K3_BOX_RDS + '_by_AD'
    frames = []
    for group in groups:
        for tissue in tissues:
            path = f'{root}/{group}/{gene}.{variant}.{tissue}.boxplot.rds'
            if not os.path.exists(path):
                print(f'  [by-AD] no {group}/{tissue} for {gene} {variant}, skipped')
                continue
            frames.append(read_dataframe_rds(path).assign(tissue=tissue, group=group))
    if not frames:
        raise FileNotFoundError(f'no by-AD boxplot RDS for {gene} {variant}')
    out = pd.concat(frames, ignore_index=True)
    out.attrs['root'] = os.path.basename(root)
    return out


def prepare_pooled_boxplot(gene, variant, tissues=('AC', 'PCC', 'DLPFC'),
                           root=None):
    """
    One genotype-boxplot frame per tissue, concatenated with a `tissue` column.

    `root` is the boxplot RDS directory: BOX_RDS for the K=5 loadings,
    '{TORINO_CODE}/results/k3_figure_202609-v2/boxplot_rds_qqnorm' for K=3.
    Both are quantile-normalised within each tissue, so the values are on a
    common scale and pooling them is meaningful; the unnormalized trees are
    not, and pooling those would mostly measure library depth.
    """
    root = root or BOX_RDS
    frames = []
    for tissue in tissues:
        path = f'{root}/{gene}.{variant}.{tissue}.boxplot.rds'
        if not os.path.exists(path):
            print(f'  [pooled] no {tissue} file for {gene} {variant}, skipped')
            continue
        frames.append(read_dataframe_rds(path).assign(tissue=tissue))
    if not frames:
        raise FileNotFoundError(f'no boxplot RDS for {gene} {variant} under {root}')
    return pd.concat(frames, ignore_index=True)


def prepare_publication_extras(data_dir='../notebooks/Figure5/figure_data', reuse=True):
    """
    The one panel pair the assembled Figure 5 needs that Figure5.ipynb does not.

    Figure5.ipynb draws the KNOP1 example in AC (the requested figure list), but
    the assembled figure labels that panel PCC, so the publication layout needs
    the PCC coverage + boxplot for the same variant. Both are plain RDS reads,
    cached as knop1_pcc.pkl.
    """
    cache = os.path.join(data_dir, 'knop1_pcc.pkl')
    if reuse and os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    gene, variant = 'ENSG00000103550', 'chr16:19708463:CTCA:C'
    iso = load_gao_isoform_bed()
    cov = prepare_coverage_panel(gene, variant, 'PCC', COV_RDS_OLD, iso)
    # keep only this gene's isoform rows; the full bed is 460k rows / 73 MB and
    # every consumer filters it to the gene anyway
    cov['iso'] = cov['iso'].loc[cov['iso'].gene_id == gene].copy()
    out = {'coverage': cov, 'boxplot': prepare_boxplot_panel(gene, variant, 'PCC')}
    os.makedirs(data_dir, exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(out, f)
    return out


# ================================================================
# gao_k3 tables and beta summaries
# ================================================================

def load_k3_tables():
    """
    LR output and the IR / no-IR gene partitions.

    Source: AD_K3_RetainedIntrons.ipynb (cells 1, 3).
    """
    LR_output = pd.read_csv(K3_LR_OUTPUT, sep='\t')
    by_gene = LR_output.groupby('gene_id').IR_factor.sum()
    gene_no_IR = by_gene.loc[by_gene == 0].index
    gene_w_IR = by_gene.loc[by_gene.isin([1, 2])].index
    return LR_output, gene_w_IR, gene_no_IR


def compute_k3_beta_stats(LR_output, gene_w_IR, gene_no_IR, fdr=0.05):
    """
    Counts and positive-beta ratios behind beta_counts_allIR /
    beta_percentage_allIR.

    Source: AD_K3_RetainedIntrons.ipynb (cells 9, 11, 12, 30, 57, 58). `fdr` is
    the significance cut applied to the logistic regression; the panel's bar
    labels are derived from the ratios, so they follow it.
    """
    sig = LR_output.pval_fdr <= fdr
    w_ir = LR_output.gene_id.isin(gene_w_IR)
    no_ir = LR_output.gene_id.isin(gene_no_IR)
    ir = LR_output.IR_factor.astype(bool)

    total_ir = LR_output.loc[w_ir & sig & ir].shape[0]
    ir_pos = int((LR_output.loc[w_ir & sig & ir].beta > 0).sum())
    total_si = LR_output.loc[w_ir & sig & (~ir)].shape[0]
    si_pos = int((LR_output.loc[w_ir & sig & (~ir)].beta > 0).sum())
    total_factors_noIR = LR_output.loc[no_ir & sig].shape[0]

    return {'fdr': fdr,
            'total_ir': total_ir, 'total_si': total_si,
            'total_factors_noIR': total_factors_noIR,
            'ir_pos_ratio': ir_pos / total_ir,
            'si_pos_ratio': si_pos / total_si,
            'total_ratio': float((LR_output.loc[sig & no_ir].beta > 0).mean())}


def load_ad_gene_names(LR_output, genes_df):
    """ad_gene_names for the GSEA panel — AD_K3_RetainedIntrons.ipynb (cell 72)."""
    ad_genes = LR_output.loc[LR_output.pval_fdr <= 0.05].gene_id.unique()
    return list(genes_df.loc[genes_df.gene_id.isin(ad_genes)].gene_name)


def run_gsea(ad_gene_names):
    """
    Gene-set enrichment of the significant AD genes.

    Source: AD_K3_RetainedIntrons.ipynb (cell 74) / Fig5_AD_figures.ipynb (75).
    Returns the `res2d` table; the dotplot itself is drawn in the plot helpers.
    """
    import warnings
    import gseapy as gp
    warnings.filterwarnings("ignore")
    enr = gp.enrich(gene_list=ad_gene_names, gene_sets=[GMT], background=None,
                    outdir=None, verbose=True)
    return enr.res2d


def prepare_knop1_braak(EL, metadata_all):
    """
    Braak-stage boxplot table for KNOP1 factor 2.

    Source: AD_K3_RetainedIntrons.ipynb (cell 90).
    """
    return metadata_all.loc[
        (metadata_all.condition.isin(['AD', 'no condition'])) &
        (metadata_all.specimenID.isin(EL.columns)),
        ['specimenID', 'braaksc', 'condition', 'tissue_id']
    ].merge(EL.loc['ENSG00000103550.factor_2'], left_on='specimenID',
            right_index=True)


# ================================================================
# Intron-retention gene-body heatmaps — shared pipeline
# ================================================================
#
# Two separate data universes feed the IR figures and must not be mixed:
#   gao_k3/  : AD_IR_only_plots.bed.gz   (AD_K3_RetainedIntrons.ipynb)
#   Fig5_ad/ : AD_IR_plots.bed.gz        (Fig5_AD_figures.ipynb, cell 77)

def load_gencode():
    """GENCODE v44 exons — AD_K3_RetainedIntrons.ipynb (cell 76)."""
    return pd.read_csv(GENCODE_EXONS, sep='\t',
                       names=['chrom', 'start', 'end', 'gene', 'transcript',
                              'strand', 'exon', 'score', 'gencode_annot',
                              'ensembl', 'mane', 'appris', 'annotation'])


def get_gencode_bed(gencode, gene):
    """
    Merged exon bed of the APPRIS-principal (else MANE Select) transcript.

    Source: AD_K3_RetainedIntrons.ipynb (cell 78).
    """
    from pybedtools import BedTool
    gencode_sub = gencode.loc[gencode.gene == gene]
    appris_mask = gencode_sub.appris.apply(
        lambda x: isinstance(x, str) and 'appris_principal_' in x)
    try:
        if appris_mask.any():
            gencode_transcript = gencode_sub.loc[appris_mask]
        else:
            mane_mask = gencode_sub.mane == 'MANE_Select'
            if mane_mask.any():
                gencode_transcript = gencode_sub.loc[mane_mask]
            else:
                return None, None
        gencode_bed = BedTool.from_dataframe(gencode_transcript).merge()
        gencode_full_gene = pd.DataFrame()
        gencode_full_gene['chrom'] = [gencode_transcript.iloc[0, 0]]
        gencode_full_gene['start'] = [gencode_transcript.start.min()]
        gencode_full_gene['end'] = [gencode_transcript.end.max()]
        gencode_full_gene = BedTool.from_dataframe(gencode_full_gene)
    except Exception as e:
        # The source swallows this with a bare except. Keep skipping the gene,
        # but surface the reason once: if `bedtools` is missing from PATH every
        # gene fails here and the heatmaps come out silently empty.
        if not getattr(get_gencode_bed, '_warned', False):
            print(f'[Figure5] get_gencode_bed failed for {gene}: '
                  f'{type(e).__name__}: {e}. Is bedtools on PATH?')
            get_gencode_bed._warned = True
        return None, None
    return gencode_bed, gencode_full_gene


def get_intron_mean_position(bed, start, end):
    """
    Length-weighted median position of a gene's introns along the gene body.

    Source: AD_K3_RetainedIntrons.ipynb (cell 78).
    """
    start_i = (bed.start - start) / (end - start)
    end_i = (bed.end - start) / (end - start)
    intervals = list(zip(start_i, end_i))
    lengths = [(e - s) for s, e in intervals]
    total_length = sum(lengths)
    half_length = total_length / 2.0
    cumulative_length = 0.0
    for (s, e), length in zip(intervals, lengths):
        cumulative_length += length
        if cumulative_length >= half_length:
            return s + (half_length - (cumulative_length - length))
    return None


def build_ir_layout(introns_bed, gencode):
    """
    Row order for the heatmaps plus the per-gene merged exon beds.

    Source: AD_K3_RetainedIntrons.ipynb (cell 79). The originals re-ran the
    pybedtools merge inside every figure loop; here it is done once and the
    resulting frames are reused by all ~50 IR figures.
    """
    from tqdm import tqdm
    gene_list, position, strand_list = [], [], []
    gencode_beds = {}
    for gene in tqdm(introns_bed.gene.unique(), position=0, leave=True):
        strand = gencode.loc[gencode.gene == gene].strand.iloc[0]
        gencode_bed, _ = get_gencode_bed(gencode, gene)
        if gencode_bed is None:
            continue
        gencode_bed = gencode_bed.to_dataframe()
        start_ = gencode_bed.end.iloc[0]
        end_ = gencode_bed.start.iloc[-1]
        pos = get_intron_mean_position(introns_bed.loc[introns_bed.gene == gene],
                                       start_, end_)
        if strand == '-':
            pos = 1 - pos
        position.append(pos)
        gene_list.append(gene)
        strand_list.append(strand)
        gencode_beds[gene] = gencode_bed

    order_df = pd.DataFrame({'gene': gene_list, 'position': position,
                             'strand': strand_list})
    if order_df.empty:
        raise RuntimeError(
            'No genes survived the gencode lookup, so every IR heatmap would be '
            'blank. The usual cause is `bedtools` missing from PATH, which makes '
            'get_gencode_bed fail for every gene.')
    return order_df, gencode_beds


def load_ad_gene_sets():
    """
    The AD gene lists used to colour the IR heatmap labels.

    Source: AD_K3_RetainedIntrons.ipynb (cells 81, 82, 84). `iso_coloc_genes`
    comes from the **genome-wide** colocalisation table, which is the one live
    in that notebook when the labels are built — distinct from the AD-gene
    subset that TWAS_summary uses.
    """
    coloc_genes = pd.read_csv(COLOC_GENES, sep='\t')
    coloc = pd.read_csv(COLOC_FILE, sep='\t')
    coloc_df = build_coloc_df(coloc, build_gene_bed(load_gao_isoform_bed()))
    return {'coloc_genes': coloc_genes,
            'iso_coloc_genes': coloc_df.gene.unique()}


K3_COLOC_SUMMARY = (f'{TORINO_CODE}/results/gao_k3/genetics/'
                    'AD_UofCsQTL_colocboost_summary.tsv')


def build_k3_coloc_table(coloc=None, LR_output=None):
    """
    The gao_k3 colocboost summary with the per-locus feature flags.

    Source: analysis/Fig5_numbers.ipynb cells 26-27 (the cell marked
    `### THIS TABLE`), reproduced verbatim — including `if has_IR and is_AD`,
    where `has_IR` is the accumulating *list* and so is truthy from the second
    row on. That makes `ir_ad_factor` effectively "an AD-significant factor
    colocalises here", not "an IR factor that is also AD-significant". It is
    kept as-is so the 18 genes match the notebook the selection came from.

    Adds: factor_QTL, eQTL, ir_factor, ad_factor, ir_ad_factor.
    """
    coloc = coloc if coloc is not None else pd.read_csv(K3_COLOC_SUMMARY, sep='\t')
    if LR_output is None:
        LR_output = pd.read_csv(K3_LR_OUTPUT, sep='\t')
    gene_factors = set(LR_output.gene_factor)
    by_factor = LR_output.drop_duplicates('gene_factor').set_index('gene_factor')

    has_sQTL, has_eQTL, has_IR, has_AD, IR_is_AD = [], [], [], [], []
    for _, clc in coloc.iterrows():
        outcomes = clc.colocalized_outcomes
        if 'sQTL_' in outcomes:
            has_sQTL.append(True)
            is_IR = AD_sig = IR_AD = False
            for e in outcomes.split('; '):
                if 'sQTL_' not in e:
                    continue
                iso = e.split('sQTL_')[1]
                if iso in gene_factors:
                    r = by_factor.loc[iso]
                    is_AD = bool(r.pval_fdr < 0.05)
                    if bool(r.IR_factor):
                        is_IR = True
                    if is_AD:
                        AD_sig = True
                    if has_IR and is_AD:      # verbatim: list, not the flag
                        IR_AD = True
            has_IR.append(is_IR); has_AD.append(AD_sig); IR_is_AD.append(IR_AD)
        else:
            has_sQTL.append(False); has_IR.append(False)
            has_AD.append(False); IR_is_AD.append(False)
        has_eQTL.append('eQTL_' in outcomes)

    out = coloc.copy()
    out['factor_QTL'] = has_sQTL
    out['eQTL'] = has_eQTL
    out['ir_factor'] = has_IR
    out['ad_factor'] = has_AD
    out['ir_ad_factor'] = IR_is_AD
    return out


def build_gene_label_k3(order_df=None, coloc_table=None, annot=None):
    """
    gene -> label colour for panel C, from the **K=3** colocalisations.

    Replaces build_gene_label(), which drew its gene set from the genome-wide
    K=5 colocalisation table while the heatmap itself is K=3. Here:
      labelled : gene_id with ir_ad_factor True in build_k3_coloc_table() -- 18
      red      : of those, the ones with eQTL True -- 13
      black    : the rest
    `order_df` restricts to the genes actually in the heatmap (all 18 are).
    """
    tbl = coloc_table if coloc_table is not None else build_k3_coloc_table()
    sel = tbl.loc[tbl.ir_ad_factor]
    red = set(sel.loc[sel.eQTL].gene_id)
    genes = list(dict.fromkeys(sel.gene_id))
    if order_df is not None:
        in_panel = set(order_df.gene)
        missing = [g for g in genes if g not in in_panel]
        if missing:
            print(f'  [panel C] {len(missing)} labelled gene(s) absent from the '
                  f'heatmap: {missing}')
        genes = [g for g in genes if g in in_panel]
    return {g: ('tab:red' if g in red else 'black') for g in genes}


def load_k3_gene_symbols(annot=None):
    """
    gene_id -> gene symbol, from the gao_k3 isoform annotation merged with
    genes.tab.gz (Fig5_numbers.ipynb cell 3). Panel C labels Ensembl ids
    otherwise.
    """
    if annot is None:
        annot = pd.read_csv(f'{TORINO_CODE}/results/gao_k3/isoforms/'
                            'annotated.isoforms.tab.gz', sep='\t')
        annot['gene_id'] = annot.transcript.apply(lambda x: x.split('.')[0])
        gd = pd.read_csv(GENES_DIR, sep='\t', names=['gene', 'gene_id', 'loc'])
        gd['gene_id'] = gd.gene_id.apply(lambda x: x.split('.')[0])
        annot = annot.merge(gd, on='gene_id')
    return dict(zip(annot.gene_id, annot.gene))


def build_gene_label(order_df, gene_sets):
    """
    gene -> label colour, or absent for unlabelled genes.

    Labels only iso_coloc genes: red if the gene is also in the FunGen coloc
    list, else black. Source: AD_K3_RetainedIntrons.ipynb (cells 113/114/119).
    """
    iso_coloc = set(gene_sets['iso_coloc_genes'])
    coloc = set(gene_sets['coloc_genes'].gene_id)
    return {gene: ('tab:red' if gene in coloc else 'black')
            for gene in order_df.gene if gene in iso_coloc}


def prepare_ir_panel(introns_bed_path, gencode, genes_df, gene_sets):
    """Bundle everything plot_ir_heatmap() needs for one data universe."""
    from Figure5_plot_helpers import place_labels
    introns_bed = pd.read_csv(introns_bed_path, sep='\t')
    order_df, gencode_beds = build_ir_layout(introns_bed, gencode)
    gene_label = build_gene_label(order_df, gene_sets)
    gene_names = dict(zip(genes_df.gene_id, genes_df.gene_name))
    return {'introns_bed': introns_bed, 'order_df': order_df,
            'gencode_beds': gencode_beds, 'gene_label': gene_label,
            'gene_names': gene_names,
            'placements': place_labels(order_df, gene_label)}


# ================================================================
# PKFP coverage (Figure5_examples.ipynb)
# ================================================================

def parse_counts_csv(csv_table, samples1, samples2):
    """
    Split a per-gene count matrix into two sample groups.

    Source: Figure5_examples.ipynb (cell 43). Note the original ignores its
    `csv_table` argument and always opens the PKFP (ENSG00000067057) matrix;
    here the argument is honoured, and the callers pass that same file.
    """
    from tqdm import tqdm
    samples1_list, samples2_list = [], []
    with gzip.open(csv_table, 'rb') as fh:
        coords_line = fh.readline().decode().rstrip().split(',')[1:]
        coords = [int(x.split(':')[1]) for x in coords_line]
        for line in tqdm(fh):
            counts_line = line.decode().rstrip().split(',')
            if counts_line[0] in samples1:
                samples1_list.append([int(x) for x in counts_line[1:]])
            elif counts_line[0] in samples2:
                samples2_list.append([int(x) for x in counts_line[1:]])
    return coords, np.array(samples1_list), np.array(samples2_list)


def prepare_pkfp_panel(tissue, metadata_all, EL3, iso):
    """
    Mean normalised PKFP coverage for AD (braak >= 6) vs control (braak <= 0).

    Source: Figure5_examples.ipynb (cells 56, 57, 59) — one per tissue.

    `tissue` is the metadata `tissue_id`, which is not always the token used in
    the output filename: the PKFP_DLPFC_counts figure selects `tissue_id ==
    'DLPC'` (cell 59), the same DLPC/DLPFC asymmetry the IR figures have. The
    filter/name pairing is handled by the caller in run_all().
    """
    ad_samples = list(metadata_all.loc[
        (metadata_all['condition'] == 'AD') &
        (metadata_all.specimenID.isin(EL3.columns)) &
        (metadata_all['braaksc'] >= 6) &
        (metadata_all.tissue_id == tissue)].specimenID)
    control_samples = list(metadata_all.loc[
        (metadata_all['condition'] == 'no condition') &
        (metadata_all.specimenID.isin(EL3.columns)) &
        (metadata_all['braaksc'] <= 0) &
        (metadata_all.tissue_id == tissue)].specimenID)

    coords, s1, s2 = parse_counts_csv(f'{COUNTS_DIR}/ENSG00000067057.csv.gz',
                                      ad_samples, control_samples)
    coords_ = pd.DataFrame(coords).rolling(100,).mean().T.iloc[0]
    df1 = pd.DataFrame(np.log10(np.mean((s1.T / np.sum(s1, axis=1)).T, axis=0) + 1))
    df2 = pd.DataFrame(np.log10(np.mean((s2.T / np.sum(s2, axis=1)).T, axis=0) + 1))
    snmf_exons = iso.loc[iso.transcript_id.isin(
        ['ENSG00000067057.isoform_1', 'ENSG00000067057.isoform_2'])]
    return {'coords_': coords_, 'df1': df1, 'df2': df2, 'snmf_exons': snmf_exons}


# ================================================================
# TWAS summary (Figure5_summary_plots.ipynb)
# ================================================================

def get_gene_var_distance(gene_bed, gene, var_pos):
    """Distance from a variant to a gene body, 0 if inside. Figure5_summary_plots (cell 2)."""
    start, end = gene_bed.loc[gene].start, gene_bed.loc[gene].end
    if (var_pos > start) and (var_pos < end):
        return 0
    return int(np.min([np.abs(var_pos - start), np.abs(var_pos - end)]))


def build_gene_bed(iso):
    """Per-gene chrom/start/end from the isoform bed. Figure5_summary_plots (cell 2)."""
    return pd.DataFrame(iso.groupby('gene_id').chrom.first()).merge(
        pd.DataFrame(iso.groupby('gene_id').start.min()),
        right_index=True, left_index=True).merge(
        iso.groupby('gene_id').end.max(), right_index=True, left_index=True)


def build_coloc_df(coloc, gene_bed):
    """
    Explode the colocalisation table into (phenotype, gene, var_id, var_dist).

    Source: Figure5_summary_plots.ipynb (cell 2). The raw file has no `gene`
    column; the gene is parsed out of each colocalised sQTL outcome name.
    """
    import re as _re
    out_df = []
    for _idx, coloc_row in coloc.loc[coloc.focal_outcome != 'FALSE'].iterrows():
        sQTLs = [x for x in coloc_row.colocalized_outcomes.split('; ') if 'factor_sQTL_' in x]
        variants = coloc_row.colocalized_variables.split('; ')
        for SQ in sQTLs:
            gene = SQ.split('factor_sQTL_')[1].split('.')[0]
            SQ_sub = SQ
            for tissue in ('DLPFC', 'AC', 'PCC'):
                if SQ.startswith(tissue):
                    SQ_sub = _re.sub(f'{tissue}_factor_sQTL_',
                                     f'ROSMAP_{tissue}_sQTL_factor_', SQ)
            for var in variants:
                var_pos = int(var.split(':')[1])
                try:
                    dist = get_gene_var_distance(gene_bed, gene, var_pos)
                except KeyError:      # gene absent from the isoform bed
                    continue
                out_df.append([SQ_sub, gene, var, dist])
    return pd.DataFrame(out_df, columns=['phenotype', 'gene', 'var_id', 'var_dist'])


def filter_twas_significant(df):
    """
    Filter significant gene-isoform associations on the TWAS criteria:
      - Bonferroni cutoff 0.05 / n imputable gene-context pairs
      - significant if it passes in >= 50% of tested methods, OR in at least one
        `is_selected_method` method

    Source: Figure5_summary_plots.ipynb (cell 18). Returns
    (df_significant, bonferroni_cutoff) — the frame is filtered to the
    significant pairs, not merely flagged.
    """
    n_pairs = df[['molecular_id', 'context']].drop_duplicates().shape[0]
    bonferroni_cutoff = 0.05 / n_pairs
    print(f"Number of imputable gene-context pairs: {n_pairs}")
    print(f"Bonferroni cutoff: {bonferroni_cutoff:.2e}")

    df = df.copy()
    df['passes_cutoff'] = df['twas_pval'] < bonferroni_cutoff
    grouped = df.groupby(['molecular_id', 'context'])

    def is_significant(group):
        n_methods = len(group)
        n_passing = group['passes_cutoff'].sum()
        crit_i = (n_passing / n_methods) >= 0.5
        crit_ii = group.loc[group['is_selected_method'] == True, 'passes_cutoff'].any()
        return crit_i or crit_ii

    significant_pairs = (grouped.filter(is_significant)[['molecular_id', 'context']]
                         .drop_duplicates())
    df_significant = df.merge(significant_pairs, on=['molecular_id', 'context'],
                              how='inner')
    print(f"Total gene-context pairs tested: {n_pairs}")
    print(f"Significant gene-context pairs: "
          f"{df_significant[['molecular_id', 'context']].drop_duplicates().shape[0]}")
    return df_significant, bonferroni_cutoff


def add_phenotype(df, id_col):
    """
    Build the `phenotype` key that joins the TWAS / MR / coloc tables:
    '{context_before_dot}_{id}.{isoform}'.

    Source: Figure5_summary_plots.ipynb (cell 34 for TWAS, cell 39 for MR).
    """
    pheno_list = []
    for _idx, row in df.iterrows():
        context, iso = row.context.split('.')
        pheno_list.append(context + '_' + row[id_col] + '.' + iso)
    df = df.copy()
    df['phenotype'] = pheno_list
    return df


def load_twas_tables():
    """
    The AD-gene colocalisation / TWAS / MR tables behind TWAS_summary.

    Source: Figure5_summary_plots.ipynb (cells 32, 34, 39). All three are the
    ADgenes subset: cell 2 loads the genome-wide versions first, but cells
    32/34/39 (exec 61/83/91) overwrite coloc_df, df_significant and mr_ad from
    the ADgenes files before the figure is drawn at exec 116.
    """
    iso = load_gao_isoform_bed()
    coloc_ad = pd.read_csv(COLOC_AD_FILE, sep='\t')
    coloc_df = build_coloc_df(coloc_ad, build_gene_bed(iso))

    twas_ad = pd.read_csv(TWAS_AD_FILE, sep='\t').dropna().copy()
    twas_ad = add_phenotype(twas_ad, 'molecular_id')
    df_significant, cutoff = filter_twas_significant(twas_ad)

    mr_ad = add_phenotype(pd.read_csv(MR_AD_FILE), 'gene_name')

    genes_dir = pd.read_csv(GENES_DIR, sep='\t', names=['gene', 'gene_id', 'loc'])
    genes_dir['gene_id'] = genes_dir.gene_id.apply(lambda x: x.split('.')[0])
    return {'coloc_df': coloc_df, 'df_significant': df_significant,
            'twas_cutoff': cutoff, 'mr_ad': mr_ad, 'genes_dir': genes_dir}


def load_twas_tables_k3():
    """
    The gao_k3 (K=3) colocalisation / TWAS / MR tables behind TWAS_summary_K3.

    Source: Figure5_Genetics_K3.ipynb (cells 8, 11, 16, 21, 23-25). The K=3
    rerun differs from load_twas_tables() in four ways:
      * the TWAS table is genome-wide, not the ADgenes subset, and is *not*
        dropna'd before the phenotype key is built (the ADgenes version is);
      * MR is dropna'd instead;
      * colocalisation comes from the colocboost summary, and the gene bed it
        is scored against is the gao_k3 isoform bed restricted to autosomes;
      * significance is still filter_twas_significant(), but the Bonferroni
        denominator is the larger genome-wide gene-context count.
    """
    twas = pd.read_csv(K3_TWAS_FILE, sep='\t')
    twas = add_phenotype(twas, 'molecular_id')
    df_significant, cutoff = filter_twas_significant(twas)

    mr = pd.read_csv(K3_MR_FILE, sep='\t').dropna()
    mr = add_phenotype(mr, 'gene_name')

    iso_k3 = load_gao_isoform_bed(K3_ISOFORMS)
    autosomes = [f'chr{x}' for x in range(1, 23)]
    iso_k3 = iso_k3.loc[iso_k3.chrom.isin(autosomes)]
    coloc_df = build_coloc_df(pd.read_csv(K3_COLOC_FILE, sep='\t'),
                              build_gene_bed(iso_k3))

    genes_dir = pd.read_csv(GENES_DIR, sep='\t', names=['gene', 'gene_id', 'loc'])
    genes_dir['gene_id'] = genes_dir.gene_id.apply(lambda x: x.split('.')[0])
    return {'coloc_df': coloc_df, 'df_significant': df_significant,
            'twas_cutoff': cutoff, 'mr_ad': mr, 'genes_dir': genes_dir}


def prepare_twas_panel(twas):
    """Reduce the TWAS tables to exactly what the plot needs."""
    mr = twas['mr_ad']
    gd = twas['genes_dir']
    return {'df_significant': twas['df_significant'],
            'coloc_phenotypes': set(twas['coloc_df'].phenotype.unique()),
            'mr_phenotypes': set(mr.loc[mr.Q_pval < 0.1].phenotype.unique()),
            'gene_names': dict(zip(gd.gene_id, gd.gene))}


# ================================================================
# 3' UTR heatmap (AD_UTR.ipynb)
# ================================================================
#
# The upstream half — building EL_utr from the UTR quantification (AD_UTR cells
# 2/3/5/33/47/70) and the per-rank logistic regression (cell 109) — is
# reproduced below. AD_UTR carries four definitions of
# `logistic_regression_groups` (cells 59, 60, 61, 64); the one used here is
# cell 64, the version taking `min_sample_frac`, which is also the
# highest-numbered execution and so the definition live when `res` was computed.

UTR_ANNOT = f'{GTEX_STM}/results/gao/isoforms/annotated.isoforms.tab.gz'


def load_utr_inputs():
    """
    annot / iso / EL / metadata for the 3' UTR analysis.

    Source: AD_UTR.ipynb (cells 4, 6, 18, 25, 101).
    """
    annot = pd.read_csv(UTR_ANNOT, sep='\t')
    iso = load_gao_isoform_bed()
    genes_dir = pd.read_csv(GENES_DIR, sep='\t', names=['gene', 'gene_id', 'loc'])
    genes_dir['gene_id'] = genes_dir.gene_id.apply(lambda x: x.split('.')[0])
    annot['gene_id'] = annot.transcript.apply(lambda x: x.split('.')[0])

    efficiency_factor = []
    for _, row in annot.iterrows():
        appris_ref = row.appris_ref
        aver = appris_ref.split('/')
        if 'RI:' in appris_ref:
            ri_numbers = int([x for x in aver if x.startswith('RI:')][0].split(':')[1])
            appris_length = int(row.appris_transcript_length)
            efficiency_factor.append(
                not (((ri_numbers <= 2) and (appris_length >= 5)) or
                     ((ri_numbers <= 1) and (appris_length >= 3))))
        else:
            efficiency_factor.append(False)
    annot['efficiency_factor'] = efficiency_factor
    annot = annot.merge(genes_dir, on='gene_id')

    annot_ref = annot.loc[annot.appris_ref.apply(
        lambda x: ('ref_transcript' in x) or ('intron_chain' in x)) |
        (annot.gencode_ref != '.')]
    per_gene = annot_ref.value_counts('gene_id')
    multi_transcript_genes = per_gene.loc[per_gene > 1].index

    EL = pd.read_csv(EL_SNMF5, sep='\t', index_col=3)
    metadata_all = pd.read_csv(METADATA, sep='\t')
    metadata = metadata_all.loc[metadata_all.specimenID.isin(EL.columns)]
    return {'annot': annot, 'annot_ref': annot_ref, 'iso': iso, 'EL': EL,
            'metadata': metadata, 'metadata_all': metadata_all,
            'multi_transcript_genes': multi_transcript_genes}


# --- terminal-exon grouping (AD_UTR cells 2, 3, 5) --------------------------

def get_exon_chain(transcript_bed):
    return [(int(row.start), int(row.end)) for _, row in transcript_bed.iterrows()]


def compatible(x, y1, y2, strict=False):
    if abs(y1 - y2) >= 200:
        return False
    if strict:
        d1, d2 = abs(y1 - x), abs(y2 - x)
        if 0.75 * d1 > d2 or 0.75 * d2 > d1:
            return False
    return True


def get_unique_utr_coords_w_iso(x, ys, isos):
    current_group_y, current_group_iso = [], []
    unique_utr_group_list, unique_iso_group_list = [], []
    for y, iso in zip(ys, isos):
        if len(current_group_y) == 0:
            current_group_y.append(y)
            current_group_iso.append(iso)
        elif any([compatible(x, y_, y) for y_ in current_group_y]):
            current_group_y.append(y)
            current_group_iso.append(iso)
        else:
            unique_utr_group_list.append(current_group_y)
            unique_iso_group_list.append(current_group_iso)
            current_group_y, current_group_iso = [y], [iso]
    if len(current_group_y) > 0:
        unique_utr_group_list.append(current_group_y)
        unique_iso_group_list.append(current_group_iso)
    return unique_utr_group_list, unique_iso_group_list


def get_last_exons_w_transcript(gene_df):
    last_exons = {}
    strand = gene_df.strand.iloc[0]
    for transcript, df in gene_df.groupby('transcript_id'):
        exon_chain = get_exon_chain(df)
        last_exons[transcript] = exon_chain[-1] if strand == '+' else exon_chain[0]
    return last_exons, strand


def group_by_exon_w_transcript(last_exons):
    last_exons_list_dict, strand = last_exons
    last_exons_dict = {}
    for transcript, coords in last_exons_list_dict.items():
        a, b = coords[0], coords[1]
        x, y = (a, b) if strand == '+' else (b, a)
        if x in last_exons_dict:
            last_exons_dict[x]['y'].append(y)
            last_exons_dict[x]['iso'].append(transcript)
        else:
            last_exons_dict[x] = {'y': [y], 'iso': [transcript]}
    return last_exons_dict


def process_utr_coordinates_torino_w_transcripts(coords_dict_torino, strand):
    y_dict = {}
    for x, val in coords_dict_torino.items():
        sorted_pairs = sorted(zip(val['y'], val['iso']), key=lambda p: abs(x - p[0]))
        ys_sorted, isos_sorted = zip(*sorted_pairs)
        coord_groups, iso_groups = get_unique_utr_coords_w_iso(x, ys_sorted, isos_sorted)
        y_dict[x] = {'coord_groups': coord_groups, 'iso_groups': iso_groups,
                     'median_coords': [np.median(y) for y in coord_groups]}
    return y_dict


def process_utr_el(utr_dict, iso_df, el_df, min_proportion=0.2):
    """Per-gene terminal-exon usage matrix. AD_UTR.ipynb (cell 33)."""
    result_rows, result_index = [], []
    for start_coord, info in utr_dict.items():
        iso_groups, median_coords = info['iso_groups'], info['median_coords']
        if len(median_coords) <= 1:
            continue
        order = np.argsort([abs(mc - start_coord) for mc in median_coords])
        gene_id = iso_groups[0][0].split('.')[0]

        all_gene_factors, seen = [], set()
        for iso_group in iso_groups:
            for transcript_id in iso_group:
                iso_rows = iso_df.loc[iso_df['transcript_id'] == transcript_id]
                if not iso_rows.empty:
                    factors_str = iso_rows['factors'].iloc[0]
                    if factors_str and factors_str != '.':
                        for f in factors_str.split(':'):
                            if f not in seen:
                                seen.add(f)
                                all_gene_factors.append(f)
        if not all_gene_factors:
            continue

        el_pids = [f"{gene_id}.{f}" for f in all_gene_factors]
        el_subset = el_df.loc[el_df.index.intersection(el_pids)]
        if el_subset.empty:
            continue
        sample_cols = [c for c in el_subset.columns
                       if c not in ('#Chr', 'start', 'end', 'gid', 'strand')]
        el_subset = el_subset[sample_cols].astype(float)
        col_sums = el_subset.sum(axis=0)
        el_norm = el_subset.div(col_sums, axis=1)
        el_norm.loc[:, col_sums < min_proportion] = np.nan

        for rank, idx in enumerate(order):
            isoforms = iso_groups[idx]
            group_factors, seen_g = [], set()
            for transcript_id in isoforms:
                iso_rows = iso_df.loc[iso_df['transcript_id'] == transcript_id]
                if not iso_rows.empty:
                    factors_str = iso_rows['factors'].iloc[0]
                    if factors_str and factors_str != '.':
                        for f in factors_str.split(':'):
                            if f not in seen_g:
                                seen_g.add(f)
                                group_factors.append(f)
            if not group_factors:
                continue
            group_pids = [f"{gene_id}.{f}" for f in group_factors
                          if f"{gene_id}.{f}" in el_norm.index]
            if not group_pids:
                continue
            summed_row = el_norm.loc[group_pids].sum(axis=0, min_count=1)
            isoform_names = [t.split('.')[1] for t in isoforms]
            result_rows.append(summed_row)
            result_index.append(f"{gene_id}_rank{rank}_{'_'.join(isoform_names)}")
    if not result_rows:
        return pd.DataFrame()
    return pd.DataFrame(result_rows, index=result_index)


def get_last_exon_coords(utr_dict, iso_df):
    """Map each rank row name to chrom:start-end:strand. AD_UTR.ipynb (cell 47)."""
    coord_map = {}
    for start_coord, info in utr_dict.items():
        iso_groups, median_coords = info['iso_groups'], info['median_coords']
        if len(median_coords) <= 1:
            continue
        order = np.argsort([abs(mc - start_coord) for mc in median_coords])
        gene_id = iso_groups[0][0].split('.')[0]
        gene_rows = iso_df.loc[iso_df['gene_id'] == gene_id]
        if gene_rows.empty:
            continue
        chrom, strand = gene_rows['chrom'].iloc[0], gene_rows['strand'].iloc[0]
        for rank, idx in enumerate(order):
            isoforms = iso_groups[idx]
            end_coord = median_coords[idx]
            coord_start = int(min(start_coord, end_coord))
            coord_end = int(max(start_coord, end_coord))
            isoform_names = [t.split('.')[1] for t in isoforms]
            row_name = f"{gene_id}_rank{rank}_{'_'.join(isoform_names)}"
            coord_map[row_name] = f"{chrom}:{coord_start}-{coord_end}:{strand}"
    return coord_map


def build_el_utr(inputs, min_proportion=0.2):
    """
    Terminal-exon usage matrix across all multi-transcript genes, plus the
    coordinate map. Source: AD_UTR.ipynb (cell 70).
    """
    from tqdm import tqdm
    annot_ref, iso, EL = inputs['annot_ref'], inputs['iso'], inputs['EL']
    utr_iso = list(annot_ref.loc[
        annot_ref.gene_id.isin(inputs['multi_transcript_genes']) &
        (~annot_ref.efficiency_factor)].transcript)
    torino_iso = iso.loc[iso.transcript_id.isin(utr_iso)]

    EL_list, coord_map = [], {}
    for _gene, torino_ref_df in tqdm(torino_iso.groupby('gene_id')):
        strand = torino_ref_df.iloc[0].strand
        utr_dict = process_utr_coordinates_torino_w_transcripts(
            group_by_exon_w_transcript(get_last_exons_w_transcript(torino_ref_df)),
            strand)
        EL_list.append(process_utr_el(utr_dict, iso, EL, min_proportion=min_proportion))
        coord_map.update(get_last_exon_coords(utr_dict, iso))
    return EL_list, coord_map


# --- per-rank logistic regression (AD_UTR cell 64) --------------------------

def logistic_regression_groups(el_result, group1_cols, group2_cols, metadata=None,
                               min_sample_frac=0.6):
    """
    Logistic regression of group membership on each terminal-exon rank.

    Source: AD_UTR.ipynb (cell 64) — the `min_sample_frac` version, selected
    over the three earlier definitions in cells 59, 60 and 61.
    """
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    keep1 = [c for c in group1_cols if c in el_result.columns]
    keep2 = [c for c in group2_cols if c in el_result.columns]
    cols = keep1 + keep2

    X = el_result[cols].T.astype(float)          # no dropna
    n_total = len(X)
    y_full = pd.Series(
        [0] * sum(c in X.index for c in keep1) +
        [1] * sum(c in X.index for c in keep2),
        index=[c for c in cols if c in X.index])

    log_X = np.log(X.values + 1e-10)

    if metadata is not None:
        meta = metadata.loc[metadata.index.intersection(X.index)].reindex(X.index)
        meta_parts = []
        for col in meta.columns:
            if pd.api.types.is_numeric_dtype(meta[col]):
                meta_parts.append(((meta[col] - meta[col].mean()) /
                                   meta[col].std()).rename(col))
            else:
                meta_parts.append(pd.get_dummies(meta[col], prefix=col, drop_first=True))
        confounders = pd.concat(meta_parts, axis=1).astype(float)
        zero_var = confounders.columns[confounders.var() == 0].tolist()
        if zero_var:
            print(f"Dropping zero-variance confounder columns: {zero_var}")
            confounders = confounders.drop(columns=zero_var)
        if confounders.empty:
            print("Warning: all confounder columns dropped, running without confounders")
            confounders = None
    else:
        confounders = None

    records = []
    for i, feat in enumerate(el_result.index.tolist()):
        x_series = pd.Series(log_X[:, i], index=X.index)
        valid_mask = x_series.notna()
        if confounders is not None:
            valid_mask &= confounders.notna().all(axis=1)
        n_valid = valid_mask.sum()

        if n_valid < min_sample_frac * n_total:
            records.append({'feature': feat, 'beta': np.nan, 'pval': np.nan,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'converged': False, 'n_samples': n_valid})
            continue

        x = x_series[valid_mask].values.reshape(-1, 1)
        y = y_full[valid_mask].values
        conf_cols = confounders[valid_mask].values if confounders is not None else None
        X_design = np.hstack([x, conf_cols]) if conf_cols is not None else x
        X_const = sm.add_constant(X_design, has_constant='add')

        try:
            result = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
            if not result.converged:
                raise ValueError('did not converge')
            conf = result.conf_int()
            records.append({'feature': feat, 'beta': result.params[1],
                            'pval': result.pvalues[1], 'ci_lower': conf[1, 0],
                            'ci_upper': conf[1, 1], 'converged': True,
                            'n_samples': n_valid})
        except Exception as e:
            print(f"Logistic regression failed for {feat}: {e}, falling back to Mann-Whitney")
            v1 = x_series[valid_mask][y_full[valid_mask] == 0].values
            v2 = x_series[valid_mask][y_full[valid_mask] == 1].values
            stat, pval = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            rbc = 1 - (2 * stat) / (len(v1) * len(v2))
            records.append({'feature': feat, 'beta': rbc, 'pval': pval,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'converged': False, 'n_samples': n_valid})

    if not records:
        print("Warning: no features were successfully tested")
        return pd.DataFrame(columns=['beta', 'pval', 'ci_lower', 'ci_upper',
                                     'converged', 'n_samples', 'pval_fdr'])

    df = pd.DataFrame(records).set_index('feature')
    tested = df['pval'].notna()
    if tested.any():
        _, df.loc[tested, 'pval_fdr'], _, _ = multipletests(
            df.loc[tested, 'pval'], method='fdr_bh')
    else:
        df['pval_fdr'] = np.nan
    return df


# --- rank-pattern summary (AD_UTR cell 87) ----------------------------------

def assign_lax_pattern(row):
    pattern = row['pattern']
    ranks = np.array(range(row['n_ranks']))
    betas = np.array(row['betas'])
    sig_mask = np.array(row['sig_mask'])
    min_rank, max_rank = ranks.min(), ranks.max()

    if pattern == 'none_sig':
        return 'none_sig'
    if pattern in ('monotone_pos', 'all_sig_positive'):
        return 'positive_pattern'
    if pattern in ('monotone_neg', 'all_sig_negative'):
        return 'negative_pattern'
    if pattern == 'single_sig':
        sig_rank, sig_beta = ranks[sig_mask][0], betas[sig_mask][0]
        if sig_rank == min_rank:
            return 'negative_pattern' if sig_beta > 0 else 'positive_pattern'
        elif sig_rank == max_rank:
            return 'negative_pattern' if sig_beta < 0 else 'positive_pattern'
        return 'ambiguous'
    if pattern == 'mixed':
        return 'mixed'
    return 'ambiguous'


def summarize_rank_beta(df, pval_col='pval_fdr', pval_thresh=0.05):
    """Per-gene summary of the rank-wise betas. AD_UTR.ipynb (cell 87)."""
    from scipy.stats import spearmanr
    parsed = df.index.str.extract(r'^(ENSG\d+)_rank(\d+)_')
    parsed.columns = ['gene_id', 'rank']
    parsed['rank'] = parsed['rank'].astype(int)
    parsed.index = df.index

    df = df.copy()
    df['gene_id'] = parsed['gene_id']
    df['rank'] = parsed['rank']

    records = []
    for gene_id, grp in df.groupby('gene_id'):
        grp = grp.sort_values('rank')
        ranks, betas = grp['rank'].values, grp['beta'].values
        pvals = grp[pval_col].values
        sig_mask = pvals < pval_thresh
        n_ranks, n_sig = len(grp), sig_mask.sum()
        sig_betas = betas[sig_mask]

        if n_sig == 0:
            pattern, rho, rho_pval, beta_direction = 'none_sig', np.nan, np.nan, None
        elif n_sig == 1:
            pattern, rho, rho_pval = 'single_sig', np.nan, np.nan
            beta_direction = 'positive' if sig_betas[0] > 0 else 'negative'
        else:
            if n_ranks > 2:
                rho, rho_pval = spearmanr(ranks, betas)
            else:
                rho = 1.0 if betas[1] > betas[0] else -1.0
                rho_pval = np.nan
            if abs(rho) > 0.8:
                pattern = 'monotone_pos' if rho > 0 else 'monotone_neg'
            elif (sig_betas > 0).all():
                pattern = 'all_sig_positive'
            elif (sig_betas < 0).all():
                pattern = 'all_sig_negative'
            else:
                pattern = 'mixed'
            beta_direction = None

        records.append({'gene_id': gene_id, 'n_ranks': n_ranks, 'n_sig': n_sig,
                        'pattern': pattern, 'spearman_rho': rho,
                        'spearman_pval': rho_pval, 'beta_direction': beta_direction,
                        'betas': list(betas), 'sig_mask': list(sig_mask),
                        'row_names': list(grp.index),
                        'sig_row_names': list(grp.index[sig_mask]),
                        'min_pval': pvals.min()})

    result = pd.DataFrame(records).set_index('gene_id')
    result['lax_pattern'] = result.apply(assign_lax_pattern, axis=1)
    return result


def get_extreme_rank_genes(summary_df, lr_df, pval_col='pval_fdr'):
    """Pick, per gene, the most significant terminal UTR rank. AD_UTR (cell 106)."""
    records = []
    for gene_id, row in summary_df.iterrows():
        row_names = row['row_names']
        betas = np.array(row['betas'])
        n_ranks = row['n_ranks']
        if row['n_sig'] == 0:
            continue
        pvals = lr_df.loc[row_names, pval_col].values
        most_sig_idx = np.argmin(pvals)
        if most_sig_idx == 0:
            rank_type = 'lowest'
        elif most_sig_idx == n_ranks - 1:
            rank_type = 'highest'
        else:
            continue
        most_sig_beta = betas[most_sig_idx]
        effective_beta = -most_sig_beta if rank_type == 'lowest' else most_sig_beta
        records.append({'gene_id': gene_id, 'rank_type': rank_type,
                        'most_sig_row_name': row_names[most_sig_idx],
                        'beta': most_sig_beta, 'effective_beta': effective_beta,
                        'pval': pvals[most_sig_idx]})
    result = pd.DataFrame(records).set_index('gene_id')
    result['signed_log_pval'] = -np.log10(result['pval']) * np.sign(result['effective_beta'])
    return result.sort_values('signed_log_pval')


def normalize_el_matrix(el_matrix, sample_groups, tissues):
    """
    Per-gene tissue-wise z-score, then a per-gene scale across all samples.

    Source: AD_UTR.ipynb (cell 104).
    """
    from sklearn.preprocessing import scale as sklearn_scale
    el_norm = el_matrix.copy().astype(float)
    for tissue in tissues:
        tissue_cols = [s for key, samples in sample_groups.items() if tissue in key
                       for s in samples if s in el_norm.columns]
        if not tissue_cols:
            continue
        tissue_data = el_norm[tissue_cols]
        tissue_mean = tissue_data.mean(axis=1)
        tissue_std = tissue_data.std(axis=1).replace(0, np.nan)
        el_norm[tissue_cols] = tissue_data.sub(tissue_mean, axis=0).div(tissue_std, axis=0)
    return pd.DataFrame(sklearn_scale(el_norm.T).T, index=el_norm.index,
                        columns=el_norm.columns)


def prepare_utr3_panel(extreme_df, EL_utr, metadata, label_genes=None,
                       label_names=None):
    """
    Build the condition x tissue mean matrix behind UTR3_changes.

    Source: AD_UTR.ipynb (cells 106, 129). Only the matrix construction lives
    here; the clustermap is drawn by Figure5_plot_helpers.plot_utr3_changes.
    """
    tissues = ['AC', 'DLPC', 'PCC']
    conditions = ['no condition', 'AD']
    sample_groups = {}
    for cond in conditions:
        for tissue in tissues:
            key = f"{'ctrl' if cond == 'no condition' else 'AD'}_{tissue}"
            sample_groups[key] = list(metadata.loc[
                (metadata['condition'] == cond) &
                (metadata['tissue_id'] == tissue) &
                (metadata['specimenID'].isin(EL_utr.columns))]['specimenID'])
    ordered_cols = [s for key in sample_groups for s in sample_groups[key]]

    # The source builds this row by row with a per-gene EL_utr.loc[...] lookup.
    # Against the Arrow-backed string index that costs ~16 MB per row for a 7 kB
    # result, so 412 genes exhaust the 8.6 GB cgroup and the process is OOM-killed
    # with no traceback. One vectorised .loc gives a byte-identical matrix.
    keep = extreme_df.loc[extreme_df['most_sig_row_name'].isin(EL_utr.index)]
    el_matrix = EL_utr.loc[keep['most_sig_row_name'].tolist(), ordered_cols].astype(float)
    el_matrix.index = keep.index
    flip = (keep['rank_type'] == 'lowest').to_numpy()
    if flip.any():
        el_matrix.iloc[flip] = 1 - el_matrix.iloc[flip]

    el_plot = normalize_el_matrix(el_matrix, sample_groups, tissues)
    el_means = pd.DataFrame(index=el_plot.index)
    for key, samples in sample_groups.items():
        valid = [s for s in samples if s in el_plot.columns]
        el_means[key] = el_plot[valid].mean(axis=1)

    positions, names = [], []
    if label_genes is not None:
        name_map = dict(zip(label_genes, label_names)) if label_names is not None else None
        label_set = set(label_genes)
        positions = [i for i, g in enumerate(el_means.index) if g in label_set]
        names = [name_map[el_means.index[i]] if name_map else el_means.index[i]
                 for i in positions]
    return {'el_matrix': el_matrix, 'el_plot': el_plot, 'el_means': el_means,
            'label_positions': positions, 'label_names': names}


UTR3_LOGREG_TABLE = (f'{TORINO_CODE}/results/gao_k5/tables/'
                     'UTR3.logreg.output.tab.gz')


def utr3_direction_summary(res, pval_thresh=0.1, label=''):
    """
    The per-analysis counts behind the 3' UTR numbers: how many genes carry two
    or more polyadenylation sites, how many sites and genes are significant, and
    how the surviving shifts split between the short and the long 3' UTR.

    `res` is a regression table indexed by '{gene}_rank{n}_...'. The direction
    follows get_extreme_rank_genes: a gene counts only when its most significant
    rank is the shortest or the longest, and `effective_beta` < 0 is a shift
    toward the short 3' UTR.
    """
    res = res.copy()
    if 'gene_id' not in res.columns:
        res['gene_id'] = [str(i).split('_rank')[0] for i in res.index]
    sig = res.loc[res.pval_fdr <= pval_thresh]
    ex = get_extreme_rank_genes(summarize_rank_beta(res, pval_thresh=pval_thresh), res)
    short = int((ex.effective_beta < 0).sum())
    long_ = int((ex.effective_beta > 0).sum())
    tot = short + long_
    per_gene = res.groupby('gene_id').size()
    return {'analysis': label, 'fdr': pval_thresh,
            'genes_2plus_polyA': int(res.gene_id.nunique()),
            'utr_sites': int(len(res)),
            'median_sites_per_gene': int(per_gene.median()),
            'max_sites_per_gene': int(per_gene.max()),
            'sig_sites': int(len(sig)),
            'sig_genes': int(sig.gene_id.nunique()),
            'genes_with_shift': tot,
            'short': short, 'short_pct': round(100 * short / tot, 0) if tot else np.nan,
            'long': long_, 'long_pct': round(100 * long_ / tot, 0) if tot else np.nan}


def utr3_covariate_sweep(covariate_sets, pval_thresh=0.05, data_dir=None):
    """
    Refit the 3' UTR regression under several covariate sets and report the
    short/long split of each -- the check behind "the 60/40 asymmetry is
    covariate-dependent".

    `covariate_sets` is {label: [column names]}. Each fit is a full pass over
    the 4,709 UTR groups (~20 s), so a sweep of four takes a couple of minutes.
    """
    data_dir = data_dir or os.path.join(FIG5_DIR_DEFAULT, 'figure_data')
    with open(os.path.join(data_dir, 'utr_el.pkl'), 'rb') as f:
        cache = pickle.load(f)
    EL_utr, ad, ctrl = cache['EL_utr'], cache['ad'], cache['ctrl']

    meta = pd.read_csv(METADATA, sep='\t')
    meta.index = meta.specimenID
    clean = pd.to_numeric(meta['age_death'].str.replace('+', '', regex=False),
                          errors='coerce')
    meta['age_group'] = pd.cut(clean, bins=[-np.inf, 85, 89.9, np.inf],
                               labels=[0, 1, 2])
    meta['log10PMI'] = np.log10(meta.pmi)
    meta['race_string'] = [{1: 'w', 2: 'b', 3: 'ai', 4: 'a'}.get(r, 'o')
                           for r in meta.race]
    samples = [s for s in ctrl + ad if s in meta.index]
    c = [s for s in ctrl if s in samples]
    a = [s for s in ad if s in samples]

    rows = []
    for label, cov in covariate_sets.items():
        res = logistic_regression_groups(EL_utr, c, a, meta.loc[samples, cov])
        rows.append({**utr3_direction_summary(res, pval_thresh, label),
                     'n_covariates': len(cov)})
    return pd.DataFrame(rows)


def prepare_utr3_panel_from_table(res_path=None, data_dir=None, gene_sets=None,
                                  pval_thresh=0.1):
    """
    The UTR3_changes matrix built from an external regression table rather than
    the notebook's cached one.

    `res_path` is the output of scripts/prepare_ad_utr3_logreg.py -- the same
    per-UTR-rank logistic regression, run with the AD_K3_LR covariate set. The
    EL_utr matrix it was fitted on is the cached terminal-exon grouping, so this
    only redoes the cheap downstream stages.

    The labelled genes come from load_ad_gene_sets(), i.e. the **genome-wide
    K=5** colocalisation table -- the right pairing here, since the whole UTR
    analysis is K=5 (snmf5_EL and the gao isoform annotation). Panel C of the
    main figure uses the gao_k3 colocboost summary instead.
    """
    res_path = res_path or UTR3_LOGREG_TABLE
    data_dir = data_dir or os.path.join(FIG5_DIR_DEFAULT, 'figure_data')
    with open(os.path.join(data_dir, 'utr_el.pkl'), 'rb') as f:
        cache = pickle.load(f)
    EL_utr, metadata = cache['EL_utr'], cache['metadata']

    res = pd.read_csv(res_path, sep='\t', index_col=0)
    print(f'[Figure5]   {res_path}: {len(res)} rows, '
          f'{int((res.pval_fdr <= 0.05).sum())} significant at FDR 5%')

    # FDR 0.1, matching the cut panels A and C use for the intron-retention
    # side. The AD_K3_LR covariate set costs power against the notebook's four
    # covariates -- CISD1's leading UTR rank goes from FDR 0.026 to 0.090 on a
    # near-identical beta (0.146 -> 0.140) -- so 0.05 would drop genes for want
    # of precision rather than effect.
    extreme = get_extreme_rank_genes(
        summarize_rank_beta(res, pval_thresh=pval_thresh), res)

    if gene_sets is None:
        gene_sets = load_ad_gene_sets()        # genome-wide K=5 colocalisations
    genes_df = pd.read_csv(GENES_BED, sep='\t',
                           names=['chrom', 'start', 'end', 'gene_id',
                                  'gene_name', 'strand'])
    labelled = genes_df.loc[genes_df.gene_id.isin(gene_sets['iso_coloc_genes'])]

    panel = prepare_utr3_panel(extreme, EL_utr, metadata,
                               label_genes=list(labelled.gene_id),
                               label_names=list(labelled.gene_name))
    panel['res'] = res
    panel['extreme'] = extreme
    panel['source'] = res_path
    panel['pval_thresh'] = pval_thresh
    return panel


def prepare_utr3_data(data_dir='../notebooks/Figure5/figure_data', gene_sets=None,
                      reuse=True):
    """
    Full 3' UTR chain: EL_utr -> logistic regression -> rank summary -> the
    condition x tissue mean matrix behind UTR3_changes.

    Source: AD_UTR.ipynb (cells 70, 109, 111, 120, 129). The regression is the
    `min_sample_frac` version from cell 64.

    The two slow stages are cached in `data_dir` and reused when present:

        utr_el.pkl   terminal-exon grouping over ~3850 genes (build_el_utr)
        utr_res.pkl  the per-rank logistic regression (logistic_regression_groups)

    With both present this returns in seconds. Pass `reuse=False`, or delete the
    two files, to force a recompute.
    """
    el_cache = os.path.join(data_dir, 'utr_el.pkl')
    res_cache = os.path.join(data_dir, 'utr_res.pkl')

    # ---- stage 1: terminal-exon usage matrix ----
    if reuse and os.path.exists(el_cache):
        print(f'[Figure5]   reusing {el_cache}')
        with open(el_cache, 'rb') as f:
            cache = pickle.load(f)
    else:
        inputs = load_utr_inputs()
        EL, metadata_all = inputs['EL'], inputs['metadata_all']
        EL_list, coord_map = build_el_utr(inputs)
        ad_samples = list(metadata_all.loc[
            (metadata_all['condition'] == 'AD') &
            (metadata_all.specimenID.isin(EL.columns))].specimenID)
        ctrl_samples = list(metadata_all.loc[
            (metadata_all['condition'] == 'no condition') &
            (metadata_all.specimenID.isin(EL.columns))].specimenID)
        meta_select = inputs['metadata'].copy()
        meta_select.index = meta_select.specimenID
        cache = {
            'EL_utr': pd.concat(EL_list)[ad_samples + ctrl_samples],
            'coord_map': coord_map,
            'ad': ad_samples, 'ctrl': ctrl_samples,
            'conf': meta_select.loc[ad_samples + ctrl_samples,
                                    ['tissue_id', 'batch', 'msex', 'race']],
            'metadata': inputs['metadata'][['specimenID', 'condition', 'tissue_id']].copy(),
        }
        os.makedirs(data_dir, exist_ok=True)
        with open(el_cache, 'wb') as f:
            pickle.dump(cache, f)
        print(f'[Figure5]   wrote {el_cache}')

    EL_utr, ad_samples, ctrl_samples = cache['EL_utr'], cache['ad'], cache['ctrl']
    metadata = cache['metadata']

    # ---- stage 2: per-rank logistic regression ----
    if reuse and os.path.exists(res_cache):
        print(f'[Figure5]   reusing {res_cache}')
        with open(res_cache, 'rb') as f:
            res = pickle.load(f)
    else:
        res = logistic_regression_groups(EL_utr, ctrl_samples, ad_samples,
                                         cache['conf'])
        res['last_exon_coords'] = res.index.map(cache['coord_map'])
        with open(res_cache, 'wb') as f:
            pickle.dump(res, f)
        print(f'[Figure5]   wrote {res_cache}')

    # ---- stage 3: cheap, always recomputed ----
    extreme = get_extreme_rank_genes(summarize_rank_beta(res), res)

    if gene_sets is None:
        gene_sets = load_ad_gene_sets()
    genes_df = pd.read_csv(GENES_BED, sep='\t',
                           names=['chrom', 'start', 'end', 'gene_id',
                                  'gene_name', 'strand'])
    labelled = genes_df.loc[genes_df.gene_id.isin(gene_sets['iso_coloc_genes'])]

    panel = prepare_utr3_panel(extreme, EL_utr, metadata,
                               label_genes=list(labelled.gene_id),
                               label_names=list(labelled.gene_name))
    panel['res'] = res
    panel['extreme'] = extreme
    return panel


# ================================================================
# Orchestrator
# ================================================================

def run_all(data_dir='../notebooks/Figure5/figure_data', sections=None):
    """
    Run the Figure 5 data pipeline, pickle every plot-ready variable into
    `data_dir`, and return them as a dict.

    `sections` selects a subset, which matters here because the parts are
    expensive and independent:
        'panels'  per-gene coverage / boxplot / factor panels (needs rpy2)
        'k3'      gao_k3 beta summaries, GSEA, KNOP1
        'ir'      the IR gene-body heatmap layouts (needs pybedtools; slow)
        'pkfp'    PKFP coverage
        'twas'    TWAS summary
        'utr'     3' UTR matrix (reuses cached utr_el.pkl / utr_res.pkl)
    Default runs all of them.
    """
    all_sections = ['panels', 'k3', 'ir', 'pkfp', 'twas', 'utr']
    sections = all_sections if sections is None else list(sections)
    os.makedirs(data_dir, exist_ok=True)
    out = {}

    def _dump(name, obj):
        with open(os.path.join(data_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)
        out[name] = obj

    genes_df = pd.read_csv(GENES_BED, sep='\t',
                           names=['chrom', 'start', 'end', 'gene_id',
                                  'gene_name', 'strand'])

    if 'panels' in sections:
        print('[Figure5] gene example panels ...')
        iso = load_gao_isoform_bed()
        _dump('iso', iso)
        coverage = {name: prepare_coverage_panel(g, v, t, root, iso)
                    for name, (g, v, t, root) in COVERAGE_PANELS.items()}
        _dump('coverage_panels', coverage)
        boxplots = {name: prepare_boxplot_panel(g, v, t)
                    for name, (g, v, t, *_rest) in BOXPLOT_PANELS.items()}
        _dump('boxplot_panels', boxplots)
        _dump('factor_panels', {name: prepare_factor_panel(g, iso, fl)
                                for name, (g, fl, _c) in FACTOR_PANELS.items()})

    if 'k3' in sections:
        print('[Figure5] gao_k3 tables and beta summaries ...')
        LR_output, gene_w_IR, gene_no_IR = load_k3_tables()
        _dump('k3_beta_stats', compute_k3_beta_stats(LR_output, gene_w_IR, gene_no_IR))
        ad_gene_names = load_ad_gene_names(LR_output, genes_df)
        print('[Figure5] GSEA enrichment ...')
        _dump('gsea_res2d', run_gsea(ad_gene_names))
        print('[Figure5] KNOP1 braak boxplot ...')
        EL = pd.read_csv(EL_SNMF3, sep='\t', index_col=3)
        metadata_all = pd.read_csv(METADATA, sep='\t')
        _dump('knop1_braak', prepare_knop1_braak(EL, metadata_all))

    if 'ir' in sections:
        print('[Figure5] IR heatmap layouts (pybedtools, slow) ...')
        gencode = load_gencode()
        gene_sets = load_ad_gene_sets()
        _dump('ir_panel_k3', prepare_ir_panel(K3_IR_ONLY_PLOTS, gencode, genes_df,
                                              gene_sets))

    if 'pkfp' in sections:
        print('[Figure5] PKFP coverage ...')
        metadata_all = pd.read_csv(METADATA, sep='\t')
        EL3 = pd.read_csv(EL_SNMF3, sep='\t')
        iso_k3 = load_gao_isoform_bed(K3_ISOFORMS)
        # (metadata tissue_id, output-name token) — see prepare_pkfp_panel
        _dump('pkfp_panels', {name: prepare_pkfp_panel(tid, metadata_all, EL3, iso_k3)
                              for tid, name in [('PCC', 'PCC'), ('AC', 'AC'),
                                                ('DLPC', 'DLPFC')]})

    if 'examples' in sections:
        print('[Figure5] K=3 / pooled example panels ...')
        iso5 = load_gao_isoform_bed()
        iso3 = load_gao_isoform_bed(K3_ISOFORMS)
        KNOP1, KNOP1_SNP = 'ENSG00000103550', 'chr16:19708463:CTCA:C'
        CISD1, CISD1_SNP = 'ENSG00000122873', 'chr10:58286894:G:A'

        # panel E of the K3 figure: KNOP1 on the K=3 factorisation
        _dump('knop1_factors_k3',
              prepare_factor_panel(KNOP1, iso3, ['factor1', 'factor2'], k=3))
        # panels F and H: the same variant in all three tissues at once
        _dump('knop1_box_pooled_k3',
              prepare_pooled_boxplot(KNOP1, KNOP1_SNP, root=K3_BOX_RDS))
        _dump('cisd1_box_pooled_k5',
              prepare_pooled_boxplot(CISD1, CISD1_SNP, root=BOX_RDS))

        # supplements: every factor of the model the main figure does not use
        _dump('knop1_factors_k5_all',
              prepare_factor_panel(KNOP1, iso5,
                                   [f'factor{i}' for i in range(1, 6)], k=5))
        _dump('knop1_box_pooled_k5',
              prepare_pooled_boxplot(KNOP1, KNOP1_SNP, root=BOX_RDS))
        _dump('cisd1_factors_k3_all',
              prepare_factor_panel(CISD1, iso3,
                                   [f'factor{i}' for i in range(1, 4)], k=3))
        _dump('cisd1_box_pooled_k3',
              prepare_pooled_boxplot(CISD1, CISD1_SNP, root=K3_BOX_RDS))
        # the marked Figure5_examples figure: pooled tissues, split by AD status
        _dump('knop1_byad_pooled_k3',
              prepare_by_ad_pooled_boxplot(KNOP1, KNOP1_SNP))
        _dump('cisd1_byad_pooled_k3',
              prepare_by_ad_pooled_boxplot(CISD1, CISD1_SNP))

        # panel C relabelled from the K=3 colocalisations (18 genes, 13 red)
        print('[Figure5] panel C labels from the K=3 colocalisations ...')
        base = out.get('ir_panel_k3')
        if base is None:
            with open(os.path.join(data_dir, 'ir_panel_k3.pkl'), 'rb') as f:
                base = pickle.load(f)
        from Figure5_plot_helpers import place_labels
        tbl = _dump('k3_coloc_table', build_k3_coloc_table())
        panel = dict(base)
        panel['gene_label'] = build_gene_label_k3(base['order_df'], tbl)
        panel['gene_names'] = {**base['gene_names'], **load_k3_gene_symbols()}
        panel['placements'] = place_labels(base['order_df'], panel['gene_label'])
        _dump('ir_panel_k3_colocK3', panel)

    if 'utr_newlr' in sections:
        print("[Figure5] 3' UTR matrix from the new logistic regression ...")
        _dump('utr3_panel_newlr', prepare_utr3_panel_from_table(data_dir=data_dir))

    if 'relaxed_sig' in sections:
        print('[Figure5] panel A at FDR 0.1 and the FDR-filtered relaxed '
              'IR layout ...')
        from Figure5_plot_helpers import place_labels
        LR_output, gene_w_IR, gene_no_IR = load_k3_tables()
        _dump('k3_beta_stats_fdr0p1',
              compute_k3_beta_stats(LR_output, gene_w_IR, gene_no_IR, fdr=0.1))

        gencode = load_gencode()
        gene_sets = load_ad_gene_sets()
        panel = prepare_ir_panel(K3_IR_ONLY_PLOTS_RELAXED_SIG, gencode, genes_df,
                                 gene_sets)
        tbl = out.get('k3_coloc_table')
        if tbl is None:
            with open(os.path.join(data_dir, 'k3_coloc_table.pkl'), 'rb') as f:
                tbl = pickle.load(f)
        panel['gene_label'] = build_gene_label_k3(panel['order_df'], tbl)
        panel['gene_names'] = {**panel['gene_names'], **load_k3_gene_symbols()}
        panel['placements'] = place_labels(panel['order_df'], panel['gene_label'])
        _dump('ir_panel_k3_relaxed_sig_colocK3', panel)

    if 'ir_relaxed' in sections:
        print('[Figure5] relaxed IR heatmap layout (pybedtools, slow) ...')
        from Figure5_plot_helpers import place_labels
        gencode = load_gencode()
        gene_sets = load_ad_gene_sets()
        panel = prepare_ir_panel(K3_IR_ONLY_PLOTS_RELAXED, gencode, genes_df,
                                 gene_sets)
        # relabel from the K=3 colocalisations, as ir_panel_k3_colocK3 does
        tbl = out.get('k3_coloc_table')
        if tbl is None:
            with open(os.path.join(data_dir, 'k3_coloc_table.pkl'), 'rb') as f:
                tbl = pickle.load(f)
        panel['gene_label'] = build_gene_label_k3(panel['order_df'], tbl)
        panel['gene_names'] = {**panel['gene_names'], **load_k3_gene_symbols()}
        panel['placements'] = place_labels(panel['order_df'], panel['gene_label'])
        _dump('ir_panel_k3_relaxed_colocK3', panel)

    if 'twas' in sections:
        print('[Figure5] TWAS summary ...')
        _dump('twas_panel', prepare_twas_panel(load_twas_tables()))
        _dump('twas_panel_k3', prepare_twas_panel(load_twas_tables_k3()))

    if 'utr' in sections:
        print("[Figure5] 3' UTR terminal-exon matrix ...")
        _dump('utr3_panel', prepare_utr3_data(data_dir))

    return out
