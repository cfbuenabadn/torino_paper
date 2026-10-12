import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns


def load_ebpmf_gene(gene, species='Human'):
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    readRDS = ro.r['readRDS']
    if species in ['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']:
        df = readRDS(f'/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/factorization/{species}/{gene}.rds')
    else:
        df = readRDS(f'/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/factorization_filtered/{species}/{gene}.rds')
    with (ro.default_converter + pandas2ri.converter).context():
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(df)
    return pd_from_r_df


def sum_every_ten_columns(dataframe):
    num_columns = dataframe.shape[1]
    new_columns = [dataframe.iloc[:, i:i+10].sum(axis=1) for i in range(0, num_columns, 10)]
    return pd.concat(new_columns, axis=1)


def factor_lm(factor, strand=None):
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
        if ((((strand == 'plus') or (strand == '+')) and slope > 0) or (((strand == 'minus') or (strand == '-')) and slope < 0)):
            y_model = x * slope + intercept
            y_corrected = y / np.maximum(y_model, 0.1)
            y_corrected = np.where(y > 0.1, y_corrected, y)
        else:
            y_corrected = factor
    return y_corrected


def _load_gtex_colors():
    """Load gtex_colors and gtex_colors2 from torino_plots.py by exec."""
    globs = {}
    exec(open('/project2/yangili1/cfbuenabadn/torino/scripts/torino_plots.py').read(), globs)
    return globs['gtex_colors'], globs['gtex_colors2']


def _prepare_factors_EF(rds, gene, strand='plus', lm=True, factor_list=None):
    """Build the lm-corrected EF DataFrame ready for plotting."""
    EF = pd.DataFrame(rds['EF_smooth'][1:-1, :])
    coords = list(rds['coords'])[1:-1]
    EF.index = coords
    if strand == 'minus':
        EF = EF.loc[EF.index[::-1]]
    EF.columns = [f'factor{str(i + 1)}' for i in range(EF.shape[1])]
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
    return EF


def _compute_xlim(coords):
    xlim1 = int(coords[0].split(':')[1]) - 1000
    xlim2 = int(coords[-1].split(':')[1]) + 1000
    return (xlim1, xlim2)


def run_all(data_dir='figure_data'):
    """Load and process all data for Figure 1, save pickles, return plot-ready variables."""
    os.makedirs(data_dir, exist_ok=True)

    # --- Load gtex color dictionaries ---
    gtex_colors, gtex_colors2 = _load_gtex_colors()

    # --- Load GTEx metadata ---
    metadata = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv', sep='\t')
    human_metadata = metadata[['entity:sample_id', 'tissue_id']]
    human_metadata.columns = ['Sample', 'Tissue']
    human_metadata.index = human_metadata.Sample

    # --- Load SRSF3 count matrix and prepare heatmap data ---
    gene = 'SRSF3'
    counts = pd.read_csv(f'/project2/yangili1/cfbuenabadn/torino_paper/code/resources/count_matrices_filtered/Human/{gene}.csv.gz', index_col=0)
    counts.index = [x.split('.')[0] for x in counts.index]

    tissues_display = sorted(metadata.loc[metadata['entity:sample_id'].isin(counts.index)].tissue_site_detail.unique())
    colors_list_display = ['#' + gtex_colors[tissue]['tissue_color_hex'] for tissue in tissues_display]

    summed_df = sum_every_ten_columns(counts)
    data_list = []
    for i, tissue in enumerate(tissues_display):
        d = summed_df.loc[pd.Index(metadata.loc[metadata.tissue_site_detail == tissue]['entity:sample_id']).intersection(counts.index)]
        data_list.append(d)
    data = pd.concat(data_list, axis=0)

    # --- Load SRSF3 RDS (factorization results) ---
    rds = load_ebpmf_gene('SRSF3')

    # --- Prepare lm-corrected EF for factor tracks ---
    EF_factors = _prepare_factors_EF(rds['torino_K10'], gene, strand='plus', lm=True)
    coords_factors = list(rds['torino_K10']['coords'])[1:-1]
    xlim_factors = _compute_xlim(coords_factors)
    K_factors = EF_factors.shape[1]

    # --- Prepare EL for loadings barplot ---
    colores_loadings = ['tab:gray', 'tab:orange', 'tab:purple', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:red', 'tab:gray', 'tab:orange', 'tab:blue']
    EL = pd.DataFrame(rds['torino_K10']['EL'])
    EL.index = list(rds['torino_K10']['samples'])
    EL.columns = [f'factor_{str(i + 1)}' for i in range(10)]

    # --- Load isoform annotations (Torino-called) ---
    human_iso = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                            names=['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon', 'notes'])
    human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

    # --- Load Gencode isoform annotations ---
    gencode_iso = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/Human/bed/exons.all.bed.gz', sep='\t')
    gencode_iso['chrom'] = gencode_iso['#chrom']
    gencode_iso['gene_id'] = gencode_iso.gene_name

    # --- Load EL table (BED format) and prepare hloadings data ---
    EL_bed = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/tables/Human_filtered/EL.K10.bed.gz', sep='\t', index_col=3)

    metadata.index = metadata['entity:sample_id']
    sorted_samples = []
    for t in tissues_display:
        sorted_samples.extend(list(EL_bed.columns.intersection(metadata.loc[metadata.tissue_site_detail == t].index)))

    EL_to_plot = EL_bed.loc[EL_bed.gid == 'SRSF3', sorted_samples].T
    factor_order = [f'SRSF3.{factor}' for factor in [f'factor_{str(x)}' for x in range(1, 11)]][::-1]
    EL_to_plot = EL_to_plot[factor_order]
    EL_to_plot.columns = [f'SRSF3.factor_{str(i)}' for i in range(1, 11)]

    human_metadata_h = human_metadata.copy()
    human_metadata_h['tissue_id'] = human_metadata_h.Tissue

    h_tissues = list(dict.fromkeys(human_metadata_h.loc[EL_to_plot.index].tissue_id))
    colors_list_h = [gtex_colors2[t]['tissue_color_hex'] for t in h_tissues]
    labels_h = [gtex_colors2[t]['tissue_abbrv'] for t in h_tissues]
    n_tissues_h = len(h_tissues)
    colores_h = sns.color_palette("tab10")

    # --- Build result dict ---
    result = {
        # Counts heatmap
        'data': data,
        'tissues_display': tissues_display,
        'colors_list_display': colors_list_display,
        # Factor tracks
        'EF_factors': EF_factors,
        'K_factors': K_factors,
        'xlim_factors': xlim_factors,
        # Loadings barplot
        'EL': EL,
        'colores_loadings': colores_loadings,
        'human_metadata': human_metadata,
        # Horizontal loadings
        'EL_to_plot': EL_to_plot,
        'human_metadata_h': human_metadata_h,
        'colores_h': colores_h,
        'colors_list_h': colors_list_h,
        'labels_h': labels_h,
        'n_tissues_h': n_tissues_h,
        # Isoform annotations
        'human_iso': human_iso,
        'gencode_iso': gencode_iso,
    }

    # --- Save pickles ---
    for key, val in result.items():
        with open(os.path.join(data_dir, f'{key}.pkl'), 'wb') as f:
            pickle.dump(val, f)

    return result
