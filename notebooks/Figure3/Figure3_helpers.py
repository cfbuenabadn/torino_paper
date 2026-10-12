"""
Figure3_helpers.py — Data loading, processing, and orchestration for Figure 3.

Contains all data-loading and data-processing functions, plus a run_all()
function that produces the final plot-ready variables and saves them as
pickles in figure_data/.

Must be run from the analysis/ directory (or have CWD set appropriately), so
that the relative '../code/...' paths resolve.

Heavy/specialised imports (pysam, scipy, tqdm, and the analysis/*.py modules
that pull them in) are done *inside* the functions that need them, so that
`from Figure3_helpers import run_all` works in a bare pandas environment.
"""

import gzip
import os
import pickle

import numpy as np
import pandas as pd


# ================================================================
# Paths / constants
# ================================================================

SSMD_BASE  = '../code/results/torino/tables'
ORTHO_DIR  = '../code/resources/annotation/orthology'
ISO_ANNOT  = '../code/results/torino/{species}/isoforms/annotated.isoforms.tab.gz'
HUMAN_ANNOT = '../code/results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz'

SPECIES_ALL = ['Human', 'Macaque', 'Mouse', 'Rat', 'Rabbit', 'Chicken', 'Cattle']

# Lazy per-gene data sources (mirrors analysis/plot_isoforms_lazy.py)
EL_PATHS = {
    sp: f'../code/results/torino/tables/{sp}/isoforms.EL.sorted.bed.gz'
    for sp in SPECIES_ALL
}
ISO_PATHS = {
    sp: f'../code/results/torino/{sp}/isoforms/isoforms.bed.gz'
    for sp in SPECIES_ALL
}
GENE_COORD_PATHS = {
    sp: path.replace('isoforms.bed.gz', 'gene_coords.tsv')
    for sp, path in ISO_PATHS.items()
}
COUNT_MATRIX_PATHS = {
    'Human':  '../code/resources/count_matrices_filtered/Human/{gene}.csv.gz',
    'Cattle': '../code/resources/count_matrices_filtered/Cattle/{gene}.csv.gz',
}
COUNT_MATRIX_PATH_OTHER = '../code/resources/count_matrices/{species}/{gene}.csv.gz'

# All pairwise orthology files used to build the multi-species ortho table
ORTHO_PAIRS = [
    ('Human',   'Macaque', 'human_macaque_orthology.txt'),
    ('Human',   'Mouse',   'human_mouse_orthology.txt'),
    ('Human',   'Rat',     'human_rat_orthology.txt'),
    ('Human',   'Cattle',  'human_cattle_orthology.txt'),
    ('Human',   'Chicken', 'human_chicken_orthology.txt'),
    ('Human',   'Rabbit',  'human_rabbit_orthology.txt'),
    ('Mouse',   'Macaque', 'mouse_macaque_orthology.txt'),
    ('Mouse',   'Rat',     'mouse_rat_orthology.txt'),
    ('Mouse',   'Chicken', 'mouse_chicken_orthology.txt'),
    ('Mouse',   'Cattle',  'mouse_cattle_orthology.txt'),
    ('Mouse',   'Rabbit',  'mouse_rabbit_orthology.txt'),
    ('Rat',     'Macaque', 'rat_macaque_orthology.txt'),
    ('Rat',     'Chicken', 'rat_chicken_orthology.txt'),
    ('Rat',     'Cattle',  'rat_cattle_orthology.txt'),
    ('Rat',     'Rabbit',  'rat_rabbit_orthology.txt'),
    ('Macaque', 'Chicken', 'macaque_chicken_orthology.txt'),
    ('Macaque', 'Cattle',  'macaque_cattle_orthology.txt'),
    ('Macaque', 'Rabbit',  'macaque_rabbit_orthology.txt'),
    ('Chicken', 'Cattle',  'chicken_cattle_orthology.txt'),
    ('Chicken', 'Rabbit',  'chicken_rabbit_orthology.txt'),
    ('Cattle',  'Rabbit',  'cattle_rabbit_orthology.txt'),
]

# Cross-species conservation heatmaps (`*_top_quantiles`).
# `label` is the name passed to SpeciesData / load_ortho_table and therefore the
# axis label on the figure — the rat panel deliberately uses the BioMart column
# name, exactly as in analysis/SSMD.ipynb.
HEATMAP_PAIRS = {
    # min_mdiff is 0.0 for macaque and 0 elsewhere, as in the source cells — the
    # filter is identical, but the value is interpolated into the panel title.
    'human_macaque': dict(alt_dir='Macaque', label='Macaque',
                          ortho='human_macaque_orthology.txt', figsize=(4.15, 4.0),
                          min_mdiff=0.0),
    'human_mouse':   dict(alt_dir='Mouse',   label='Mouse',
                          ortho='human_mouse_orthology.txt',   figsize=(4.15, 4.0),
                          min_mdiff=0),
    'human_rabbit':  dict(alt_dir='Rabbit',  label='Rabbit',
                          ortho='human_rabbit_orthology.txt',  figsize=(4.15, 4.0),
                          min_mdiff=0),
    'human_rat':     dict(alt_dir='Rat',     label='Norway rat - BN/NHsdMcwi',
                          ortho='human_rat_orthology.txt',     figsize=(4.15, 4.0),
                          min_mdiff=0),
    'human_chicken': dict(alt_dir='Chicken', label='Chicken',
                          ortho='human_chicken_orthology.txt', figsize=(4.15, 4.0),
                          min_mdiff=0),
    'human_cattle':  dict(alt_dir='Cattle',  label='Cattle',
                          ortho='human_cattle_orthology.txt',  figsize=(4.15, 3.5),
                          min_mdiff=0),
}

# Example genes: (gene, tissue, species subset or None for all seven)
# PHC1 is plotted in Cerebellum for every species that has one; Cattle has no
# Cerebellum, so its panel comes from a separate Brain call.
EXAMPLE_PANELS = [
    ('PHC1',   'Cerebellum', ['Human', 'Macaque', 'Mouse', 'Rat', 'Rabbit', 'Chicken']),
    ('PHC1',   'Brain',      ['Cattle']),
    ('SYNGR1', 'Brain',      None),
    ('VRK3',   'Brain',      None),
]

TISSUES_CONS = ['Brain', 'Cerebellum', 'Liver', 'Heart', 'Testis']


# ================================================================
# Metadata + isoform annotation loading  (analysis/load_fig3_metadata.py)
# ================================================================

def _add_efficiency_factor(annot):
    """Flag isoforms whose retained introns make them likely degradation
    products (identical logic to analysis/load_fig3_metadata.py)."""
    efficiency_factor = []
    for idx, row in annot.iterrows():
        appris_ref = row.appris_ref
        aver = appris_ref.split('/')
        if 'RI:' in appris_ref:
            ri_factor = [x for x in aver if x.startswith('RI:')][0]
            ri_numbers = int(ri_factor.split(':')[1])
            appris_length = int(row.appris_transcript_length)

            if ((ri_numbers <= 2) and (appris_length >= 5)) or \
               ((ri_numbers <= 1) and (appris_length >= 3)):
                efficiency_factor.append(False)
            else:
                efficiency_factor.append(True)
        else:
            efficiency_factor.append(False)

    annot['efficiency_factor'] = efficiency_factor
    return annot


def _load_annot(path):
    annot = pd.read_csv(path, sep='\t')
    annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
    return _add_efficiency_factor(annot)


def load_metadata():
    """
    Load sample metadata and the per-species annotated isoform tables.

    Returns dict with 'species_metadata' and 'species_annotation', both keyed
    by species name.
    """
    metadata = pd.read_csv(
        '/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv', sep='\t')
    human_metadata = metadata[['entity:sample_id', 'tissue_id']]
    human_metadata.columns = ['Sample', 'Tissue']
    human_metadata.index = human_metadata.Sample

    cattle_metadata = pd.read_csv(
        '../code/config/cattleGTEx_metadata.txt', sep='\t', index_col=0)
    cattle_metadata.columns = ['Tissue', 'counts']

    cardoso = pd.read_csv('../code/config/cardoso_samples.tsv', sep='\t', index_col=0)

    species_annotation = {'Human': _load_annot(HUMAN_ANNOT)}
    for sp in ['Macaque', 'Mouse', 'Rat', 'Rabbit', 'Chicken', 'Cattle']:
        species_annotation[sp] = _load_annot(ISO_ANNOT.format(species=sp))

    species_metadata = {
        'Human':   human_metadata,
        'Macaque': cardoso.loc[cardoso.Species == 'Macaque'],
        'Mouse':   cardoso.loc[cardoso.Species == 'Mouse'],
        'Rat':     cardoso.loc[cardoso.Species == 'Rat'],
        'Rabbit':  cardoso.loc[cardoso.Species == 'Rabbit'],
        'Chicken': cardoso.loc[cardoso.Species == 'Chicken'],
        'Cattle':  cattle_metadata,
    }

    return {'species_metadata': species_metadata,
            'species_annotation': species_annotation}


# ================================================================
# SSMD results
# ================================================================

def load_ssmd_results(species_list=None):
    """Return {species: SSMDResults} for the PVal output directories."""
    from plot_ssmd_results import SSMDResults
    species_list = species_list or SPECIES_ALL
    return {sp: SSMDResults(f'{SSMD_BASE}/{sp}/SSMD/PVal') for sp in species_list}


def compute_significant_ssmd_barplot(species='Human', fdr=0.01, min_mdiff=0.05,
                                     drop_tissues=('Ovary',)):
    """
    Plot-ready data for `<Species>_significant_SSMD`: per-tissue counts of genes
    with a significant tissue-specific isoform, sorted descending.

    Source: analysis/SSMD.ipynb (Cell 26 — the loop over species that writes
    `{species}_significant_SSMD`).  Returns a DataFrame indexed by tissue.
    """
    from plot_ssmd_results import SSMDResults
    res = SSMDResults(f'{SSMD_BASE}/{species}/SSMD/PVal')
    df = res.summary(fdr=fdr, min_mdiff=min_mdiff)
    df = df.loc[[x for x in df.index if x not in drop_tissues]]
    return df.sort_values('n_sig', ascending=False)


# ================================================================
# Orthology table
# ================================================================

def build_ortho2(max_missing=2):
    """
    Build the multi-species orthology table and keep genes with orthologues in
    all but `max_missing` species (the `ortho2` of the source notebooks).
    """
    from cumulative_isoform_plot import build_ortho_table
    ortho_paths = {(sp1, sp2): f'{ORTHO_DIR}/{fn}' for sp1, sp2, fn in ORTHO_PAIRS}
    ortho_table = build_ortho_table(ortho_paths=ortho_paths, high_confidence=False)
    ortho2 = ortho_table.loc[ortho_table.isna().sum(axis=1) <= max_missing]
    return ortho_table, ortho2


# ================================================================
# Conservation of tissue-specific isoform usage (cumulative area plots)
# ================================================================

def load_species_sig_low(species, fdr=0.01, min_mdiff=0.05, q=0.5):
    """
    Significant / low-SSMD gene sets for one species, made mutually exclusive.
    Source: analysis/Fig3_SSMD_species.ipynb (Cell 5, `load_species`).
    """
    from plot_ssmd_results import SSMDResults
    res = SSMDResults(f'{SSMD_BASE}/{species}/SSMD/PVal')
    sig_ = res.sig_genes_by_quantile(fdr=fdr, min_mdiff=min_mdiff)
    low_ = res.low_ssmd_genes_by_quantile(q=q)
    sig, low = {}, {}
    for tissue in sig_.keys():
        sig[tissue] = sig_[tissue].difference(low_[tissue])
        low[tissue] = low_[tissue].difference(sig_[tissue])
    return sig, low


def isin_or_nan(series, values):
    return series.isin(values) | series.isna()


def isnotin(series, values):
    return ~series.isin(values)


def isnotin_obs(series, values):
    return (~series.isin(values)) & (~series.isna())


def get_tissue_cons_genes(tissue, ortho2, sig):
    """
    Human-referenced conservation categories for one tissue.
    Source: analysis/Fig3_SSMD_species.ipynb (Cell 52 — the `isnotin` version,
    which is the one `df_table` and the published figure are built from).

    Cattle has no Cerebellum, so it drops out of the Cerebellum filters.
    """
    use_cattle = tissue != 'Cerebellum'

    def _cattle(f):
        # No-op mask when Cattle is excluded (Cerebellum)
        return f(ortho2.Cattle, sig['Cattle'][tissue]) if use_cattle \
            else pd.Series(True, index=ortho2.index)

    human_only = set(ortho2.loc[
        ortho2.Human.isin(sig['Human'][tissue]) &
        isnotin(ortho2.Mouse,   sig['Mouse'][tissue]) &
        isnotin(ortho2.Rat,     sig['Rat'][tissue]) &
        isnotin(ortho2.Macaque, sig['Macaque'][tissue]) &
        isnotin(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isnotin) &
        isnotin(ortho2.Chicken, sig['Chicken'][tissue])
    ].Human)

    primates = set(ortho2.loc[
        ortho2.Human.isin(sig['Human'][tissue]) &
        isnotin(ortho2.Mouse,  sig['Mouse'][tissue]) &
        isnotin(ortho2.Rat,    sig['Rat'][tissue]) &
        ortho2.Macaque.isin(sig['Macaque'][tissue]) &
        isnotin(ortho2.Rabbit, sig['Rabbit'][tissue]) &
        _cattle(isnotin) &
        isnotin(ortho2.Chicken, sig['Chicken'][tissue])
    ].Human)

    mammals = set(ortho2.loc[
        ortho2.Human.isin(sig['Human'][tissue]) &
        isin_or_nan(ortho2.Mouse,   sig['Mouse'][tissue]) &
        isin_or_nan(ortho2.Rat,     sig['Rat'][tissue]) &
        isin_or_nan(ortho2.Macaque, sig['Macaque'][tissue]) &
        isin_or_nan(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isin_or_nan) &
        isnotin_obs(ortho2.Chicken, sig['Chicken'][tissue])
    ].Human)

    all_animals = set(ortho2.loc[
        ortho2.Human.isin(sig['Human'][tissue]) &
        isin_or_nan(ortho2.Mouse,   sig['Mouse'][tissue]) &
        isin_or_nan(ortho2.Rat,     sig['Rat'][tissue]) &
        isin_or_nan(ortho2.Macaque, sig['Macaque'][tissue]) &
        isin_or_nan(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isin_or_nan) &
        isin_or_nan(ortho2.Chicken, sig['Chicken'][tissue])
    ].Human)

    return human_only, primates, mammals, all_animals


def get_tissue_cons_genes_mouse(tissue, ortho2, sig):
    """
    Mouse-referenced conservation categories for one tissue.
    Source: analysis/Fig3_SSMD_species.ipynb (Cell 71).
    """
    use_cattle = tissue != 'Cerebellum'

    def _cattle(f):
        return f(ortho2.Cattle, sig['Cattle'][tissue]) if use_cattle \
            else pd.Series(True, index=ortho2.index)

    mouse_only = set(ortho2.loc[
        ortho2.Mouse.isin(sig['Mouse'][tissue]) &
        isnotin(ortho2.Human,   sig['Human'][tissue]) &
        isnotin(ortho2.Rat,     sig['Rat'][tissue]) &
        isnotin(ortho2.Macaque, sig['Macaque'][tissue]) &
        isnotin(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isnotin) &
        isnotin(ortho2.Chicken, sig['Chicken'][tissue])
    ].Mouse)

    mouse_rat = set(ortho2.loc[
        ortho2.Mouse.isin(sig['Mouse'][tissue]) &
        ortho2.Rat.isin(sig['Rat'][tissue]) &
        isnotin(ortho2.Human,   sig['Human'][tissue]) &
        isnotin(ortho2.Macaque, sig['Macaque'][tissue]) &
        isnotin(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isnotin) &
        isnotin(ortho2.Chicken, sig['Chicken'][tissue])
    ].Mouse)

    mouse_rat_rabbit = set(ortho2.loc[
        ortho2.Mouse.isin(sig['Mouse'][tissue]) &
        ortho2.Rat.isin(sig['Rat'][tissue]) &
        ortho2.Rabbit.isin(sig['Rabbit'][tissue]) &
        isnotin(ortho2.Human,   sig['Human'][tissue]) &
        isnotin(ortho2.Macaque, sig['Macaque'][tissue]) &
        _cattle(isnotin) &
        isnotin(ortho2.Chicken, sig['Chicken'][tissue])
    ].Mouse)

    mammals = set(ortho2.loc[
        ortho2.Mouse.isin(sig['Mouse'][tissue]) &
        isin_or_nan(ortho2.Rat,     sig['Rat'][tissue]) &
        isin_or_nan(ortho2.Human,   sig['Human'][tissue]) &
        isin_or_nan(ortho2.Macaque, sig['Macaque'][tissue]) &
        isin_or_nan(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isin_or_nan) &
        isnotin_obs(ortho2.Chicken, sig['Chicken'][tissue])
    ].Mouse)

    all_animals = set(ortho2.loc[
        ortho2.Mouse.isin(sig['Mouse'][tissue]) &
        isin_or_nan(ortho2.Rat,     sig['Rat'][tissue]) &
        isin_or_nan(ortho2.Human,   sig['Human'][tissue]) &
        isin_or_nan(ortho2.Macaque, sig['Macaque'][tissue]) &
        isin_or_nan(ortho2.Rabbit,  sig['Rabbit'][tissue]) &
        _cattle(isin_or_nan) &
        isin_or_nan(ortho2.Chicken, sig['Chicken'][tissue])
    ].Mouse)

    return mouse_only, mouse_rat, mouse_rat_rabbit, mammals, all_animals


def classify_genes(gene_sets, category):
    """
    Split one conservation category into tissue-specificity groups.
    Brain + Cerebellum are treated as a single group for the multi-tissue call.
    Source: analysis/Fig3_SSMD_species.ipynb (Cells 52 / 71).
    """
    other_tissues = ['Liver', 'Heart', 'Testis']

    brain      = gene_sets['Brain'][category]
    cerebellum = gene_sets['Cerebellum'][category]
    other      = {t: gene_sets[t][category] for t in other_tissues}
    all_other  = set.union(*other.values())

    brain_only       = (brain - cerebellum) - all_other
    cereb_only       = (cerebellum - brain) - all_other
    brain_cereb_only = (brain & cerebellum) - all_other

    results = {
        'Brain only':              len(brain_only),
        'Cerebellum only':         len(cereb_only),
        'Brain & Cerebellum only': len(brain_cereb_only),
    }
    for t in other_tissues:
        t_only = other[t] - brain - cerebellum - set.union(
            *[other[t2] for t2 in other_tissues if t2 != t])
        results[f'{t} only'] = len(t_only)

    def tissue_groups(gene):
        groups = []
        if gene in brain or gene in cerebellum:   # brain/cereb = one group
            groups.append('Brain/Cerebellum')
        for t in other_tissues:
            if gene in other[t]:
                groups.append(t)
        return groups

    all_genes = brain | cerebellum | all_other
    results['Multiple tissues'] = len(
        {g for g in all_genes if len(tissue_groups(g)) > 1})
    return results


def classify_gene_sets(gene_sets, category):
    """
    The gene SETS behind `classify_genes`, which returns only their sizes.

    Same partition, same Brain+Cerebellum-as-one-group rule for the multi-tissue
    call — this function is the reason the Figure 3B cells can be resolved back
    to gene lists, since only the counts are stored.
    """
    other_tissues = ['Liver', 'Heart', 'Testis']

    brain      = gene_sets['Brain'][category]
    cerebellum = gene_sets['Cerebellum'][category]
    other      = {t: gene_sets[t][category] for t in other_tissues}
    all_other  = set.union(*other.values())

    out = {
        'Brain only':              (brain - cerebellum) - all_other,
        'Cerebellum only':         (cerebellum - brain) - all_other,
        'Brain & Cerebellum only': (brain & cerebellum) - all_other,
    }
    for t in other_tissues:
        out[f'{t} only'] = other[t] - brain - cerebellum - set.union(
            *[other[t2] for t2 in other_tissues if t2 != t])

    def tissue_groups(gene):
        groups = []
        if gene in brain or gene in cerebellum:   # brain/cereb = one group
            groups.append('Brain/Cerebellum')
        for t in other_tissues:
            if gene in other[t]:
                groups.append(t)
        return groups

    all_genes = brain | cerebellum | all_other
    out['Multiple tissues'] = {g for g in all_genes if len(tissue_groups(g)) > 1}
    return out


def conservation_gene_table(ortho2, sig, reference='Human'):
    """
    {(tissue-specificity row, conservation category): set of genes} — the gene
    lists behind every cell of the Figure 3B stack.  `reference='Human'` gives
    human gene names, `'Mouse'` gives mouse ones.
    """
    if reference == 'Human':
        getter     = get_tissue_cons_genes
        categories = ['human_only', 'primates', 'mammals', 'all_animals']
        columns    = ['Human only', 'Primates', 'Mammals', 'All animals']
    else:
        getter     = get_tissue_cons_genes_mouse
        categories = ['mouse_only', 'mouse_rat', 'mouse_rat_rabbit',
                      'mammals', 'all_animals']
        columns    = ['Mouse only', 'Rodents', 'Gilres', 'Mammals', 'All animals']

    gene_sets = {}
    for tissue in TISSUES_CONS:
        gene_sets[tissue] = dict(zip(categories, getter(tissue, ortho2, sig)))

    out = {}
    for cat, col in zip(categories, columns):
        for row, genes in classify_gene_sets(gene_sets, cat).items():
            out[(row, col)] = genes
    return out


TISSUE_LABELS = [
    'Brain only',
    'Cerebellum only',
    'Brain & Cerebellum only',
    'Liver only',
    'Heart only',
    'Testis only',
    'Multiple tissues',
]


def build_conservation_table(ortho2, sig, reference='Human'):
    """
    Plot-ready table for the cumulative area plots: rows = tissue-specificity
    group, columns = conservation breadth.

    reference='Human' → `df_table`      (Fig3_SSMD_species.ipynb Cells 52-53)
    reference='Mouse' → `df_table_mouse`(Fig3_SSMD_species.ipynb Cell 71)
    """
    if reference == 'Human':
        getter     = get_tissue_cons_genes
        categories = ['human_only', 'primates', 'mammals', 'all_animals']
        columns    = ['Human only', 'Primates', 'Mammals', 'All animals']
    else:
        getter     = get_tissue_cons_genes_mouse
        categories = ['mouse_only', 'mouse_rat', 'mouse_rat_rabbit',
                      'mammals', 'all_animals']
        columns    = ['Mouse only', 'Rodents', 'Gilres', 'Mammals', 'All animals']

    gene_sets = {}
    for tissue in TISSUES_CONS:
        gene_sets[tissue] = dict(zip(categories, getter(tissue, ortho2, sig)))

    rows = {label: {} for label in TISSUE_LABELS}
    for cat in categories:
        res = classify_genes(gene_sets, cat)
        for label in TISSUE_LABELS:
            rows[label][cat] = res[label]

    df_table = pd.DataFrame(rows).T
    df_table.index.name = 'Tissue Specificity'
    df_table.columns = columns
    return df_table


# ================================================================
# Cross-species conservation heatmaps (`*_top_quantiles`)
# ================================================================

def compute_species_heatmap(pair_key, quantile=0.9, min_mdiff=None,
                            pval_cutoff=None):
    """
    Plot-ready match / enrichment matrices for one `human_<species>_top_quantiles`
    heatmap.  Source: analysis/SSMD.ipynb (Cells 153-158).

    Returns dict with match_df, enrich_df, tissue lists, axis labels and figsize.
    """
    from species_heatmap import (SpeciesData, load_ortho_table,
                                 compute_match_matrices, _DEFAULT_TISSUES)

    cfg = HEATMAP_PAIRS[pair_key]
    if min_mdiff is None:
        min_mdiff = cfg['min_mdiff']
    data1 = SpeciesData('Human', f'{SSMD_BASE}/Human/SSMD/PVal')
    data2 = SpeciesData(cfg['label'], f"{SSMD_BASE}/{cfg['alt_dir']}/SSMD/PVal")

    ortho = load_ortho_table(
        f"{ORTHO_DIR}/{cfg['ortho']}",
        'Human', cfg['label'],
        ref_genes=data1.ssmd.index, alt_genes=data2.ssmd.index,
    )

    tissue_list1 = [t for t in _DEFAULT_TISSUES.get('Human', _DEFAULT_TISSUES['_other'])
                    if t in data1.tissues]
    tissue_list2 = [t for t in _DEFAULT_TISSUES.get(cfg['label'], _DEFAULT_TISSUES['_other'])
                    if t in data2.tissues]

    match_df, _, enrich_df = compute_match_matrices(
        ortho, data1, data2, tissue_list1, tissue_list2,
        fdr_cutoff=None, pval_cutoff=pval_cutoff,
        quantile=quantile, min_mdiff=min_mdiff,
    )

    title = f"Human vs {cfg['label']}  (top-{int(quantile * 100)}% SSMD"
    title += f'  |Δmed|≥{min_mdiff})' if min_mdiff is not None else ')'

    return {
        'match_df':     match_df,
        'enrich_df':    enrich_df,
        'tissue_list1': tissue_list1,
        'tissue_list2': tissue_list2,
        'sp1':          'Human',
        'sp2':          cfg['label'],
        'figsize':      cfg['figsize'],
        'title':        title,
    }


# ================================================================
# Per-gene example panels (isoform structure + factor loadings)
# ================================================================

_gene_coord_cache: dict = {}


def _load_gene_coords(species):
    if species not in _gene_coord_cache:
        path = GENE_COORD_PATHS[species]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Gene coord index not found at {path}. '
                'Run plot_isoforms_lazy.setup_indexes() first.')
        df = pd.read_csv(path, sep='\t',
                         names=['gene_id', 'chrom', 'start', 'end'],
                         dtype={'start': int, 'end': int})
        _gene_coord_cache[species] = df.set_index('gene_id')
    return _gene_coord_cache[species]


def get_gene_coords(species, gene_id):
    coords = _load_gene_coords(species)
    if gene_id not in coords.index:
        raise KeyError(f"Gene '{gene_id}' not found in {species} coordinate index.")
    row = coords.loc[gene_id]
    return row['chrom'], int(row['start']), int(row['end'])


_el_header_cache: dict = {}


def _get_el_header(species):
    if species not in _el_header_cache:
        with gzip.open(EL_PATHS[species], 'rt') as fh:
            _el_header_cache[species] = fh.readline().rstrip('\n').split('\t')
    return _el_header_cache[species]


def _tabix_fetch(path, chrom, start, end):
    import pysam
    tbx = pysam.TabixFile(path)
    rows = []
    try:
        for rec in tbx.fetch(chrom, start, end):
            rows.append(rec.split('\t'))
    except ValueError:
        pass
    finally:
        tbx.close()
    return rows


def get_EL_for_gene(species, gene_id):
    """Fetch the isoform expression-loadings rows for one gene via tabix."""
    chrom, start, end = get_gene_coords(species, gene_id)
    header = _get_el_header(species)
    rows = _tabix_fetch(EL_PATHS[species], chrom, start, end)
    if not rows:
        return pd.DataFrame(columns=header)

    df = pd.DataFrame(rows, columns=header)
    gid_col = 'gid' if 'gid' in df.columns else header[3]
    df = df.loc[df[gid_col] == gene_id].copy()

    sample_cols = df.columns[6:]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors='coerce')
    df['pid'] = df[gid_col] + '.' + df['pid']
    df.index = df['pid']
    return df


def get_iso_for_gene(species, gene_id):
    """Fetch the isoform exon BED rows for one gene via tabix."""
    chrom, start, end = get_gene_coords(species, gene_id)
    iso_cols = ['chrom', 'start', 'end', 'gene_id', 'transcript_id',
                'strand', 'factors', 'exon', 'notes']
    rows = _tabix_fetch(ISO_PATHS[species], chrom, start, end)
    if not rows:
        return pd.DataFrame(columns=iso_cols)

    df = pd.DataFrame(rows, columns=iso_cols)
    df = df.loc[df['gene_id'] == gene_id].copy()
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['transcript_id'] = df['gene_id'] + '.' + df['transcript_id']
    return df


def _rolling_median(arr, window):
    arr = np.asarray(arr)
    return np.array([np.median(arr[i:i + window])
                     for i in range(len(arr) - window + 1)])


# Per-species sample-loading colour lists, keyed by the sorted tissue order the
# metadata produces (verbatim from plot_isoforms_lazy.plot_isoforms_and_loadings_lazy)
LOADING_TISSUE_COLORS = {
    'Human':  ['#FFA54F', '#8B1C62', '#EEEE00', '#EEEE00', '#EEEE00',
               '#EEEE00', '#EEEE00', '#00CDCD', '#B452CD', '#7A378B',
               '#CDB79E', '#9ACD32', '#7A67EE', '#CD9B1D', '#3A5FCD',
               '#1E90FF', '#A6A6A6', '#FF00FF'],
    'Cattle': ['#FF00FF', '#EEEE00', '#CDB79E', '#B452CD', '#EEEE00', '#CDB79E',
               '#9ACD32', '#00CDCD', '#7A67EE', '#3A5FCD', '#FFA54F', '#A6A6A6'],
    '_other': ['#EEEE00', '#EEEE00', '#B452CD', '#CDB79E', '#CDB79E',
               '#FFB6C1', '#A6A6A6'],
}


def _compute_coverage(species, gene, iso, xlim, strand,
                      tissue_keys, plot_cerebellum=True, window=20):
    """
    Read the per-gene count matrix and build the smoothed per-tissue coverage
    tracks.  Extracted verbatim from plot_isoforms_lazy.plot_isoforms_species so
    that the plotting function itself touches no files.
    """
    if species in COUNT_MATRIX_PATHS:
        cm_path = COUNT_MATRIX_PATHS[species].format(gene=gene)
    else:
        cm_path = COUNT_MATRIX_PATH_OTHER.format(species=species, gene=gene)

    use_cerebellum = (species not in ('Cattle',)) and plot_cerebellum
    if not use_cerebellum:
        tissue_keys = [x for x in tissue_keys if x != 'cerebellum']

    y, ns = {}, {}
    with gzip.open(cm_path, 'rb') as fh:
        coords = fh.readline().decode().rstrip().split(',')[1:]
        x = [int(c.split(':')[1]) for c in coords]

        for k in tissue_keys:
            y[k] = np.zeros(len(x))
            ns[k] = 0

        for line in fh:
            linea = line.decode().rstrip().split(',')
            label = linea[0]
            vals = np.array(linea[1:]).astype(int)

            # Every branch is guarded on `tissue_keys`: the upstream helper always
            # passed all five keys, so its Cattle and cerebellum branches were
            # unguarded and raise KeyError on a subset of tracks.
            if species == 'Cattle':
                if ('CNS' in label or 'Hypothalamus' in label) and ('brain' in tissue_keys):
                    y['brain'] += vals;  ns['brain'] += 1
                elif ('Heart'  in label) and ('heart'  in tissue_keys):
                    y['heart']  += vals;  ns['heart']  += 1
                elif ('Liver'  in label) and ('liver'  in tissue_keys):
                    y['liver']  += vals;  ns['liver']  += 1
                elif ('Testis' in label) and ('testis' in tissue_keys):
                    y['testis'] += vals;  ns['testis'] += 1
                # Cattle mammary samples are labelled 'Mammary_Gland'
                elif ('Mammary' in label) and ('mammary' in tissue_keys):
                    y['mammary'] += vals;  ns['mammary'] += 1
            else:
                if ('Brain' in label and 'Cereb' not in label) and ('brain' in tissue_keys):
                    y['brain'] += vals;  ns['brain'] += 1
                elif ('Cereb' in label) and use_cerebellum and ('cerebellum' in tissue_keys):
                    y['cerebellum'] += vals;  ns['cerebellum'] += 1
                elif ('Heart'  in label) and ('heart'  in tissue_keys):
                    y['heart']  += vals;  ns['heart']  += 1
                elif ('Liver'  in label) and ('liver'  in tissue_keys):
                    y['liver']  += vals;  ns['liver']  += 1
                elif ('Testis' in label) and ('testis' in tissue_keys):
                    y['testis'] += vals;  ns['testis'] += 1
                elif ('Mammary' in label) and ('mammary' in tissue_keys):
                    y['mammary'] += vals;  ns['mammary'] += 1

    for k in tissue_keys:
        if ns[k] > 0:
            y[k] = np.convolve(y[k] / ns[k], np.ones(window, dtype=int), 'valid')
        else:
            y[k] = np.zeros(max(len(x) - window + 1, 0))

    if strand == '-':
        x = np.array(x)
        x = xlim[0] + xlim[1] - x
    x = _rolling_median(x, window)

    return x, y, sorted(tissue_keys), ns


# ================================================================
# Which two isoforms to highlight
# ================================================================

# Some cardoso species carry a literal 'Brain,Cerebellum' tissue.  The coverage
# tracks already fold it into cerebellum ("'Cereb' in label"), and the per-species
# colour lists have no slot for it — left separate it shifts every colour after it
# and drops the last tissue.  Group it the same way wherever tissues are listed.
TISSUE_MERGE = {'Brain,Cerebellum': 'Cerebellum'}

# The two isoforms to colour, chosen by eye from the structure plots in
# Figure3.ipynb.  Positions are 1-BASED indices into `iso_order` as drawn there.
# `'max_heart'` picks the isoform with the highest median loading in heart.
ISOFORM_PICKS = {
    ('PHC1',   'Human'):   (1, 3),
    ('PHC1',   'Macaque'): (1, 4),
    ('PHC1',   'Mouse'):   (1, 'max_heart'),   # resolves to position 5
    ('PHC1',   'Rat'):     (1, 2),
    ('PHC1',   'Rabbit'):  (3, 4),
    ('PHC1',   'Chicken'): (1, 3),
    ('PHC1',   'Cattle'):  (1, 4),

    ('SYNGR1', 'Human'):   (1, 3),
    ('SYNGR1', 'Macaque'): (1, 3),
    ('SYNGR1', 'Mouse'):   (1, 3),
    ('SYNGR1', 'Rat'):     (1, 4),
    ('SYNGR1', 'Rabbit'):  (1, 4),
    ('SYNGR1', 'Chicken'): (1, 3),
    ('SYNGR1', 'Cattle'):  (1, 4),

    ('VRK3',   'Human'):   (2, 3),   # cerebellum isoform_8, cerebellum median 0.62
    # iso_order for VRK3 is built from Brain SSMD, so position 1 IS the brain
    # top isoform; position 2 is the cerebellum-driven one (median 0.61 in
    # cerebellum, 0.00 in brain).  Macaque is shown on brain, human on cerebellum.
    ('VRK3',   'Macaque'): (1, 5),
    ('VRK3',   'Rat'):     (1, 3),
    ('VRK3',   'Cattle'):  (2, 4),
}

ISO_RED, ISO_NAVY, ISO_OTHER = 'crimson', 'navy', 'gray'
ISO_MID = '#0099DD'      # deep sky blue — see the colour note in the notebook

# An optional THIRD highlighted isoform, drawn orange between the red and the
# navy.  Used where a panel needs both the brain-top and the cerebellum-top
# isoform visible: whichever of the pair is not already the red one goes here.
# Positions are 1-BASED indices into `iso_order`, as in ISOFORM_PICKS.
# Currently empty — every panel highlights just two isoforms.  Adding an entry
# here is all it takes to bring the orange one back for a given (gene, species):
#   ('VRK3', 'Human'): 1, ('VRK3', 'Macaque'): 2
ISOFORM_MID = {}


def merged_tissue_ids(panel):
    """Per-sample tissue ids with 'Brain,Cerebellum' folded into Cerebellum."""
    EL = panel['el_annot']
    idx = (EL.index.intersection(pd.Index(panel['sublist']))
                   .intersection(panel['meta'].index))
    return panel['meta'].loc[idx, 'tissue_id'].replace(TISSUE_MERGE)


# The sample-table tissue id standing for each role, per species.  Human panels
# carry GTEx ids, so a bare 'Heart' matches nothing there; the id used is the one
# that role's loadings column displays, so a 'max_<role>' isoform is visibly the
# highest point in its own column.
ROLE_TISSUE_ID = {
    'Human':  {'Brain':      'Brain_Anterior_cingulate_cortex_BA24',
               'Cerebellum': 'Brain_Cerebellar_Hemisphere',
               'Heart':      'Heart_Atrial_Appendage',
               'Liver':      'Liver',
               'Testis':     'Testis'},
    'Cattle': {'Brain': 'CNS', 'Cerebellum': None, 'Heart': 'Heart',
               'Liver': 'Liver', 'Testis': 'Testis'},
    '_other': {'Brain': 'Brain', 'Cerebellum': 'Cerebellum', 'Heart': 'Heart',
               'Liver': 'Liver', 'Testis': 'Testis'},
}


def role_tissue(panel, role):
    """Sample-table tissue id for `role` in this panel's species."""
    m = ROLE_TISSUE_ID.get(panel['species'], ROLE_TISSUE_ID['_other'])
    return m.get(role)


def heart_tissue(panel):
    return role_tissue(panel, 'Heart')


def _highest_median_in(panel, tissue, exclude=()):
    """iso_order index whose median loading in `tissue` is highest."""
    EL = panel['el_annot']
    ids = merged_tissue_ids(panel)
    samples = ids.index[ids == tissue]
    best, best_med = None, -np.inf
    for i, factor in enumerate(panel['iso_order']):
        if i in exclude:
            continue
        vals = EL.loc[samples, factor].dropna()
        med = float(np.median(vals)) if len(vals) else -np.inf
        if med > best_med:
            best, best_med = i, med
    return best


def _exons_in_order(panel, transcript_id):
    """Exons of one transcript in transcription order."""
    r = panel['iso']
    r = r[r.transcript_id == transcript_id][['start', 'end']].values
    r = r[np.argsort(r[:, 0])]
    return r[::-1] if panel['strand'] == '-' else r


def cassette_isoforms(panel, lo=0.0, hi=0.35, min_frac=0.5, skip_first=True):
    """
    iso_order indices carrying a cassette exon whose RELATIVE position along the
    transcript falls in (lo, hi] — an exon with no overlap at all in at least
    `min_frac` of the other isoforms.

    Relative rather than absolute position, because these genes are annotated
    with 6 to 20 exons depending on species: "the second exon" and "the middle
    exon" are not comparable across them as integers.  Exon 1 is skipped, since
    alternative transcription starts would otherwise read as cassettes.
    """
    order = panel['iso_order']
    ex = {t: _exons_in_order(panel, t) for t in order}
    out = []
    for i, t in enumerate(order):
        others = [o for o in order if o != t]
        n = len(ex[t])
        if not others or n < 2:
            continue
        for k, (s_, e_) in enumerate(ex[t]):
            if skip_first and k == 0:
                continue
            rel = (k + 1) / n
            if not (lo < rel <= hi):
                continue
            absent = sum(1 for o in others
                         if not any(s_ < e2 and s2 < e_ for s2, e2 in ex[o]))
            if absent >= min_frac * len(others):
                out.append(i)
                break
    return out


CASSETTE_WINDOW = {'cassette5': (0.0, 0.35),      # 5'-proximal
                   'cassettemid': (0.35, 0.85),   # mid-gene
                   'cassette': (0.0, 0.35)}       # back-compatible alias


def cassette5_isoforms(panel, positions=(2, 3, 4), min_frac=0.5):
    """
    iso_order indices of isoforms carrying a 5'-proximal CASSETTE exon: an exon
    at one of `positions` (1-based, transcription order) that has no overlap at
    all in at least `min_frac` of the other isoforms.

    The fraction matters.  Nearly every isoform has a first exon that differs
    from one or two others — alternative transcription starts, not cassettes —
    so requiring the exon to be absent from most of the gene's isoforms is what
    separates a genuine skipped exon from a shifted boundary.  Exon 1 is
    excluded for the same reason.
    """
    order = panel['iso_order']
    ex = {t: _exons_in_order(panel, t) for t in order}
    out = []
    for i, t in enumerate(order):
        others = [o for o in order if o != t]
        if not others:
            continue
        for pos in positions:
            if pos > len(ex[t]):
                continue
            s_, e_ = ex[t][pos - 1]
            absent = sum(1 for o in others
                         if not any(s_ < e2 and s2 < e_ for s2, e2 in ex[o]))
            if absent >= min_frac * len(others):
                out.append(i)
                break
    return out


def _introns_in_order(panel, transcript_id):
    """Introns of one transcript, in transcription order."""
    e = _exons_in_order(panel, transcript_id)
    return [tuple(sorted((e[i][1], e[i + 1][0]) if panel['strand'] == '+'
                         else (e[i + 1][1], e[i][0])))
            for i in range(len(e) - 1)]


def retained_intron_isoform(panel, red, exclude=(), after_longest=True):
    """
    The isoform that RETAINS the first intron, at or after the longest one in
    transcription order, which the RED isoform splices.

    Anchoring on red's own introns is the point: a retained intron somewhere
    else in the gene gives a navy isoform that shows nothing about the event red
    illustrates.

    Transcription order is strand-aware: on the minus strand "after" means lower
    coordinates, so a plain coordinate comparison picks the wrong intron.
    Of the retaining isoforms the one with the highest median loading outside
    testis is returned — the retention is what shows in the other tissues.
    """
    order = panel['iso_order']
    ex = {t: _exons_in_order(panel, t) for t in order}
    introns = _introns_in_order(panel, order[red])      # red's own, in order
    if not introns:
        return None
    longest = max(introns, key=lambda k: k[1] - k[0])
    start = introns.index(longest) if after_longest else 0

    ids = merged_tissue_ids(panel)
    EL = panel['el_annot']

    def med(i, role):
        tis = role_tissue(panel, role)
        smp = ids.index[ids == tis] if tis else []
        v = EL.loc[smp, order[i]].dropna() if len(smp) else []
        return float(np.median(v)) if len(v) else -np.inf

    # Prefer an intron at or after the longest, but fall back to the whole gene:
    # in cattle the only retained intron sits upstream of the longest one.
    for lo in ((start, 0) if start else (0,)):
        for a, b in introns[lo:]:
            retainers = [i for i, t in enumerate(order)
                         if i not in exclude and i != red
                         and any(s_ <= a and e_ >= b for s_, e_ in ex[t])]
            if retainers:
                return max(retainers,
                           key=lambda i: max(med(i, 'Heart'), med(i, 'Brain')))
    return None


def _resolve_noncassette(panel, sentinel, gene, exclude=()):
    """'noncassette_<role>': highest in that role among isoforms WITHOUT a 5'
    cassette exon — the partner that makes the cassette visible as a
    difference, rather than an isoform that also carries it."""
    role = sentinel.split('_', 1)[1].capitalize()
    tissue = role_tissue(panel, role)
    if tissue is None:
        raise ValueError(f"{gene}/{panel['species']}: no tissue for role {role!r}")
    cass = set(cassette5_isoforms(panel))
    cands = [i for i in range(len(panel['iso_order']))
             if i not in cass and i not in exclude]
    if not cands:
        cands = [i for i in range(len(panel['iso_order'])) if i not in exclude]
    ids = merged_tissue_ids(panel)
    samples = ids.index[ids == tissue]
    EL = panel['el_annot']

    def med(i):
        v = EL.loc[samples, panel['iso_order'][i]].dropna()
        return float(np.median(v)) if len(v) else -np.inf

    return max(cands, key=med)


def _resolve_cassette(panel, sentinel, gene, exclude=()):
    """'cassette5_<role>' / 'cassettemid_<role>': of the isoforms carrying a
    cassette exon in that part of the gene, the one highest in the role's
    tissue."""
    kind, role = sentinel.split('_', 1)
    role = role.capitalize()
    lo, hi = CASSETTE_WINDOW[kind]
    tissue = role_tissue(panel, role)
    if tissue is None:
        raise ValueError(f"{gene}/{panel['species']}: no tissue for role {role!r}")
    cands = [i for i in cassette_isoforms(panel, lo, hi) if i not in exclude]
    if not cands:
        raise ValueError(f"{gene}/{panel['species']}: no isoform carries a "
                         f'cassette exon in {sentinel!r}')
    ids = merged_tissue_ids(panel)
    samples = ids.index[ids == tissue]
    EL = panel['el_annot']

    def med(i):
        v = EL.loc[samples, panel['iso_order'][i]].dropna()
        return float(np.median(v)) if len(v) else -np.inf

    return max(cands, key=med)


def _resolve_max(panel, sentinel, gene, exclude=()):
    """
    'max_<role>' -> iso_order index with the highest median loading in that
    role's tissue, skipping `exclude` (so navy falls through when the top
    isoform there is already red).
    """
    role = sentinel[4:].capitalize()
    tissue = role_tissue(panel, role)
    if tissue is None:
        raise ValueError(f"{gene}/{panel['species']}: no tissue for role "
                         f'{role!r}, cannot resolve {sentinel!r}')
    idx = _highest_median_in(panel, tissue, exclude=exclude)
    if idx is None:
        raise ValueError(f"{gene}/{panel['species']}: no samples for "
                         f'{tissue!r}, cannot resolve {sentinel!r}')
    return idx


def resolve_picks(panel, gene):
    """
    (red index, navy index) into iso_order, 0-based.

    Genes without a hand-curated entry fall back to the first two of `iso_order`
    — the two highest-SSMD isoforms in the panel's tissue, which is what the
    upstream code always used.  That is a starting point, not a curated choice:
    for the three main-figure genes the picks were chosen by eye instead.
    """
    red, navy = ISOFORM_PICKS.get((gene, panel['species']), (1, 2))

    # Red may itself be a 'max_<role>' sentinel; it is resolved first, with no
    # exclusion, so navy can then be resolved against it.
    if isinstance(red, str) and red.startswith('noncassette_'):
        red = _resolve_noncassette(panel, red, gene)
    elif isinstance(red, str) and red.split('_', 1)[0] in CASSETTE_WINDOW:
        red = _resolve_cassette(panel, red, gene)
    elif isinstance(red, str) and red.startswith('max_'):
        red = _resolve_max(panel, red, gene)
    else:
        red = red - 1

    if isinstance(navy, str) and navy.startswith('max_'):
        # 'max_<role>': the isoform with the highest median loading in that
        # role's tissue.  Excluding red means the rule falls through to the next
        # highest when the top isoform there is already the red one.
        navy = _resolve_max(panel, navy, gene, exclude=(red,))
    elif isinstance(navy, str) and navy.split('_', 1)[0] in CASSETTE_WINDOW:
        navy = _resolve_cassette(panel, navy, gene, exclude=(red,))
    elif navy == 'retained_intron':
        navy = retained_intron_isoform(panel, red, exclude=(red,))
        if navy is None:
            # Not every species annotates a retained-intron isoform; fall back
            # to the ordinary heart pick rather than dropping the species.
            navy = _resolve_max(panel, 'max_heart', gene, exclude=(red,))
    else:
        navy = navy - 1
    n = len(panel['iso_order'])
    if not (0 <= red < n and 0 <= navy < n):
        raise IndexError(f"{gene}/{panel['species']}: picks ({red + 1}, {navy + 1}) "
                         f'outside the {n} isoforms in iso_order')
    return red, navy


def resolve_mid(panel, gene):
    """iso_order index of the orange isoform, or None if this panel has none."""
    pos = ISOFORM_MID.get((gene, panel['species']))
    if pos is None:
        return None
    idx = pos - 1
    if not 0 <= idx < len(panel['iso_order']):
        raise IndexError(f"{gene}/{panel['species']}: mid pick {pos} outside the "
                         f"{len(panel['iso_order'])} isoforms in iso_order")
    red, navy = resolve_picks(panel, gene)
    if idx in (red, navy):
        raise ValueError(f"{gene}/{panel['species']}: mid pick {pos} is already "
                         f'the red ({red + 1}) or navy ({navy + 1}) isoform')
    return idx


def n_picked(panel, gene):
    """How many isoforms this panel highlights: 2, or 3 where there is a mid."""
    return 3 if resolve_mid(panel, gene) is not None else 2


def picked_isoforms(panel, gene, other_color=ISO_OTHER):
    """
    (isoforms, colours) with the chosen isoforms first — red, then orange where
    the panel has one, then navy — and the rest in their original order.  Use
    this where only the chosen isoforms are drawn, and for the loadings rows.
    """
    order = list(panel['iso_order'])
    red, navy = resolve_picks(panel, gene)
    mid = resolve_mid(panel, gene)
    chosen = [red] + ([mid] if mid is not None else []) + [navy]
    colors = [ISO_RED] + ([ISO_MID] if mid is not None else []) + [ISO_NAVY]
    rest = [x for i, x in enumerate(order) if i not in chosen]
    return ([order[i] for i in chosen] + rest,
            colors + [other_color] * len(rest))


def picked_colors_in_place(panel, gene, other_color=ISO_OTHER):
    """
    Colours for iso_order *in its original order*, with only the chosen isoforms
    coloured.  Use this where every isoform is still drawn.
    """
    red, navy = resolve_picks(panel, gene)
    mid = resolve_mid(panel, gene)
    out = []
    for i in range(len(panel['iso_order'])):
        out.append(ISO_RED if i == red else
                   ISO_MID if i == mid else
                   ISO_NAVY if i == navy else other_color)
    return out


# The loadings columns are sample-level tissue ids; the SSMD tables use a coarser
# vocabulary.  This maps a loadings role to the tissue name the SSMD/FDR tables
# are keyed by, per species.
ROLE_SSMD_TISSUE = {
    'Human':  {'Brain': 'Brain', 'Cerebellum': 'Cerebellum', 'Heart': 'Heart',
               'Mammary': 'Breast_Mammary_Tissue', 'Liver': 'Liver',
               'Testis': 'Testis'},
    'Cattle': {'Brain': 'Brain', 'Cerebellum': None, 'Heart': 'Heart',
               'Mammary': 'Mammary_Gland', 'Liver': 'Liver', 'Testis': 'Testis'},
    '_other': {'Brain': 'Brain', 'Cerebellum': 'Cerebellum', 'Heart': 'Heart',
               'Mammary': None, 'Liver': 'Liver', 'Testis': 'Testis'},
}


def attach_ssmd_stats(panels, roles=('Brain', 'Cerebellum', 'Heart',
                                     'Mammary', 'Liver', 'Testis')):
    """
    Attach the gene-level SSMD statistics to each prepared panel, as
    `panel['ssmd_stats'][role] = {'ssmd', 'fdr', 'pval', 'argmax'}`.

    The p-value is per (gene, tissue), not per isoform — it is
    `1 - Phi(max_Z)^m` over the gene's isoforms — so `argmax` records which
    isoform it actually describes.  Callers should check that against the
    isoform they highlight before quoting the number.
    """
    res = load_ssmd_results()
    for gene, by_species in panels.items():
        for species, panel in by_species.items():
            r = res[species]
            gid = panel['gene']
            m = ROLE_SSMD_TISSUE.get(species, ROLE_SSMD_TISSUE['_other'])
            stats = {}
            for role in roles:
                t = m.get(role)
                if t is None or t not in r.tissues or gid not in r.pval.index:
                    continue
                stats[role] = {
                    'ssmd':   float(r.obs_ssmd.loc[gid, t]),
                    'pval':   float(r.pval.loc[gid, t]),
                    'fdr':    float(r.fdr.loc[gid, t]),
                    'argmax': (r.argmax_iso.loc[gid, t]
                               if gid in r.argmax_iso.index else None),
                    'tissue': t,
                }
            panel['ssmd_stats'] = stats
    return panels


def prepare_gene_panel(species, gene_id, tissue, metadata, annot, ssmd_results,
                       gene_view=(0, 1),
                       tissue_keys=('brain', 'cerebellum', 'heart', 'liver', 'testis')):
    """
    Produce everything needed to draw the `<GENE>.<Species>.<Tissue>.gene` and
    `.loadings` panels, with no further computation or file access.

    Extracted from plot_isoforms_lazy.plot_isoforms_and_loadings_lazy (steps 1-5)
    plus the coverage/track set-up of plot_isoforms_species.

    Returns None when the gene has no data for this species/tissue — exactly the
    conditions under which the original function printed `[skip]` and returned.
    """
    tissue_keys = list(tissue_keys)

    el = get_EL_for_gene(species, gene_id)
    iso = get_iso_for_gene(species, gene_id)
    if el.empty or iso.empty:
        print(f'  [skip] No data for {gene_id} in {species}')
        return None

    # --- per-isoform SSMD from the pre-loaded tables ---
    gid_col = ssmd_results.ssmd_iso.columns[0]
    gene_rows = ssmd_results.ssmd_iso.loc[ssmd_results.ssmd_iso[gid_col] == gene_id]
    if gene_rows.empty or tissue not in gene_rows.columns:
        print(f'  [skip] {gene_id}/{tissue} not found in ssmd_iso')
        return None

    ssmd_vals = gene_rows[tissue].dropna()
    if ssmd_vals.empty:
        print(f'  [skip] No SSMD values for {gene_id}/{tissue}')
        return None

    ssmd = pd.DataFrame({
        'pid':         ssmd_vals.index,
        'ssmd':        ssmd_vals.values,
        'median_diff': float('nan'),
        'gene':        gene_id,
    })
    ssmd['abs_ssmd'] = ssmd['ssmd'].abs()

    # --- isoform order and colours ---
    annot_gene = annot.loc[(annot.gene == gene_id) & (~annot.efficiency_factor)]
    if annot_gene.shape[0] <= 1:
        annot_gene = annot.loc[(annot.gene == gene_id)]
    ssmd = ssmd.loc[ssmd.pid.isin(annot_gene.transcript)]
    top_iso    = ssmd.sort_values('ssmd', ascending=False).iloc[0].pid
    bottom_iso = ssmd.sort_values('ssmd', ascending=False).iloc[1].pid

    if species == 'Human':
        iso = iso.loc[iso.transcript_id.isin(annot_gene.transcript)]
        iso_order = [top_iso, bottom_iso]
        colores   = ['crimson', 'navy']

        for _, isoform in annot_gene.iterrows():
            if isoform.transcript in (top_iso, bottom_iso):
                continue
            if any(tag in isoform.appris_ref or tag in isoform.gencode_ref
                   for tag in ('intron_chain', 'ref_transcript')):
                colores.append('dimgray')
                iso_order.append(isoform.transcript)

        for _, isoform in annot_gene.iterrows():
            if isoform.transcript not in iso_order:
                iso_order.append(isoform.transcript)
                colores.append('darkgray')
    else:
        other_iso = [p for p in ssmd.sort_values('ssmd', ascending=False).pid
                     if p not in (top_iso, bottom_iso)]
        iso_order = [top_iso, bottom_iso] + other_iso
        colores   = ['crimson', 'navy'] + ['gray'] * len(other_iso)

    # --- gene-structure panel geometry + coverage tracks ---
    gene_iso = iso.loc[iso.gene_id == gene_id]
    strand = gene_iso.strand.iloc[0]
    xlen = gene_iso.end.max() - gene_iso.start.min()
    xpad = xlen / 100
    full_start = gene_iso.start.min() - xpad
    full_end   = gene_iso.end.max() + xpad
    W = full_end - full_start
    xlim = (full_start + gene_view[0] * W, full_start + gene_view[1] * W)

    x, y, track_order, track_n = _compute_coverage(
        species, gene_id, iso, xlim, strand, tissue_keys)

    # --- factor-loadings panel ---
    el_annot = el[el.columns[6:]].T
    el_annot.columns = list(el['pid'])

    if species in ('Human', 'Cattle'):
        meta = metadata.loc[el.columns[6:]].copy()
    else:
        meta = metadata.loc[metadata.Species == species].copy()
    meta['tissue_id'] = meta['Tissue']
    tissue_colors = LOADING_TISSUE_COLORS.get(species, LOADING_TISSUE_COLORS['_other'])

    return {
        'species':       species,
        'gene':          gene_id,
        'tissue':        tissue,
        # gene-structure panel
        'iso':           iso,
        'iso_order':     iso_order,
        'colores':       colores,
        'strand':        strand,
        'xlim':          xlim,
        'tissues_n':     len(tissue_keys),
        'track_order':   track_order,
        'track_n':       track_n,      # samples averaged per track; 0 = no data
        'x':             x,
        'y':             y,
        # loadings panel
        'el_annot':      el_annot,
        'meta':          meta,
        'sublist':       el.columns[6:],
        'tissue_colors': tissue_colors,
    }


def get_species_ortho(ortho_table, gene, species):
    """Human gene name → orthologue in `species` (None when absent)."""
    try:
        species_gene = str(ortho_table.loc[ortho_table.Human == gene, species].iloc[0])
        if species_gene == 'nan':
            return None
    except Exception:
        return None
    return species_gene


def prepare_gene_panels(gene, tissue, ortho_table, species_metadata,
                        species_annotation, species_res, species_list=None):
    """
    Prepare the example panels for one gene across species.
    Mirrors `plot_gene_species` in analysis/Fig3_examples_final.ipynb, including
    its silent skipping of species where the gene/tissue combination is absent.
    """
    species_list = species_list or SPECIES_ALL
    panels = {}
    for species in species_list:
        species_gene = gene if species == 'Human' \
            else get_species_ortho(ortho_table, gene, species)
        if species_gene is None:
            continue
        try:
            panel = prepare_gene_panel(
                species, species_gene, tissue,
                species_metadata[species], species_annotation[species],
                species_res[species],
            )
        except Exception as err:
            print(f'  [skip] {species_gene} / {species}: {err}')
            continue
        if panel is not None:
            panels[species] = panel
    return panels


# ================================================================
# Panels for the assembled publication figure
# ================================================================

# The assembled figure shows a hand-picked subset of coverage tracks per gene,
# rather than the five the exploratory helper always drew, and only the top two
# isoforms (three for VRK3 in human).  `tracks` is the superset to compute;
# `show` is the preferred display order, filtered at plot time to tracks that
# actually have samples.
FIGURE_PANELS = {
    'PHC1': dict(
        tissue='Cerebellum', tissue_cattle='Brain',
        tracks=['brain', 'cerebellum', 'heart', 'testis'],
        show=['brain', 'cerebellum', 'heart', 'testis'],
        n_iso=2,
        species=['Human', 'Macaque', 'Rabbit', 'Mouse', 'Rat', 'Chicken', 'Cattle'],
    ),
    'SYNGR1': dict(
        tissue='Brain', tissue_cattle='Brain',
        tracks=['brain', 'heart', 'mammary', 'testis'],
        show=['brain', 'heart', 'mammary', 'testis'],
        n_iso=2,
        species=['Human', 'Mouse', 'Cattle', 'Macaque', 'Rat', 'Rabbit', 'Chicken'],
    ),
    'VRK3': dict(
        tissue='Brain', tissue_cattle='Brain',
        tracks=['brain', 'cerebellum', 'heart', 'testis'],
        show=['cerebellum', 'heart', 'testis'],
        show_fallback=['brain', 'heart', 'testis'],   # Cattle has no cerebellum
        n_iso=2, n_iso_human=3,
        species=['Human', 'Macaque', 'Rat', 'Cattle'],
    ),
}


# ================================================================
# Supplementary example genes
# ================================================================

# Tissue is the one the manuscript makes its claim about, which is not always
# the top-SSMD tissue (e.g. ATG13 is a liver story but ranks highest in testis).
# It sets the isoform ranking, so it matters.
#
# `star`: 'last' / 'first' mark the alternative terminal exon; None means the
# mechanism is a cassette exon, intron retention, a UTR change or a splicing
# efficiency change, where marking a terminal exon would point at the wrong
# feature.
SUPP_FIGURES = {
    'PHC1':    dict(tissue='Cerebellum', star='first',
                    title='Conserved brain-specific alternative promoter in'),
    'LAMP2':   dict(tissue='Brain', star='last',
                    title='Conserved brain-specific alternative last exon in'),
    'RUFY3':   dict(tissue='Brain', star='last',
                    title='Conserved brain-specific alternative last exon in'),
    'TSC22D4': dict(tissue='Brain', star='first',
                    title='Conserved brain-specific alternative first exon in'),
    'CLTA':    dict(tissue='Brain', star=None,
                    title='Conserved brain-specific cassette exon in'),
    'CLTB':    dict(tissue='Brain', star=None,
                    title='Conserved brain-specific cassette exon in'),
    'ATP2A2':  dict(tissue='Heart', star='last',
                    title='Conserved heart-specific alternative last exon in'),
    'TPM2':    dict(tissue='Heart', star='last',
                    title='Conserved heart-specific alternative last exon in'),
    'MFF':     dict(tissue='Heart', star=None,
                    title='Conserved heart-specific cassette exon in'),
    'ATG13':   dict(tissue='Liver', star=None,
                    title='Conserved liver-specific exon skipping in'),
    'UBXN11':  dict(tissue='Testis', star=None,
                    title='Conserved testis-specific splicing efficiency in'),
    'MBOAT7':  dict(tissue='Cerebellum', star=None,
                    title='Conserved brain and cerebellum intron retention in'),
    'AGO1':    dict(tissue='Brain', star=None,
                    title="Lineage-restricted neuronal 3' UTR lengthening in"),
    'CABLES1': dict(tissue='Brain', star=None,
                    title='Conserved brain-specific isoform in'),
}

# ----------------------------------------------------------------------------
# The constrained human-only exceptions (SFigures3 / F3B "Human only" cell).
#
# These are NOT the conserved examples above.  Each is a gene whose
# tissue-specific isoform reaches significance only in human, yet which sits in
# the most LoF-intolerant tail of gnomAD v4.1.2 (pLI >= 0.99, LOEUF < 0.13) —
# the exceptions to the "deeply shared isoforms live in constrained genes"
# trend, and the place a power artefact in another species would hide.
#
# `tissue` is the top-SSMD tissue among those where the gene is significant by
# the same rule Figure 3b uses (fdr 0.01, min_mdiff 0.05, minus the low-SSMD
# set), so the isoform ranking follows the call being illustrated rather than a
# judgement of mine.  `star` is None throughout: the splicing mechanism has not
# been checked gene by gene, and marking a terminal exon would assert one.
SUPP_HUMAN_FIGURES = {
    'EIF4G1': dict(tissue='Liver',      star=None),
    'SON':    dict(tissue='Testis',     star=None),
    'CNOT3':  dict(tissue='Brain',      star=None),
    'KAT6A':  dict(tissue='Testis',     star=None),
    'HUWE1':  dict(tissue='Brain',      star=None),
    'KAT6B':  dict(tissue='Testis',     star=None),
    'BPTF':   dict(tissue='Testis',     star=None),
    'GRIN2B': dict(tissue='Liver',      star=None),
    'TCF4':   dict(tissue='Testis',     star=None),
    'STXBP1': dict(tissue='Cerebellum', star=None),
    'CREBBP': dict(tissue='Cerebellum', star=None),
    'KMT2B':  dict(tissue='Testis',     star=None),
    'NR2F1':  dict(tissue='Cerebellum', star=None),
}

_TISSUE_WORD = {'Brain': 'brain', 'Cerebellum': 'cerebellum',
                'Heart': 'heart', 'Liver': 'liver', 'Testis': 'testis'}

for _g, _cfg in SUPP_HUMAN_FIGURES.items():
    _cfg['title'] = (f'Human-only {_TISSUE_WORD[_cfg["tissue"]]}-specific '
                     f'isoform in the constrained gene')

# ----------------------------------------------------------------------------
# Additional cross-species example genes, in the SUPP_FIGURES style.
#
# `tissue` follows the usual rule (top SSMD among the tissues significant under
# the Figure 3b criterion).  `star` is None: the splicing mechanism has not been
# checked gene by gene, and marking a terminal exon would assert one.
# SH3GLB1/SH3GL1 and RUFY2 are prepared as candidates for ambiguous names in the
# request; only the chosen ones end up in a figure.
SUPP_EXTRA_FIGURES = {
    'TSC22D2': dict(tissue='Brain',      star=None,
                    title='Brain-specific isoform in'),
    'TXNL1':   dict(tissue='Cerebellum', star=None,
                    title='Cerebellum-specific isoform in'),
    'RUFY2':   dict(tissue='Cerebellum', star=None,
                    title='Cerebellum-specific isoform in'),
    'SH3GLB1': dict(tissue='Testis',     star=None,
                    title='Testis-specific isoform in'),
    'SH3GL1':  dict(tissue='Testis',     star=None,
                    title='Testis-specific isoform in'),
}

# ----------------------------------------------------------------------------
# Multi-gene, human-only figures: one cell per GENE rather than per species.
#
# `tissue` follows the same rule as SUPP_HUMAN_FIGURES — top SSMD among the
# tissues the gene is significant in under the Figure 3b criterion — except
# CLTB, which keeps the Brain ranking it has in FigureS_CLTB so the same
# isoforms are drawn in both figures.
SUPP_GENE_FIGURES = {
    'ENPP2': dict(tissue='Brain',      star=None,
                  title='Brain-specific isoform in'),
    'REEP6': dict(tissue='Cerebellum', star=None,
                  title='Cerebellum-specific isoform in'),
    'EPN1':  dict(tissue='Cerebellum', star=None,
                  title='Cerebellum-specific isoform in'),
    'SERP2': dict(tissue='Testis',     star=None,
                  title='Testis-specific isoform in'),
    # Second pair of multi-gene figures.  SH3GLB1 is not repeated here: its
    # human panel already exists in supp_panels_extra.pkl.
    'MBD2':   dict(tissue='Cerebellum', star=None,
                   title='Cerebellum-specific isoform in'),
    'VMA21':  dict(tissue='Heart',      star=None,
                   title='Heart-specific isoform in'),
    'TUBA1C': dict(tissue='Brain',      star=None,
                   title='Brain-specific isoform in'),
    'ZNF41':  dict(tissue='Cerebellum', star=None,
                   title='Cerebellum-specific isoform in'),
    'ZSWIM7': dict(tissue='Testis',     star=None,
                   title='Testis-specific isoform in'),
}

# The two composed figures: a list of genes, all drawn in human.
MULTIGENE_FIGURES = {
    'brain': dict(
        genes=['CLTB', 'ENPP2', 'REEP6'],
        title='Brain- and cerebellum-specific isoforms in human'),
    # The contrast here is a cassette exon, so the pair is ordered by which
    # isoform carries an exon the other skips entirely — red is the inclusion
    # isoform.  EPN1's cassette exon is 74 bp; SERP2's is 221 bp.
    'EPN1_SERP2': dict(
        genes=['EPN1', 'SERP2'], order='cassette_exon',
        title='Isoforms differing by a cassette exon, in human'),

    # One figure, two lettered groups.  The contrast being shown is the length
    # of the LAST exon, so the pair is chosen by expression but then ordered by
    # last-exon length: red is always the long-3' UTR isoform.
    #
    # ZSWIM7 takes its partner from cerebellum instead of heart: its top-brain
    # and top-heart isoforms have identical 584 bp last exons, so that pair
    # shows no 3' UTR difference at all.
    'UTR3': dict(
        genes=['MBD2', 'SH3GLB1', 'VMA21', 'TUBA1C', 'ZNF41', 'ZSWIM7'],
        groups=[['MBD2', 'SH3GLB1', 'VMA21'],
                ['TUBA1C', 'ZNF41', 'ZSWIM7']],
        red='max_brain', navy='max_heart',
        navy_by_gene={'ZSWIM7': 'max_cerebellum'},
        order='long_last_exon', row_gap=2.0,
        title="Isoforms differing in the length of the last exon, in human"),
}

# Default pick rules for a multi-gene figure, overridden per entry above.
MULTIGENE_PICKS = {'red': 'first', 'navy': 'max_heart'}

# Everything the drawing code may be asked for, conserved examples first.
SUPP_FIGURES_ALL = {**SUPP_FIGURES, **SUPP_HUMAN_FIGURES, **SUPP_GENE_FIGURES,
                    **SUPP_EXTRA_FIGURES}

SUPP_TRACKS = ['brain', 'cerebellum', 'heart', 'liver', 'testis']


def prepare_supp_panels(genes=None, ortho_table=None, species_metadata=None,
                        species_annotation=None, species_res=None):
    """
    Prepare the supplementary example-gene panels.  All five coverage tracks are
    computed regardless of the gene's claim tissue, so any of them can be shown
    later without re-reading the count matrices.  Species where the gene has no
    orthologue are skipped, as upstream does.
    """
    genes = list(SUPP_FIGURES) if genes is None else list(genes)
    if ortho_table is None:
        ortho_table, _ = build_ortho2()
    if species_metadata is None or species_annotation is None:
        meta = load_metadata()
        species_metadata = species_metadata or meta['species_metadata']
        species_annotation = species_annotation or meta['species_annotation']
    if species_res is None:
        species_res = load_ssmd_results()

    out = {}
    for gene in genes:
        cfg = SUPP_FIGURES_ALL[gene]
        panels = {}
        for species in SPECIES_ALL:
            species_gene = gene if species == 'Human' \
                else get_species_ortho(ortho_table, gene, species)
            if species_gene is None:
                continue
            tissue = cfg['tissue']
            if species == 'Cattle' and tissue == 'Cerebellum':
                tissue = 'Brain'              # cattle has no cerebellum
            try:
                panel = prepare_gene_panel(
                    species, species_gene, tissue,
                    species_metadata[species], species_annotation[species],
                    species_res[species], tissue_keys=SUPP_TRACKS)
            except Exception as err:
                print(f'  [skip] {gene}/{species_gene} in {species}: {err}')
                continue
            if panel is None:
                continue
            panel['show_tracks'] = [t for t in SUPP_TRACKS
                                    if panel['track_n'].get(t, 0) > 0]
            panels[species] = panel
        out[gene] = panels
        print(f'{gene:9s} {len(panels)} species: {list(panels)}')
    return out


def prepare_figure_panels(ortho_table=None, species_metadata=None,
                          species_annotation=None, species_res=None):
    """
    Prepare the example-gene panels for the assembled figure, with the coverage
    tracks the figure actually shows (including the mammary track for SYNGR1).

    Returns {gene: {species: panel}}, each panel carrying an extra 'show_tracks'
    key giving the tracks to draw, in order.
    """
    if ortho_table is None:
        ortho_table, _ = build_ortho2()
    if species_metadata is None or species_annotation is None:
        meta = load_metadata()
        species_metadata = species_metadata or meta['species_metadata']
        species_annotation = species_annotation or meta['species_annotation']
    if species_res is None:
        species_res = load_ssmd_results()

    out = {}
    for gene, cfg in FIGURE_PANELS.items():
        panels = {}
        for species in cfg['species']:
            species_gene = gene if species == 'Human' \
                else get_species_ortho(ortho_table, gene, species)
            if species_gene is None:
                continue
            tissue = cfg['tissue_cattle'] if species == 'Cattle' else cfg['tissue']
            try:
                panel = prepare_gene_panel(
                    species, species_gene, tissue,
                    species_metadata[species], species_annotation[species],
                    species_res[species], tissue_keys=cfg['tracks'],
                )
            except Exception as err:
                print(f'  [skip] {species_gene} / {species}: {err}')
                continue
            if panel is None:
                continue

            def _avail(keys):
                return [t for t in keys if panel['track_n'].get(t, 0) > 0]

            show = _avail(cfg['show'])
            if len(show) < len(cfg['show']) and 'show_fallback' in cfg:
                show = _avail(cfg['show_fallback'])
            panel['show_tracks'] = show
            panels[species] = panel
        out[gene] = panels
    return out


# ================================================================
# Orchestration
# ================================================================

def run_all(data_dir='../notebooks/Figure3/figure_data'):
    """
    Run the full Figure 3 data pipeline, pickle every plot-ready variable into
    `data_dir`, and return them as a dict.
    """
    os.makedirs(data_dir, exist_ok=True)

    def _dump(name, obj):
        with open(os.path.join(data_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)

    # --- Human_significant_SSMD -------------------------------------------
    print('[Figure3] significant-SSMD barplot ...')
    ssmd_barplot_df = compute_significant_ssmd_barplot('Human')
    _dump('ssmd_barplot_df', ssmd_barplot_df)

    # --- *_top_quantiles heatmaps -----------------------------------------
    print('[Figure3] cross-species conservation heatmaps ...')
    heatmaps = {key: compute_species_heatmap(key) for key in HEATMAP_PAIRS}
    _dump('heatmaps', heatmaps)

    # --- cumulative isoform conservation tables ---------------------------
    print('[Figure3] orthology table ...')
    ortho_table, ortho2 = build_ortho2()

    print('[Figure3] per-species significant / low gene sets ...')
    sig, low = {}, {}
    for sp in SPECIES_ALL:
        sig[sp], low[sp] = load_species_sig_low(sp)

    df_table       = build_conservation_table(ortho2, sig, reference='Human')
    df_table_mouse = build_conservation_table(ortho2, sig, reference='Mouse')
    _dump('df_table', df_table)
    _dump('df_table_mouse', df_table_mouse)

    # --- example gene panels ----------------------------------------------
    print('[Figure3] example gene panels ...')
    meta = load_metadata()
    species_res = load_ssmd_results()

    gene_panels = {}
    for gene, tissue, species_list in EXAMPLE_PANELS:
        panels = prepare_gene_panels(
            gene, tissue, ortho_table,
            meta['species_metadata'], meta['species_annotation'], species_res,
            species_list=species_list,
        )
        gene_panels.setdefault(gene, {}).update(panels)
    _dump('gene_panels', gene_panels)

    return {
        'ssmd_barplot_df': ssmd_barplot_df,
        'heatmaps':        heatmaps,
        'df_table':        df_table,
        'df_table_mouse':  df_table_mouse,
        'gene_panels':     gene_panels,
    }
