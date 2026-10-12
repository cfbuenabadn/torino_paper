"""
SFigure4_helpers.py — data loading and processing for Supplementary Figure 4,
the corrected version of the `colocs` plot (fraction of GWAS loci whose
colocalisation implicates each RNA-processing class).

Must be run from the analysis/ directory (or with CWD set there) so the
relative '../code/...' paths resolve.

The plot exists in two exploratory versions that disagree:
  analysis/Figure_4.ipynb  cell 93 (exec 223)
  analysis/PruebaF4.ipynb  cell 31

Three things are corrected here.

1. The denominator is the set of **tested regions**, taken from the lead-SNP
   windows the colocalisation was actually run over
   (gwas/LeadSnpWindows.bed), not from hyprcoloc's own surviving rows. Both
   exploratory versions shrink the denominator with filters that discard
   evidence of *absence* of colocalisation:
     * PruebaF4 cell 15 drops every row with no TopCandidateSNP — 99,134 of
       140,517 rows (71%);
     * both notebooks drop rows where hyprcoloc dropped the GWAS trait
       (`Dropped_GWAS_trait`), but a dropped trait means the GWAS signal did
       not colocalise in that region, which is exactly the outcome the
       percentage is meant to measure.
   Here a region counts as tested if hyprcoloc returned a result for it,
   whether or not the GWAS trait was dropped. That raises the denominator for
   60 of the 65 traits (8,705 -> 8,966 regions in total) and so lowers every
   percentage.

2. Both notebooks build the x tick labels with
       elif idx == 'TRANSETHNICRA':
           new_name == 'Rheumatoid arthritis'          # `==`, not `=`
       elif idx == 'ADHD':
           new_name == 'Attention-deficit/hyperactivity disorder'
   Those branches compare instead of assigning, so `new_name` keeps the previous
   iteration's value and two bars inherit their neighbour's label. In PruebaF4
   that mislabels the ADHD bar as a second "Multiple sclerosis". Fixed in
   trait_display_names(), after which all 65 labels are unique.

3. The "All traits" column was annotated with a hard-coded 1740, the number of
   *colocalized* regions, while every per-trait bar is annotated with `counter`,
   the number of *tested* regions. It now carries the matching total and is a
   pooled percentage rather than a mean of per-trait percentages: all 9,016
   regions of all 68 tested traits, including the three whose result was zero
   colocalizations, which is a result and not a missing value. Only the 65
   traits with at least one colocalized region get their own bar, so the
   All-traits column is over a slightly larger denominator (9,016) than the
   drawn bars cover (8,966). 1740 is still reported by build_filtering_trace()
   — it is the number of events in the Figure 4 upset plot.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

# ================================================================
# Paths
# ================================================================

ROOT = '/project2/yangili1/cfbuenabadn/torino_paper/code'
FIG4_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure4'
GTEX_STM_CONFIG = '/project2/mstephens/cfbuenabadn/gtex-stm/code/config'

HYPRCOLOC = f'{ROOT}/hyprcoloc/Results/ForGWASColoc/GWAS_GTEx_01/results.txt.gz'

# The regions the colocalisation was run over: one 1 Mb window per GWAS lead
# SNP per trait. Its name field is '{chrom}_{pos}_N_N_{trait}', two placeholder
# allele fields wider than the '{chrom}_{pos}_{trait}' ids hyprcoloc reports.
GWAS_DIR = '/project2/mstephens/cfbuenabadn/gtex-stm/code/gwas'
LEAD_SNP_WINDOWS = f'{GWAS_DIR}/LeadSnpWindows.bed'
LEAD_SNP_BEDS = f'{GWAS_DIR}/leadSnps'
GWAS_NAMES = f'{ROOT}/resources/gwas_table2.tsv'

# the three files code/rules/common.py reads to build gwas_traits_for_coloc
GWAS_ANNOT_MAP = f'{GTEX_STM_CONFIG}/gwas_annot_map.csv'
GWAS_FEATURES = f'{GTEX_STM_CONFIG}/gwas_features.txt'
BJF79_TABLE = f'{GTEX_STM_CONFIG}/bjf79_gwas_table.tsv'

# the hyprcoloc id that names a GWAS locus: '{chrom}_{pos}_{ref}_{alt}_{rsid}_{trait}'
LOCUS = 'GWASLeadSnpChrom_Pos_RefAllele_AltAllele_rsID_trait'

MIN_REGIONS = 10       # a trait needs this many tested regions to be plotted


# ================================================================
# Inputs
# ================================================================

def load_gwas_trait_pool():
    """
    `gwas_traits_for_coloc` exactly as code/rules/common.py builds it: the pool
    of GWAS traits the colocalisation was run over.
    """
    gwas_annot = pd.read_csv(GWAS_ANNOT_MAP)
    with open(GWAS_FEATURES) as fh:
        gwas_traits = [x.rstrip() for x in fh]
    bjf79 = pd.read_csv(BJF79_TABLE, index_col='gwas', sep='\t',
                        usecols=['gwas', 'trait', 'FTPPath',
                                 'SummaryStatsLocalFilepath', 'ProcessingMethod',
                                 'Continuous'])
    return [x for x in list(gwas_annot.trait) if x in gwas_traits] + list(bjf79.index)


def load_isoform_annotation():
    """
    The corrected isoform annotation with the 'efficiency factor' flag. Reused
    from Figure4_helpers so the two figures cannot drift apart.
    """
    if FIG4_DIR not in sys.path:
        sys.path.append(FIG4_DIR)
    from Figure4_helpers import load_isoform_annotation as _load
    return _load()


def load_hyprcoloc_results():
    """
    The full hyprcoloc output with the three derived columns, and no row filter.

    Source: Figure_4.ipynb cells 19-20. `coloc` marks loci where the GWAS trait
    itself is among the colocalised traits; `Dropped_GWAS_trait` marks loci
    where hyprcoloc dropped the GWAS trait before the analysis, which makes the
    locus untestable for that trait.
    """
    df = pd.read_csv(HYPRCOLOC, sep='\t')
    df['GWAS_trait'] = df[LOCUS].apply(lambda x: '_'.join(x.split('_')[2:]))
    df['lead_snp'] = df[LOCUS].apply(lambda x: '_'.join(x.split('_')[:2]))
    df['coloc'] = [(r[LOCUS] in r.ColocalizedTraits)
                   if isinstance(r.ColocalizedTraits, str) else False
                   for _, r in df.iterrows()]
    df['Dropped_GWAS_trait'] = [False if str(r.DroppedTrait) == 'nan'
                                else (r.GWAS_trait in r.DroppedTrait)
                                for _, r in df.iterrows()]
    return df


# ================================================================
# The filtering funnel
# ================================================================

def load_lead_snp_windows():
    """
    The regions the colocalisation was run over, from gwas/LeadSnpWindows.bed.

    Adds a `locus` column in hyprcoloc's own id format ('{chrom}_{pos}_{trait}')
    by dropping the two 'N' placeholder allele fields, so windows and hyprcoloc
    results join directly.
    """
    w = pd.read_csv(LEAD_SNP_WINDOWS, sep='\t',
                    names=['chrom', 'start', 'end', 'name'])
    parts = w.name.str.split('_')
    w['trait'] = parts.apply(lambda p: '_'.join(p[4:]))
    w['locus'] = parts.apply(lambda p: '_'.join(p[:2] + p[4:]))
    return w


def tested_regions(windows, df, pool):
    """
    The regions that were actually tested for each of the pooled traits: a
    lead-SNP window that hyprcoloc returned a result for.

    A window with no result had no molecular phenotype in it to colocalise
    against, so it is not testable. A window whose result has
    `Dropped_GWAS_trait` set IS testable — hyprcoloc dropping the GWAS trait
    means the signal did not colocalise there, which is the outcome being
    measured, so it belongs in the denominator.
    """
    w = windows.loc[windows.trait.isin(set(pool))]
    return w.loc[w.locus.isin(set(df[LOCUS]))]


def regions_per_trait(tested):
    """`counter`: distinct tested regions for each trait."""
    return tested.groupby('trait').locus.nunique()


def build_filtering_trace(pool, df, windows):
    """
    Every filtering step from the GWAS trait pool to the traits and regions that
    reach the plot, as a table of (step, traits kept, regions kept, what dropped).
    """
    pool_set = set(pool)
    submitted = windows.loc[windows.trait.isin(pool_set)]
    tested = tested_regions(windows, df, pool)
    counter = regions_per_trait(tested)
    keep_n = counter.loc[counter >= MIN_REGIONS].index
    coloc_n = df.loc[df.coloc].groupby('GWAS_trait')[LOCUS].nunique()
    plotted = [t for t in keep_n if coloc_n.get(t, 0) > 0]

    not_in_pool = sorted(set(windows.trait) - pool_set)
    no_result = sorted(pool_set - set(tested.trait))
    rows = [
        ('0. gwas_traits_for_coloc (code/rules/common.py)',
         len(pool), np.nan, 'every one has a leadSnps bed and windows'),
        ('1. lead-SNP windows submitted for those traits',
         submitted.trait.nunique(), submitted.locus.nunique(),
         f'-{windows.locus.nunique() - submitted.locus.nunique()} windows from '
         f'{", ".join(not_in_pool)}, not in the pool'),
        ('2. window tested (hyprcoloc returned a result)',
         tested.trait.nunique(), tested.locus.nunique(),
         f'-{submitted.locus.nunique() - tested.locus.nunique()} windows with '
         f'nothing to colocalise against'
         + (f'; -{len(no_result)} trait ({", ".join(no_result)})' if no_result else '')),
        (f'3. trait has >= {MIN_REGIONS} tested regions',
         len(keep_n), int(counter.loc[keep_n].sum()),
         f'-{int((counter < MIN_REGIONS).sum())} traits'),
        ('4. trait has >= 1 colocalized region (percentage_coloc > 0)',
         len(plotted), int(counter.loc[plotted].sum()),
         f'-{len(keep_n) - len(plotted)} traits '
         f'({", ".join(sorted(set(keep_n) - set(plotted)))})'),
    ]
    trace = pd.DataFrame(rows, columns=['step', 'traits', 'GWAS_regions', 'dropped'])
    trace.attrs['colocalized_regions'] = int(
        df.loc[df.coloc & df.GWAS_trait.isin(plotted), LOCUS].nunique())
    trace.attrs['lead_snps'] = int(submitted.chrom.str.cat(
        submitted.start.astype(str), sep='_').nunique())
    return trace


# ================================================================
# Per-trait RNA-processing classes
# ================================================================

def classify_traits(df, annot, counter):
    """
    For every trait, count its colocalized regions by RNA-processing class.

    `counter` is the tested-region count per trait from regions_per_trait();
    the class loop itself is Figure_4.ipynb cell 39 / PruebaF4 cell 39 kept
    verbatim, including the `else: add_other = True` that rides on the UTR3
    test, so the class counts match the exploratory notebooks exactly.
    """
    out = {}
    for trait, sub in df.groupby('GWAS_trait'):
        t = dict(ir=0, utr=0, other=0, cassette=0, multiple=0, ir_utr=0,
                 ir_cassette=0, cassette_utr=0, coloc_counter=0,
                 percentage_coloc=0, efficiency_factor=0, counter=0)
        t['counter'] = int(counter.get(trait, 0))

        for x in list(sub.loc[sub.coloc == True].ColocalizedTraits):
            add_ir = add_cassette = add_utr = add_other = False
            isoforms = sorted({y.split(';')[1] for y in x.split(', ') if 'isoform_' in y})

            if len(isoforms) > 0:
                if all(annot.loc[annot.transcript.isin(isoforms), 'efficiency_factor']):
                    t['efficiency_factor'] += 1
                    t['coloc_counter'] += 1
                    continue
                appris_refs = annot.loc[annot.transcript.isin(isoforms)
                                        & (~annot.efficiency_factor), 'appris_ref']
                for ref in appris_refs:
                    if 'RI' in ref:
                        add_ir = True
                    if ('CE' in ref) or ('SE' in ref):
                        add_cassette = True
                    if ('MXE' in ref):
                        add_cassette = True
                    if ('UTR3' in ref):
                        add_utr = True
                    else:
                        add_other = True

            if add_cassette and not add_utr and not add_ir:
                t['cassette'] += 1
            elif add_cassette and add_utr and not add_ir:
                t['cassette_utr'] += 1
            elif add_cassette and not add_utr and add_ir:
                t['ir_cassette'] += 1
            elif add_cassette and add_utr and add_ir:
                t['multiple'] += 1
            elif not add_cassette and add_utr and not add_ir:
                t['utr'] += 1
            elif not add_cassette and add_utr and add_ir:
                t['ir_utr'] += 1
            elif not add_cassette and not add_utr and add_ir:
                t['ir'] += 1
            else:
                t['other'] += 1
            t['coloc_counter'] += 1

        t['percentage_coloc'] = t['coloc_counter'] / t['counter'] if t['counter'] else 0
        out[trait] = t

    trait_df = pd.DataFrame(out).T
    return trait_df.loc[trait_df.counter > 0].sort_values('percentage_coloc')


# ================================================================
# Labels
# ================================================================

def trait_display_names(trait_ids, gwas_names=None):
    """
    Display label for each trait id — the corrected version of the name_list
    loop in Figure_4 cell 77 / PruebaF4 cell 31, whose TRANSETHNICRA and ADHD
    branches used `==` instead of `=` and so relabelled those two bars with
    whatever name the previous bar had.
    """
    if gwas_names is None:
        gwas_names = pd.read_csv(GWAS_NAMES, sep='\t')
    special = {'IMSGC2019': 'Multiple sclerosis 2019',
               'TRANSETHNICRA': 'Rheumatoid arthritis',
               'ADHD': 'Attention-deficit/hyperactivity disorder'}
    names = []
    for idx in trait_ids:
        if idx.startswith('GCS'):
            names.append(gwas_names.loc[gwas_names.gwas == idx].trait.iloc[0])
        elif idx in special:
            names.append(special[idx])
        else:
            names.append(' '.join(idx.capitalize().split('_')))
    return names


# ================================================================
# Panel assembly
# ================================================================

def build_trait_table(pool, df, windows, gwas_names=None):
    """
    One row per submitted GWAS trait, plus a Total row.

    Columns
      trait            display name (trait_display_names)
      accession        the id used throughout the pipeline
      total_leadSnps   lead-SNP windows submitted for the trait
      tested_regions   of those, the ones hyprcoloc returned a result for
      tested           tested_regions >= MIN_REGIONS
      colocs           colocalized windows
      percent_coloc    colocs / tested_regions, blank when nothing was tested

    Starts from all 85 traits in `pool`, so the traits that drop out at each
    step stay visible rather than silently disappearing.
    """
    pool = list(pool)
    submitted = windows.loc[windows.trait.isin(set(pool))].groupby('trait').locus.nunique()
    tested = regions_per_trait(tested_regions(windows, df, pool))
    colocs = df.loc[df.coloc].groupby('GWAS_trait')[LOCUS].nunique()

    t = pd.DataFrame({'accession': pool})
    t['trait'] = trait_display_names(pool, gwas_names)
    t['total_leadSnps'] = t.accession.map(submitted).fillna(0).astype(int)
    t['tested_regions'] = t.accession.map(tested).fillna(0).astype(int)
    # object dtype so the per-trait rows keep True/False when the Total row
    # (which carries a count) is appended
    t['tested'] = (t.tested_regions >= MIN_REGIONS).astype(object)
    t['colocs'] = t.accession.map(colocs).fillna(0).astype(int)
    t['percent_coloc'] = (100 * t.colocs / t.tested_regions.replace(0, np.nan)).round(1)
    t = t[['trait', 'accession', 'total_leadSnps', 'tested_regions', 'tested',
           'colocs', 'percent_coloc']].sort_values(
        ['tested', 'colocs'], ascending=[False, False], key=lambda c:
        c.astype(int) if c.name == 'tested' else c).reset_index(drop=True)

    # Total row: pooled over the tested traits, which is what the figure's
    # "All traits" column reports.
    ted = t.loc[t.tested == True]
    total = {'trait': 'Total',
             'accession': f'{len(t)} traits, {int((t.tested == True).sum())} tested',
             'total_leadSnps': int(t.total_leadSnps.sum()),
             'tested_regions': int(ted.tested_regions.sum()),
             'tested': int((t.tested == True).sum()),
             'colocs': int(ted.colocs.sum()),
             'percent_coloc': round(100 * ted.colocs.sum() / ted.tested_regions.sum(), 1)}
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True)


def prepare_colocs_panel(trait_df):
    """Everything plot_colocs() needs, and nothing it does not."""
    tested = trait_df.loc[trait_df.counter >= MIN_REGIONS]     # 68 traits
    sel = tested.loc[tested.percentage_coloc > 0]              # 65 with a bar
    # The "All traits" column pools every tested trait, the three with zero
    # colocalizations included, so its denominator (9,016) is larger than the
    # regions the 65 drawn bars cover (8,966).
    total_regions = int(tested.counter.sum())
    total_coloc = int(tested.coloc_counter.sum())
    classes = ['ir', 'utr', 'cassette', 'ir_utr', 'ir_cassette', 'cassette_utr',
               'multiple']
    return {'trait_selected_df': sel,
            'name_list': trait_display_names(list(sel.index)),
            'total_loci': total_regions,
            'total_coloc': total_coloc,
            'total_percent': 100 * total_coloc / total_regions,
            'n_tested_traits': int(tested.shape[0]),
            # pooled fractions for the All-traits stack, in drawing order
            'pooled': {
                'any': total_coloc / total_regions,
                'classified': tested[classes].sum().sum() / total_regions,
                'cassette': (tested.ir + tested.utr + tested.cassette).sum() / total_regions,
                'utr': (tested.ir + tested.utr).sum() / total_regions,
                'ir': tested.ir.sum() / total_regions,
            }}


def run_all(data_dir='figure_data'):
    """Produce the plot-ready objects and pickle them into `data_dir`."""
    out = {}

    def _dump(name, obj):
        with open(os.path.join(data_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)
        out[name] = obj
        return obj

    os.makedirs(data_dir, exist_ok=True)

    print('[SFigure4] hyprcoloc results and lead-SNP windows ...')
    df = load_hyprcoloc_results()
    pool = load_gwas_trait_pool()
    windows = load_lead_snp_windows()
    _dump('filtering_trace', build_filtering_trace(pool, df, windows))
    counter = regions_per_trait(tested_regions(windows, df, pool))

    print('[SFigure4] isoform annotation ...')
    annot = load_isoform_annotation()

    print('[SFigure4] per-trait RNA-processing classes ...')
    trait_df = _dump('trait_df', classify_traits(df, annot, counter))
    _dump('colocs_panel', prepare_colocs_panel(trait_df))
    _dump('trait_table', build_trait_table(pool, df, windows))
    return out
