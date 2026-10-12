"""SFigure2 3' UTR / polyA panels, from the results already on disk.

Three panels, from analysis/UTR3.ipynb.

A note on cost. `polyA_barplot` is pure literals and draws instantly. The other
two are the expensive panels of this figure and there is no way around it:

  polyA_heatmap   needs long_df / short_df, built by a per-gene loop over the
                  ~4,600 genes with an extended 3' UTR (In[88]). About an hour.
  UTRs_cumulative needs bam_dict2, which counts long reads per polyA site in
                  nine BAMs (In[1224]). Also about an hour, and it is I/O bound.

Neither result was ever written to disk, so both have to be recomputed -- these
are computations, not re-runs of a pipeline step whose output exists. Run them
on a compute node; nothing here writes to the work tree.
"""

import os
import sys

from SFigure2_intron_retention import (      # noqa: F401  -- shared setup
    ANALYSIS_DIR, np, pd, plt, sns, BedTool, bedtools_safe, _cached, _panel,
    free, _subset, _tag, gtex_colors2, load_annot,
)

import SFigure2_intron_retention as _R
from tqdm import tqdm


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
@_panel
def panel_polyA_barplot_published():
    """Notebook cell 65 (In[158]), verbatim -- the published bar heights.

    Kept because it reproduces supp_fig/polyA/barplot.png pixel for pixel. Its
    counts are literals, in the gene:exon_start unit, and one of them (5,959
    GENCODE APA sites) appears in no cell output and is ten short of the 5,969
    the data gives. panel_polyA_barplot below recomputes everything instead.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar([0, 1, 3, 4], [9587, 10905, 15411, 5959],
           color=['tab:blue', 'tab:blue', 'salmon', 'salmon'])
    ax.bar([0, 1, 3, 4], [9155, 6973, 13736, 3516],
           color=['navy', 'navy', 'tab:red', 'tab:red'])
    ax.text(-0.1, 800, '9155 / 9587', c='white', rotation=90)
    ax.text(0.9, 800, '6973 / 10905', c='white', rotation=90)
    ax.text(2.9, 800, '13736 / 15411', c='white', rotation=90)
    ax.text(3.9, 1000, '3516 / 5959', c='white', rotation=90)
    ax.text(3.9, 6300, '9', c='black', rotation=90)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(['Single polyA', 'Multiple polyA',
                        'Single polyA', 'Multiple polyA'], rotation=90)
    ax.set_ylabel('counts')
    ax.spines[['top', 'right']].set_visible(False)
    return fig


@_panel
def panel_polyA_barplot():
    """The same four bars, recomputed in the gene:end unit.

    Light bar: sites called on that side. Dark bar: sites the other annotation
    also has. Both Torino populations are the unambiguous ones -- APA restricted
    to genes labelled multi-site, single-site restricted to genes with exactly
    one gene:end -- so a site is counted once and in one place. Follows
    GENE_SUBSET, so the corrected notebook draws the corrected bars.
    """
    n = polya_numbers(with_heatmap=False)
    totals = [n['single-site genes confirmed'], n['APA sites (gene:end)'],
              n['GENCODE constitutive sites (gene:end)'],
              n['GENCODE APA sites (gene:end)']]
    found = [n['single sites matched by GENCODE'],
             n['APA sites matched by GENCODE'],
             n['GENCODE constitutive found by Torino'],
             n['GENCODE APA found by Torino']]
    x = [0, 1, 3, 4]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(x, totals, color=['tab:blue', 'tab:blue', 'salmon', 'salmon'])
    ax.bar(x, found, color=['navy', 'navy', 'tab:red', 'tab:red'])
    for xi, tot, fnd in zip(x, totals, found):
        ax.text(xi - 0.1, 0.03 * max(totals), f'{fnd:,} / {tot:,}',
                c='white', rotation=90, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Single polyA', 'Multiple polyA',
                        'Single polyA', 'Multiple polyA'], rotation=90)
    ax.set_ylabel('polyA sites')
    top = max(totals)
    ax.set_ylim(0, top * 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.5, top * 1.0, 'Torino', ha='center', fontsize=9)
    ax.text(3.5, top * 1.0, 'GENCODE', ha='center', fontsize=9)
    return fig


PANELS = {'polyA_barplot': panel_polyA_barplot,
          'polyA_barplot_published': panel_polyA_barplot_published}


# --------------------------------------------------------------------------
# polyA_heatmap
#
# long_df / short_df come from a per-gene loop over the genes with an extended
# 3' UTR (In[87]); nothing caches them, so this is about an hour. The helper
# definitions are the ones live at that point: In[40] first, then In[84]
# overriding the *_w_transcript variants, exactly as the kernel had them.
# --------------------------------------------------------------------------
# --- get_exon_chain and the first UTR helpers
# Copied from UTR3.ipynb cells 6 (In[6]) and 11 (In[9]);
# definitions only. get_alt_utrs and get_exon_chain live only here.
def get_exon_chain(transcript_bed):
    exon_chain = []
    for idx, row in transcript_bed.iterrows():
        exon_chain.append((int(row.start), int(row.end)))

    return exon_chain


def alt_utr(ref, query, strand, prime = 'three'):
    ref_len = ref.shape[0]
    query_len = query.shape[0]
    # print(ref_len)
    # print(query_len)
    if strand == '-':
        ref = ref[::-1]
        query = query[::-1]

    if prime == 'five':
        i = 0
    elif prime == 'three':
        i = -1
    else:
        raise Exception('invalid prime end')
        
    ref_start = int(ref.iloc[i].start)
    ref_end = int(ref.iloc[i].end)
    ref_exon_len = ref_end - ref_start

    ref_exon_len = np.max([ref_exon_len, 1])

    query_start = int(query.iloc[i].start)
    query_end = int(query.iloc[i].end)
    query_exon_len = query_end - query_start

    query_exon_len = np.max([query_exon_len, 1])

    ratio_ref = ref_exon_len/query_exon_len
    ratio_query = query_exon_len/ref_exon_len
    

    if ((prime == 'five') and (strand == '+')) or ((prime == 'three') and (strand == '-')):
        if (ref_end != query_end) and ((ref_len > 1) or (query_len > 1)):
            utr = 'alt.exon'
        elif (np.abs(ref_start - query_start) < 200) and (ratio_ref > 0.75) and (ratio_query > 0.75):
            utr = 'same'
        else:
            utr = 'alt.utr'
    else:
        if (ref_start != query_start) and ((ref_len > 1) or (query_len > 1)):
            utr = 'alt.exon'
        elif (np.abs(ref_end - query_end) < 200) and (ratio_ref > 0.75) and (ratio_query > 0.75):
            utr = 'same'
        else:
            utr = 'alt.utr'
    return utr


def compatible(x, y1, y2, strict = False):
    if abs(y1 - y2) >= 200:
        return False

    if strict:
        d1, d2 = abs(y1 - x), abs(y2 - x)
        if 0.75 * d1 > d2:
            return False
        if 0.75 * d2 > d1:
            return False
    return True


def group_by_exon(last_exons):
    strand = last_exons[1]
    last_exons_list = last_exons[0]
    last_exons_dict = {}
    for a, b in last_exons_list:
        if strand == '+':
            x = a
            y = b
        else:
            x = b
            y = a

        if x in last_exons_dict.keys():
            last_exons_dict[x].append(y)
        else:
            last_exons_dict.update({x:[y]})

    return last_exons_dict


def get_alt_utrs(last_exon_dict):
    alt_utrs = 0
    for x, y_list in last_exon_dict.items():
        if len(y_list) <= 1:
            continue
        else:
            unique_utrs = get_unique_utrs(x, y_list)
            if unique_utrs > 1:
                alt_utrs += unique_utrs
    return alt_utrs


def get_unique_utrs(x, ys):
    ys = sorted(ys)
    current_group = []
    unique_utr = 1
    for y in ys:
        if len(current_group) == 0:
            current_group.append(y)
            
        else:
            if any([compatible(x, y_, y) for y_ in current_group]): # all
                current_group.append(y)
            else:
                unique_utr += 1
                current_group = [y]
                
    return unique_utr


def get_last_exons(gene_df):
    last_exons = []
    strand = gene_df.strand.iloc[0]
    for transcript, df in gene_df.groupby('transcript_id'):
        exon_chain = get_exon_chain(df)
        if strand == '+':
            last_exons.append(exon_chain[-1])
        else:
            last_exons.append(exon_chain[0])

    return last_exons, strand


# --- get_EL_for_gene
# Copied verbatim from UTR3.ipynb cell 115 (In[78]).
import pysam  # pip install pysam


# ---------------------------------------------------------------------------
# Paths — edit these to match your project layout
# ---------------------------------------------------------------------------

EL_PATHS = {
    'Human':   '../code/results/torino/tables/Human/isoforms.EL.sorted.bed.gz',
    'Macaque': '../code/results/torino/tables/Macaque/isoforms.EL.sorted.bed.gz',
    'Mouse':   '../code/results/torino/tables/Mouse/isoforms.EL.sorted.bed.gz',
    'Rat':     '../code/results/torino/tables/Rat/isoforms.EL.sorted.bed.gz',
    'Rabbit':  '../code/results/torino/tables/Rabbit/isoforms.EL.sorted.bed.gz',
    'Chicken': '../code/results/torino/tables/Chicken/isoforms.EL.sorted.bed.gz',
    'Cattle':  '../code/results/torino/tables/Cattle/isoforms.EL.sorted.bed.gz',
}

ISO_PATHS = {
    'Human':   '../code/results/torino/Human/isoforms/isoforms.bed.gz',
    'Macaque': '../code/results/torino/Macaque/isoforms/isoforms.bed.gz',
    'Mouse':   '../code/results/torino/Mouse/isoforms/isoforms.bed.gz',
    'Rat':     '../code/results/torino/Rat/isoforms/isoforms.bed.gz',
    'Rabbit':  '../code/results/torino/Rabbit/isoforms/isoforms.bed.gz',
    'Chicken': '../code/results/torino/Chicken/isoforms/isoforms.bed.gz',
    'Cattle':  '../code/results/torino/Cattle/isoforms/isoforms.bed.gz',
}

# Small lookup: gene_id -> (chrom, start, end)  — built once from the iso BED
# Stored as a lightweight TSV per species (see setup_tabix_indexes.py)
GENE_COORD_PATHS = {
    species: path.replace('isoforms.bed.gz', 'gene_coords.tsv')
    for species, path in ISO_PATHS.items()
}

# ---------------------------------------------------------------------------
# Gene coordinate cache  (tiny, safe to keep all species in RAM)
# ---------------------------------------------------------------------------

_gene_coord_cache: dict[str, pd.DataFrame] = {}

def _load_gene_coords(species: str) -> pd.DataFrame:
    """Load the pre-built gene→coord lookup (one row per gene_id)."""
    if species not in _gene_coord_cache:
        path = GENE_COORD_PATHS[species]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Gene coord index not found at {path}. "
                "Run setup_tabix_indexes.py first."
            )
        df = pd.read_csv(path, sep='\t',
                         names=['gene_id', 'chrom', 'start', 'end'],
                         dtype={'start': int, 'end': int})
        df = df.set_index('gene_id')
        _gene_coord_cache[species] = df
    return _gene_coord_cache[species]


def get_gene_coords(species: str, gene_id: str):
    """Return (chrom, start, end) for a gene in a given species."""
    coords = _load_gene_coords(species)
    if gene_id not in coords.index:
        raise KeyError(f"Gene '{gene_id}' not found in {species} coordinate index.")
    row = coords.loc[gene_id]
    return row['chrom'], int(row['start']), int(row['end'])


# ---------------------------------------------------------------------------
# EL header cache  (sample names; first line of each file)
# ---------------------------------------------------------------------------

_el_header_cache: dict[str, list[str]] = {}

def _get_el_header(species: str) -> list[str]:
    """Read and cache the header line of the EL BED file."""
    if species not in _el_header_cache:
        path = EL_PATHS[species]
        with gzip.open(path, 'rt') as fh:
            header = fh.readline().rstrip('\n').split('\t')
        _el_header_cache[species] = header
    return _el_header_cache[species]


# ---------------------------------------------------------------------------
# Core lazy loaders
# ---------------------------------------------------------------------------

def get_EL_for_gene(species: str, gene_id: str) -> pd.DataFrame:
    """
    Fetch only the rows for `gene_id` from the EL table using tabix.

    Returns a DataFrame with the same columns as the full EL table,
    but only the isoforms belonging to this gene.

    Prerequisite: the EL BED file must be tabix-indexed:
        tabix -p bed file.bed.gz
    """
    chrom, start, end = get_gene_coords(species, gene_id)
    path = EL_PATHS[species]
    header = _get_el_header(species)

    tbx = pysam.TabixFile(path)
    rows = []
    try:
        for rec in tbx.fetch(chrom, start, end):
            fields = rec.split('\t')
            rows.append(fields)
    except ValueError:
        # pysam raises ValueError if region has no reads — return empty
        pass
    finally:
        tbx.close()

    if not rows:
        return pd.DataFrame(columns=header)

    df = pd.DataFrame(rows, columns=header)

    # Filter to exact gene (tabix fetches overlapping intervals, not exact matches)
    gid_col = 'gid' if 'gid' in df.columns else header[3]
    df = df.loc[df[gid_col] == gene_id].copy()

    # Restore numeric types for sample columns (tabix returns everything as str)
    sample_cols = df.columns[6:]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors='coerce')

    # Replicate the pid construction done at load time
    df['pid'] = df[gid_col] + '.' + df['pid']
    df.index = df['pid']

    return df

import os
import gzip


# --- median_by_tissue
# Copied from UTR3.ipynb cell 117 (In[80]); definitions only, since
# the cell also did module-level work against names built elsewhere.
import pandas as pd


import numpy as np


def median_by_tissue(tissue_df, data_df, tissue_col='Tissue'):
    """
    Parameters:
    -----------
    tissue_df : DataFrame with a column marking tissues (index = sample names)
    data_df   : DataFrame with numeric values (rows = features, columns = samples)
    tissue_col: Name of the column in tissue_df that contains tissue labels

    Returns:
    --------
    DataFrame with rows = features (same as data_df), columns = unique tissues
    """
    # Only keep samples that appear in both
    common_samples = tissue_df.index.intersection(data_df.columns)
    
    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between tissue_df index and data_df columns.")
    
    tissue_df = tissue_df.loc[common_samples]
    data_df   = data_df[common_samples]
    
    # Group columns by tissue and compute median across columns (axis=1)
    tissues = tissue_df[tissue_col].unique()
    
    result = pd.DataFrame(index=data_df.index)
    
    for tissue in tissues:
        samples_in_tissue = tissue_df[tissue_df[tissue_col] == tissue].index
        result[tissue] = data_df[samples_in_tissue].median(axis=1)
    
    return result


# --- dict_to_dataframe
# Copied from UTR3.ipynb cell 47 (In[42]); definitions only, since
# the cell also did module-level work against names built elsewhere.
import pandas as pd


def dict_to_dataframe(data):
    """
    Convert a nested dictionary to a DataFrame.
    
    Parameters:
    -----------
    data : dict
        Dictionary with structure {gene: {key: {'coords': [...], 'median_coords': [...], 
        'exon_match': [...], 'utr_match': [...]}}}
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: gene, x, coords, median_coords, exon_match, utr_match
    """
    rows = []
    for gene, gene_data in data.items():
        for key, value in gene_data.items():
            # Get the length of the lists (they should all be the same length)
            n = len(value['median_coords'])
            for i in range(n):
                rows.append({
                    'gene': gene,
                    'exon_start': key,
                    #'coords': value['coords'][i],
                    'exon_end': value['median_coords'][i],
                    'exon_match': value['exon_match'][i],
                    'utr_match': value['utr_match'][i]
                })
    
    return pd.DataFrame(rows)


# --- the UTR helpers (In[40])
# Copied from UTR3.ipynb cell 45 (In[40]); definitions only, since
# the cell also did module-level work against names built elsewhere.
def process_utr_loadings(grouped_torino_ref, gene, median_EL):

    
    torino_ref_df = grouped_torino_ref.get_group(gene)
    strand = torino_ref_df.iloc[0].strand
    isos = process_utr_coordinates_torino_w_transcripts(group_by_exon_w_transcript(get_last_exons_w_transcript(torino_ref_df)), strand)
    long_EL_list = []
    short_EL_list = []
    for x, iso_dict in isos.items():
        iso_groups = iso_dict['iso_groups']
        if len(iso_groups) > 1:
            iso_short = iso_groups[0]
            iso_long = iso_groups[-1]
        else:
            continue

        short_EL = median_EL.loc[iso_short].sum(axis=0)
        short_EL_list.append(short_EL)
        
        long_EL = median_EL.loc[iso_long].sum(axis=0)
        long_EL_list.append(long_EL)

    return short_EL_list, long_EL_list


def process_gene_with_ref(gene_df, gene_df_appris):
    # gene_df_appris = gene_df.loc[~gene_df.appris.isna()]
    last_exons_appris = group_by_exon(get_last_exons(gene_df_appris))

    last_exons, strand = get_last_exons(gene_df)

    for a, b in last_exons:
        if strand == '+':
            x = a
            y = b
        else:
            x = b
            y = a

        if x in last_exons_appris.keys():
            if y not in last_exons_appris[x]:
                last_exons_appris[x].append(y)

    return last_exons_appris


def get_last_exons_w_transcript(gene_df):
    last_exons = {}
    strand = gene_df.strand.iloc[0]
    for transcript, df in gene_df.groupby('transcript_id'):
        exon_chain = get_exon_chain(df)
        if strand == '+':
            last_exons.update({transcript:exon_chain[-1]})
                # append(exon_chain[-1])
        else:
            last_exons.update({transcript:exon_chain[0]})
            # last_exons.append(exon_chain[0])

    return last_exons, strand


def group_by_exon_w_transcript(last_exons):
    strand = last_exons[1]
    last_exons_list_dict = last_exons[0]
    last_exons_dict = {}
    for transcript, coords in last_exons_list_dict.items():
        a = coords[0]
        b = coords[1]
        if strand == '+':
            x = a
            y = b
        else:
            x = b
            y = a

        if x in last_exons_dict.keys():
            last_exons_dict[x]['y'].append(y)
            last_exons_dict[x]['iso'].append(transcript)
        else:
            last_exons_dict.update({x:{'y':[y], 'iso':[transcript]}})

    return last_exons_dict


def get_unique_utr_coords(x, ys):
    current_group = []
    unique_utr_group_list = []
    # print(ys)
    for y in ys:
        if len(current_group) == 0:
            # print('wut')
            current_group.append(y)
            
        else:
            # print(y)
            # print(current_group)
            if any([compatible(x, y_, y) for y_ in current_group]): ## Note how we use any
                current_group.append(y)
            else:
                # print('como')
                unique_utr_group_list.append(current_group)
                current_group = [y]

    if len(current_group) > 0:
        unique_utr_group_list.append(current_group)
                
    return unique_utr_group_list


def process_utr_coordinates_torino_w_transcripts(coords_dict_torino):
    y_dict = {}
    for x, ys in coords_dict_torino.items():
        ys = sorted(ys)
        coord_groups = get_unique_utr_coords(x, ys)
        coords_groups_median = [np.median(y) for y in coord_groups]

    return coord_groups, coord_groups_median


def get_unique_utr_coords_w_iso(x, ys, isos):
    current_group_y = []
    current_group_iso = []
    unique_utr_group_list = []
    unique_iso_group_list = []

    for y, iso in zip(ys, isos):
        if len(current_group_y) == 0:
            current_group_y.append(y)
            current_group_iso.append(iso)
        else:
            if any([compatible(x, y_, y) for y_ in current_group_y]):
                current_group_y.append(y)
                current_group_iso.append(iso)
            else:
                unique_utr_group_list.append(current_group_y)
                unique_iso_group_list.append(current_group_iso)
                current_group_y = [y]
                current_group_iso = [iso]

    if len(current_group_y) > 0:
        unique_utr_group_list.append(current_group_y)
        unique_iso_group_list.append(current_group_iso)

    return unique_utr_group_list, unique_iso_group_list


def process_utr_coordinates_torino_w_transcripts(coords_dict_torino):
    y_dict = {}
    for x, val in coords_dict_torino.items():
        ys   = val['y']
        isos = val['iso']

        # Sort both by y, keeping iso aligned
        sorted_pairs = sorted(zip(ys, isos), key=lambda p: p[0])
        ys_sorted, isos_sorted = zip(*sorted_pairs)

        coord_groups, iso_groups = get_unique_utr_coords_w_iso(x, ys_sorted, isos_sorted)
        coords_groups_median = [np.median(y) for y in coord_groups]

        y_dict[x] = {
            'coord_groups':  coord_groups,
            'iso_groups':    iso_groups,
            'median_coords': coords_groups_median,
        }

    return y_dict


def process_utr_coordinates_torino_w_transcripts(coords_dict_torino, strand):
    y_dict = {}
    for x, val in coords_dict_torino.items():
        ys   = val['y']
        isos = val['iso']

        # Sort by actual |x - y| distance, shortest first
        sorted_pairs = sorted(zip(ys, isos), key=lambda p: abs(x - p[0]))
        ys_sorted, isos_sorted = zip(*sorted_pairs)

        coord_groups, iso_groups = get_unique_utr_coords_w_iso(x, ys_sorted, isos_sorted)
        coords_groups_median = [np.median(y) for y in coord_groups]

        y_dict[x] = {
            'coord_groups':  coord_groups,
            'iso_groups':    iso_groups,
            'median_coords': coords_groups_median,
        }

    return y_dict


def process_utr_coordinates_torino(coords_dict_torino, coords_dict_gencode):
    y_dict = {}
    for x, ys in coords_dict_torino.items():
        ys = sorted(ys)
        coord_groups = get_unique_utr_coords(x, ys)
        coords_groups_median = [np.median(y) for y in coord_groups]

        if x in coords_dict_gencode.keys():
            gencode_ys = coords_dict_gencode[x]

            exon_compat_list = []
            for torino_y in coords_groups_median:
                if any([compatible(x, gen_y, torino_y) for gen_y in gencode_ys]):
                    exon_compat_list.append(True)
                else:
                    exon_compat_list.append(False)

        else:
            exon_compat_list = [False]*len(coords_groups_median)
            
        all_gencode_ys = []
        for gen_x in coords_dict_gencode.keys():
            all_gencode_ys.extend(coords_dict_gencode[gen_x])

        utr_compat_list = []
        for torino_y in coords_groups_median:
            if any([compatible(x, gen_y, torino_y) for gen_y in all_gencode_ys]):
                utr_compat_list.append(True)
            else:
                utr_compat_list.append(False)

        y_dict.update({x:{
            'coords':coord_groups,
            'median_coords':coords_groups_median,
            'exon_match':exon_compat_list,
            'utr_match':utr_compat_list
        }})

    return y_dict


# --- the *_w_transcript helpers (In[84]), overriding those above
# Copied from UTR3.ipynb cell 123 (In[84]); definitions only, since
# the cell also did module-level work against names built elsewhere.
def get_last_exons_w_transcript(gene_df):
    last_exons = {}
    strand = gene_df.strand.iloc[0]
    for transcript, df in gene_df.groupby('transcript_id'):
        exon_chain = get_exon_chain(df)
        if strand == '+':
            last_exons[transcript] = exon_chain[-1]
        else:
            last_exons[transcript] = exon_chain[0]
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
    return last_exons_dict, strand


def merge_extra_isoforms(coords_dict, gene_df_extra, strand):
    """
    Given an appris-derived coords_dict {x: {y:[...], iso:[...]}},
    check extra isoforms: if their last-exon start (x) matches an existing
    key, add their y and transcript_id if not already present.
    """
    last_exons_extra, _ = get_last_exons_w_transcript(gene_df_extra)

    for transcript, coords in last_exons_extra.items():
        a, b = coords[0], coords[1]
        x, y = (a, b) if strand == '+' else (b, a)

        if x in coords_dict:  # only add if x matches an appris UTR start
            if y not in coords_dict[x]['y']:
                coords_dict[x]['y'].append(y)
                coords_dict[x]['iso'].append(transcript)

    return coords_dict


def process_utr_coordinates_torino_w_transcripts(coords_dict_torino, strand):
    y_dict = {}
    for x, val in coords_dict_torino.items():
        ys   = val['y']
        isos = val['iso']

        sorted_pairs = sorted(zip(ys, isos), key=lambda p: abs(x - p[0]))
        ys_sorted, isos_sorted = zip(*sorted_pairs)

        coord_groups, iso_groups = get_unique_utr_coords_w_iso(x, ys_sorted, isos_sorted)
        coords_groups_median = [np.median(y) for y in coord_groups]

        y_dict[x] = {
            'coord_groups':  coord_groups,
            'iso_groups':    iso_groups,
            'median_coords': coords_groups_median,
        }

    return y_dict


def process_utr_loadings_w_added(grouped_torino_ref, grouped_extra, gene, median_EL):
    """
    grouped_torino_ref : DataFrameGroupBy over appris/reference isoforms
    grouped_extra      : DataFrameGroupBy over all isoforms (same structure)
                         Pass None to skip the extra-isoform merging step.
    """
    torino_ref_df = grouped_torino_ref.get_group(gene)
    strand = torino_ref_df.iloc[0].strand

    # Build appris-derived coords dict
    coords_dict, strand = group_by_exon_w_transcript(
        get_last_exons_w_transcript(torino_ref_df)
    )

    # Optionally merge extra isoforms that share an appris exon start
    if grouped_extra is not None and gene in grouped_extra.groups:
        gene_df_extra = grouped_extra.get_group(gene)
        coords_dict = merge_extra_isoforms(coords_dict, gene_df_extra, strand)

    isos = process_utr_coordinates_torino_w_transcripts(coords_dict, strand)

    long_EL_list  = []
    short_EL_list = []

    for x, iso_dict in isos.items():
        iso_groups = iso_dict['iso_groups']
        if len(iso_groups) > 1:
            iso_short = iso_groups[0]
            iso_long  = iso_groups[-1]
        else:
            continue

        short_EL = median_EL.loc[median_EL.index.intersection(iso_short)].sum(axis=0)
        long_EL  = median_EL.loc[median_EL.index.intersection(iso_long)].sum(axis=0)

        short_EL_list.append(short_EL)
        long_EL_list.append(long_EL)

    return short_EL_list, long_EL_list


# --- process_utr_loadings_remove_extreme
# Copied from UTR3.ipynb cell 124 (In[85]); definitions only, since
# the cell also did module-level work against names built elsewhere.
def process_utr_loadings_remove_extreme(grouped_torino_ref, grouped_extra, gene, median_EL):
    torino_ref_df = grouped_torino_ref.get_group(gene)
    strand = torino_ref_df.iloc[0].strand

    # Build appris-derived coords dict
    coords_dict, strand = group_by_exon_w_transcript(
        get_last_exons_w_transcript(torino_ref_df)
    )

    # Find the longest appris UTR endpoint
    # on + strand: largest y; on - strand: smallest y
    all_appris_ys = [y for val in coords_dict.values() for y in val['y']]
    if strand == '+':
        appris_extreme = max(all_appris_ys)
    else:
        appris_extreme = min(all_appris_ys)

    # Check extra isoforms for ambiguity before merging
    if grouped_extra is not None and gene in grouped_extra.groups:
        gene_df_extra = grouped_extra.get_group(gene)
        last_exons_extra, _ = get_last_exons_w_transcript(gene_df_extra)

        for transcript, coords in last_exons_extra.items():
            a, b = coords[0], coords[1]
            x, y = (a, b) if strand == '+' else (b, a)

            # Check if this isoform ends beyond the appris extreme
            # AND does not share an appris exon start (i.e. would be silently skipped)
            extends_beyond = (y > appris_extreme) if strand == '+' else (y < appris_extreme)
            shares_exon_start = x in coords_dict

            if extends_beyond and not shares_exon_start:
                return None  # ambiguous — skip gene entirely

        # Safe to merge
        coords_dict = merge_extra_isoforms(coords_dict, gene_df_extra, strand)

    isos = process_utr_coordinates_torino_w_transcripts(coords_dict, strand)

    long_EL_list  = []
    short_EL_list = []

    for x, iso_dict in isos.items():
        iso_groups = iso_dict['iso_groups']
        if len(iso_groups) > 1:
            iso_short = iso_groups[0]
            iso_long  = iso_groups[-1]
        else:
            continue

        short_EL = median_EL.loc[median_EL.index.intersection(iso_short)].sum(axis=0)
        long_EL  = median_EL.loc[median_EL.index.intersection(iso_long)].sum(axis=0)

        short_EL_list.append(short_EL)
        long_EL_list.append(long_EL)

    return short_EL_list, long_EL_list


def load_utr_loadings():
    """long_df / short_df / scaled_all -- cells 1, 3, 46, 48, 125, 126, 128, 131."""
    def build():
        from sklearn.preprocessing import scale
        # -- cell 1 (In[2])
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

        # -- cell 3 (In[4])
        human_iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                                names = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon', 'notes'])

        human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

        # -- cell 120 (In[82])
        human_EL = pd.read_csv('../code/results/torino/tables/Human/isoforms_filtered.merged_isoforms.EL.bed.gz',
                        sep='\t')


        human_EL['pid'] = human_EL.gid + '.' + human_EL.pid

        human_EL.index = human_EL.pid

        # -- cell 116 (In[79])
        EL = get_EL_for_gene('Human', 'SRSF3')
        metadata = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv', sep='\t')
        samples = [x for x in EL.columns if x.startswith('GTEX')]

        human_metadata = metadata[['entity:sample_id', 'tissue_id']]
        human_metadata.columns = ['Sample', 'Tissue']
        human_metadata.index = human_metadata.Sample

        # -- cell 121 (In[83]) -- median_EL, which the loop in cell 126 needs.
        # Omitting it made every gene raise NameError inside that loop's bare
        # `except: continue`, so long_df came out empty with no error shown.
        median_EL = median_by_tissue(human_metadata, human_EL)

        # -- cell 46 (In[41])
        # El bueno

        gencode_iso = pd.read_csv('../code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz', sep='\t',
                                 names = ['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id', 'strand', 'exon',
                                          'score', 'basic', 'ensembl', 'MANE', 'appris', 'annot'])
        # gencode_iso['chrom'] = gencode_iso['#chrom']
        # gencode_iso['gene_id'] = gencode_iso.gene_name

        annot_ref = annot.loc[annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x)) | (annot.gencode_ref != '.')]
        ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]

        grouped_torino_ref = ref_chain_iso.groupby('gene_id')
        grouped_torino_all = human_iso.groupby('gene_id')#human_iso.groupby('gene_id')
        grouped_gencode = gencode_iso.groupby('gene_name')

        genes_dict = {}

        # Iterate through one and access the others
        for gene, torino_ref_df in tqdm(grouped_torino_ref):
            torino_all_df = grouped_torino_all.get_group(gene)
            gencode_df = grouped_gencode.get_group(gene)
            # your code here
            coords_dict_torino = process_gene_with_ref(torino_all_df, torino_ref_df)
            coords_dict_gencode = group_by_exon(get_last_exons(gencode_df))
            genes_dict.update({gene:process_utr_coordinates_torino(coords_dict_torino, coords_dict_gencode)})

        # flecha

        # -- cell 48 (In[43])
        torino_utrs_all_gene_start = dict_to_dataframe(genes_dict)

        # -- cell 55 (In[116]) -- adds gene_start, which cell 125 groups on
        torino_utrs_all_gene_start['gene_start'] = torino_utrs_all_gene_start.gene +':' + torino_utrs_all_gene_start.exon_start.astype(str)
        idx = torino_utrs_all_gene_start.value_counts('gene_start').loc[torino_utrs_all_gene_start.value_counts('gene_start')>=2].index
        torino_utrs_all_gene_start.loc[torino_utrs_all_gene_start.gene_start.isin(idx)].utr_match.sum()

        # This build replays the notebook cells, so it rebuilds its own UTR
        # table instead of going through load_torino_utrs -- apply the gene
        # subset here or it never reaches this panel.
        torino_utrs_all_gene_start = _subset(
            torino_utrs_all_gene_start, torino_utrs_all_gene_start.gene)

        # -- cell 125 (In[86])
        genes_utr_ext = [x.split(':')[0] for x in torino_utrs_all_gene_start.value_counts('gene_start').loc[torino_utrs_all_gene_start.value_counts('gene_start')>=2].index]

        # -- cell 126 (In[87])
        annot_ref = annot.loc[annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x)) | (annot.gencode_ref != '.')]
        ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]

        grouped_torino_ref = ref_chain_iso.groupby('gene_id')
        grouped_torino_all = human_iso.groupby('gene_id')#human_iso.groupby('gene_id')


        aver_long_list = []
        aver_short_list = []
        gene_list = []
        for gene in tqdm(genes_utr_ext):
            try:
                aver = process_utr_loadings_remove_extreme(grouped_torino_ref, grouped_torino_all, gene, median_EL)
                if aver is None:
                    continue
                if len(aver[0]) > 1:
                    gene_name = f'{gene}_1'
                    j = 1
                else:
                    gene_name = gene
                for i in range(len(aver[0])):
                    aver_long_list.append(aver[1][i])
                    aver_short_list.append(aver[0][i])
                    gene_list.append(gene_name)
                    if len(aver[0]) > 1:
                        j += 1
                        gene_name = f'{gene}_{str(j)}'
            except:
                continue
        
                # monteverdi

        # -- cell 128 (In[88])
        long_df = pd.concat(aver_long_list, axis=1).T
        long_df.index = gene_list

        short_df = pd.concat(aver_short_list, axis=1).T
        short_df.index = gene_list

        from sklearn.preprocessing import scale
        from scipy.stats import zscore

        # -- cell 129 (In[89])
        brain_tissues = [x for x in long_df.columns if (x.startswith('Brain') and ('Cereb' not in x))]
        cereb_tissues = [x for x in long_df.columns if (x.startswith('Brain') and ('Cereb' in x))]
        musc_tissues = []
        other_tissues = [x for x in long_df.columns if (not x.startswith('Brain'))]
        tissues_order = brain_tissues + cereb_tissues + other_tissues

        # -- cell 131 (In[90])
        select_long = long_df.loc[(long_df.max(axis=1) - long_df.min(axis=1)) >= 0.05]
        scaled_long = pd.DataFrame(scale(select_long, axis=1))
        scaled_long.columns = select_long.columns
        scaled_long.index = select_long.index

        select_short = short_df.loc[(short_df.max(axis=1) - short_df.min(axis=1)) >= 0.05]
        scaled_short = pd.DataFrame(scale(select_short, axis=1))
        scaled_short.columns = select_short.columns
        scaled_short.index = select_short.index

        idx = select_short.index.intersection(select_long.index)


        aver = (select_long.loc[idx]/(select_long.loc[idx] + select_short.loc[idx])).dropna()

        scaled_all = pd.DataFrame(scale(aver, axis=1))

        scaled_all.columns = aver.columns
        scaled_all.index = aver.index
        return long_df, short_df, scaled_all, brain_tissues, cereb_tissues
    return _cached(f'utr_loadings:{_tag()}', build)


@_panel
def panel_polyA_heatmap():
    """Notebook cell 135 (In[164])."""
    long_df, short_df, scaled_all, brain_tissues, cereb_tissues = load_utr_loadings()
    color_list = [gtex_colors2[x]['tissue_color_hex']
                  for x in scaled_all.mean(axis=0).sort_values().index[::-1]]

    # Row colours: is the longer 3' UTR used more in brain than elsewhere?
    # Brain means the brain and cerebellum columns; "other" is every remaining
    # tissue. Counted here rather than hardcoded, and the count goes in the
    # legend so the figure states its own n.
    brain_cols = list(brain_tissues) + list(cereb_tissues)
    other_cols = [c for c in scaled_all.columns if c not in set(brain_cols)]
    brain_higher = (scaled_all[brain_cols].mean(axis=1)
                    > scaled_all[other_cols].mean(axis=1))
    n_higher = int(brain_higher.sum())
    n_lower = int((~brain_higher).sum())
    row_colors = brain_higher.map({True: '#b2182b', False: '#2166ac'})
    row_colors.name = 'Brain'

    row_order = scaled_all[brain_cols].mean(axis=1).sort_values().index[::-1]
    col_order = scaled_all.mean(axis=0).sort_values().index[::-1]
    g = sns.clustermap(scaled_all.loc[row_order, col_order],
                       col_colors=color_list, row_colors=row_colors.loc[row_order],
                       col_cluster=False, row_cluster=False,
                       cmap='bwr', figsize=(6, 8), rasterized=True)

    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel(f"3' UTR across {len(scaled_all):,} genes with"
                            "\n> 0.05 usage difference across tissues")
    g.ax_cbar.set_ylabel("Scaled usage of longer\nalt. 3' UTR", fontsize=9)

    import matplotlib.patches as _mpatches
    g.ax_heatmap.legend(
        handles=[_mpatches.Patch(color='#b2182b',
                                 label=f'higher in brain (n = {n_higher:,})'),
                 _mpatches.Patch(color='#2166ac',
                                 label=f'lower in brain (n = {n_lower:,})')],
        title="Longer 3' UTR usage", loc='upper left',
        bbox_to_anchor=(1.18, 1.0), frameon=False, fontsize=8, title_fontsize=8)

    return g.figure


PANELS['polyA_heatmap'] = panel_polyA_heatmap


# --------------------------------------------------------------------------
# polyA numbers
#
# Three per-gene passes over the isoform annotation, a few minutes each:
#   utr counts        In[27]   how many polyA sites each gene has
#   torino utrs       In[41]   every Torino 3' UTR, and whether GENCODE has it
#   gencode utrs      In[76]   the same from GENCODE's side
# --------------------------------------------------------------------------
def load_utr_counts():
    """Per-gene polyA site counts -- UTR3.ipynb cell 31 (In[27])."""
    def build():
        # -- cell 1 (In[2])
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

        # -- cell 3 (In[4])
        human_iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                                names = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon', 'notes'])

        human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

        # -- cell 31 (In[27])
        def group_all_utrs(last_exons):
            strand = last_exons[1]
            if strand == '+':
                x = last_exons[0][0][0]
                ys = [y[1] for y in last_exons[0]]
            else:
                x = last_exons[0][0][1]
                ys = [y[0] for y in last_exons[0]]
            return {x:ys}
            # annot_ref = annot.loc[~annot.efficiency_factor]
        # ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]
        from tqdm import tqdm
        gene_list_all = []
        utr_list_all = []
        for gene, gene_df in tqdm(human_iso.groupby('gene_id')):
            utrs_ = get_alt_utrs(group_all_utrs(get_last_exons(gene_df)))
            gene_list_all.append(gene)
            utr_list_all.append(utrs_)

        utr_df_all_all = pd.DataFrame({'gene':gene_list_all, 'utrs':utr_list_all})
        return _subset(utr_df_all_all, utr_df_all_all.gene)
    return _cached(f'utr_counts:{_tag()}', build)


def load_torino_utrs():
    """Torino 3' UTRs with gene_start and utr_match -- cells 46, 47, 48, 55."""
    def build():
        # -- cell 1 (In[2])
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

        # -- cell 3 (In[4])
        human_iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                                names = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon', 'notes'])

        human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

        # -- cell 46 (In[41])
        # El bueno

        gencode_iso = pd.read_csv('../code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz', sep='\t',
                                 names = ['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id', 'strand', 'exon',
                                          'score', 'basic', 'ensembl', 'MANE', 'appris', 'annot'])
        # gencode_iso['chrom'] = gencode_iso['#chrom']
        # gencode_iso['gene_id'] = gencode_iso.gene_name

        annot_ref = annot.loc[annot.appris_ref.apply(lambda x: ('ref_transcript' in x) or ('intron_chain' in x)) | (annot.gencode_ref != '.')]
        ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]

        grouped_torino_ref = ref_chain_iso.groupby('gene_id')
        grouped_torino_all = human_iso.groupby('gene_id')#human_iso.groupby('gene_id')
        grouped_gencode = gencode_iso.groupby('gene_name')

        genes_dict = {}

        # Iterate through one and access the others
        for gene, torino_ref_df in tqdm(grouped_torino_ref):
            torino_all_df = grouped_torino_all.get_group(gene)
            gencode_df = grouped_gencode.get_group(gene)
            # your code here
            coords_dict_torino = process_gene_with_ref(torino_all_df, torino_ref_df)
            coords_dict_gencode = group_by_exon(get_last_exons(gencode_df))
            genes_dict.update({gene:process_utr_coordinates_torino(coords_dict_torino, coords_dict_gencode)})

        # flecha

        # -- cell 48 (In[43])
        torino_utrs_all_gene_start = dict_to_dataframe(genes_dict)

        # -- cell 55 (In[116])
        torino_utrs_all_gene_start['gene_start'] = torino_utrs_all_gene_start.gene +':' + torino_utrs_all_gene_start.exon_start.astype(str)
        idx = torino_utrs_all_gene_start.value_counts('gene_start').loc[torino_utrs_all_gene_start.value_counts('gene_start')>=2].index
        torino_utrs_all_gene_start.loc[torino_utrs_all_gene_start.gene_start.isin(idx)].utr_match.sum()
        return _subset(torino_utrs_all_gene_start, torino_utrs_all_gene_start.gene)
    return _cached(f'torino_utrs:{_tag()}', build)


def load_gencode_utrs():
    """GENCODE 3' UTRs with gene_start -- cells 219, 220, 222.

    gencode_iso is read at the top of cell 46, the Torino loop, so it has to be
    repeated here: cell 219 uses it but never reads it.
    """
    def build():
        gencode_iso = pd.read_csv(
            '../code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
            sep='\t',
            names=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id',
                   'strand', 'exon', 'score', 'basic', 'ensembl', 'MANE', 'appris',
                   'annot'])
        # -- cell 1 (In[2])
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

        # -- cell 3 (In[4])
        human_iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                                names = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon', 'notes'])

        human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

        # -- cell 219 (In[76])
        # grouped_torino_ref = ref_chain_iso.groupby('gene_id')
        grouped_torino_all = human_iso.groupby('gene_id')
        grouped_gencode = gencode_iso.groupby('gene_name')
        grouped_gencode_appris = gencode_iso.loc[(~gencode_iso.appris.isna()) & gencode_iso.gene_name.isin(annot.gene)].groupby('gene_name')

        genes_dict_gencode = {}

        # Iterate through one and access the others
        for gene, gencode_appris_df in tqdm(grouped_gencode_appris):
            gencode_all_df = grouped_gencode.get_group(gene)
            torino_all_df = grouped_torino_all.get_group(gene)
            # your code here
            coords_dict_gencode = process_gene_with_ref(gencode_appris_df, gencode_appris_df)
            coords_dict_torino = group_by_exon(get_last_exons(torino_all_df))
            genes_dict_gencode.update({gene:process_utr_coordinates_torino(coords_dict_gencode, coords_dict_torino)})

        # -- cell 220 (In[77])
        gencode_utrs = dict_to_dataframe(genes_dict_gencode)

        # -- cell 222 (In[87])
        gencode_utrs['gene_start'] = gencode_utrs.gene + ':' + gencode_utrs.exon_start.astype(str)
        return _subset(gencode_utrs, gencode_utrs.gene)
    return _cached(f'gencode_utrs:{_tag()}', build)


def polya_numbers(with_heatmap=True):
    """The polyA counts, honouring GENE_SUBSET.

    Units matter here, and mixing them is what made the first version of this
    table look self-contradictory. Three different objects are involved:

      utr_df_all_all   one row per tested gene, with its polyA-site count. A
                       "site" is a cluster of terminal-exon ends within 200 bp
                       (get_unique_utrs), not a distinct coordinate.
      torino_utrs      one row per (gene, terminal-exon start, end), only for
                       genes with an appris or GENCODE reference -- about 72%
                       of genes. `utr_match` exists only here.
      gencode_utrs     the same from GENCODE's side, over all genes.

    So the site counts are reported per distinct `gene:end`, and the two
    populations are made unambiguous:

      APA              in the shared-terminal-exon table AND labelled
                       multi-site. 71 genes are in the first but not the
                       second -- their ends cluster into one site but sit on a
                       shared terminal-exon start -- and are dropped.
      single-site      labelled single-site AND exactly one gene:end pair, so
                       one gene is one site.
    """
    counts = load_utr_counts()
    tor = load_torino_utrs()
    gen = load_gencode_utrs()
    single = set(counts.loc[counts.utrs == 0].gene)
    multi = set(counts.loc[counts.utrs > 0].gene)

    n = {}
    # --- the gene-level partition (all tested genes)
    n['tested genes'] = len(counts)
    n['genes, single site'] = len(single)
    n['genes, two or more sites'] = len(multi)
    n['polyA sites'] = int(counts.utrs.sum() + (counts.utrs == 0).sum())
    n['APA sites (sum over genes)'] = int(counts.utrs.sum())

    # --- Torino APA, unambiguous
    vc = tor.value_counts('gene_start')
    idx = vc.loc[vc >= 2].index
    apa = tor.loc[tor.gene_start.isin(idx) & tor.gene.isin(multi)]
    apa_ends = apa.groupby(['gene', 'exon_end']).utr_match.max().reset_index()
    n['APA genes'] = int(apa.gene.nunique())
    n['APA sites (gene:end)'] = len(apa_ends)
    n['APA sites matched by GENCODE'] = int(apa_ends.utr_match.sum())
    n['APA genes dropped as ambiguous'] = len(
        set(tor.loc[tor.gene_start.isin(idx)].gene) - multi)

    # --- Torino single-site, unambiguous
    s_ends = (tor.loc[tor.gene.isin(single)]
                 .groupby(['gene', 'exon_end']).utr_match.max().reset_index())
    per_gene = s_ends.groupby('gene').size()
    conf = s_ends.loc[s_ends.gene.isin(set(per_gene.loc[per_gene == 1].index))]
    n['single-site genes confirmed'] = len(conf)
    n['single sites matched by GENCODE'] = int(conf.utr_match.sum())
    n['single-site genes dropped (>1 end)'] = int((per_gene > 1).sum())
    n['single-site genes not in the reference table'] = len(single - set(tor.gene))

    # --- GENCODE side, same gene:end unit
    gvc = gen.value_counts('gene_start')
    gidx = gvc.loc[gvc >= 2].index
    for key, sub in [('GENCODE APA', gen.loc[gen.gene_start.isin(gidx)]),
                     ('GENCODE constitutive', gen.loc[~gen.gene_start.isin(gidx)])]:
        ends = sub.groupby(['gene', 'exon_end']).utr_match.max().reset_index()
        n[f'{key} sites (gene:end)'] = len(ends)
        n[f'{key} found by Torino'] = int(ends.utr_match.sum())

    # The last two need the loadings loop, the hour-long part; the rest are
    # independent of it, so a failure there must not hide them.
    if with_heatmap:
        _, _, scaled_all, brain, cereb = load_utr_loadings()
        other = [c for c in scaled_all.columns if not c.startswith('Brain')]
        n['genes with variable APA'] = int(len(scaled_all))
        n['longer 3UTR in brain'] = int(
            (scaled_all[brain + cereb].mean(axis=1)
             > scaled_all[other].mean(axis=1)).sum())
    return n


# (row -> the row it is a fraction of)
_POLYA_DEN = {
    'genes, single site': 'tested genes',
    'genes, two or more sites': 'tested genes',
    'APA sites matched by GENCODE': 'APA sites (gene:end)',
    'single sites matched by GENCODE': 'single-site genes confirmed',
    'GENCODE APA found by Torino': 'GENCODE APA sites (gene:end)',
    'GENCODE constitutive found by Torino': 'GENCODE constitutive sites (gene:end)',
    'longer 3UTR in brain': 'genes with variable APA',
}

_POLYA_SECTIONS = [
    ('Genes and polyA sites', ['tested genes', 'genes, single site',
                               'genes, two or more sites', 'polyA sites',
                               'APA sites (sum over genes)']),
    ('Torino APA (multi-site genes, unambiguous)',
     ['APA genes', 'APA sites (gene:end)', 'APA sites matched by GENCODE',
      'APA genes dropped as ambiguous']),
    ('Torino single-site (one gene:end each)',
     ['single-site genes confirmed', 'single sites matched by GENCODE',
      'single-site genes dropped (>1 end)',
      'single-site genes not in the reference table']),
    ('GENCODE side (same gene:end unit)',
     ['GENCODE APA sites (gene:end)', 'GENCODE APA found by Torino',
      'GENCODE constitutive sites (gene:end)',
      'GENCODE constitutive found by Torino']),
    ('Variable APA across tissues',
     ['genes with variable APA', 'longer 3UTR in brain']),
]


def polya_table(with_heatmap=True):
    """The polyA numbers, uncorrected and corrected side by side, as markdown."""
    keep = _R.GENE_SUBSET
    try:
        _R.GENE_SUBSET = None
        free()
        a = polya_numbers(with_heatmap)
        _R.GENE_SUBSET = _R.load_select_genes()
        free()
        c = polya_numbers(with_heatmap)
    finally:
        _R.GENE_SUBSET = keep
        free()

    rows = ["**3' UTR / polyadenylation**", '',
            '| | all genes | corrected gene set | of |', '|---|---:|---:|---|']
    for title, keys in _POLYA_SECTIONS:
        keys = [k for k in keys if k in a]
        if not keys:
            continue
        rows.append(f'| **{title}** | | | |')
        for k in keys:
            den = _POLYA_DEN.get(k)
            cells = []
            for n in (a, c):
                v = n[k]
                cells.append(f'{v:,}' + (f' ({100 * v / n[den]:.1f}%)' if den else ''))
            rows.append(f'| {k} | {cells[0]} | {cells[1]} | {den or ""} |')
    rows += ['', '**What the exclusion rows mean.** A site is a *cluster* of '
                 'terminal-exon ends within 200 bp (`get_unique_utrs`), not a '
                 'distinct coordinate, so the site counter and the '
                 'terminal-exon table can disagree about a gene. The three '
                 'rows below record the genes where they do, which are left '
                 'out rather than assigned:', '',
             '- **APA genes dropped as ambiguous** -- two or more isoforms '
             'share a terminal-exon start (so the terminal-exon table shows '
             'APA), but the gene is *not* labelled multi-site, because its '
             'ends fall within 200 bp and cluster into a single site. The two '
             'criteria contradict each other.',
             '- **single-site genes dropped (>1 end)** -- the same '
             'contradiction reversed: labelled single-site, yet carrying more '
             'than one distinct `gene:end`. Removing them is what makes one '
             'gene equal one site exactly in the confirmed set, so the '
             'single-polyA bar can count genes and sites interchangeably.',
             '- **single-site genes not in the reference table** -- no row at '
             'all in the reference-matched terminal-exon table, which is what '
             'defines the exon end. With no coordinate there is no `gene:end` '
             'to count, so these are excluded for missing data rather than '
             'for conflicting labels.']
    larger = [k for k in a if c[k] > a[k]]
    if larger:
        rows += ['', f'> **check**: corrected exceeds all genes for {larger} '
                     '-- impossible for a subset.']
    return '\n'.join(rows)


# --------------------------------------------------------------------------
# UTRs_cumulative
#
# Chain: UTR3.ipynb cells 265, 271, 278, 279, 280, 281, 283, 288
# (In[585], In[597], In[1224], In[1225], In[1226], In[1227], In[1229],
# In[1280]).
#
# The expensive step is cell 278: nine ~8.7 GB ONT BAMs, fetching every gene's
# span and counting read ends within 200 bp of each polyA site. Nothing on disk
# caches it, so it is built once into figure_data/bam_counts.pkl and read back
# from then on -- build_bam_counts() refuses to recompute if the pickle exists.
#
# Three lines of cell 278's loop are dropped deliberately:
#
#     coords_dict_gencode = process_gene_with_ref(gencode_appris_df, gencode_appris_df)
#     coords_dict_torino  = group_by_exon(get_last_exons(torino_all_df))
#     genes_dict_gencode.update({gene: process_utr_coordinates_torino(...)})
#
# They sit inside the per-gene loop but take the same arguments on every
# iteration, so they recompute one identical result thousands of times per BAM,
# and nothing downstream reads genes_dict_gencode -- its only consumer is cell
# 220 (In[77]), which builds gencode_utrs, an *input* to this cell. As written
# they would also raise NameError, since `genes_dict_gencode = {}` is commented
# out in cell 278; the name survived only from an earlier kernel. Dropping them
# changes no output and removes five dependencies.
# --------------------------------------------------------------------------
BAM_DIR = '../code/resources/GTEx/ONT/bam'

# -- cell 265 (In[585])
brain_bams = ['GTEx_v9_ONT_data_sequence_data_GTEX-1192X-0011-R10a-SM-4RXXZ.FAK49046.bam',
              'GTEx_v9_ONT_data_sequence_data_GTEX-145MH-0011-R3b-SM-5CTCM.FAK41439.bam',
              'GTEx_v9_ONT_data_sequence_data_GTEX-11H98-0011-R11b-SM-4SFLZ.FAK46829.bam',
              'GTEx_v9_ONT_data_sequence_data_GTEX-QDT8-0011-R10A-SM-2FKJB.FAK49182.bam'
             ]
muscle_bams = ['GTEx_v9_ONT_data_sequence_data_GTEX-13QJ3-0726-SM-7LDHS_rep.FAK93371.bam',
               'GTEx_v9_ONT_data_sequence_data_GTEX-15RIE-1726-SM-5V7WB.FAK41916.bam',
               'GTEx_v9_ONT_data_sequence_data_GTEX-17MFQ-1926-SM-6659L.FAK44757.bam',
               'GTEx_v9_ONT_data_sequence_data_GTEX-1C64N-0326-SM-6ZHT1.FAK53752.bam',
               'GTEx_v9_ONT_data_sequence_data_GTEX-1GN1W-0226-SM-7AGLJ.FAK44655.bam'
              ]

SAMPLE_NAMES = [f'brain_{i+1}' for i in range(4)] + [f'muscle_{i+1}' for i in range(5)]
PCT_COLS = [f'{s}_pct' for s in SAMPLE_NAMES]


# -- cell 271 (In[597])
def get_utr_dict(utr_df_in):
    utr_dict = {}
    for end in utr_df_in.exon_end:
        utr_dict.update({int(end):0})
    return utr_dict


def load_torino_utrs_ref_match():
    """torino_utrs_ref_match -- cell 71 (In[58]), named at cell 180 (In[59]).

    The same per-gene loop as load_utr_loadings' (cell 46), over the same
    grouping, with the same GENCODE side -- differing in one argument:

        cell 46:  process_gene_with_ref(torino_all_df, torino_ref_df)
        cell 71:  process_gene_with_ref(torino_ref_df, torino_ref_df)

    so the query set is the reference-matched isoforms rather than all of them.
    """
    def build():
        annot = load_annot()      # same frame cells 1-2 build inline
        human_iso = pd.read_csv(
            '../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
            names=['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand',
                   'factors', 'exon', 'notes'])
        human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id
        gencode_iso = pd.read_csv(
            '../code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
            sep='\t',
            names=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id',
                   'strand', 'exon', 'score', 'basic', 'ensembl', 'MANE', 'appris',
                   'annot'])

        annot_ref = annot.loc[annot.appris_ref.apply(
            lambda x: ('ref_transcript' in x) or ('intron_chain' in x))
            | (annot.gencode_ref != '.')]
        ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]

        grouped_torino_ref = ref_chain_iso.groupby('gene_id')
        grouped_gencode = gencode_iso.groupby('gene_name')

        genes_dict = {}
        for gene, torino_ref_df in tqdm(grouped_torino_ref):
            try:
                gencode_df = grouped_gencode.get_group(gene)
            except KeyError:
                continue
            coords_dict_torino = process_gene_with_ref(torino_ref_df, torino_ref_df)
            coords_dict_gencode = group_by_exon(get_last_exons(gencode_df))
            genes_dict.update(
                {gene: process_utr_coordinates_torino(coords_dict_torino,
                                                      coords_dict_gencode)})

        out = dict_to_dataframe(genes_dict)
        return _subset(out, out.gene)
    return _cached(f'torino_utrs_ref_match:{_tag()}', build)


# One pickle per gene set. Without the tag, a single bam_counts.pkl was shared
# and the corrected notebook read counts built with GENE_SUBSET unset: 17,335
# gene:end pairs over 12,155 genes, where the corrected table has 17,005 over
# 11,915. Same failure mode as the hardcoded totals -- a cached result silently
# standing in for a different gene set.
def _bam_counts_pkl():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'figure_data', f'bam_counts.{_tag()}.pkl')


def build_bam_counts(force=False, out_path=None):
    """bam_dict2 -- cell 278 (In[1224]). Reads nine ONT BAMs. Hours.

    Writes the result to figure_data/bam_counts.pkl and REFUSES to recompute
    if that file already exists: this is the one step here that costs real
    time, and it never needs doing twice. Pass force=True only to rebuild it
    deliberately.

    Run this from a compute node, not a login node.
    """
    import pickle
    out_path = out_path or _bam_counts_pkl()
    if os.path.exists(out_path) and not force:
        print(f'{out_path} already exists -- nothing to do. '
              f'force=True to rebuild.')
        return out_path

    import pysam
    torino_utrs_ref_match = load_torino_utrs_ref_match()
    gencode_utrs = load_gencode_utrs()
    gencode_iso = pd.read_csv(
        '../code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        sep='\t',
        names=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id',
               'strand', 'exon', 'score', 'basic', 'ensembl', 'MANE', 'appris',
               'annot'])

    torino_utrs_ref_match = torino_utrs_ref_match.copy()
    gencode_utrs = gencode_utrs.copy()
    torino_utrs_ref_match.exon_end = torino_utrs_ref_match.exon_end.astype(int)
    gencode_utrs.exon_end = gencode_utrs.exon_end.astype(int)

    torino_unique_utr = (torino_utrs_ref_match.groupby(['gene', 'exon_end'])
                         .utr_match.max().reset_index())
    gencode_unique_utr = (gencode_utrs.groupby(['gene', 'exon_end'])
                          .utr_match.max().reset_index())

    torino_grouped_utrs = torino_unique_utr.groupby('gene')
    gencode_grouped_utrs = gencode_unique_utr.groupby('gene')
    gencode_grouped_annot = gencode_iso.groupby('gene_name')

    bam_dict2 = {}
    for bamf in brain_bams + muscle_bams:
        print(f'[bam] {bamf}', flush=True)
        torino_gene_col, torino_exon_end_col, torino_utr_match_col = [], [], []
        gencode_gene_col, gencode_exon_end_col, gencode_utr_match_col = [], [], []
        torino_match_col, gencode_match_col = [], []
        torino_total_col, gencode_total_col = [], []

        samfile = pysam.AlignmentFile(f'{BAM_DIR}/{bamf}', 'rb')
        for gene, torino_gene_utr in tqdm(torino_grouped_utrs):
            try:
                gencode_gene_utr = gencode_grouped_utrs.get_group(gene)
                gencode_gene_annot = gencode_grouped_annot.get_group(gene)
            except KeyError:
                continue

            start = gencode_gene_annot.start.min()
            end = gencode_gene_annot.end.max()
            chrom = gencode_gene_annot.chrom.iloc[0]
            strand = gencode_gene_annot.strand.iloc[0]

            torino_dict = get_utr_dict(torino_gene_utr)
            gencode_dict = get_utr_dict(gencode_gene_utr)

            total_reads = 0
            for read in samfile.fetch(chrom, start, end):
                if read.is_unmapped:
                    continue
                if strand == '+':
                    read_start, read_end = read.reference_start, read.reference_end
                else:
                    read_start, read_end = read.reference_end, read.reference_start
                if abs(read_end - read_start) >= 200:
                    for torino_end in torino_dict:
                        if abs(read_end - torino_end) < 200:
                            torino_dict[torino_end] += 1
                    for gencode_end in gencode_dict:
                        if abs(read_end - gencode_end) < 200:
                            gencode_dict[gencode_end] += 1
                    total_reads += 1

            for torino_end in torino_dict:
                torino_match_col.append(torino_dict[torino_end])
                torino_total_col.append(total_reads)
                torino_gene_col.append(gene)
                torino_exon_end_col.append(torino_end)
            torino_utr_match_col.extend(list(torino_gene_utr.utr_match))

            for gencode_end in gencode_dict:
                gencode_match_col.append(gencode_dict[gencode_end])
                gencode_total_col.append(total_reads)
                gencode_gene_col.append(gene)
                gencode_exon_end_col.append(gencode_end)
            gencode_utr_match_col.extend(list(gencode_gene_utr.utr_match))

        samfile.close()
        bam_dict2[bamf] = {
            'torino': {'gene': torino_gene_col, 'utr': torino_exon_end_col,
                       'match': torino_utr_match_col, 'counts': torino_match_col,
                       'total_counts': torino_total_col},
            'gencode': {'gene': gencode_gene_col, 'utr': gencode_exon_end_col,
                        'match': gencode_utr_match_col, 'counts': gencode_match_col,
                        'total_counts': gencode_total_col}}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(bam_dict2, fh)
    print(f'wrote {out_path}')
    return out_path


def load_lr_dfs():
    """torino_lr_df, gencode_lr_df -- cell 279 (In[1225]), from the pickle."""
    def build():
        import pickle
        path = _bam_counts_pkl()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} does not exist. It holds the long-read counts for the '
                f'nine ONT BAMs for the {_tag()} gene set and takes ~45 min to '
                f'build, so it is not built on demand. Run build_bam_counts() '
                f'on a compute node with GENE_SUBSET set the same way.')
        with open(path, 'rb') as fh:
            bam_dict2 = pickle.load(fh)

        bamf_list = list(bam_dict2.keys())
        cols = ['gene', 'utr', 'match']

        def _first(kind, name):
            df = pd.DataFrame(bam_dict2[bamf_list[0]][kind])
            df.columns = cols + [f'{name}_counts', f'{name}_total_counts']
            df[f'{name}_pct'] = df[f'{name}_counts'] / df[f'{name}_total_counts']
            return df

        torino_lr_df = _first('torino', SAMPLE_NAMES[0])
        gencode_lr_df = _first('gencode', SAMPLE_NAMES[0])

        for i, bamf in enumerate(bamf_list[1:]):
            name = SAMPLE_NAMES[i + 1]
            for kind, base in (('torino', 'torino'), ('gencode', 'gencode')):
                df = pd.DataFrame(bam_dict2[bamf][kind])
                df.columns = cols + [f'{name}_counts', f'{name}_total_counts']
                df[f'{name}_pct'] = df[f'{name}_counts'] / df[f'{name}_total_counts']
                if kind == 'torino':
                    torino_lr_df = torino_lr_df.merge(df, on=cols)
                else:
                    gencode_lr_df = gencode_lr_df.merge(df, on=cols)
        return torino_lr_df, gencode_lr_df
    return _cached(f'lr_dfs:{_tag()}', build)


# -- cell 281 (In[1227])
def plot_cumulative_utr(x, steps=100):
    '''
    Function to get cumulative distribution.
    '''
    x = np.array(x)
    X = np.linspace(np.min(x), np.max(x), steps)
    y = [np.mean(x <= z) for z in X]
    return X, y


@_panel
def panel_UTRs_cumulative():
    """Cells 283 and 288 (In[1229], In[1280])."""
    torino_lr_df, gencode_lr_df = load_lr_dfs()

    # -- cell 283 (In[1229])
    _tg = list(torino_lr_df.loc[torino_lr_df.match][PCT_COLS].fillna(0).mean(axis=1))
    _t = list(torino_lr_df.loc[~torino_lr_df.match][PCT_COLS].fillna(0).mean(axis=1))
    _g = list(gencode_lr_df.loc[~gencode_lr_df.match][PCT_COLS].fillna(0).mean(axis=1))
    # The shared curve has two counts, because matching is not one-to-one: one
    # GENCODE end can match several Torino ends. Labelling only the Torino side
    # invites adding it to 'Gencode only' to get a GENCODE total, which
    # overcounts -- the GENCODE total is _n_gen_match + len(_g).
    _n_gen_match = int(gencode_lr_df.match.sum())
    tgx, tgy = plot_cumulative_utr(_tg)
    tx, ty = plot_cumulative_utr(_t)
    gx, gy = plot_cumulative_utr(_g)

    # -- cell 288 (In[1280])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(tgy, tgx, c='tab:purple', linewidth=3, label=f'Torino & Gencode ({len(_tg):,} Torino / {_n_gen_match:,} Gencode ends)')
    ax.plot(ty, tx, c='tab:blue', linewidth=3, label=f'Torino only (n = {len(_t):,})')
    ax.plot(gy, gx, c='tab:red', linewidth=3, label=f'Gencode only (n = {len(_g):,})')
    ax.set_xlabel('Percentage of polyA sites')
    ax.set_ylabel('Cumulative site usage percentage')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_title('Long-read mapping to polyA site')
    ax.legend(frameon=False)
    return fig


PANELS['UTRs_cumulative'] = panel_UTRs_cumulative


# ==========================================================================
# UTRs_cumulative, rebuilt on the table's site definitions
#
# Replaces panel_UTRs_cumulative. Two panels:
#
#   APA sites      Torino (unambiguous APA, multi-site genes) vs GENCODE APA
#   Single sites   Torino (confirmed, one gene:end each) vs GENCODE constitutive
#
# The published version counted every gene:end pair in torino_utrs_ref_match
# with no ambiguity filtering, which is why its numbers match nothing in the
# polyA table. Here the site sets ARE the table's, so the two agree: the curves
# carry 10,735 / 5,361 Torino sites and 5,927 / 14,599 GENCODE sites for the
# corrected gene set.
#
# Read counts come from the same pysam pass as before, but keyed on the ends
# these site sets use (load_torino_utrs / load_gencode_utrs, the cell-46 loop),
# not torino_utrs_ref_match (cell 71). That is a different set of ends, so the
# counts must be built separately -- hence a second pickle rather than a filter
# of the first.
# ==========================================================================
def _site_sets():
    """The four site sets, exactly as polya_numbers defines them."""
    counts = load_utr_counts()
    tor = load_torino_utrs()
    gen = load_gencode_utrs()
    single = set(counts.loc[counts.utrs == 0].gene)
    multi = set(counts.loc[counts.utrs > 0].gene)

    vc = tor.value_counts('gene_start')
    idx = vc.loc[vc >= 2].index
    apa = tor.loc[tor.gene_start.isin(idx) & tor.gene.isin(multi)]
    apa_ends = apa.groupby(['gene', 'exon_end']).utr_match.max().reset_index()

    s_ends = (tor.loc[tor.gene.isin(single)]
                 .groupby(['gene', 'exon_end']).utr_match.max().reset_index())
    per_gene = s_ends.groupby('gene').size()
    conf = s_ends.loc[s_ends.gene.isin(set(per_gene.loc[per_gene == 1].index))]

    gvc = gen.value_counts('gene_start')
    gidx = gvc.loc[gvc >= 2].index
    g_apa = (gen.loc[gen.gene_start.isin(gidx)]
                .groupby(['gene', 'exon_end']).utr_match.max().reset_index())
    g_sing = (gen.loc[~gen.gene_start.isin(gidx)]
                 .groupby(['gene', 'exon_end']).utr_match.max().reset_index())
    for df in (apa_ends, conf, g_apa, g_sing):
        df['exon_end'] = df.exon_end.astype(int)
    return {'torino_apa': apa_ends, 'torino_single': conf,
            'gencode_apa': g_apa, 'gencode_single': g_sing}


def _site_counts_pkl():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'figure_data', f'site_counts.{_tag()}.pkl')


def build_site_counts(force=False):
    """Long-read counts for the four site sets. Nine ONT BAMs, ~45 min.

    One pickle per gene set; refuses to rebuild if it exists.
    """
    import pickle
    out_path = _site_counts_pkl()
    if os.path.exists(out_path) and not force:
        print(f'{out_path} already exists -- nothing to do. force=True to rebuild.')
        return out_path

    import pysam
    sets = _site_sets()
    gencode_iso = pd.read_csv(
        '../code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        sep='\t',
        names=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id',
               'strand', 'exon', 'score', 'basic', 'ensembl', 'MANE', 'appris',
               'annot'])
    gannot = gencode_iso.groupby('gene_name')

    # every gene that appears in any site set, and the ends wanted per gene
    wanted = {}
    for name, df in sets.items():
        for gene, end in zip(df.gene, df.exon_end):
            wanted.setdefault(gene, {}).setdefault(name, []).append(int(end))

    out = {name: {'gene': [], 'end': [], 'matched': []} for name in sets}
    for name, df in sets.items():
        out[name]['matched'] = list(df.utr_match.astype(bool))
        out[name]['gene'] = list(df.gene)
        out[name]['end'] = list(df.exon_end.astype(int))
        out[name]['counts'] = {}        # bam -> list aligned to the rows above

    for bamf in brain_bams + muscle_bams:
        print(f'[bam] {bamf}', flush=True)
        samfile = pysam.AlignmentFile(f'{BAM_DIR}/{bamf}', 'rb')
        per_gene_counts = {}
        for gene, by_set in wanted.items():
            try:
                ga = gannot.get_group(gene)
            except KeyError:
                continue
            chrom = ga.chrom.iloc[0]
            strand = ga.strand.iloc[0]
            start, end = ga.start.min(), ga.end.max()
            ends = sorted({e for lst in by_set.values() for e in lst})
            tally = {e: 0 for e in ends}
            total = 0
            for read in samfile.fetch(chrom, start, end):
                if read.is_unmapped:
                    continue
                if strand == '+':
                    r_start, r_end = read.reference_start, read.reference_end
                else:
                    r_start, r_end = read.reference_end, read.reference_start
                if abs(r_end - r_start) >= 200:
                    for e in ends:
                        if abs(r_end - e) < 200:
                            tally[e] += 1
                    total += 1
            per_gene_counts[gene] = (tally, total)
        samfile.close()

        for name, df in sets.items():
            col = []
            for gene, e in zip(out[name]['gene'], out[name]['end']):
                tally, total = per_gene_counts.get(gene, ({}, 0))
                col.append((tally.get(e, 0), total))
            out[name]['counts'][bamf] = col

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(out, fh)
    print(f'wrote {out_path}')
    return out_path


def load_site_usage():
    """{set name: DataFrame with gene, end, matched, and one pct per sample}."""
    def build():
        import pickle
        path = _site_counts_pkl()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} does not exist. Run build_site_counts() on a compute '
                f'node once, with GENE_SUBSET set the same way (~45 min).')
        with open(path, 'rb') as fh:
            raw = pickle.load(fh)
        out = {}
        for name, d in raw.items():
            df = pd.DataFrame({'gene': d['gene'], 'end': d['end'],
                               'matched': d['matched']})
            for i, (bamf, col) in enumerate(d['counts'].items()):
                hits = np.array([c for c, _ in col], dtype=float)
                tots = np.array([t for _, t in col], dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    df[f'{SAMPLE_NAMES[i]}_pct'] = np.where(tots > 0, hits / tots,
                                                            np.nan)
            out[name] = df
        return out
    return _cached(f'site_usage:{_tag()}', build)


def _pooled_sites():
    """Every polyA site once, per annotation, with whether the other has it.

    APA and single-site sets are pooled: a site is a site regardless of which
    class its own annotation put it in. That matters because the two
    annotations disagree about the class -- a gene Torino calls multi-site may
    be constitutive in GENCODE -- so splitting the figure by class asks the
    reader to compare sets that are not the same sites.

    Deduplicated on gene:end, keeping `matched` if any copy was matched: the
    GENCODE split is by gene_start, so one end reachable from two starts can
    appear in both halves.
    """
    usage = load_site_usage()
    out = {}
    for side, keys in (('torino', ('torino_apa', 'torino_single')),
                       ('gencode', ('gencode_apa', 'gencode_single'))):
        df = pd.concat([usage[k] for k in keys], ignore_index=True)
        n_before = len(df)
        df['matched'] = df.matched.astype(bool)
        # keep the matched copy of any duplicated gene:end
        df = (df.sort_values('matched', ascending=False)
                .drop_duplicates(['gene', 'end'], keep='first')
                .reset_index(drop=True))
        if len(df) != n_before:
            print(f'[sites] {side}: {n_before - len(df):,} duplicate gene:end '
                  f'pair(s) across the APA/single split, kept once')
        out[side] = df
    return out


@_panel
def panel_UTRs_cumulative_sites():
    """Long-read support per polyA site: shared, Torino-only, GENCODE-only."""
    sites = _pooled_sites()
    tor, gen = sites['torino'], sites['gencode']
    t_match = tor.matched.values
    g_match = gen.matched.values

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for df, lab, colour in (
            # n is the number of points on the line: the Torino sites GENCODE
            # also annotates. The GENCODE-side count of the same agreement
            # (16,454 corrected) is larger, because that flag is tested against
            # every Torino isoform rather than the APPRIS-matched sites -- but
            # it is not a property of this curve, so it does not belong here.
            (tor.loc[t_match],
             f'Torino & Gencode (n = {int(t_match.sum()):,})', 'tab:purple'),
            (tor.loc[~t_match],
             f'Torino only (n = {int((~t_match).sum()):,})', 'tab:blue'),
            (gen.loc[~g_match],
             f'Gencode only (n = {int((~g_match).sum()):,})', 'tab:red')):
        if not len(df):
            continue
        vals = df[PCT_COLS].fillna(0).mean(axis=1)
        x, y = plot_cumulative_utr(list(vals))
        ax.plot(y, x, c=colour, linewidth=3, label=lab)

    ax.set_xlabel('Percentage of polyA sites')
    ax.set_ylabel('Cumulative site usage percentage')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_title('Long-read mapping to polyA site', fontsize=11)
    ax.legend(frameon=False, fontsize=7, loc='lower right',
              title='polyA site called by', title_fontsize=7)
    return fig


PANELS['UTRs_cumulative_sites'] = panel_UTRs_cumulative_sites


def variable_apa_table():
    """The polyA heatmap's numbers: how 2,702 arises, and the brain direction.

    The heatmap's rows are alternative 3' UTR PAIRS, not genes -- a gene with
    two long/short pairs contributes two rows (GENE_1, GENE_2) -- and they are
    not a subset of the APA gene:end sites: this pipeline runs over genes with
    an extended 3' UTR through median usage per tissue, and never applies the
    gene:end site definitions or the ambiguity cleanup.
    """
    import re as _re
    keep = _R.GENE_SUBSET
    rows = {}
    try:
        for label, sub in (('all genes', None), ('corrected', True)):
            _R.GENE_SUBSET = _R.load_select_genes() if sub else None
            free()
            long_df, short_df, scaled_all, brain_t, cereb_t = load_utr_loadings()
            sl = long_df.loc[(long_df.max(axis=1) - long_df.min(axis=1)) >= 0.05]
            ss = short_df.loc[(short_df.max(axis=1) - short_df.min(axis=1)) >= 0.05]
            genes = pd.Index([_re.sub(r'_\d+$', '', g)
                              for g in scaled_all.index]).unique()
            bc = list(brain_t) + list(cereb_t)
            oc = [c for c in scaled_all.columns if c not in set(bc)]
            hi = scaled_all[bc].mean(axis=1) > scaled_all[oc].mean(axis=1)
            sets = _site_sets()
            apa_genes = set(sets['torino_apa'].gene)
            inside = sum(1 for g in genes if g in apa_genes)
            rows[label] = {
                "UTR pairs tested": len(long_df),
                "long usage varies >= 0.05": len(sl),
                "short usage varies >= 0.05": len(ss),
                "both vary (intersection)": len(sl.index.intersection(ss.index)),
                "heatmap rows (variable APA)": len(scaled_all),
                "distinct genes behind them": len(genes),
                "higher in brain": int(hi.sum()),
                "lower in brain": int((~hi).sum()),
                "APA gene:end sites, for reference": len(sets['torino_apa']),
                "distinct APA genes": len(apa_genes),
                "variable-APA genes inside the APA set": inside,
                "variable-APA genes outside it": len(genes) - inside,
            }
    finally:
        _R.GENE_SUBSET = keep
        free()

    a, c = rows['all genes'], rows['corrected']
    out = ["**Variable APA across tissues (polyA heatmap)**", '',
           '| | all genes | corrected gene set |', '|---|---:|---:|']
    for k in a:
        pct = ''
        if k in ('higher in brain', 'lower in brain'):
            out.append(f'| {k} | {a[k]:,} '
                       f'({100 * a[k] / a["heatmap rows (variable APA)"]:.1f}%) '
                       f'| {c[k]:,} '
                       f'({100 * c[k] / c["heatmap rows (variable APA)"]:.1f}%) |')
        else:
            out.append(f'| {k} | {a[k]:,} | {c[k]:,} |')
    out += ['',
            "> **The rows are 3' UTR pairs, not genes.** A gene with two "
            "long/short pairs appears twice (`GENE_1`, `GENE_2`), so the "
            "percentage is of pairs.",
            '',
            "> **Not a subset of the APA sites.** Different unit (UTR pairs vs "
            "`gene:end` sites) and different membership: this pipeline runs "
            "over genes with an extended 3' UTR through median usage per "
            "tissue, and applies neither the site definitions nor the "
            "ambiguity cleanup -- hence the genes falling outside the APA set.",
            '',
            "> Brain means the brain and cerebellum columns; other is every "
            "remaining tissue. The split is identical under the "
            "`not startswith('Brain')` rule, so cerebellum does not affect it."]
    return '\n'.join(out)


# ==========================================================================
# APA heatmap: relative usage of the top two UTRs per locus
#
# A heatmap like polyA_heatmap, but over the APA sites instead of the
# extended-3'UTR pairs. For each APA locus -- a gene:terminal-exon-start with
# two or more distinct UTR end clusters, which is what makes the 10,735 sites
# APA -- the two MOST-USED UTR ends are taken and the row is
#
#     usage(most used) / (usage(most used) + usage(second))
#
# per tissue, then row-scaled. That differs from polyA_heatmap, which takes the
# SHORTEST and LONGEST ends regardless of how much they are used. Rows are kept
# when that ratio varies by >= APA_HEATMAP_MIN_RANGE across tissues.
#
# Usage per end comes from median_EL -- the per-tissue median isoform usage --
# summed over the isoforms in that end's group, exactly as
# process_utr_loadings_remove_extreme does for the long/short pair.
# ==========================================================================
APA_HEATMAP_MIN_RANGE = 0.1


def _utr_setup():
    """annot / human_iso / median_EL / the two groupings, as the loaders build them."""
    def build():
        annot = load_annot()
        human_iso = pd.read_csv(
            '../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
            names=['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand',
                   'factors', 'exon', 'notes'])
        human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id
        human_iso = _subset(human_iso, human_iso.gene_id)

        human_EL = pd.read_csv(
            '../code/results/torino/tables/Human/isoforms_filtered.merged_isoforms.EL.bed.gz',
            sep='\t')
        # These two lines are what make median_EL addressable by isoform id.
        # Without them its index is 0..n and every
        # `median_EL.index.intersection(iso_group)` comes back empty, so usage
        # sums to zero and every locus is silently dropped -- which is exactly
        # what happened on the first run: 3,004 loci found, 0 kept.
        human_EL['pid'] = human_EL.gid + '.' + human_EL.pid
        human_EL.index = human_EL.pid

        metadata = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv',
                               sep='\t')
        human_metadata = metadata[['entity:sample_id', 'tissue_id']].copy()
        human_metadata.columns = ['Sample', 'Tissue']
        human_metadata.index = human_metadata.Sample
        median_EL = median_by_tissue(human_metadata, human_EL)

        annot_ref = annot.loc[
            annot.appris_ref.apply(lambda x: ('ref_transcript' in x)
                                   or ('intron_chain' in x))
            | (annot.gencode_ref != '.')]
        ref_chain_iso = human_iso.loc[human_iso.transcript_id.isin(annot_ref.transcript)]
        return (median_EL, ref_chain_iso.groupby('gene_id'),
                human_iso.groupby('gene_id'))
    return _cached(f'utr_setup:{_tag()}', build)


def load_apa_top2_usage(pair='top2'):
    """One row per APA locus: relative usage of two of its UTR ends.

    pair='top2'     the two MOST-USED ends; the row is top/(top + second).
    pair='extremes' the SHORTEST and LONGEST ends; the row is
                    long/(long + short), the same quantity polyA_heatmap plots,
                    so the two figures differ only in the locus population.

    iso_groups is ordered shortest to longest, which is why 'extremes' is
    [0] and [-1] -- the indices process_utr_loadings_remove_extreme uses.

    Index is `gene:exon_start`, columns are tissues. Restricted to the genes
    behind the unambiguous APA sites, so this is a subset of the same
    population the 10,735 come from.
    """
    def build():
        median_EL, grouped_ref, grouped_all = _utr_setup()
        apa_genes = sorted(set(_site_sets()['torino_apa'].gene))

        rows, index, n_loci, n_skipped, n_nan = [], [], 0, 0, 0
        for gene in tqdm(apa_genes):
            if gene not in grouped_ref.groups:
                n_skipped += 1
                continue
            try:
                aver = process_utr_loadings_remove_extreme(
                    grouped_ref, grouped_all, gene, median_EL)
            except Exception:
                n_skipped += 1
                continue
            if aver is None:
                n_skipped += 1
                continue

            # rebuild the per-end isoform groups for this gene
            torino_ref_df = grouped_ref.get_group(gene)
            coords_dict, strand = group_by_exon_w_transcript(
                get_last_exons_w_transcript(torino_ref_df))
            if gene in grouped_all.groups:
                coords_dict = merge_extra_isoforms(
                    coords_dict, grouped_all.get_group(gene), strand)
            isos = process_utr_coordinates_torino_w_transcripts(coords_dict, strand)

            for x, iso_dict in isos.items():
                groups = iso_dict['iso_groups']
                if len(groups) < 2:
                    continue        # not an APA locus: one UTR end only
                n_loci += 1
                usage = [median_EL.loc[median_EL.index.intersection(g)].sum(axis=0)
                         for g in groups]
                if pair == 'top2':
                    # the two most-used ends, by mean usage across tissues
                    order = sorted(range(len(usage)),
                                   key=lambda i: usage[i].mean(), reverse=True)
                    top, second = usage[order[0]], usage[order[1]]
                elif pair == 'extremes':
                    # longest vs shortest, numerator = longest
                    top, second = usage[-1], usage[0]
                else:
                    raise ValueError(f'unknown pair: {pair!r}')
                denom = top + second
                ratio = (top / denom).where(denom > 0)
                if ratio.isna().any():
                    n_nan += 1
                    continue
                rows.append(ratio)
                index.append(f'{gene}:{x}')

        df = pd.DataFrame(rows)
        df.index = index
        print(f'[apa heatmap/{pair}] {len(df):,} loci from {len(apa_genes):,} APA genes '
              f'({n_loci:,} loci seen, {n_skipped:,} genes skipped, '
              f'{n_nan:,} loci with no usable usage)')
        if n_loci and not len(df):
            raise RuntimeError(
                f'every one of {n_loci:,} APA loci was dropped for having no '
                f'usable usage. The usual cause is median_EL not being indexed '
                f'by isoform id, which makes every isoform lookup empty.')
        return df
    return _cached(f'apa_pair_{pair}:{_tag()}', build)


def apa_heatmap_numbers(min_range=None, pair='top2'):
    """Counts behind the APA heatmaps."""
    min_range = APA_HEATMAP_MIN_RANGE if min_range is None else min_range
    df = load_apa_top2_usage(pair)
    sel = df.loc[(df.max(axis=1) - df.min(axis=1)) >= min_range]
    brain = [c for c in df.columns if c.startswith('Brain')]
    other = [c for c in df.columns if not c.startswith('Brain')]
    hi = sel[brain].mean(axis=1) > sel[other].mean(axis=1)
    return {'APA loci with two or more UTR ends': len(df),
            f'varying >= {min_range} across tissues': len(sel),
            'higher top-UTR usage in brain': int(hi.sum()),
            'lower in brain': int((~hi).sum()),
            'distinct genes': int(pd.Index(
                [i.rsplit(':', 1)[0] for i in sel.index]).nunique()),
            'brain columns': len(brain), 'other columns': len(other)}


@_panel
def panel_polyA_heatmap_apa(min_range=None, pair='top2'):
    """Relative usage of two UTR ends per APA locus; see load_apa_top2_usage."""
    from sklearn.preprocessing import scale
    import matplotlib.patches as _mpatches
    min_range = APA_HEATMAP_MIN_RANGE if min_range is None else min_range

    df = load_apa_top2_usage(pair)
    sel = df.loc[(df.max(axis=1) - df.min(axis=1)) >= min_range]
    scaled = pd.DataFrame(scale(sel, axis=1), index=sel.index, columns=sel.columns)

    brain = [c for c in scaled.columns if c.startswith('Brain')]
    other = [c for c in scaled.columns if not c.startswith('Brain')]
    hi = scaled[brain].mean(axis=1) > scaled[other].mean(axis=1)
    n_hi, n_lo = int(hi.sum()), int((~hi).sum())
    row_colors = hi.map({True: '#b2182b', False: '#2166ac'})

    col_order = scaled.mean(axis=0).sort_values().index[::-1]
    row_order = scaled[brain].mean(axis=1).sort_values().index[::-1]
    color_list = [gtex_colors2[c]['tissue_color_hex'] for c in col_order]

    g = sns.clustermap(scaled.loc[row_order, col_order], col_colors=color_list,
                       row_colors=row_colors.loc[row_order],
                       col_cluster=False, row_cluster=False, cmap='bwr',
                       figsize=(6, 8), rasterized=True)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel(
        f"{len(sel):,} APA loci with >= {min_range} usage difference\n"
        f"across tissues (of {len(df):,} tested)")
    _which = ('most-used UTR end' if pair == 'top2' else "longest 3' UTR end")
    g.ax_cbar.set_ylabel(f'Scaled usage of the\n{_which}', fontsize=9)
    g.ax_heatmap.legend(
        handles=[_mpatches.Patch(color='#b2182b',
                                 label=f'higher in brain (n = {n_hi:,})'),
                 _mpatches.Patch(color='#2166ac',
                                 label=f'lower in brain (n = {n_lo:,})')],
        title=_which[0].upper() + _which[1:], loc='upper left',
        bbox_to_anchor=(1.18, 1.0), frameon=False, fontsize=8, title_fontsize=8)
    return g.figure


@_panel
def panel_polyA_heatmap_apa_extremes(min_range=None):
    """The same APA loci, but shortest vs longest end.

    Isolates the pair choice: same loci as polyA_heatmap_apa, same threshold,
    the only difference being which two ends are compared.
    """
    return panel_polyA_heatmap_apa.__wrapped__(min_range=min_range,
                                               pair='extremes')


PANELS['polyA_heatmap_apa'] = panel_polyA_heatmap_apa
PANELS['polyA_heatmap_apa_extremes'] = panel_polyA_heatmap_apa_extremes


# ==========================================================================
# APA heatmap built strictly on the 10,735 sites
#
# One row per gene. The two ends compared are the gene's LONGEST and SHORTEST
# end TAKEN FROM THE APA SITE TABLE ITSELF -- no terminal-exon groups are
# rebuilt and merge_extra_isoforms is not called -- so every end plotted is one
# of the 10,735, and every gene is one of the genes behind them. That is the
# difference from polyA_heatmap_apa_extremes, whose loci are recomputed and so
# included 26 gene:starts the site table does not carry.
#
# Usage for an end is median_EL summed over the isoforms whose own terminal
# exon ends within UTR_CLUSTER_BP of it -- the same 200 bp window that defines
# a site in get_unique_utrs, so an end and the isoforms assigned to it belong
# to the same cluster.
#
# Longest/shortest is strand-aware: on + the longest end is the largest
# coordinate, on - the smallest.
# ==========================================================================
UTR_CLUSTER_BP = 200


def load_apa_site_extremes():
    """Per gene: usage ratio of the longest vs shortest APA site end.

    Returns (ratio_df, record). ratio_df is indexed by gene, columns tissues,
    values long/(long + short). record is one row per gene naming the two ends
    used, so the plotted sites can be traced back to the site table.
    """
    def build():
        median_EL, grouped_ref, grouped_all = _utr_setup()
        apa = _site_sets()['torino_apa']

        rows, index, rec = [], [], []
        n_one_end = n_no_iso = n_nan = 0
        for gene, sub in tqdm(apa.groupby('gene')):
            ends = sorted(set(int(e) for e in sub.exon_end))
            if len(ends) < 2:
                n_one_end += 1
                continue
            if gene not in grouped_all.groups:
                n_no_iso += 1
                continue
            gene_df = grouped_all.get_group(gene)
            last_exons, strand = get_last_exons_w_transcript(gene_df)
            # terminal end per isoform, in the strand's own orientation
            iso_end = {t: (c[1] if strand == '+' else c[0])
                       for t, c in last_exons.items()}

            long_end = max(ends) if strand == '+' else min(ends)
            short_end = min(ends) if strand == '+' else max(ends)

            def _usage(end):
                isos = [t for t, e in iso_end.items()
                        if abs(e - end) < UTR_CLUSTER_BP]
                if not isos:
                    return None
                return median_EL.loc[median_EL.index.intersection(isos)].sum(axis=0)

            u_long, u_short = _usage(long_end), _usage(short_end)
            if u_long is None or u_short is None:
                n_no_iso += 1
                continue
            denom = u_long + u_short
            ratio = (u_long / denom).where(denom > 0)
            if ratio.isna().any():
                n_nan += 1
                continue
            rows.append(ratio)
            index.append(gene)
            rec.append({'gene': gene, 'strand': strand, 'n_sites': len(ends),
                        'long_end': long_end, 'short_end': short_end})

        df = pd.DataFrame(rows)
        df.index = index
        record = pd.DataFrame(rec)
        print(f'[apa sites] {len(df):,} genes of {apa.gene.nunique():,} APA genes '
              f'({n_one_end:,} with one end only, {n_no_iso:,} without usable '
              f'isoforms, {n_nan:,} with a zero denominator)')
        if apa.gene.nunique() and not len(df):
            raise RuntimeError('every APA gene was dropped -- check that '
                               'median_EL is indexed by isoform id.')
        return df, record
    return _cached(f'apa_site_extremes:{_tag()}', build)


def apa_site_numbers(min_range=None):
    """Counts behind panel_polyA_heatmap_apa_sites, and the subset checks."""
    min_range = APA_HEATMAP_MIN_RANGE if min_range is None else min_range
    df, record = load_apa_site_extremes()
    sel = df.loc[(df.max(axis=1) - df.min(axis=1)) >= min_range]
    brain = [c for c in df.columns if c.startswith('Brain')]
    other = [c for c in df.columns if not c.startswith('Brain')]
    hi = sel[brain].mean(axis=1) > sel[other].mean(axis=1)

    apa = _site_sets()['torino_apa']
    site_pairs = set(zip(apa.gene, apa.exon_end.astype(int)))
    used = set(zip(record.gene, record.long_end)) | set(zip(record.gene,
                                                            record.short_end))
    return {'APA sites (gene:end)': len(apa),
            'APA genes': int(apa.gene.nunique()),
            'genes with two or more sites, usable': len(df),
            f'varying >= {min_range} across tissues': len(sel),
            'higher longest-UTR usage in brain': int(hi.sum()),
            'lower in brain': int((~hi).sum()),
            'sites plotted (2 per gene)': len(used),
            'sites plotted that are in the 10,735': len(used & site_pairs),
            'sites plotted NOT in the 10,735': len(used - site_pairs),
            'genes plotted that are APA genes': int(
                record.gene.isin(set(apa.gene)).sum())}


@_panel
def panel_polyA_heatmap_apa_sites(min_range=None):
    """Longest vs shortest APA site end, per gene, strictly within the 10,735."""
    from sklearn.preprocessing import scale
    import matplotlib.patches as _mpatches
    min_range = APA_HEATMAP_MIN_RANGE if min_range is None else min_range

    df, record = load_apa_site_extremes()
    sel = df.loc[(df.max(axis=1) - df.min(axis=1)) >= min_range]
    scaled = pd.DataFrame(scale(sel, axis=1), index=sel.index, columns=sel.columns)

    brain = [c for c in scaled.columns if c.startswith('Brain')]
    other = [c for c in scaled.columns if not c.startswith('Brain')]
    hi = scaled[brain].mean(axis=1) > scaled[other].mean(axis=1)
    n_hi, n_lo = int(hi.sum()), int((~hi).sum())
    row_colors = hi.map({True: '#b2182b', False: '#2166ac'})

    col_order = scaled.mean(axis=0).sort_values().index[::-1]
    row_order = scaled[brain].mean(axis=1).sort_values().index[::-1]
    color_list = [gtex_colors2[c]['tissue_color_hex'] for c in col_order]

    g = sns.clustermap(scaled.loc[row_order, col_order], col_colors=color_list,
                       row_colors=row_colors.loc[row_order],
                       col_cluster=False, row_cluster=False, cmap='bwr',
                       figsize=(6, 8), rasterized=True)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel(
        f"{len(sel):,} genes with >= {min_range} usage difference across\n"
        f"tissues, longest vs shortest APA site (of {len(df):,} tested)")
    g.ax_cbar.set_ylabel("Scaled usage of the\nlongest 3' UTR site", fontsize=9)
    g.ax_heatmap.legend(
        handles=[_mpatches.Patch(color='#b2182b',
                                 label=f'higher in brain (n = {n_hi:,})'),
                 _mpatches.Patch(color='#2166ac',
                                 label=f'lower in brain (n = {n_lo:,})')],
        title="Longest APA site", loc='upper left',
        bbox_to_anchor=(1.18, 1.0), frameon=False, fontsize=8, title_fontsize=8)
    return g.figure


PANELS['polyA_heatmap_apa_sites'] = panel_polyA_heatmap_apa_sites


def apa_site_table(thresholds=(0.05, 0.1, 0.15, 0.2)):
    """Everything behind panel_polyA_heatmap_apa_sites.

    Three sections: how the gene set narrows from the tested genes to the ones
    plotted, the verification that the plotted sites really are a subset of the
    APA sites, and the brain split at each cutoff.
    """
    counts = load_utr_counts()
    tor = load_torino_utrs()
    multi = set(counts.loc[counts.utrs > 0].gene)
    vc = tor.value_counts('gene_start')
    shared = set(tor.loc[tor.gene_start.isin(vc.loc[vc >= 2].index)].gene)
    apa = _site_sets()['torino_apa']
    df, record = load_apa_site_extremes()

    site_pairs = set(zip(apa.gene, apa.exon_end.astype(int)))
    used = (set(zip(record.gene, record.long_end))
            | set(zip(record.gene, record.short_end)))

    out = ["**APA heatmap: longest vs shortest site per gene**", '',
           'One row per gene. The two ends compared are the gene\'s longest and '
           'shortest end **taken from the APA site table itself** -- no '
           'terminal-exon groups are rebuilt -- so every end plotted is one of '
           'the APA sites and every gene is one of the genes behind them. '
           'Longest/shortest is strand-aware. Usage for an end is the '
           'per-tissue median isoform usage summed over the isoforms whose own '
           'terminal exon ends within 200 bp of it, the same window that '
           'defines a site.', '',
           '**How the gene set narrows**', '',
           '| step | genes | dropped |', '|---|---:|---:|']
    rows = [
        ('tested genes', len(counts), ''),
        ('labelled multi-site (utrs > 0)', len(multi),
         f'{len(counts) - len(multi):,} single-site'),
        ('shared terminal-exon start, >= 2 rows', len(shared), ''),
        ('both -> APA genes', len(shared & multi),
         f'{len(multi - shared):,} multi-site without a shared start; '
         f'{len(shared - multi):,} ambiguous'),
        ('usable usage ratio (tested here)', len(df),
         f'{apa.gene.nunique() - len(df):,} with a zero denominator '
         f'in some tissue'),
    ]
    for label, n, drop in rows:
        out.append(f'| {label} | {n:,} | {drop} |')
    out += ['',
            '> The two other possible losses are both zero: no APA gene has '
            'only one distinct end, and none lacks usable isoforms. The '
            f'{apa.gene.nunique() - len(df):,} dropped genes each had a tissue '
            'where both the longest and shortest end had zero usage, leaving '
            'the ratio undefined there.', '',
            '**The plotted sites are a subset of the APA sites**', '',
            '| | count |', '|---|---:|',
            f'| APA sites (gene:end) | {len(apa):,} |',
            f'| APA genes | {apa.gene.nunique():,} |',
            f'| sites plotted (2 per gene) | {len(used):,} |',
            f'| of those, present among the APA sites | {len(used & site_pairs):,} |',
            f'| **not** among them | {len(used - site_pairs):,} |',
            f'| genes plotted that are APA genes | '
            f'{int(record.gene.isin(set(apa.gene)).sum()):,} of {len(record):,} |',
            '',
            '**Results by cutoff**', '',
            '| min. change across tissues | genes plotted | higher in brain | '
            'lower in brain |', '|---|---:|---:|---:|']
    brain = [c for c in df.columns if c.startswith('Brain')]
    other = [c for c in df.columns if not c.startswith('Brain')]
    for mr in thresholds:
        sel = df.loc[(df.max(axis=1) - df.min(axis=1)) >= mr]
        hi = sel[brain].mean(axis=1) > sel[other].mean(axis=1)
        n_hi = int(hi.sum())
        out.append(f'| >= {mr} | {len(sel):,} | {n_hi:,} '
                   f'({100 * n_hi / max(len(sel), 1):.1f}%) '
                   f'| {int((~hi).sum()):,} |')
    out += ['',
            f'> The figure uses >= {APA_HEATMAP_MIN_RANGE} '
            '(`U.APA_HEATMAP_MIN_RANGE`). The brain fraction is flat across '
            'the whole range, so nothing hinges on that choice.', '',
            "> Brain means the brain and cerebellum columns; other is every "
            "remaining tissue.", '',
            '> **Related panels.** `polyA_heatmap` plots the same '
            "longest/shortest quantity over genes with an annotated extended "
            "3' UTR instead (2,702 UTR pairs, 84.8% higher in brain). "
            '`polyA_heatmap_apa` keeps these APA loci but compares the two '
            'MOST-USED ends rather than the extremes, which drops the brain '
            'fraction to about 63% -- so the brain skew is a property of the '
            'length axis, not of dominant-isoform switching.']
    return '\n'.join(out)
