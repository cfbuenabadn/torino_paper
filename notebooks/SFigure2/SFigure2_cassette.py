"""SFigure2 cassette-exon panels and counts, from the results already on disk.

From analysis/CassetteExonsDeNovo.ipynb. The Torino and Gencode cassette-exon
tables are read back from the bed files the notebook wrote (read_cassette_bed
below) rather than rebuilt by the per-gene bedtools loop that produced them -- that loop is the better part of an
hour and its output is already on disk.

Nothing here writes; the guard in SFigure2_intron_retention is active, and
GENE_SUBSET there applies to these loaders too.
"""

import os
import sys

from SFigure2_intron_retention import (      # noqa: F401  -- shared setup
    ANALYSIS_DIR, np, pd, plt, sns, BedTool, bedtools_safe, _cached, _panel,
    free, _subset, _tag, _as_symbols, load_select_genes,
)
import SFigure2_intron_retention as _R

from tqdm import tqdm


# --------------------------------------------------------------------------
# Inputs
#
# REFERENCE ANNOTATIONS are never restricted by GENE_SUBSET. The correction says
# which Torino genes we trust, not what we may compare them against: subsetting
# Gencode as well made cassette exons look novel purely because the annotation
# at their locus belonged to a gene the subset had dropped -- and since
# select_genes excludes genes that overlap another gene, that is exactly the
# annotation being removed. It showed up as more unannotated exons in the
# corrected set than in the full one, which a subset cannot produce.
# --------------------------------------------------------------------------
CASSETTE_BED_DIR = ('/project2/yangili1/cfbuenabadn/torino_paper/code/results/'
                    'torino/CoveragePlots/CassetteExons/bed')
CASSETTE_COLS = ['chrom', 'start', 'end', 'cassette_exons_coord', 'gene', 'strand']


def read_cassette_bed(prefix, flanking=True):
    """The cassette-exon table, read back from the bed files it was written to.

    The notebook built this with a per-gene bedtools intersect over ~15,000
    genes -- the better part of an hour -- and then wrote it out as three bed
    files. They hold the same data: <prefix>.cassette_exons.bed is
    chrom/start/end/coord/gene/strand, the flanking1 bed's start column is
    flanking1_start, and the flanking2 bed's end column is flanking2_end, keyed
    on the same coord. The files exist, so the loop is not re-run.
    """
    def build():
        def _bed(kind):
            return pd.read_csv(f'{CASSETTE_BED_DIR}/{prefix}.{kind}.bed',
                               sep='\t', names=CASSETTE_COLS)
        df = _bed('cassette_exons')
        if flanking:
            f1 = _bed('flanking1_exons')[['cassette_exons_coord', 'start']]
            f1.columns = ['cassette_exons_coord', 'flanking1_start']
            f2 = _bed('flanking2_exons')[['cassette_exons_coord', 'end']]
            f2.columns = ['cassette_exons_coord', 'flanking2_end']
            df = (df.merge(f1, on='cassette_exons_coord')
                    .merge(f2, on='cassette_exons_coord'))
        df['exon'] = (df.chrom + ':' + df.start.astype(str)
                      + ':' + df.end.astype(str))
        # Torino's own table is subset; Gencode's is a reference -- see
        # REFERENCE ANNOTATIONS below.
        return _subset(df, df.gene) if prefix == 'torino' else df
    return _cached(f'cassette_bed:{prefix}:{flanking}:'
                   f'{_tag() if prefix == "torino" else "all"}', build)


def load_gencode_exons():
    """Gencode exon annotation -- notebook cell 3/4 (In[4], In[5])."""
    def build():
        # 2.1M rows: read only the columns used, or the frame alone is ~2 GB
        ex = pd.read_csv('../code/resources/annotation/Human/bed/exons.all.bed.gz',
                         sep='\t', usecols=['#chrom', 'start', 'end', 'gene_name'])
        ex['exon'] = (ex['#chrom'] + ':' + ex.start.astype(str)
                      + ':' + ex.end.astype(str))
        return ex                      # reference annotation: never subset
    return _cached('gencode_exons', build)


def load_vastdb_exons():
    """VastDB cassette-exon events -- notebook cells 33, 34, 41 (In[29..37]).

    EVENT_INFO restricted to simple exon-skipping events, with the cassette and
    flanking coordinates unpacked into the same `cassette_exons_coord` key the
    Torino table uses (chrom:start:end:intron_start:intron_end) and the plain
    `exon` key (chrom:start:end).
    """
    def build():
        # EVENT_INFO is wide and the sequence columns are huge; read only what
        # the coordinate unpacking below needs, or the frame is >2 GB
        df = pd.read_csv('../code/resources/vastdb/EVENT_INFO-hg38.tab.gz', sep='\t',
                         usecols=['EVENT', 'GENE', 'COMPLEX', 'CO_A', 'CO_C1',
                                  'CO_C2', 'REF_CO'])
        df = df.dropna()
        cols = list(df.columns)
        df['chrom'] = df.CO_A.apply(lambda x: x.split(':')[0])
        df['start'] = df.CO_A.apply(lambda x: x.split(':')[1].split('-')[0])
        df['end'] = df.CO_A.apply(lambda x: x.split('-')[1])
        df = df[['chrom', 'start', 'end'] + cols]
        df = df.loc[(~df.COMPLEX.isin(['MIC-S', 'MIC_S', 'IR', 'Alt3', 'Alt5', 'MIC-M']))
                    & (df.EVENT.apply(lambda x: x.startswith('HsaEX')))]
        df = df.loc[~df.COMPLEX.apply(lambda x: (x.startswith('ME') or (x == 'A_NA')
                                                 or (x == 'ANN') or x == 'NA*'))]
        df['strand'] = df.REF_CO.apply(lambda x: x.split(':')[-1])

        coords, i_start, i_end, f1_start, f2_end = [], [], [], [], []
        for _, row in df.iterrows():
            if row.strand == '+':
                intron_start = row.CO_C1.split('-')[-1]
                flanking1_start = row.CO_C1.split(':')[-1].split('-')[0]
                intron_end = row.CO_C2.split(':')[-1].split('-')[0]
                flanking2_end = row.CO_C2.split('-')[-1]
            else:
                intron_start = row.CO_C2.split('-')[-1]
                flanking1_start = row.CO_C2.split(':')[-1].split('-')[0]
                intron_end = row.CO_C1.split(':')[-1].split('-')[0]
                flanking2_end = row.CO_C1.split('-')[-1]
            coords.append(f'{row.chrom}:{row.start}:{row.end}:'
                          f'{intron_start}:{intron_end}')
            i_start.append(intron_start)
            i_end.append(intron_end)
            f1_start.append(flanking1_start)
            f2_end.append(flanking2_end)

        df['cassette_exons_coord'] = coords
        df['intron_start'] = i_start
        df['intron_end'] = i_end
        df['flanking1_start'] = pd.Series(f1_start, index=df.index).astype(int)
        df['flanking2_end'] = pd.Series(f2_end, index=df.index).astype(int)
        df['exon'] = df.chrom + ':' + df.start.astype(str) + ':' + df.end.astype(str)
        # EVENT_INFO is wide; nothing downstream needs the rest of it
        df = df[['chrom', 'start', 'end', 'strand', 'GENE', 'EVENT', 'COMPLEX',
                 'cassette_exons_coord', 'exon', 'intron_start', 'intron_end',
                 'flanking1_start', 'flanking2_end']]
        return df                      # reference annotation: never subset
    return _cached('vastdb_exons', build)


def load_cassette_overlaps():
    """Each Torino cassette exon, with how many Gencode exons it overlaps.

    Notebook cells 57-59 (In[51]-In[53]): cassette_bed.intersect(gencode_exons_bed,
    c=True). blockCount == 0 means the region overlaps no annotated exon at all.
    """
    def build():
        cassette = BedTool.from_dataframe(bedtools_safe(read_cassette_bed('torino')))
        gencode = BedTool.from_dataframe(bedtools_safe(load_gencode_exons()))
        return cassette.intersect(gencode, c=True).to_dataframe()
    return _cached(f'cassette_overlaps:{_tag()}', build)


# --------------------------------------------------------------------------
# The numbers quoted in the text
# --------------------------------------------------------------------------
def cassette_numbers():
    """Cassette-exon counts, honouring GENE_SUBSET.

    Each traces to a cell of CassetteExonsDeNovo.ipynb:

      cassette exons        In[98]   cassette_df.shape
      genes                 In[26]
      match by exon         In[43]   isin(vastdb.exon) | isin(gencode.exon)
      match with flanks     In[41]   same, on cassette_exons_coord
      gencode               In[46]   isin(gencode_casette_df.cassette_exons_coord)
      vastdb                In[45]   isin(vastdb_exons_df.cassette_exons_coord)
      no annotated overlap  In[55]   cassette_overlaps.blockCount == 0
      partial / constitutive In[191] (total - no overlap) - match by exon

    "match by exon" asks only whether the cassette exon itself is annotated;
    "match with flanks" requires the flanking exons to agree too, which is why
    it is the smaller number.
    """
    cassette = read_cassette_bed('torino')
    gencode = read_cassette_bed('gencode')
    vastdb = load_vastdb_exons()
    overlaps = load_cassette_overlaps()

    n = {}
    n['cassette_exons'] = len(cassette)
    n['genes'] = cassette.gene.nunique()
    n['match_exon'] = int((cassette.exon.isin(vastdb.exon)
                           | cassette.exon.isin(gencode.exon)).sum())
    n['match_flanks'] = int(
        (cassette.cassette_exons_coord.isin(vastdb.cassette_exons_coord)
         | cassette.cassette_exons_coord.isin(gencode.cassette_exons_coord)).sum())
    n['match_gencode'] = int(
        cassette.cassette_exons_coord.isin(gencode.cassette_exons_coord).sum())
    n['match_vastdb'] = int(
        cassette.cassette_exons_coord.isin(vastdb.cassette_exons_coord).sum())
    n['no_overlap'] = int((overlaps.blockCount == 0).sum())
    n['overlapping'] = n['cassette_exons'] - n['no_overlap']
    n['partial_or_constitutive'] = n['overlapping'] - n['match_exon']
    return n


def cassette_table():
    """The cassette-exon numbers, both gene sets side by side, as markdown."""
    keep = _R.GENE_SUBSET
    try:
        _R.GENE_SUBSET = None
        free()
        a = cassette_numbers()
        _R.GENE_SUBSET = load_select_genes()
        free()
        c = cassette_numbers()
    finally:
        _R.GENE_SUBSET = keep
        free()

    def row(label, key, denom=None, src=''):
        cells = []
        for n in (a, c):
            v = n[key]
            cells.append(f'{v:,}' + (f' ({100 * v / n[denom]:.1f}%)' if denom else ''))
        return f'| {label} | {cells[0]} | {cells[1]} | `{src}` |'

    rows = [
        row('Cassette exons called by Torino', 'cassette_exons', src='In[98]'),
        row('&nbsp;&nbsp;across genes', 'genes', src='In[26]'),
        row('Matching Gencode or VastDB (exon only)', 'match_exon',
            'cassette_exons', 'In[43]'),
        row('&nbsp;&nbsp;flanking exons also matching', 'match_flanks',
            'cassette_exons', 'In[41]'),
        row('&nbsp;&nbsp;&nbsp;&nbsp;matching Gencode', 'match_gencode',
            'cassette_exons', 'In[46]'),
        row('&nbsp;&nbsp;&nbsp;&nbsp;matching VastDB', 'match_vastdb',
            'cassette_exons', 'In[45]'),
        row('Overlapping an annotated exon', 'overlapping',
            'cassette_exons', 'derived'),
        row('&nbsp;&nbsp;constitutive or partial match only',
            'partial_or_constitutive', 'cassette_exons', 'In[191]'),
        row('No match to Gencode or VastDB structures', 'no_overlap',
            'cassette_exons', 'In[55]'),
    ]
    larger = [k for k in a if c[k] > a[k]]
    out = ['**Cassette exons**', '',
           '| | all genes | corrected gene set | source |',
           '|---|---:|---:|---|'] + rows
    if larger:
        out += ['', f'> **check**: corrected is larger than all genes for {larger} '
                    '-- that should be impossible, it is a subset.']
    return '\n'.join(out)


# --------------------------------------------------------------------------
# Plotting helpers, copied verbatim from the notebook cells that were live
# when the panels ran.
# --------------------------------------------------------------------------
# --- plot_cassette_exon_metaplots and its helpers
# Copied verbatim from CassetteExonsDeNovo.ipynb cell 65 (In[59]).
MATRIX_DIR = '../code/results/torino/CoveragePlots/CassetteExons/matrices'

_META_COLS = ['chrom', 'start', 'end', 'gene', 'score', 'strand']
_POS_NAMES = [f'bin_{i+1}' for i in range(120)]
_ALL_COLS  = _META_COLS + _POS_NAMES

# the 60 bins adjacent to the junction in each flanking matrix — used for normalisation
# (mirrors `exon_cols` in norm_exons / plot_cassette_exon)
_NORM_BINS = [f'bin_{i}' for i in range(31, 91)]


def _load_tissue_matrices(tissue):
    """Load the three cassette-exon matrices for *tissue*.

    Returns (fk1, exon, fk2) DataFrames indexed by the 'gene' coordinate string.
    All three are aligned to the same index (their intersection).
    """
    def _read(region):
        path = f'{MATRIX_DIR}/{tissue}.{region}.mat.gz'
        df = pd.read_csv(path, sep='\t', skiprows=1, names=_ALL_COLS)
        return df.set_index('gene')

    fk1  = _read('flanking1_exons')
    exon = _read('cassette_exons')
    fk2  = _read('flanking2_exons')

    # align to shared coordinates (row order can differ across the three files)
    common = fk1.index.intersection(exon.index).intersection(fk2.index)
    return fk1.loc[common], exon.loc[common], fk2.loc[common]


def _plot_cassette_exon(fk1, exon, fk2, ax, q=0.5, color='gray'):
    """Plot one metaplot track — mirrors plot_cassette_exon() in Cassette_exons.ipynb.

    Normalises by the mean coverage of _NORM_BINS in fk1 and fk2, then plots:
      • fk1 median at x = 0–119  (black line)
      • exon q-quantile filled at x = 130–249  (colored fill)
      • fk2 median at x = 260–379  (black line)
    """
    norm_factor = pd.concat(
        [fk1[_NORM_BINS], fk2[_NORM_BINS]], axis=1
    ).mean(axis=1)
    valid = norm_factor[norm_factor >= 0.01].index

    if valid.empty:
        return

    X_fk1 = fk1.loc[valid, _POS_NAMES].T.divide(norm_factor.loc[valid]).T
    X     = exon.loc[valid, _POS_NAMES].T.divide(norm_factor.loc[valid]).T
    X_fk2 = fk2.loc[valid, _POS_NAMES].T.divide(norm_factor.loc[valid]).T

    ax.plot(
        np.array(X_fk1.quantile(0.5, axis=0)),
        linewidth=2, c='black'
    )
    ax.fill_between(
        np.arange(120) + 130, [0] * 120,
        np.array(X.quantile(q, axis=0)),
        color=color
    )
    ax.plot(
        np.arange(120) + 260,
        np.array(X_fk2.quantile(0.5, axis=0)),
        linewidth=2, c='black'
    )


def _draw_schematic(ax, color='gray'):
    """Draw the cassette-exon cartoon — mirrors the annotation row in plot_tissue_quintiles_annotation."""
    ax.plot([0,   120], [0.5, 0.5], c='black', zorder=1, linewidth=3)
    ax.plot([130, 250], [0.5, 0.5], c=color,   zorder=1, linewidth=3)
    ax.plot([260, 380], [0.5, 0.5], c='black', zorder=1, linewidth=3)
    ax.plot([120, 130], [0.25, 0.75], c='red', zorder=1, linewidth=2)
    ax.plot([250, 260], [0.25, 0.75], c='red', zorder=1, linewidth=2)
    ax.fill_between(np.arange(30, 90),       [0]*60, [1]*60, color='black', zorder=2)
    ax.fill_between(np.arange(30, 90) + 130, [0]*60, [1]*60, color=color,   zorder=2)
    ax.fill_between(np.arange(30, 90) + 260, [0]*60, [1]*60, color='black', zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_cassette_exon_metaplots(
    tissue,
    coord_lists,
    list_labels=None,
    q=0.5,
    color='gray',
    figsize=(10, 3),
):
    """
    Cassette-exon metaplot analogous to plot_tissue_quintiles_annotation.

    Instead of three fixed exon-type rows (cassette / exon_only / intron_only)
    the rows are defined by *coord_lists*: each inner list selects a subset of
    cassette-exon coordinates from the tissue matrices.

    Parameters
    ----------
    tissue : str
        GTEx tissue name, e.g. ``'Heart_Atrial_Appendage'``.
        The three matrices
        ``{tissue}.cassette_exons.mat.gz``,
        ``{tissue}.flanking1_exons.mat.gz``,
        ``{tissue}.flanking2_exons.mat.gz``
        must exist in ``MATRIX_DIR``.
    coord_lists : list of lists  (or a single flat list)
        Each inner list contains coordinate strings matching the 'gene' column
        (e.g. ``'chr1:931039:931089:930336:935772'``).
        A plain flat list is treated as a single group (one row).
    list_labels : list of str, optional
        Y-axis label for each metaplot row.
    q : float
        Quantile to fill for the cassette-exon body (default 0.5 = median).
        The flanking tracks always use the median.
    color : str
        Colour for the cassette-exon fill and schematic (default ``'gray'``).
    figsize : tuple
        Figure (width, height) in inches.

    Returns
    -------
    fig, axes : matplotlib Figure and 2-D ndarray of Axes  (shape: [n_rows+1, 1])
    """
    # ── normalise input ──────────────────────────────────────────────────────
    if not coord_lists:
        raise ValueError('coord_lists must not be empty')
    if not isinstance(coord_lists[0], (list, tuple, np.ndarray, set)):
        coord_lists = [coord_lists]

    n_rows = len(coord_lists)

    # ── load matrices ────────────────────────────────────────────────────────
    fk1_all, exon_all, fk2_all = _load_tissue_matrices(tissue)

    # ── figure layout — mirrors plot_tissue_quintiles_annotation ─────────────
    height_ratios = [1] * n_rows + [0.3]
    fig, axes = plt.subplots(
        nrows=n_rows + 1,
        ncols=5,
        figsize=figsize,
        gridspec_kw={'wspace': 0.1, 'hspace': 0.05, 'height_ratios': height_ratios},
        squeeze=False,   # always 2-D array
    )

    # ── metaplot rows ────────────────────────────────────────────────────────
    for i, coords in enumerate(coord_lists):

        coords   = list(coords)
        valid    = [c for c in coords if c in fk1_all.index]
        fk1_sub  = fk1_all.loc[valid]
        exon_sub = exon_all.loc[valid]
        fk2_sub  = fk2_all.loc[valid]


        body_cols = [f'bin_{i+1}' for i in range(30, 90)]
        body_mean = exon_sub[body_cols].mean(axis=1)
        q_cuts = pd.qcut(body_mean, 5, labels=['Q1','Q2','Q3','Q4','Q5'])
        
        quintile_lists  = [list(body_mean[q_cuts == lbl].index) for lbl in ['Q1','Q2','Q3','Q4','Q5']]
        quintile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        for j, qlist in enumerate(quintile_lists):
            fk1_sub_q = fk1_sub.loc[qlist]
            fk2_sub_q = fk2_sub.loc[qlist]
            exon_sub_q = exon_sub.loc[qlist]
        

            _plot_cassette_exon(fk1_sub_q, exon_sub_q, fk2_sub_q, axes[i, j], q=q, color=color)
    
            axes[i, j].set_ylim([-0.1, 1.2])
            axes[i, j].set_xticks([])
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)

            if i == 0:
                axes[i, j].set_title(quintile_labels[j])

            if j > 0:
                axes[i,j].set_yticks([])
    
        if i < len(coord_lists):
            axes[i, 0].set_ylabel('PSI')

        # else:

    # ── schematic row ────────────────────────────────────────────────────────
    for j in range(5):
        _draw_schematic(axes[n_rows, j], color='gray')

    return fig, axes


# --- plot_cassette_exon_metaplots_comparison
# Copied verbatim from CassetteExonsDeNovo.ipynb cell 127 (In[109]).
def _load_tissue_matrices_annot(tissue, matrix_path):
    """Load the three cassette-exon matrices for *tissue*.

    Returns (fk1, exon, fk2) DataFrames indexed by the 'gene' coordinate string.
    All three are aligned to the same index (their intersection).
    """
    def _read(region):
        path = f'{matrix_path}/{tissue}.{region}.mat.gz'
        df = pd.read_csv(path, sep='\t', skiprows=1, names=_ALL_COLS)
        return df.set_index('gene')

    fk1  = _read('flanking1_exons')
    exon = _read('cassette_exons')
    fk2  = _read('flanking2_exons')

    # align to shared coordinates (row order can differ across the three files)
    common = fk1.index.intersection(exon.index).intersection(fk2.index)
    return fk1.loc[common], exon.loc[common], fk2.loc[common]

def plot_cassette_exon_metaplots_comparison(
    tissue,
    coord_lists,
    list_labels=None,
    q=0.5,
    color='gray',
    figsize=(10, 3),
    annot_color = 'tab:orange',
    annot_name = 'Gencode',
    annotation = 'gencode'
):
    """
    Cassette-exon metaplot analogous to plot_tissue_quintiles_annotation.

    Instead of three fixed exon-type rows (cassette / exon_only / intron_only)
    the rows are defined by *coord_lists*: each inner list selects a subset of
    cassette-exon coordinates from the tissue matrices.

    Parameters
    ----------
    tissue : str
        GTEx tissue name, e.g. ``'Heart_Atrial_Appendage'``.
        The three matrices
        ``{tissue}.cassette_exons.mat.gz``,
        ``{tissue}.flanking1_exons.mat.gz``,
        ``{tissue}.flanking2_exons.mat.gz``
        must exist in ``MATRIX_DIR``.
    coord_lists : list of lists  (or a single flat list)
        Each inner list contains coordinate strings matching the 'gene' column
        (e.g. ``'chr1:931039:931089:930336:935772'``).
        A plain flat list is treated as a single group (one row).
    list_labels : list of str, optional
        Y-axis label for each metaplot row.
    q : float
        Quantile to fill for the cassette-exon body (default 0.5 = median).
        The flanking tracks always use the median.
    color : str
        Colour for the cassette-exon fill and schematic (default ``'gray'``).
    figsize : tuple
        Figure (width, height) in inches.

    Returns
    -------
    fig, axes : matplotlib Figure and 2-D ndarray of Axes  (shape: [n_rows+1, 1])
    """
    # ── normalise input ──────────────────────────────────────────────────────
    if not coord_lists:
        raise ValueError('coord_lists must not be empty')
    if len(coord_lists) != 3:
        raise ValueError('must provide 3 coord_lists')

    n_rows = len(coord_lists)

    matrix_path_torino = f'{MATRIX_DIR}/torino/'
    matrix_path_annot = f'{MATRIX_DIR}/{annotation}/'
    # ── load matrices ────────────────────────────────────────────────────────
    # fk1_all, exon_all, fk2_all = _load_tissue_matrices_annot(tissue, matrix_path_torino)
    fk1_all_, exon_all_, fk2_all_ = _load_tissue_matrices_annot(tissue, matrix_path_annot)

    # ── figure layout — mirrors plot_tissue_quintiles_annotation ─────────────
    height_ratios = [1] * n_rows + [0.3]
    fig, axes = plt.subplots(
        nrows=n_rows + 1,
        ncols=7,
        figsize=figsize,
        gridspec_kw={'wspace': 0.1, 'hspace': 0.05, 'height_ratios': height_ratios},
        squeeze=False,   # always 2-D array
    )

    # ── metaplot rows ────────────────────────────────────────────────────────
    for i, coords in enumerate(coord_lists):

        coords   = list(coords)

        # if i < 2:
        
        #     valid    = [c for c in coords if c in fk1_all.index]
        #     fk1_sub  = fk1_all.loc[valid]
        #     exon_sub = exon_all.loc[valid]
        #     fk2_sub  = fk2_all.loc[valid]
        # else:
        valid    = [c for c in coords if c in fk1_all_.index]
        fk1_sub  = fk1_all_.loc[valid]
        exon_sub = exon_all_.loc[valid]
        fk2_sub  = fk2_all_.loc[valid]
            


        body_cols = [f'bin_{i+1}' for i in range(30, 90)]
        body_mean = exon_sub[body_cols].mean(axis=1)
        q_cuts = pd.qcut(body_mean, 5, labels=['Q1','Q2','Q3','Q4','Q5'])
        
        quintile_lists  = [list(body_mean[q_cuts == lbl].index) for lbl in ['Q1','Q2','Q3','Q4','Q5']]
        quintile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        for j, qlist in enumerate(quintile_lists):
            fk1_sub_q = fk1_sub.loc[qlist]
            fk2_sub_q = fk2_sub.loc[qlist]
            exon_sub_q = exon_sub.loc[qlist]
        

            _plot_cassette_exon(fk1_sub_q, exon_sub_q, fk2_sub_q, axes[i, j], q=q, color=color)
    
            axes[i, j].set_ylim([-0.1, 1.2])
            axes[i, j].set_xticks([])
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)

            if i == 0:
                axes[i, j].set_title(quintile_labels[j])

            if j > 0:
                axes[i,j].set_yticks([])
    
        if i < len(coord_lists):
            axes[i, 0].set_ylabel('PSI')

        # else:

    # ── schematic row ────────────────────────────────────────────────────────
    for j in range(5):
        _draw_schematic(axes[n_rows, j], color='gray')

    for j in range(3):
        axes[j, 5].set_ylim([-3, 4])
        axes[j, 6].set_ylim([-3, 4])
        if j in [0, 1]:
            a = 1
        else:
            a = 0.1
        axes[j, 5].plot([0, 380], [0.5, 0.5], c='tab:blue', linewidth=3, alpha=a)
        axes[j, 5].fill_between(np.arange(30, 90), [0]*60, [1]*60, color='tab:blue', zorder=2, alpha=a)
        axes[j, 5].fill_between(np.arange(30, 90)+130, [0]*60, [1]*60, color='tab:blue', zorder=2, alpha=a)
        axes[j, 5].fill_between(np.arange(30, 90)+260, [0]*60, [1]*60, color='tab:blue', zorder=2, alpha=a)

        if j in [0, 2]:
            a = 1
        else:
            a = 0.1
        axes[j, 5].plot([0, 380], [2.5, 2.5], c='tab:blue', linewidth=3, alpha=a)
        axes[j, 5].fill_between(np.arange(30, 90), [2]*60, [3]*60, color='tab:blue', zorder=2, alpha=a)
        # axes[j, 5].fill_between(np.arange(30, 90)+130, [2]*60, [3]*60, color='black', zorder=2)
        axes[j, 5].fill_between(np.arange(30, 90)+260, [2]*60, [3]*60, color='tab:blue', zorder=2, alpha=a)

        axes[j, 6].plot([0, 380], [0.5, 0.5], c=annot_color, linewidth=3)
        axes[j, 6].fill_between(np.arange(30, 90), [0]*60, [1]*60, color=annot_color, zorder=2)
        axes[j, 6].fill_between(np.arange(30, 90)+130, [0]*60, [1]*60, color=annot_color, zorder=2)
        axes[j, 6].fill_between(np.arange(30, 90)+260, [0]*60, [1]*60, color=annot_color, zorder=2)

        axes[j, 6].plot([0, 380], [2.5, 2.5], c=annot_color, linewidth=3)
        axes[j, 6].fill_between(np.arange(30, 90), [2]*60, [3]*60, color=annot_color, zorder=2)
        # axes[j, 6].fill_between(np.arange(30, 90)+130, [2]*60, [3]*60, color='tab:orange', zorder=2)
        axes[j, 6].fill_between(np.arange(30, 90)+260, [2]*60, [3]*60, color=annot_color, zorder=2)

    for j in range(4):
        for i in range(5, 7):
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
            axes[j, i].spines['top'].set_visible(False)
            axes[j, i].spines['right'].set_visible(False)
            axes[j, i].spines['bottom'].set_visible(False)
            axes[j, i].spines['left'].set_visible(False)


    axes[0, 5].set_title('Torino', c='tab:blue')
    axes[0, 6].set_title(annot_name, c=annot_color)

    x = len(coord_lists[0])
    y = len(coord_lists[1])
    z = len(coord_lists[2])

    axes[0, 6].text(100, -2, f'({x})', c=annot_color)
    axes[1, 6].text(100, -2, f'({y})', c=annot_color)
    axes[2, 6].text(100, -2, f'({z})', c=annot_color)
    return fig, axes


# --- plot_cumulative
# Copied verbatim from CassetteExonsDeNovo.ipynb cell 93 (In[127]).
def plot_cumulative(x, label, ax=None, zorder=1):
    x = sorted(x)
    y = np.arange(len(x))/len(x)
    if ax is None:
        plt.plot(x, y, linewidth=3, label=label)
    else:
        ax.plot(x, y, linewidth=3, label=label, zorder=zorder)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def load_cassette_groups():
    """The five groups of cassette exons the metaplots split on -- cell 76 (In[67]).

    gencode_vastdb   annotated by both
    gencode_only     annotated by Gencode, absent from VastDB
    vastdb_only      the reverse
    overlap_no_annot unannotated, but the region overlaps some annotated exon
    no_overlap       unannotated and overlapping nothing
    """
    cassette = read_cassette_bed('torino')
    gencode = read_cassette_bed('gencode')
    vastdb = load_vastdb_exons()
    overlaps = load_cassette_overlaps()

    coord = cassette.cassette_exons_coord
    in_v = coord.isin(vastdb.cassette_exons_coord)
    in_g = coord.isin(gencode.cassette_exons_coord)

    gencode_vastdb = list(coord[in_v & in_g])
    gencode_only = list(coord[(~in_v) & in_g])
    vastdb_only = list(coord[in_v & (~in_g)])
    no_annot = list(coord[(~in_v) & (~in_g)])
    no_overlap = list(overlaps.loc[overlaps.blockCount == 0].name)
    overlap_no_annot = [x for x in no_annot if x not in set(no_overlap)]

    return [gencode_vastdb, gencode_only, vastdb_only, overlap_no_annot, no_overlap]


# Short forms of CASSETTE_ROW_LABELS, for the row labels on the figure.
CASSETTE_ROW_SHORT = ['Gencode & VastDB', 'Gencode only', 'VastDB only',
                      'Unannotated,\noverlapping', 'Unannotated,\nno overlap']


def _cassette_metaplot(tissue, colour, title):
    groups = load_cassette_groups()
    fig, axes = plot_cassette_exon_metaplots(tissue, groups,
                                             color=colour, figsize=(4, 4))
    fig.suptitle(title, fontsize=12, y=1.01)

    # n per row, on the right-hand side so the 'PSI' label on column 0 stays.
    # Counted from the group actually passed to the plot, not from a stored
    # number, so the label cannot drift from what is drawn.
    for i, (label, grp) in enumerate(zip(CASSETTE_ROW_SHORT, groups)):
        ax = axes[i, 4]
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(f'{label}\n(n = {len(grp):,})', rotation=270,
                      fontsize=6.5, labelpad=16, va='center')
    return fig


@_panel
def panel_cassette_exons_brain_cortex():
    """Notebook cell 76 (In[67])."""
    return _cassette_metaplot('Brain_Cortex', '#FFD700', 'Brain - Cortex')


@_panel
def panel_cassette_exons_heart_atrial_appendage():
    """Notebook cell 77 (In[68])."""
    return _cassette_metaplot('Heart_Atrial_Appendage', '#B452CD',
                              'Heart - Atrial Appendage')


@_panel
def panel_cassette_exons_liver():
    """Notebook cell 78 (In[69])."""
    return _cassette_metaplot('Liver', '#CDB79E', 'Liver')


PANELS = {
    'cassette_exons_brain_cortex': panel_cassette_exons_brain_cortex,
    'cassette_exons_heart_atrial_appendage': panel_cassette_exons_heart_atrial_appendage,
    'cassette_exons_liver': panel_cassette_exons_liver,
}


# --------------------------------------------------------------------------
# Inputs for the remaining two panels
# --------------------------------------------------------------------------
BASE_EXON_COLS = ('chrom', 'start', 'end', 'gene', 'strand', 'intron_start',
                  'intron_end', 'flanking1_start', 'flanking2_end',
                  'exon_coordinates', 'exon_id', 'transcripts')


def load_torino_internal_exons():
    """Torino internal exons, with the intron and cassette keys -- cells 1, 8, 109, 110."""
    def build():
        df = pd.read_csv('../code/results/torino/annotations/'
                         'torino_internal_exons_w_transcript.bed.gz', sep='\t')
        df['exon'] = (df.chrom + ':' + df.start.astype(str)
                      + ':' + df.end.astype(str))
        df['intron'] = (df.chrom + ':' + df.intron_start.astype(str)
                        + ':' + df.intron_end.astype(str))
        df['cassette_exons_coord'] = (df.exon + ':' + df.intron_start.astype(str)
                                      + ':' + df.intron_end.astype(str))
        return _subset(df, df.gene)
    return _cached(f'torino_internal_exons:{_tag()}', build)


def load_torino_introns_df():
    """Torino introns -- cell 1 (In[2])."""
    def build():
        df = pd.read_csv('../code/results/torino/annotations/torino_introns.bed.gz',
                         sep='\t')
        return _subset(df, df.gene_id)
    return _cached(f'torino_introns_df:{_tag()}', build)


def load_cassette_df_w_transcripts():
    """Cassette exons with the transcripts carrying them -- cell 12 (In[13]).

    The only loop here that cannot be replaced by reading a file: this frame's
    `transcripts` column was never written out. It is restricted to the genes
    that yielded a cassette exon at all, which read_cassette_bed already names,
    so it runs over a few thousand genes instead of ~15,000 -- same rows out.
    """
    def build():
        exons = load_torino_internal_exons()
        introns = load_torino_introns_df()
        genes = set(read_cassette_bed('torino').gene.unique())
        # The 20 column names below assume the frame the notebook had at this
        # point: the bed file's 12 columns plus `exon` from cell 8. This loader
        # also carries `intron` and `cassette_exons_coord` for other panels, and
        # passing those into the intersect shifts every field, which silently
        # empties the result.
        exons = exons[list(BASE_EXON_COLS) + ['exon']]
        exons_grouped = exons.loc[exons.gene.isin(genes)].groupby('gene')
        introns_grouped = introns.groupby('gene_id')

        out = []
        for gene, exon_df in tqdm(exons_grouped, leave=True, position=0):
            exon_df = exon_df.loc[exon_df.start < exon_df.end]
            try:
                intron_df = introns_grouped.get_group(gene)
            except KeyError:
                continue
            intron_df = intron_df.loc[intron_df.start < intron_df.end]
            if (exon_df.shape[0] == 0) or (intron_df.shape[0] == 0):
                continue
            hits = BedTool.from_dataframe(exon_df).intersect(
                BedTool.from_dataframe(intron_df), f=1, wo=True).to_dataframe(
                names=['chrom', 'start', 'end', 'gene', 'strand', 'intron_start',
                       'intron_end', 'flanking1_start', 'flanking2_end',
                       'exon_coordinates', 'exon_id', 'transcripts', 'exon',
                       'chrom_int', 'start_int', 'end_int', 'gene_int',
                       'transcript_int', 'strand_int', 'overlap'])
            if hits.shape[0] == 0:
                continue
            hits = hits.loc[(hits.intron_start == hits.start_int)
                            & (hits.intron_end == hits.end_int)]
            if hits.shape[0] == 0:
                continue
            hits['cassette_exons_coord'] = (
                hits.chrom + ':' + hits.start.astype(str) + ':' + hits.end.astype(str)
                + ':' + hits.start_int.astype(str) + ':' + hits.end_int.astype(str))
            hits = hits[['chrom', 'start', 'end', 'cassette_exons_coord', 'gene',
                         'strand', 'transcripts']].drop_duplicates()
            if hits.shape[0] > 0:
                out.append(hits)

        df = pd.concat(out, axis=0)
        df['exon'] = (df.chrom + ':' + df.start.astype(str)
                      + ':' + df.end.astype(str))
        return df
    return _cached(f'cassette_df_w_transcripts:{_tag()}', build)


def load_human_EL_cassette():
    """The EL table this notebook uses -- cell 80 (In[75]).

    Note this is isoforms_filtered.merged_isoforms, NOT the corrected_ table the
    intron-retention notebook reads.
    """
    def build():
        el = pd.read_csv('../code/results/torino/tables/Human/'
                         'isoforms_filtered.merged_isoforms.EL.bed.gz', sep='\t')
        el['pid'] = el.gid + '.' + el.pid
        el.index = el.pid
        return _subset(el, el.gid)
    return _cached(f'human_EL_cassette:{_tag()}', build)


def load_torino_v_gencode_groups():
    """The three groups for the Torino-vs-Gencode comparison -- cell 121 (In[103]).

    gencode_and_torino  cassette exons both call
    torino_constant     Torino interior exons Gencode calls cassette, where
                        Torino sees no intron (so: constitutive in Torino)
    gencode_only        Gencode cassette exons whose intron Torino observes but
                        whose exon Torino never calls
    """
    gencode = read_cassette_bed('gencode')
    exons = load_torino_internal_exons()
    introns_df = load_torino_introns_df()
    cassette = read_cassette_bed('torino')

    gencode_bed = BedTool.from_dataframe(bedtools_safe(gencode))
    exons_bed = BedTool.from_dataframe(bedtools_safe(exons))
    gencode_overlaps = gencode_bed.intersect(exons_bed, c=True).to_dataframe()
    gencode_overlaps['intron'] = gencode_overlaps.name.apply(
        lambda s: (lambda x: f'{x[0]}:{x[3]}:{x[4]}')(s.split(':')))

    positive = introns_df.loc[(introns_df.end - introns_df.start) > 0]
    exon_intron = exons_bed.intersect(
        BedTool.from_dataframe(bedtools_safe(positive)), c=True).to_dataframe(
        names=list(exons.columns) + ['blockCount'])

    torino_introns = list(introns_df.chrom + ':' + introns_df.start.astype(str)
                          + ':' + introns_df.end.astype(str))

    gencode_only = list(gencode_overlaps.loc[
        gencode_overlaps.intron.isin(torino_introns)
        & (gencode_overlaps.blockCount == 0)].name)
    torino_constant = list(exon_intron.loc[
        exon_intron.cassette_exons_coord.isin(gencode_overlaps.name)
        & (exon_intron.blockCount == 0)].cassette_exons_coord.unique())
    gencode_and_torino = list(cassette.loc[
        cassette.cassette_exons_coord.isin(
            gencode.cassette_exons_coord)].cassette_exons_coord)

    return [gencode_and_torino, torino_constant, gencode_only]


def load_mean_loadings():
    """Mean isoform loading per cassette-exon group -- cell 85 (In[105]).

    For each group, the isoforms that carry exactly one of its cassette exons,
    then the mean loading of each across GTEx samples.

    `gencode_only` is bound twice: cell 76 (In[67]) to the Torino cassette
    exons Gencode also calls, cell 121 (In[103]) to Gencode cassette exons
    Torino misses. This cell ran at In[105], after both, so execution order
    says the second -- but rendering it both ways against the published
    cassette_exons.median_loadings.png settles it the other way: In[67] gives
    corr 1.000 and In[103] only 0.993. All five groups come from In[67].
    """
    groups = load_cassette_groups()

    w_transcripts = load_cassette_df_w_transcripts()
    el = load_human_EL_cassette()
    gtex_samples = el.columns[6:]

    out = []
    for case in groups:
        sel = w_transcripts.loc[w_transcripts.cassette_exons_coord.isin(case)]
        counts = sel.groupby('transcripts').transcripts.value_counts()
        isoforms = counts.loc[counts == 1].index
        idx = [x for x in isoforms if x in el.index]
        out.append(list(el.loc[idx, gtex_samples].mean(axis=1)))
    return out


@_panel
def panel_cassette_exons_median_loadings():
    """Notebook cell 94 (In[129])."""
    labels = ['Gencode & VastDB', 'Gencode', 'VastDB', 'Overlap', 'No Overlap']
    fig, ax = plt.subplots(figsize=(4, 4))
    # n per curve, taken from the array each curve is drawn from, so the label
    # cannot drift from the data.
    #
    # The unit is ISOFORMS, not exons: load_mean_loadings keeps transcripts
    # carrying exactly one cassette exon from the group, then those present in
    # the usage matrix, and plots each one's mean usage across GTEx samples. So
    # these n are unrelated to the group sizes in cassette_row_table, which
    # count exons -- two filters sit between them (the one-exon rule, and
    # presence in the matrix).
    for i, x in enumerate(load_mean_loadings()):
        plot_cumulative(x, f'{labels[i]} (n = {len(x):,})', ax, 5 - i)
    ax.legend(frameon=False, fontsize=8, title_fontsize=7,
              title='Torino cassette exons\n'
                    'n = isoforms carrying exactly one such exon,\n'
                    'with measured usage')
    ax.set_ylabel('Cumulative percentage')
    ax.set_xlabel('Mean isoform usage')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    return fig


@_panel
def panel_cassette_exons_torino_v_gencode_brain_cortex():
    """Notebook cell 128 (In[111])."""
    fig, axes = plot_cassette_exon_metaplots_comparison(
        'Brain_Cortex', load_torino_v_gencode_groups(), list_labels=None, q=0.5,
        color='gray', figsize=(10, 3), annot_color='tab:orange',
        annot_name='Gencode', annotation='gencode')
    fig.suptitle('Cassette exons annotated by Torino and Gencode',
                 fontsize=12, y=1.01)
    return fig


PANELS['cassette_exons_median_loadings'] = panel_cassette_exons_median_loadings
PANELS['cassette_exons_torino_v_gencode_brain_cortex'] = (
    panel_cassette_exons_torino_v_gencode_brain_cortex)


# --------------------------------------------------------------------------
# Row counts for the cassette-exon metaplots
#
# The five rows of panel_cassette_exons_<tissue> are the five annotation
# groups from load_cassette_groups(); the five columns within a row are
# expression quintiles of that row's exons. So "entries per row" is the
# number of exons in the group that the tissue's matrices actually carry:
# plot_cassette_exon_metaplots keeps only `[c for c in coords if c in
# fk1_all.index]`, and the index is the intersection of the three matrices.
# That is why the counts differ between tissues.
#
# Only the coordinate column is read, not the 200 position columns.
# --------------------------------------------------------------------------
CASSETTE_ROW_LABELS = ['Gencode & VastDB', 'Gencode only', 'VastDB only',
                       'Unannotated, overlaps an annotated exon',
                       'Unannotated, no overlap']

CASSETTE_TISSUES = ('Brain_Cortex', 'Heart_Atrial_Appendage', 'Liver')


def _tissue_matrix_index(tissue):
    """The coordinates a tissue's metaplot can draw: the shared index."""
    def _idx(region):
        path = f'{MATRIX_DIR}/{tissue}.{region}.mat.gz'
        col = pd.read_csv(path, sep='\t', skiprows=1, names=_ALL_COLS,
                          usecols=['gene'])['gene']
        return pd.Index(col)
    idx = _idx('flanking1_exons')
    for region in ('cassette_exons', 'flanking2_exons'):
        idx = idx.intersection(_idx(region))
    return idx


def cassette_row_counts(tissues=CASSETTE_TISSUES):
    """{tissue: [n per row]}, plus 'group' for the unfiltered group sizes."""
    def build():
        groups = load_cassette_groups()
        out = {'group': [len(g) for g in groups]}
        for tissue in tissues:
            idx = set(_tissue_matrix_index(tissue))
            out[tissue] = [sum(1 for c in g if c in idx) for g in groups]
        # The rows are not a strict partition: no_overlap comes from the
        # bedtools overlap table (blockCount == 0), a different source from the
        # isin-based annotation groups, so it is not guaranteed disjoint from
        # them -- only overlap_no_annot explicitly subtracts it. Report the
        # double-counted exons rather than leave the sum unexplained.
        union = set().union(*[set(g) for g in groups])
        out['union'] = len(union)
        out['double_counted'] = sum(len(g) for g in groups) - len(union)
        return out
    return _cached(f'cassette_row_counts:{_tag()}:{",".join(tissues)}', build)


def cassette_row_table(tissues=CASSETTE_TISSUES):
    """Exons per metaplot row, for both gene sets."""
    keep = _R.GENE_SUBSET
    try:
        _R.GENE_SUBSET = None
        free()
        a = cassette_row_counts(tissues)
        _R.GENE_SUBSET = _R.load_select_genes()
        free()
        c = cassette_row_counts(tissues)
    finally:
        _R.GENE_SUBSET = keep
        free()

    head = (['**Cassette-exon metaplot rows: exons per row**', '',
             'The five rows are annotation groups; the five columns within a row '
             'are expression quintiles of that row, so each quintile holds about '
             'a fifth of the row count.', '',
             '| row | group total | ' + ' | '.join(t.replace("_", " ")
                                                   for t in tissues) + ' |',
             '|---|---:|' + '---:|' * len(tissues)])
    rows = []
    for label, (a_tot, c_tot) in zip(CASSETTE_ROW_LABELS,
                                     zip(a['group'], c['group'])):
        i = CASSETTE_ROW_LABELS.index(label)
        cells = [f'{a_tot:,} / {c_tot:,}']
        for t in tissues:
            cells.append(f'{a[t][i]:,} / {c[t][i]:,}')
        rows.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    tot = [f'{sum(a["group"]):,} / {sum(c["group"]):,}']
    for t in tissues:
        tot.append(f'{sum(a[t]):,} / {sum(c[t]):,}')
    rows.append('| **total** | ' + ' | '.join(tot) + ' |')
    foot = ['', '> Each cell is **all genes / corrected**. The group total is '
                'every exon in that group; a tissue column counts only those '
                'its three matrices carry, which is what the row actually plots. '
                'The three tissues agree because their matrices share one index '
                'covering every group member -- the rows plot the same exons in '
                'each panel, only the coverage differs.']
    if a['double_counted'] or c['double_counted']:
        foot.append(
            f"> The rows are not a strict partition: {a['double_counted']} exon(s) "
            f"({c['double_counted']} corrected) fall in two rows and are drawn "
            f"twice, so the rows sum to {sum(a['group']):,} over "
            f"{a['union']:,} distinct exons. These are VastDB-annotated exons "
            f"that the overlap table reports as overlapping nothing, so they are "
            f"in both 'VastDB only' and 'Unannotated, no overlap'. This is the "
            f"source notebook's own grouping (cell 76), left as is.")
    for t in tissues:
        if any(a[t][i] < 5 for i in range(len(CASSETTE_ROW_LABELS))):
            foot.append(f'> NOTE: a row in {t} has fewer than 5 exons, so its '
                        f'five quintiles cannot be formed cleanly.')
    return '\n'.join(head + rows + foot)
