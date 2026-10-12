"""SFigure2 intron-retention panels, drawn from the results already on disk.

Seven panels, from analysis/Retained_Introns_figure.ipynb. Each has a small
`panel_<name>()` that loads what it needs and draws it. Nothing here recomputes
an analysis output: every input is a file that already exists, and a guard below
raises if anything tries to write into the work tree.

The plotting functions and `get_metaplot_input` are copied verbatim from the
notebook -- they are not importable from anywhere else -- with their source cell
noted. Everything else is loading.

A note on `remove_low`, which the notebook sets twice. The execution counts fix
which setting each panel actually used: the matrices were built with
remove_low=False at In[49], and intron_metaplot_Liver (In[54]) and
intron_metaplot_quintiles (In[65]) both ran against those. They were rebuilt
with remove_low=True at In[80], and intron_metaplot_quintiles_vastdb ran after,
at In[87]. The CDF panel is the one case the execution trail gets wrong; see
load_intron_coverage_groups.

Every panel here was checked against the figure already in
code/results/figures/revision/supp_fig/intron_retention/ and reproduces it
pixel for pixel at 500 dpi.
"""

import builtins
import functools
import gc
import os
import pickle
import re
import sys

ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
NB_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
sys.path.append('/project2/yangili1/cfbuenabadn/torino_paper/code/scripts')
sys.path.append(ANALYSIS_DIR)

# Every path below is relative to analysis/, as in the source notebook, and some
# of the modules imported next read their data at import time.
os.chdir(ANALYSIS_DIR)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.figure
from pybedtools import BedTool
# plot_pca_scatter_EL annotates its return as -> PCA but imports PCA inside
# its own body; the notebook had it in globals already.
from sklearn.decomposition import PCA

from load_fig3_metadata import *      # noqa: F401,F403


# --------------------------------------------------------------------------
# Write guard: these panels read. Nothing may write into the work tree.
# --------------------------------------------------------------------------
_PROTECTED = ('/project2/yangili1/cfbuenabadn/torino_paper',
              '/project2/mstephens/cfbuenabadn/gtex-stm')

# The two exceptions inside the work tree: cached panel data, and the rendered
# panels. Everything else under _PROTECTED stays read-only.
#
# This must hold for ANY notebook directory, not just this module's own: the
# corrected notebook lives in SFigure2_corrected/ and imports these modules
# from SFigure2/, so a _WRITABLE pinned to __file__ would make the guard reject
# SFigure2_corrected/panels/ -- it would block the very figures it is meant to
# let through. So: any directory named panels/ or figure_data/ anywhere under
# notebooks/ is writable, and nothing else in the work tree is.
_NB_DIR = os.path.dirname(os.path.abspath(__file__))
_NOTEBOOKS_ROOT = os.path.dirname(_NB_DIR)
_WRITABLE_NAMES = ('panels', 'figure_data')
PANEL_DIR = os.path.join(_NB_DIR, 'panels')


def _is_writable(p):
    if not (p + os.sep).startswith(_NOTEBOOKS_ROOT + os.sep):
        return False
    parts = os.path.relpath(p, _NOTEBOOKS_ROOT).split(os.sep)
    return any(part in _WRITABLE_NAMES for part in parts)


def _guard_path(path, what):
    try:
        p = os.path.abspath(os.fspath(path))
    except TypeError:
        return
    if _is_writable(p):
        return
    if p.startswith(_PROTECTED):
        raise PermissionError(
            f'{what} tried to write into the work tree: {p}\n'
            'These panels only read; the inputs already exist.')


_orig_open = builtins.open


def _guarded_open(file, mode='r', *a, **k):
    if any(c in str(mode) for c in 'wxa'):
        _guard_path(file, 'open')
    return _orig_open(file, mode, *a, **k)


builtins.open = _guarded_open

for _cls, _meth, _label in [(matplotlib.figure.Figure, 'savefig', 'savefig'),
                            (pd.DataFrame, 'to_csv', 'to_csv'),
                            (pd.DataFrame, 'to_pickle', 'to_pickle'),
                            (BedTool, 'saveas', 'BedTool.saveas'),
                            (BedTool, 'moveto', 'BedTool.moveto')]:
    def _wrap(orig, label):
        def guarded(self, path_or_buf=None, *a, **k):
            if path_or_buf is not None:
                _guard_path(path_or_buf, label)
            return orig(self, path_or_buf, *a, **k)
        return guarded
    setattr(_cls, _meth, _wrap(getattr(_cls, _meth), _label))


def bedtools_safe(df, limit=3900):
    """Blank BED fields until every line fits bedtools' ~4 kB input buffer.

    Over-long lines lose their trailing strand field silently, and the next
    intersect in the chain then dies with "has 5 fields, but 6 were expected".
    332 lines of gencode.retained_introns are over the limit. Only the 4th
    column (name) is read downstream, so blanking the widest field of an
    over-long line leaves every result identical. Note dtype is not consulted:
    under pandas 3 a column of strings is `str`, not `object`.
    """
    lens = pd.DataFrame({c: df[c].astype(str).str.len() for c in df.columns})
    sep = len(df.columns) - 1
    over = lens.sum(axis=1) + sep > limit
    if not over.any():
        return df
    out = df.copy()
    while over.any():
        widest = lens.loc[over].idxmax(axis=1)
        for col in widest.unique():
            rows = widest.index[widest == col]
            out.loc[rows, col] = '.'
            lens.loc[rows, col] = 1
        over = lens.sum(axis=1) + sep > limit
    return out


# --------------------------------------------------------------------------
# Inputs. Each is a file that already exists; results are cached per session.
# --------------------------------------------------------------------------
BED_DIR = '../code/results/CoveragePlots/bed_files/introns'
_CACHE = {}

# Inputs are cached so one panel does not read the same file twice, and dropped
# again once that panel is drawn: the EL table alone is a couple of GB in
# memory, and holding several panels' inputs at once kills the kernel. Set
# KEEP_CACHE = True to trade memory for speed when drawing several panels that
# share inputs (PCA_all and PCA_introns both read the EL table).
KEEP_CACHE = False

# The EL table's 2,016 sample columns as float32 rather than float64: 1.5 GB
# instead of 2.5 GB, before the PCA copies and imputes it. Set False to restore
# the notebook's exact dtypes at twice the memory.
EL_FLOAT32 = True

# The corrected analysis is the same panels over a subset of genes: the 16,073
# higher-confidence genes from analysis/Fig1Numbers.ipynb. Set
# GENE_SUBSET = load_select_genes() and every loader below filters to it --
# nothing is recomputed, the same files are read and then cut down.
GENE_SUBSET = None
SELECT_GENES_N = 16073


def load_select_genes():
    """The higher-confidence gene set, as analysis/Fig1Numbers.ipynb defines it.

    Torino reached K=10 in >= 100 samples, the gene carries a corrected isoform
    annotation, and it does not overlap another gene. The size is asserted
    against the notebook's recorded 16,073, so a change in any of the three
    source tables fails loudly rather than shifting every number silently.
    """
    def build():
        root = '/project2/yangili1/cfbuenabadn/torino_paper/code'
        status = pd.read_csv(f'{ANALYSIS_DIR}/torino_K10_status_Human.csv')
        annot = pd.read_csv(f'{root}/results/torino/Human/isoforms/'
                            'corrected_annotated.isoforms.tab.gz', sep='\t')
        annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
        no_overlap = pd.read_csv(f'{root}/config/human.protein_coding.bed.gz',
                                 sep='\t', index_col=0)
        sel = (pd.Index(status.loc[status.n_samples >= 100].gene)
               .intersection(pd.Index(annot.gene.unique()))
               .intersection(pd.Index(no_overlap.gene_name)))
        if len(sel) != SELECT_GENES_N:
            raise AssertionError(f'select_genes has {len(sel)}, '
                                 f'expected {SELECT_GENES_N}')
        return sel
    return _cached('select_genes', build)


def _symbol_of_id():
    """Ensembl gene id -> symbol, from code/config/human.protein_coding.bed.gz."""
    def build():
        bed = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/config/'
                          'human.protein_coding.bed.gz', sep='\t', index_col=0)
        return dict(zip(bed.gene_id, bed.gene_name))
    return _cached('symbol_of_id', build)


def _as_symbols(genes):
    """Gene symbols, translating Ensembl ids where a table uses them.

    The intron tables are not consistent: appris and gencode name introns
    GENE_<annot>_intron_<n> with a symbol, but vastdb's ir_id is a bare Ensembl
    id (ENSG00000187634). select_genes is a set of symbols, so filtering vastdb
    without this mapping matches 90 of its 179,081 rows instead of ~99%.
    """
    genes = pd.Index(genes)
    if not genes.str.startswith('ENSG').any():
        return genes
    mapping = _symbol_of_id()
    return pd.Index([mapping.get(g.split('.')[0], g) for g in genes])


def _gene_of(names):
    """Gene symbol from an intron id: GENE_<annotation>_intron_<n>."""
    return pd.Index(names).str.split('_').str[0]


def _subset(df, genes):
    """Keep only rows whose gene is in GENE_SUBSET (no-op when unset)."""
    if GENE_SUBSET is None:
        return df
    return df.loc[_as_symbols(genes).isin(GENE_SUBSET)]


def _tag():
    """Cache-key suffix, so switching GENE_SUBSET never returns stale data."""
    return 'corrected' if GENE_SUBSET is not None else 'all'


# --------------------------------------------------------------------------
# Disk cache
#
# Every loader goes through _cached, so persisting here gives all of them a
# pickle layer at once: a loader's result is computed from the source tables
# exactly once and read back from figure_data/cache/ afterwards. Without this,
# each render re-read the .bed.gz / .mat.gz inputs -- the three cassette panels
# parsed the same three matrices three times, 120 s each.
#
# CACHE_VERSION is part of every filename. Bump it when a loader's definition
# changes, and nothing can silently read a result computed by the old code --
# which is what the orphaned Sep-24 pickles in figure_data/ would have done,
# since they predate the reference-subsetting, _as_symbols and inner->left
# merge fixes.
#
# Keys already carry _tag(), so the corrected and uncorrected results are
# separate files and switching GENE_SUBSET cannot return the wrong one.
# --------------------------------------------------------------------------
CACHE_VERSION = 'v1'
DISK_CACHE = True
DISK_CACHE_DIR = os.path.join(_NB_DIR, 'figure_data', 'cache')
# Above this, a result is kept in memory but not written: the EL table pickles
# to ~1.5 GB, and writing it costs more than re-reading the source.
DISK_CACHE_MAX_MB = 400


def _cache_path(key):
    safe = re.sub(r'[^A-Za-z0-9_.=-]', '_', key)
    return os.path.join(DISK_CACHE_DIR, f'{safe}.{CACHE_VERSION}.pkl')


def _cached(key, build):
    if key in _CACHE:
        return _CACHE[key]

    path = _cache_path(key)
    if DISK_CACHE and os.path.exists(path):
        try:
            with open(path, 'rb') as fh:
                _CACHE[key] = pickle.load(fh)
            return _CACHE[key]
        except Exception as exc:
            # A truncated or stale-format file must not take the panel down.
            print(f'[cache] re-building {key}: {type(exc).__name__}')

    value = build()
    _CACHE[key] = value
    if DISK_CACHE:
        _save_cache(key, value, path)
    return value


def _save_cache(key, value, path):
    """Write a loader result, skipping what cannot or should not be written."""
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        # BedTool objects and anything holding a file handle land here. They
        # stay in the in-memory cache; only the disk copy is skipped.
        print(f'[cache] not picklable, skipped: {key} ({type(exc).__name__})')
        return
    mb = len(blob) / 1e6
    if mb > DISK_CACHE_MAX_MB:
        print(f'[cache] {key} is {mb:.0f} MB > {DISK_CACHE_MAX_MB} MB, not written')
        return
    os.makedirs(DISK_CACHE_DIR, exist_ok=True)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as fh:
            fh.write(blob)
        os.replace(tmp, path)       # atomic: a killed job leaves no half file
    except Exception as exc:
        print(f'[cache] write failed for {key}: {type(exc).__name__}: {exc}')
        if os.path.exists(tmp):
            os.remove(tmp)


def clear_disk_cache(pattern=None):
    """Delete cached loader results. pattern filters by substring."""
    if not os.path.isdir(DISK_CACHE_DIR):
        print('no disk cache')
        return 0
    n = 0
    for f in sorted(os.listdir(DISK_CACHE_DIR)):
        if pattern and pattern not in f:
            continue
        os.remove(os.path.join(DISK_CACHE_DIR, f))
        n += 1
    print(f'removed {n} cached file(s)')
    return n


def free():
    """Drop every input loaded so far."""
    _CACHE.clear()
    gc.collect()


# Set SAVE_DIR to a directory and every panel writes itself as soon as it is
# drawn -- no separate save step, no remembering to call anything. The notebooks
# set it in their first cell. PANEL_DIR ('panels/' next to this file) is one of
# the two paths the write guard allows inside the work tree.
SAVE_DIR = None
SAVE_FORMATS = ('png', 'pdf')
SAVE_DPI = 400


def _save_panel(name, result=None):
    """Write the figure just drawn into SAVE_DIR as <name>.<ext>."""
    if not SAVE_DIR:
        return []
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Panels return a Figure, a seaborn grid, or nothing at all; fall back to
    # whatever matplotlib currently has open.
    fig = getattr(result, 'figure', result)
    if not hasattr(fig, 'savefig'):
        fig = plt.gcf()
    written = []
    for ext in SAVE_FORMATS:
        path = os.path.join(SAVE_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight')
        written.append(path)
    print(f'[saved] {name}: ' + ', '.join(os.path.basename(w) for w in written)
          + f'  -> {SAVE_DIR}')
    return written


def _panel(fn):
    """A panel: load, draw, save if SAVE_DIR is set, release the inputs."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            name = fn.__name__[6:] if fn.__name__.startswith('panel_') \
                else fn.__name__
            try:
                _save_panel(name, result)
            except Exception as exc:        # a failed save must not lose the figure
                print(f'[saved] {name}: FAILED -- {type(exc).__name__}: {exc}')
            return result
        finally:
            if not KEEP_CACHE:
                free()
    return wrapped


def load_intron_table(name):
    """One of the retained-intron bed tables (appris / vastdb / gencode)."""
    def build():
        df = pd.read_csv(f'{BED_DIR}/{name}.retained_introns.bed.gz', sep='\t',
                         names=['chrom', 'start', 'end', 'ir_id', 'isoforms', 'strand'])
        df['gene'] = df.ir_id.apply(lambda x: x.split('_')[0])
        return _subset(df, df.gene)
    return _cached(f'table:{name}:{_tag()}', build)


def load_intron_beds():
    """torino_all_bed, torino_bed, gencode_bed -- notebook cell 49 (In[44])."""
    def build():
        cols = ['chrom', 'start', 'end', 'transcript', 'factors', 'strand']
        beds = {}
        for key, fname in [('torino_all', 'torino.all_retained_introns.bed'),
                           ('torino', 'torino.retained_introns.bed.gz'),
                           ('gencode', 'gencode.retained_introns.bed.gz')]:
            df = pd.read_csv(f'{BED_DIR}/{fname}', sep='\t', names=cols)
            df = _subset(df, _gene_of(df.transcript))
            beds[key] = BedTool.from_dataframe(bedtools_safe(df))
        return beds
    return _cached(f'beds:{_tag()}', build)


def load_metaplot_X(remove_low):
    """torino_X, gencode_X, appris_X for Liver -- notebook cells 61 / 69.

    get_metaplot_input reads the per-tissue .mat.gz matrices that are already
    under code/results/CoveragePlots/bed_files/IntronsDeNovo/matrices/.
    """
    def build():
        X = {a: get_metaplot_input('Liver', a, remove_low=remove_low)
             for a in ('torino', 'gencode', 'appris')}
        return {a: _subset(x, _gene_of(x.index)) for a, x in X.items()}
    return _cached(f'X:{remove_low}:{_tag()}', build)


EL_PATH = ('../code/results/torino/tables/Human/'
           'corrected_isoforms_filtered.merged_isoforms.EL.bed.gz')


def load_human_EL(float32=True):
    """The corrected EL table -- notebook cell 146 (In[115]).

    90,965 x 2,022. The 2,016 sample columns come back as float64 by default,
    which is 2.5 GB before the PCA has copied or imputed anything -- enough to
    kill a kernel on a login node. Reading them as float32 halves that; the
    loadings are ratios in [0, 1] and the PCA scatter is unchanged (checked
    against the published figure).
    """
    def build():
        dtypes = None
        if float32:
            header = pd.read_csv(EL_PATH, sep='\t', nrows=0).columns
            dtypes = {c: np.float32 for c in header[6:]}
        el = pd.read_csv(EL_PATH, sep='\t', dtype=dtypes)
        el['pid'] = el.gid + '.' + el.pid
        el.index = el.pid
        return _subset(el, el.gid)
    return _cached(f'human_EL:{float32}:{_tag()}', build)


def load_human_metadata():
    """Sample -> tissue.

    The notebook never builds this; it was left in the kernel from an earlier
    session. Reconstructed from the table every other notebook uses.
    """
    def build():
        meta = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv',
                           sep='\t')[['entity:sample_id', 'tissue_id']]
        meta.columns = ['Sample', 'Tissue']
        meta.index = meta.Sample
        return meta
    return _cached('human_metadata', build)


def load_annot():
    """The corrected isoform annotation, with efficiency_factor.

    efficiency_factor marks isoforms whose retained introns look like
    incomplete splicing rather than a real isoform; the rule is the notebook's
    (cell 160): an isoform is NOT an efficiency factor when it has at most two
    retained introns in a reference of five or more exons, or at most one in a
    reference of three or more.
    """
    def build():
        annot = pd.read_csv('../code/results/torino/Human/isoforms/'
                            'corrected_annotated.isoforms.tab.gz', sep='\t')
        annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
        eff = []
        for _, row in annot.iterrows():
            ref = row.appris_ref
            if 'RI:' in ref:
                n_ri = int([x for x in ref.split('/') if x.startswith('RI:')][0].split(':')[1])
                length = int(row.appris_transcript_length)
                eff.append(not (((n_ri <= 2) and (length >= 5))
                                or ((n_ri <= 1) and (length >= 3))))
            else:
                eff.append(False)
        annot['efficiency_factor'] = eff
        return _subset(annot, annot.gene)
    return _cached(f'annot:{_tag()}', build)


def load_IR_transcripts():
    """Transcripts whose appris reference calls a retained intron -- cell 160."""
    annot = load_annot()
    return annot.loc[(~annot.efficiency_factor)
                     & (annot.appris_ref.apply(lambda x: 'RI:' in x))]


def intron_retention_numbers():
    """The retained-intron counts quoted in the text, honouring GENE_SUBSET.

    Every line traces to a cell of Retained_Introns_figure.ipynb:

      introns              In[82]   torino_introns_all.shape
      transcripts, genes   In[90], In[84]
      tested genes         annot.gene.nunique()
      true IR              In[95]   torino_introns.shape
      true IR genes        In[86]
      gencode RI           In[99]   gencode_introns.shape
      gencode observed     In[48]   gencode_bed.intersect(torino_all, F=1, f=1)
      gencode as true IR   In[49]   gencode_bed.intersect(torino_bed, F=1, f=1)
      gencode missing      In[52]   gencode_bed.intersect(torino_all, v=True)
      torino not gencode   In[53]   torino_bed.intersect(gencode_bed, v=True)
    """
    beds = load_intron_beds()
    annot = load_annot()
    is_ri = annot.appris_ref.apply(lambda x: 'RI:' in x)

    n = {}
    n['introns'] = len(beds['torino_all'])
    n['transcripts'] = annot.loc[is_ri].transcript.nunique()
    n['genes'] = annot.loc[is_ri].gene.nunique()
    n['tested_genes'] = annot.gene.nunique()
    n['true_ir'] = len(beds['torino'])
    n['true_ir_genes'] = annot.loc[is_ri & (~annot.efficiency_factor)].gene.nunique()
    n['gencode_ri'] = len(beds['gencode'])
    n['gencode_observed'] = len(beds['gencode'].intersect(beds['torino_all'], F=1, f=1))
    n['gencode_true_ir'] = len(beds['gencode'].intersect(beds['torino'], F=1, f=1))
    n['gencode_missing'] = len(beds['gencode'].intersect(beds['torino_all'], v=True))
    # 'observed' demands reciprocal 100% overlap (F=1, f=1) while 'missing'
    # demands none at all (v=True), so the two are NOT complements: an intron
    # that overlaps a Torino intron without matching its coordinates exactly is
    # in neither. Without this row, gencode_ri - gencode_observed does not equal
    # gencode_missing and the table looks wrong. (Checked: intersect emits one
    # line per overlapping PAIR, but here pairs == unique features, so nothing
    # is double counted.)
    n['gencode_partial'] = (len(beds['gencode'].intersect(beds['torino_all'], u=True))
                            - n['gencode_observed'])
    # The three now partition the GENCODE retained introns exactly; if they
    # ever stop doing so, say so rather than print rows that do not add up.
    _parts = n['gencode_observed'] + n['gencode_missing'] + n['gencode_partial']
    if _parts != n['gencode_ri']:
        n['gencode_UNACCOUNTED'] = n['gencode_ri'] - _parts
    n['torino_not_gencode'] = len(beds['torino'].intersect(beds['gencode'], v=True))
    # observed introns that are not true IR: splicing-efficiency factors
    eff = beds['torino_all'].intersect(beds['torino'], v=True)
    n['efficiency_introns'] = len(eff)
    n['efficiency_gencode'] = len(eff.intersect(beds['gencode'], F=1, f=1))
    n['efficiency_not_gencode'] = len(eff.intersect(beds['gencode'], v=True))
    # observed by Torino but called an efficiency factor rather than true IR
    n['gencode_efficiency'] = (n['gencode_ri'] - n['gencode_missing']
                               - n['gencode_true_ir'])
    return n


# --------------------------------------------------------------------------
# Plotting helpers, copied verbatim from the notebook (they live nowhere else).
# --------------------------------------------------------------------------
# --- gtex_colors2: the GTEx tissue colour table.
# Copied verbatim from UTR3.ipynb cell 132 (In[91]).
gtex_colors2 = {
  "Adipose_Subcutaneous": {
    "tissue_abbrv": "ADPSBQ", 
    "tissue_color_hex": "#FFA54F", 
    "tissue_color_rgb": "255,165,79"
  }, 
  "Adipose_Visceral_Omentum": {
    "tissue_abbrv": "ADPVSC", 
    "tissue_color_hex": "#EE9A00", 
    "tissue_color_rgb": "238,154,0"
  }, 
  "Adrenal_Gland": {
    "tissue_abbrv": "ADRNLG", 
    "tissue_color_hex": "#8FBC8F", 
    "tissue_color_rgb": "143,188,143"
  }, 
  "Artery_Aorta": {
    "tissue_abbrv": "ARTAORT", 
    "tissue_color_hex": "#8B1C62", 
    "tissue_color_rgb": "139,28,98"
  }, 
  "Artery_Coronary": {
    "tissue_abbrv": "ARTCRN", 
    "tissue_color_hex": "#EE6A50", 
    "tissue_color_rgb": "238,106,80"
  }, 
  "Artery_Femoral": {
    "tissue_abbrv": "ARTFMR", 
    "tissue_color_hex": "#FF4500", 
    "tissue_color_rgb": "255,69,0"
  }, 
  "Artery_Tibial": {
    "tissue_abbrv": "ARTTBL", 
    "tissue_color_hex": "#FF0000", 
    "tissue_color_rgb": "255,0,0"
  }, 
  "Bladder": {
    "tissue_abbrv": "BLDDER", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Brain_Amygdala": {
    "tissue_abbrv": "BRNAMY", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Anterior_cingulate_cortex_BA24": {
    "tissue_abbrv": "BRNACC", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Caudate_basal_ganglia": {
    "tissue_abbrv": "BRNCDT", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Cerebellar_Hemisphere": {
    "tissue_abbrv": "BRNCHB", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Cerebellum": {
    "tissue_abbrv": "BRNCHA", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Cortex": {
    "tissue_abbrv": "BRNCTXA", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Frontal_Cortex_BA9": {
    "tissue_abbrv": "BRNCTXB", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Hippocampus": {
    "tissue_abbrv": "BRNHPP", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Hypothalamus": {
    "tissue_abbrv": "BRNHPT", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Nucleus_accumbens_basal_ganglia": {
    "tissue_abbrv": "BRNNCC", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Putamen_basal_ganglia": {
    "tissue_abbrv": "BRNPTM", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Spinal_cord_cervical_c_1": {
    "tissue_abbrv": "BRNSPC", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Brain_Substantia_nigra": {
    "tissue_abbrv": "BRNSNG", 
    "tissue_color_hex": "#EEEE00", 
    "tissue_color_rgb": "238,238,0"
  }, 
  "Breast_Mammary_Tissue": {
    "tissue_abbrv": "BREAST", 
    "tissue_color_hex": "#00CDCD", 
    "tissue_color_rgb": "0,205,205"
  }, 
  "Cells_EBV_transformed_lymphocytes": {
    "tissue_abbrv": "LCL", 
    "tissue_color_hex": "#EE82EE", 
    "tissue_color_rgb": "238,130,238"
  }, 
  "Cells_Transformed_fibroblasts": {
    "tissue_abbrv": "FIBRBLS", 
    "tissue_color_hex": "#9AC0CD", 
    "tissue_color_rgb": "154,192,205"
  }, 
  "Cervix_Ectocervix": {
    "tissue_abbrv": "CVXECT", 
    "tissue_color_hex": "#EED5D2", 
    "tissue_color_rgb": "238,213,210"
  }, 
  "Cervix_Endocervix": {
    "tissue_abbrv": "CVSEND", 
    "tissue_color_hex": "#EED5D2", 
    "tissue_color_rgb": "238,213,210"
  }, 
  "Colon_Sigmoid": {
    "tissue_abbrv": "CLNSGM", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Colon_Transverse": {
    "tissue_abbrv": "CLNTRN", 
    "tissue_color_hex": "#EEC591", 
    "tissue_color_rgb": "238,197,145"
  }, 
  "Esophagus_Gastroesophageal_Junction": {
    "tissue_abbrv": "ESPGEJ", 
    "tissue_color_hex": "#8B7355", 
    "tissue_color_rgb": "139,115,85"
  }, 
  "Esophagus_Mucosa": {
    "tissue_abbrv": "ESPMCS", 
    "tissue_color_hex": "#8B7355", 
    "tissue_color_rgb": "139,115,85"
  }, 
  "Esophagus_Muscularis": {
    "tissue_abbrv": "ESPMSL", 
    "tissue_color_hex": "#CDAA7D", 
    "tissue_color_rgb": "205,170,125"
  }, 
  "Fallopian_Tube": {
    "tissue_abbrv": "FLLPNT", 
    "tissue_color_hex": "#EED5D2", 
    "tissue_color_rgb": "238,213,210"
  }, 
  "Heart_Atrial_Appendage": {
    "tissue_abbrv": "HRTAA", 
    "tissue_color_hex": "#B452CD", 
    "tissue_color_rgb": "180,82,205"
  }, 
  "Heart_Left_Ventricle": {
    "tissue_abbrv": "HRTLV", 
    "tissue_color_hex": "#7A378B", 
    "tissue_color_rgb": "122,55,139"
  }, 
  "Kidney_Cortex": {
    "tissue_abbrv": "KDNCTX", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Kidney_Medulla": {
    "tissue_abbrv": "KDNMDL", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Liver": {
    "tissue_abbrv": "LIVER", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Lung": {
    "tissue_abbrv": "LUNG", 
    "tissue_color_hex": "#9ACD32", 
    "tissue_color_rgb": "154,205,50"
  }, 
  "Minor_Salivary_Gland": {
    "tissue_abbrv": "SLVRYG", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Muscle_Skeletal": {
    "tissue_abbrv": "MSCLSK", 
    "tissue_color_hex": "#7A67EE", 
    "tissue_color_rgb": "122,103,238"
  }, 
  "Nerve_Tibial": {
    "tissue_abbrv": "NERVET", 
    "tissue_color_hex": "#FFD700", 
    "tissue_color_rgb": "255,215,0"
  }, 
  "Ovary": {
    "tissue_abbrv": "OVARY", 
    "tissue_color_hex": "#FFB6C1", 
    "tissue_color_rgb": "255,182,193"
  }, 
  "Pancreas": {
    "tissue_abbrv": "PNCREAS", 
    "tissue_color_hex": "#CD9B1D", 
    "tissue_color_rgb": "205,155,29"
  }, 
  "Pituitary": {
    "tissue_abbrv": "PTTARY", 
    "tissue_color_hex": "#B4EEB4", 
    "tissue_color_rgb": "180,238,180"
  }, 
  "Prostate": {
    "tissue_abbrv": "PRSTTE", 
    "tissue_color_hex": "#D9D9D9", 
    "tissue_color_rgb": "217,217,217"
  }, 
  "Skin_Not_Sun_Exposed_Suprapubic": {
    "tissue_abbrv": "SKINNS", 
    "tissue_color_hex": "#3A5FCD", 
    "tissue_color_rgb": "58,95,205"
  }, 
  "Skin_Sun_Exposed_Lower_leg": {
    "tissue_abbrv": "SKINS", 
    "tissue_color_hex": "#1E90FF", 
    "tissue_color_rgb": "30,144,255"
  }, 
  "Small_Intestine_Terminal_Ileum": {
    "tissue_abbrv": "SNTTRM", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Spleen": {
    "tissue_abbrv": "SPLEEN", 
    "tissue_color_hex": "#CDB79E", 
    "tissue_color_rgb": "205,183,158"
  }, 
  "Stomach": {
    "tissue_abbrv": "STMACH", 
    "tissue_color_hex": "#FFD39B", 
    "tissue_color_rgb": "255,211,155"
  }, 
  "Testis": {
    "tissue_abbrv": "TESTIS", 
    "tissue_color_hex": "#A6A6A6", 
    "tissue_color_rgb": "166,166,166"
  }, 
  "Thyroid": {
    "tissue_abbrv": "THYROID", 
    "tissue_color_hex": "#008B45", 
    "tissue_color_rgb": "0,139,69"
  }, 
  "Uterus": {
    "tissue_abbrv": "UTERUS", 
    "tissue_color_hex": "#EED5D2", 
    "tissue_color_rgb": "238,213,210"
  }, 
  "Vagina": {
    "tissue_abbrv": "VAGINA", 
    "tissue_color_hex": "#EED5D2", 
    "tissue_color_rgb": "238,213,210"
  }, 
  "Whole_Blood": {
    "tissue_abbrv": "WHLBLD", 
    "tissue_color_hex": "#FF00FF", 
    "tissue_color_rgb": "255,0,255"
  }
}


# --- get_boxplot_input
# Copied verbatim from Retained_Introns_figure.ipynb cell 138 (In[292]).
# NOT the same as GeneticsPlotsFunctions.get_boxplot_input, which shares
# the name but takes (gtex_vcf, isoform) and does something else.
def get_boxplot_input(mat_list, label_list):

    ir_column = []
    label_column = []
    for i, mat in enumerate(mat_list):
        label = label_list[i]
    
        # norm_factor = mat[exon_bins].mean(axis=1)
        # idx = norm_factor.loc[norm_factor >= 0.01].index
        
        X = mat
        xcols = X.columns[25:-25]
        y = np.minimum(X[xcols], 1)
        ir_column += list(np.sqrt(y.mean(axis=1)))
        label_column += [label]*X.shape[0]
        
    out_df = pd.DataFrame()
    out_df['ir_percent'] = ir_column
    out_df['label'] = label_column

    return out_df

# --- get_metaplot_input / load_matrices_per_tissue
# Copied verbatim from Retained_Introns_figure.ipynb cell 48 (In[113]).


def load_matrices_per_tissue(tissue, annotation):
    col_names = ['chrom', 'start', 'end', 'gene', 'score', 'strand']
    pos_names_intron = [f'bin_{str(i+1)}' for i in range(110)]
    pos_names = [f'bin_{str(i+1)}' for i in range(30)]

    intron_file = f'../code/results/CoveragePlots/bed_files/IntronsDeNovo/matrices/{tissue}.{annotation}.retained_introns.mat.gz'
    fk1_file = f'../code/results/CoveragePlots/bed_files/IntronsDeNovo/matrices/{tissue}.{annotation}.flanking_exons_1.mat.gz'
    fk2_file = f'../code/results/CoveragePlots/bed_files/IntronsDeNovo/matrices/{tissue}.{annotation}.flanking_exons_2.mat.gz'

    # if 'torino' in annotation:
    #     fk1_file = f'../code/results/CoveragePlots/matrices/{tissue}.torino.flanking_exons_1.mat.gz'
    #     fk2_file = f'../code/results/CoveragePlots/matrices/{tissue}.torino.flanking_exons_2.mat.gz'
    #     intron_file = f'../code/results/CoveragePlots/matrices/{tissue}.{annotation}.retained_introns.mat.gz'
    # elif 'gencode' in annotation:
    #     fk1_file = f'../code/results/CoveragePlots/matrices/{tissue}.gencode.flanking_exons_1.mat.gz'
    #     fk2_file = f'../code/results/CoveragePlots/matrices/{tissue}.gencode.flanking_exons_2.mat.gz'
    #     intron_file = f'../code/results/CoveragePlots/matrices/{tissue}.{annotation}.retained_introns.mat.gz'
    # elif 'appris' in annotation:
    #     fk1_file = f'../code/results/CoveragePlots/matrices/{tissue}.appris_flanking_exons_1.mat.gz'
    #     fk2_file = f'../code/results/CoveragePlots/matrices/{tissue}.appris_flanking_exons_2.mat.gz'
    #     intron_file = f'../code/results/CoveragePlots/matrices/{tissue}.appris_introns.mat.gz'

    fk1 = pd.read_csv(fk1_file, sep='\t', skiprows=1, names=col_names + pos_names, index_col=3)

    fk2 = pd.read_csv(fk2_file, sep='\t', skiprows=1, names=col_names + pos_names, index_col=3)
    
    intron = pd.read_csv(intron_file, sep='\t', skiprows=1, names=col_names + pos_names_intron, index_col=3)

    return fk1, intron, fk2
    
def get_metaplot_input(tissue, annotation, remove_low = True):
    fk1, intron, fk2 = load_matrices_per_tissue(tissue, annotation)
    
    col_names = ['chrom', 'start', 'end', 'gene', 'score', 'strand']
    pos_names_intron = [f'bin_{str(i+1)}' for i in range(110)]
    pos_names = [f'bin_{str(i+1)}' for i in range(30)]
    
    
    pos_introns = intron.loc[intron.strand == '+'].index
    neg_introns = intron.loc[intron.strand == '-'].index
    
    norm_factor_pos = pd.concat([fk1.loc[pos_introns, pos_names[5:25]], fk2.loc[pos_introns, pos_names[5:25]]], axis=1).mean(axis=1)
    norm_factor_neg = pd.concat([fk1.loc[neg_introns, pos_names[5:25]], fk2.loc[neg_introns, pos_names[5:25]]], axis=1).mean(axis=1)
    
    fk1_meta = fk1.loc[norm_factor_pos.index, pos_names[:25]]
    intron_meta = intron.loc[norm_factor_pos.index, pos_names_intron[6:-5]]
    fk2_meta = fk2.loc[norm_factor_pos.index, pos_names[5:]]
    meta_X_pos = pd.concat([fk1_meta, intron_meta, fk2_meta], axis=1)
    meta_X_pos.columns = [f'bin_{str(i+1)}' for i in range(meta_X_pos.shape[1])]
    if remove_low:
        idx = norm_factor_pos.loc[norm_factor_pos >= 0.01].index
    else:
        idx = norm_factor_pos.index
    norm_X_pos = meta_X_pos.loc[idx].divide(norm_factor_pos.loc[idx], axis=0)
    
    
    fk1_meta = fk2.loc[norm_factor_neg.index, pos_names[:25]]
    intron_meta = intron.loc[norm_factor_neg.index, pos_names_intron[6:-5]]
    fk2_meta = fk1.loc[norm_factor_neg.index, pos_names[5:]]
    meta_X_neg = pd.concat([fk1_meta, intron_meta, fk2_meta], axis=1)
    meta_X_neg.columns = [f'bin_{str(i+1)}' for i in range(meta_X_pos.shape[1])]
    if remove_low:
        idx = norm_factor_neg.loc[norm_factor_neg >= 0.01].index
    else:
        idx = norm_factor_neg.index
    norm_X_neg = meta_X_neg.loc[idx].divide(norm_factor_neg.loc[idx], axis=0)
    
    norm_X = pd.concat([norm_X_pos, norm_X_neg], axis=0)

    return norm_X


# --- plot_intron_metaplot_comparison
# Copied verbatim from Retained_Introns_figure.ipynb cell 66 (In[84]).
def plot_intron_metaplot(norm_X, axes=None):
    if axes is None:
        fig, axes = plt.subplots(ncols = 5, figsize=(20, 3))


    avg_coverage = norm_X[norm_X.columns[25:-25]].mean(axis=1)
    Q5 = avg_coverage.loc[avg_coverage >= avg_coverage.quantile(0.8)].index
    Q4 = avg_coverage.loc[
        (avg_coverage <= avg_coverage.quantile(0.8)) & (avg_coverage >= avg_coverage.quantile(0.6))
    ].index
    Q3 = avg_coverage.loc[
        (avg_coverage <= avg_coverage.quantile(0.6)) & (avg_coverage >= avg_coverage.quantile(0.4))
    ].index
    Q2 = avg_coverage.loc[
        (avg_coverage <= avg_coverage.quantile(0.4)) & (avg_coverage >= avg_coverage.quantile(0.2))
    ].index
    Q1 = avg_coverage.loc[avg_coverage <= avg_coverage.quantile(0.2)].index

    axes[0].fill_between(np.arange(20), [0]*20, np.array(norm_X.loc[Q1].median(axis=0))[5:25], color='black', zorder = 2)
    axes[0].fill_between(np.arange(98)+19, [0]*98, np.array(norm_X.loc[Q1].median(axis=0))[25:-26], color='gray', zorder = 1)
    axes[0].fill_between(np.arange(21)+116,[0]*21, np.array(norm_X.loc[Q1].median(axis=0))[-26:-5], color='black', zorder = 2)
    axes[0].set_xlim([-5, 142])

    axes[1].fill_between(np.arange(20), [0]*20, np.array(norm_X.loc[Q2].median(axis=0))[5:25], color='black', zorder = 2)
    axes[1].fill_between(np.arange(98)+19, [0]*98, np.array(norm_X.loc[Q2].median(axis=0))[25:-26], color='gray', zorder = 1)
    axes[1].fill_between(np.arange(21)+116,[0]*21, np.array(norm_X.loc[Q2].median(axis=0))[-26:-5], color='black', zorder = 2)
    axes[1].set_xlim([-5, 142])
    
    axes[2].fill_between(np.arange(20), [0]*20, np.array(norm_X.loc[Q3].median(axis=0))[5:25], color='black', zorder = 2)
    axes[2].fill_between(np.arange(98)+19, [0]*98, np.array(norm_X.loc[Q3].median(axis=0))[25:-26], color='gray', zorder = 1)
    axes[2].fill_between(np.arange(21)+116,[0]*21, np.array(norm_X.loc[Q3].median(axis=0))[-26:-5], color='black', zorder = 2)
    axes[2].set_xlim([-5, 142])
    
    axes[3].fill_between(np.arange(20), [0]*20, np.array(norm_X.loc[Q4].median(axis=0))[5:25], color='black', zorder = 2)
    axes[3].fill_between(np.arange(98)+19, [0]*98, np.array(norm_X.loc[Q4].median(axis=0))[25:-26], color='gray', zorder = 1)
    axes[3].fill_between(np.arange(21)+116,[0]*21, np.array(norm_X.loc[Q4].median(axis=0))[-26:-5], color='black', zorder = 2)
    axes[3].set_xlim([-5, 142])
    
    axes[4].fill_between(np.arange(20), [0]*20, np.array(norm_X.loc[Q5].median(axis=0))[5:25], color='black', zorder = 2)
    axes[4].fill_between(np.arange(98)+19, [0]*98, np.array(norm_X.loc[Q5].median(axis=0))[25:-26], color='gray', zorder = 1)
    axes[4].fill_between(np.arange(21)+116,[0]*21, np.array(norm_X.loc[Q5].median(axis=0))[-26:-5], color='black', zorder = 2)
    axes[4].set_xlim([-5, 142])

def plot_intron_metaplot_comparison(annot1_X, annot2_X, annot3_X, bed1, bed2, bed3, bed4,
                                   alt_color = 'tab:orange', alt_name = 'Gencode'):


    shared_ = pd.Index(list(bed1.intersect(bed2, F=1, f=1).to_dataframe().name))
    annot1_ = pd.Index(list(bed1.intersect(bed2, v = True).to_dataframe().name))
    annot2_ = pd.Index(list(bed2.intersect(bed1, v = True).intersect(bed4, v = True).to_dataframe().name))
    # print(bed3.intersect(bed1, v = True).intersect(bed3, v = True).to_dataframe().head())
    # assert 1 == 2
    annot3_ = pd.Index(list(bed3.intersect(bed1, v = True).intersect(bed2, v = True).to_dataframe().name))

    print(annot3_X.loc[annot3_.intersection(annot3_X.index)].shape)

    fig, axes = plt.subplots(nrows = 5, ncols = 7, figsize=(10, 4), gridspec_kw={'wspace':0.1, 'height_ratios': [1, 1, 1, 1, 0.3]})

    plot_intron_metaplot(annot1_X.loc[shared_.intersection(annot1_X.index)], axes[0,:])
    plot_intron_metaplot(annot1_X.loc[annot1_.intersection(annot1_X.index)], axes[1,:])
    plot_intron_metaplot(annot2_X.loc[annot2_.intersection(annot2_X.index)], axes[2,:])
    plot_intron_metaplot(annot3_X.loc[annot3_.intersection(annot3_X.index)], axes[3,:])
    
    # plot_intron_metaplot(norm_X_snmf_gencode, axes[0,:])
    # plot_intron_metaplot(norm_X_snmf, axes[1,:])
    # plot_intron_metaplot(norm_X_gencode, axes[2,:])
    # plot_intron_metaplot(norm_X_appris, axes[3,:])
    
    for i in range(5):
        axes[4, i].plot([-3, 139], [0.5, 0.5], c='black', zorder=1, linewidth = 3)
        axes[4, i].fill_between(np.arange(20), [0]*20, [1]*20, color='black', zorder=2)
        axes[4, i].fill_between(np.arange(21)+116, [0]*21, [1]*21, color='black', zorder=2)
        axes[4, i].set_xlim([-5, 142])
        # axes[4, i].fill_between(np.arange(30, 90)+260, [0]*60, [1]*60, color='black', zorder=2)
        axes[4, i].set_xticks([])
        axes[4, i].set_yticks([])
        axes[4, i].spines['top'].set_visible(False)
        axes[4, i].spines['right'].set_visible(False)
        axes[4, i].spines['bottom'].set_visible(False)
        axes[4, i].spines['left'].set_visible(False)
        
        for j in range(4):
            if i > 0:
                axes[j, i].set_yticks([])
        
            axes[j, i].set_ylim([-0.1, 1.2])
            axes[j, i].set_xticks([])
            axes[j, i].spines['top'].set_visible(False)
            axes[j, i].spines['right'].set_visible(False)
    
    axes[0, 0].set_ylabel('Coverage')
    axes[1, 0].set_ylabel('Coverage')
    axes[2, 0].set_ylabel('Coverage')
    axes[3, 0].set_ylabel('Coverage')
    
    axes[0, 0].set_title('Q1')
    axes[0, 1].set_title('Q2')
    axes[0, 2].set_title('Q3')
    axes[0, 3].set_title('Q4')
    axes[0, 4].set_title('Q5')
    
    for j in range(4):
        axes[j, 5].set_ylim([-1, 4])
        axes[j, 6].set_ylim([-1, 4])
        
        axes[j, 5].plot([-3, 139], [2.5, 2.5], c='tab:blue', zorder=1, linewidth = 3)
        axes[j, 5].fill_between(np.arange(20), [2]*20, [3]*20, color='tab:blue', zorder=2)
        axes[j, 5].fill_between(np.arange(21)+116, [2]*21, [3]*21, color='tab:blue', zorder=2)
    
        axes[j, 6].plot([-3, 139], [2.5, 2.5], c=alt_color, zorder=1, linewidth = 3)
        axes[j, 6].fill_between(np.arange(20), [2]*20, [3]*20, color=alt_color, zorder=2)
        axes[j, 6].fill_between(np.arange(21)+116, [2]*21, [3]*21, color=alt_color, zorder=2)
        
        if j in [0, 1]:
            a = 1
        else:
            a = 0.1
        
    
        axes[j, 5].plot([-3, 139], [0.5, 0.5], c='tab:blue', zorder=1, alpha=a, linewidth = 3)
        axes[j, 5].fill_between(np.arange(137), [0]*137, [1]*137, color='tab:blue', zorder=2, alpha=a)
        
        if j in [0, 2]:
            a = 1
        else:
            a = 0.1
    
        axes[j, 6].plot([-3, 139], [0.5, 0.5], c=alt_color, zorder=1, alpha=a, linewidth = 3)
        axes[j, 6].fill_between(np.arange(137), [0]*137, [1]*137, color=alt_color, zorder=2, alpha=a)
        
    
        # axes[j, 6].plot([0, 380], [0.5, 0.5], c='tab:orange', linewidth=3)
        # axes[j, 6].fill_between(np.arange(30, 90), [0]*60, [1]*60, color='tab:orange', zorder=2)
        # axes[j, 6].fill_between(np.arange(30, 90)+130, [0]*60, [1]*60, color='tab:orange', zorder=2)
        # axes[j, 6].fill_between(np.arange(30, 90)+260, [0]*60, [1]*60, color='tab:orange', zorder=2)
    
        # axes[j, 6].plot([0, 380], [2.5, 2.5], c='tab:orange', linewidth=3)
        # axes[j, 6].fill_between(np.arange(30, 90), [2]*60, [3]*60, color='tab:orange', zorder=2)
        # # axes[j, 6].fill_between(np.arange(30, 90)+130, [2]*60, [3]*60, color='tab:orange', zorder=2)
        # axes[j, 6].fill_between(np.arange(30, 90)+260, [2]*60, [3]*60, color='tab:orange', zorder=2)
    
    for j in range(5):
        for i in range(5, 7):
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
            axes[j, i].spines['top'].set_visible(False)
            axes[j, i].spines['right'].set_visible(False)
            axes[j, i].spines['bottom'].set_visible(False)
            axes[j, i].spines['left'].set_visible(False)
    
    axes[0, 5].set_title('Torino', c='tab:blue')
    axes[0, 6].set_title(alt_name, c=alt_color)

    


    x = annot1_X.loc[shared_.intersection(annot1_X.index)].shape[0]#, axes[0,:])
    y = annot1_X.loc[annot1_.intersection(annot1_X.index)].shape[0]#, axes[1,:])
    z = annot2_X.loc[annot2_.intersection(annot2_X.index)].shape[0]#, axes[2,:])
    w = annot3_X.loc[annot3_.intersection(annot3_X.index)].shape[0]#, axes[3,:]
    

    axes[0, 5].text(30, -1.2, f'({x})', c='tab:blue')
    axes[1, 5].text(30, -1.2, f'({y})', c='tab:blue')
    axes[2, 6].text(30, -1.2, f'({z})', c=alt_color)
    axes[3, 5].text(30, -1.2, f'({w})', c='tab:blue')
        
    
    # plt.show()
# plt.savefig('../code/results/figures/Fig2_isoforms/intron_metaplot_quintiles.pdf', bbox_inches='tight')
# plt.savefig('../code/results/figures/Fig2_isoforms/intron_metaplot_quintiles.png', bbox_inches='tight', dpi=500)


# --- plot_intron_single_metaplot_comparison
# Copied verbatim from Retained_Introns_figure.ipynb cell 78 (In[53]).

def plot_intron_single_metaplot_comparison(annot1_X, annot2_X, annot3_X, bed1, bed2, bed3, bed4):


    shared_ = pd.Index(list(bed1.intersect(bed2, F=1, f=1).to_dataframe().name))
    annot1_ = pd.Index(list(bed1.intersect(bed2, v = True).to_dataframe().name))
    annot2_ = pd.Index(list(bed2.intersect(bed1, v = True).intersect(bed4, v = True).to_dataframe().name))
    # print(bed3.intersect(bed1, v = True).intersect(bed3, v = True).to_dataframe().head())
    # assert 1 == 2
    annot3_ = pd.Index(list(bed3.intersect(bed1, v = True).intersect(bed2, v = True).to_dataframe().name))

    # print(annot3_X.loc[annot3_.intersection(annot3_X.index)].shape)

    # fig, axes = plt.subplots(nrows = 5, ncols = 7, figsize=(10, 4), gridspec_kw={'wspace':0.1, 'height_ratios': [1, 1, 1, 1, 0.3]})

    # plot_intron_metaplot(annot1_X.loc[shared_.intersection(annot1_X.index)], axes[0,:])
    # plot_intron_metaplot(annot1_X.loc[annot1_.intersection(annot1_X.index)], axes[1,:])
    # plot_intron_metaplot(annot2_X.loc[annot2_.intersection(annot2_X.index)], axes[2,:])
    # plot_intron_metaplot(annot3_X.loc[annot3_.intersection(annot3_X.index)], axes[3,:])

    fig, ax = plt.subplots(nrows = 2, figsize=(3, 3), gridspec_kw={'hspace':0.1, 'height_ratios': [1, 0.1]})
    # n per category: the introns actually behind each median, i.e. after the
    # intersection with the matrix index, not the size of the bed set.
    _n = {
        'shared': len(shared_.intersection(annot1_X.index)),
        'annot1': len(annot1_.intersection(annot1_X.index)),
        'annot2': len(annot2_.intersection(annot2_X.index)),
        'annot3': len(annot3_.intersection(annot3_X.index)),
    }
    ax[0].plot(np.sqrt(np.array(annot1_X.loc[shared_.intersection(annot1_X.index)].median(axis=0))[5:-5]), zorder=5, linewidth=3, label=f'Torino & Gencode ({_n["shared"]:,})', c='tab:purple')
    # ax[0].plot(np.sqrt(np.array(norm_X_snmf_overlap.median(axis=0))[5:-5]), zorder=4, linewidth=3, label='Torino/Gencode overlap', c='tab:cyan')
    
    ax[0].plot(np.sqrt(np.array(annot1_X.loc[annot1_.intersection(annot1_X.index)].median(axis=0))[5:-5]), zorder=3, linewidth=3, label=f'Torino only ({_n["annot1"]:,})', c='tab:blue')
    ax[0].plot(np.sqrt(np.array(annot2_X.loc[annot2_.intersection(annot2_X.index)].median(axis=0))[5:-5]), zorder=2, linewidth=3, label=f'Gencode only ({_n["annot2"]:,})', c='tab:red')
    ax[0].plot(np.sqrt(np.array(annot3_X.loc[annot3_.intersection(annot3_X.index)].median(axis=0))[5:-5]), zorder=1, c='tab:gray', linewidth=3, label=f'Random intron ({_n["annot3"]:,})')
    
    ax[0].set_xticks([])
    
    
    ax[1].plot([-3, 139], [0.5, 0.5], c='black', zorder=1, linewidth = 3)
    ax[1].fill_between(np.arange(20), [0]*20, [1]*20, color='black', zorder=2)
    ax[1].fill_between(np.arange(21)+116, [0]*21, [1]*21, color='black', zorder=2)
    ax[1].set_xlim([-5, 142])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    
    ax[0].set_yticks([np.sqrt(x) for x in [0, 0.01, 0.1, 0.5, 1]], ['0', '1e-2', '1e-1', '0.5', '1'])
    ax[0].set_ylabel('Relative coverage')
    
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    
    ax[0].legend(frameon=False, bbox_to_anchor=(1, 0., 0.5, 0.5),
                 title='Retained introns\nin liver-expressed genes',
                 title_fontsize=7, fontsize=7)
    
    # plt.show()
# plt.savefig('../code/results/figures/Fig2_isoforms/intron_metaplot.Liver.pdf', bbox_inches='tight')
# plt.savefig('../code/results/figures/Fig2_isoforms/intron_metaplot.Liver.png', bbox_inches='tight', dpi=500)
# plt.savefig('../code/scratch_figures/figure2/intron_metaplot.pdf', bbox_inches='tight')
# plt.savefig('../code/scratch_figures/figure2/intron_metaplot.png', bbox_inches='tight', dpi=500)


# --- plot_pca_scatter_EL
# Copied verbatim from Retained_Introns_figure.ipynb cell 172 (In[69]).
def plot_pca_scatter_EL(
    human_EL: pd.DataFrame,
    human_metadata: pd.DataFrame,
    gtex_colors2: dict,
    subindex: list = None,
    n_components: int = 50,
    figsize=(7, 4),
    title: str = "PCA – samples coloured by tissue",
    save_path: str = None,
    random_state: int = 42,
    max_nan_frac: float = 0.1
) -> PCA:
    import matplotlib.patches as mpatches
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    gtex_samples = human_EL.columns[6:]

    # Subset rows if requested
    if subindex is not None:
        data = human_EL.loc[subindex, gtex_samples]
    else:
        data = human_EL[gtex_samples]

    # Align samples with metadata
    common_samples = gtex_samples.intersection(human_metadata.index)
    data = data[common_samples]
    tissue_labels = human_metadata.loc[common_samples, 'Tissue']
    colors = [gtex_colors2[t]['tissue_color_hex'] for t in tissue_labels]
    _n_samples = len(common_samples)      # samples actually plotted

    # Drop rows with too many NaNs, impute the rest
    nan_frac = data.isna().mean(axis=1)
    n_before = len(data)
    data = data.loc[nan_frac <= max_nan_frac]
    n_after = len(data)
    if n_before != n_after:
        print(f"[NaN filter]  dropped {n_before - n_after} rows "
              f"(>{max_nan_frac*100:.0f}% NaN); {n_after} rows retained")

    X = data.values.T.astype(float)  # (n_samples, n_features)
    X = SimpleImputer(strategy='mean').fit_transform(X)

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"[PCA]  PC1+PC2 variance explained: {(var[0]+var[1])*100:.1f}%")

    fig, ax = plt.subplots(figsize=figsize)

    for tissue in sorted(tissue_labels.unique()):
        mask = (tissue_labels == tissue).values
        color = gtex_colors2[tissue]['tissue_color_hex']
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], s=18, alpha=0.75, linewidths=0,
                   rasterized=True, zorder=2)

    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    # n = samples plotted. Printed from the data, so if the sample set is not
    # the full 2,016 the figure says what it actually is.
    ax.text(0.99, 0.01, f'n = {_n_samples:,} samples', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(color=gtex_colors2[t]['tissue_color_hex'], label=t)
        for t in sorted(tissue_labels.unique())
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=False)

    plt.tight_layout()

    # if save_path is None:
    #     plt.show()
    # else:
    #     plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    #     plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=500)
    #     print(f"[plot]  PCA scatter saved -> {save_path}")
    #     plt.close()

    return pca


# --------------------------------------------------------------------------
# Panels. Each loads what it needs and draws; the notebook's savefig calls are
# left out, so the figure is returned to the caller rather than written.
# --------------------------------------------------------------------------
@_panel
def panel_intron_metaplot_quintiles():
    """Notebook cell 68 (In[65]). Matrices from In[49], so remove_low=False."""
    X = load_metaplot_X(remove_low=False)
    beds = load_intron_beds()
    appris_introns = load_intron_table('appris')
    plot_intron_metaplot_comparison(
        X['torino'], X['gencode'], X['appris'], beds['torino'], beds['gencode'],
        BedTool.from_dataframe(bedtools_safe(appris_introns)).intersect(
            beds['torino_all'], v=True),
        beds['torino_all'])


@_panel
def panel_intron_metaplot_quintiles_vastdb():
    """Notebook cell 70 (In[87]). Ran after In[80], so remove_low=True."""
    X = load_metaplot_X(remove_low=True)
    beds = load_intron_beds()
    appris = BedTool.from_dataframe(bedtools_safe(load_intron_table('appris')))
    vastdb = BedTool.from_dataframe(bedtools_safe(load_intron_table('vastdb')))
    plot_intron_metaplot_comparison(
        X['appris'], X['appris'], X['appris'],
        appris.intersect(beds['torino_all'], F=1, f=1),
        appris.intersect(vastdb, F=1, f=1),
        appris.intersect(vastdb, v=True),
        beds['torino_all'], alt_color='tab:green', alt_name='VastDB')


@_panel
def panel_intron_metaplot_Liver():
    """Notebook cell 79 (In[54]). Matrices from In[49], so remove_low=False."""
    X = load_metaplot_X(remove_low=False)
    beds = load_intron_beds()
    appris_introns = load_intron_table('appris')
    plot_intron_single_metaplot_comparison(
        X['torino'], X['gencode'], X['appris'], beds['torino'], beds['gencode'],
        BedTool.from_dataframe(bedtools_safe(appris_introns)).intersect(
            beds['torino_all'], v=True),
        beds['torino_all'])


def intron_retention_table():
    """The retained-intron numbers, both gene sets side by side, as markdown.

    Each row names the cell of Retained_Introns_figure.ipynb it comes from. The
    corrected column is the same computation over the 16,073-gene subset, so
    every count should be at most its all-genes counterpart.
    """
    global GENE_SUBSET
    keep = GENE_SUBSET
    try:
        GENE_SUBSET = None
        free()
        a = intron_retention_numbers()
        GENE_SUBSET = load_select_genes()
        free()
        c = intron_retention_numbers()
    finally:
        GENE_SUBSET = keep
        free()

    def row(label, key, denom=None, src=''):
        cells = []
        for n in (a, c):
            v = n[key]
            cells.append(f'{v:,}' + (f' ({100 * v / n[denom]:.1f}%)' if denom else ''))
        return f'| {label} | {cells[0]} | {cells[1]} | `{src}` |'

    rows = [
        row('Retained introns observed by Torino', 'introns', src='In[82]'),
        row('&nbsp;&nbsp;across transcripts', 'transcripts', src='In[90]'),
        row('&nbsp;&nbsp;across genes', 'genes', 'tested_genes', 'In[84]'),
        row('&nbsp;&nbsp;genes tested', 'tested_genes', src='annot'),
        row('Classified as true IR', 'true_ir', 'introns', 'In[95]'),
        row('&nbsp;&nbsp;across genes', 'true_ir_genes', src='In[86]'),
        row('&nbsp;&nbsp;not annotated by GENCODE', 'torino_not_gencode', 'true_ir', 'In[53]'),
        row('Splicing-efficiency factors', 'efficiency_introns', 'introns', 'derived'),
        row('&nbsp;&nbsp;matched by GENCODE', 'efficiency_gencode',
            'efficiency_introns', 'derived'),
        row('&nbsp;&nbsp;not matched by GENCODE', 'efficiency_not_gencode',
            'efficiency_introns', 'derived'),
        row('GENCODE retained introns', 'gencode_ri', src='In[99]'),
        row('&nbsp;&nbsp;observed by Torino', 'gencode_observed', 'gencode_ri', 'In[48]'),
        row('&nbsp;&nbsp;&nbsp;&nbsp;matching a true IR event', 'gencode_true_ir',
            'gencode_ri', 'In[49]'),
        row('&nbsp;&nbsp;&nbsp;&nbsp;a splicing-efficiency factor', 'gencode_efficiency',
            'gencode_ri', 'derived'),
        row('&nbsp;&nbsp;missing from Torino', 'gencode_missing', 'gencode_ri', 'In[52]'),
        row('&nbsp;&nbsp;partial overlap, not an exact match', 'gencode_partial',
            'gencode_ri', 'derived'),
    ]
    larger = [k for k in a if c[k] > a[k]]
    out = ['**Retained introns**', '',
           '| | all genes | corrected gene set | source |',
           '|---|---:|---:|---|'] + rows
    if larger:
        out += ['', f'> **check**: corrected is larger than all genes for {larger} '
                    '-- that should be impossible, it is a subset.']
    return '\n'.join(out)


PUBLISHED_INTRON_COUNTS = [174635, 158506, 115176, 25305, 9131]


def intron_counts():
    """The five bar heights, honouring GENE_SUBSET.

    Four are simply table lengths. The VastDB bar is not: the notebook computed
    it at In[102] as the VastDB introns lying in genes Torino annotates --
    vastdb.gene is an Ensembl id, matched against the ids of protein-coding
    genes whose symbol appears in the corrected annotation. With GENE_SUBSET
    unset this returns the published [174635, 158506, 115176, 25305, 9131]
    exactly; with it set, every bar is recounted over the subset.
    """
    beds = load_intron_beds()
    root = '/project2/yangili1/cfbuenabadn/torino_paper/code'
    human_bed = pd.read_csv(f'{root}/config/human.protein_coding.bed.gz',
                            sep='\t', index_col=0)
    annot = pd.read_csv(f'{root}/results/torino/Human/isoforms/'
                        'corrected_annotated.isoforms.tab.gz', sep='\t')
    annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
    torino_genes = pd.Index(annot.gene.unique())
    if GENE_SUBSET is not None:
        torino_genes = torino_genes.intersection(pd.Index(GENE_SUBSET))
    ids = human_bed.loc[human_bed.gene_name.isin(torino_genes)].gene_id.unique()
    vastdb = load_intron_table('vastdb')
    return [len(load_intron_table('appris')),
            int(vastdb.gene.isin(ids).sum()),
            len(beds['torino_all']), len(beds['torino']), len(beds['gencode'])]


@_panel
def panel_intron_barplot(counts=None):
    """Notebook cell 137 (In[291]).

    The notebook hardcodes the five counts and writes them into the bar labels.
    Here they are recomputed by intron_counts(), which reproduces the published
    numbers exactly when GENE_SUBSET is unset and recounts them when it is set,
    so the panel stays honest under the corrected gene set.
    """
    counts = counts or intron_counts()
    fig, ax = plt.subplots(figsize=(1.5, 3))
    ax.bar(['Appris introns', 'VastDB RI', 'Torino (all)', 'Torino RI', 'Gencode RI'],
           counts,
           color=['tab:gray', 'tab:orange', 'navy', 'tab:blue', 'tab:red'])
    ax.set_xticks([])
    for x, y, text, colour in [
            (-0.2, 4000,  f'Appris intron ({counts[0]:,})',  'white'),
            (0.8,  4000,  f'VastDB ({counts[1]:,})',         'white'),
            (1.8,  4000,  f'Torino - all ({counts[2]:,})',   'white'),
            (2.8,  30000, f'Torino RI ({counts[3]:,})',      'tab:blue'),
            (3.8,  15000, f'Gencode RI ({counts[4]:,})',     'tab:red')]:
        ax.text(x, y, text, rotation=90, c=colour)
    ax.set_yticks([0, 50000, 100000, 150000])
    ax.set_yticklabels(['0', '5e5', '1e6', '1.5e6'])
    ax.set_ylabel('Total introns')
    ax.spines[['right', 'top']].set_visible(False)
    return fig


def load_intron_coverage_groups():
    """The four intron groups behind the CDF -- notebook cell 142 (In[298]).

    Introns are split by which annotation calls them: shared by Torino and
    Gencode, Torino only, Gencode only, and Appris introns that neither calls
    (the "random intron" background).

    remove_low=False. The execution trail is ambiguous here -- In[298] comes
    after the remove_low=True rebuild at In[80], but gencode_X had since been
    overwritten by the saved gencode_introns_X.tab.gz at In[84]. Rendering both
    ways against the published intron_cdf.Liver.png settles it: remove_low=False
    reproduces it exactly, remove_low=True does not.
    """
    X = load_metaplot_X(remove_low=False)
    beds = load_intron_beds()
    appris_introns = load_intron_table('appris')
    bed1, bed2, bed4 = beds['torino'], beds['gencode'], beds['torino_all']
    bed3 = BedTool.from_dataframe(bedtools_safe(appris_introns)).intersect(bed4, v=True)

    shared_ = pd.Index(list(bed1.intersect(bed2, F=1, f=1).to_dataframe().name))
    annot1_ = pd.Index(list(bed1.intersect(bed2, v=True).to_dataframe().name))
    annot2_ = pd.Index(list(bed2.intersect(bed1, v=True)
                              .intersect(bed4, v=True).to_dataframe().name))
    annot3_ = pd.Index(list(bed3.intersect(bed1, v=True)
                              .intersect(bed2, v=True).to_dataframe().name))

    a1, a2, a3 = X['torino'], X['gencode'], X['appris']
    return [a1.loc[shared_.intersection(a1.index)],
            a1.loc[annot1_.intersection(a1.index)],
            a2.loc[annot2_.intersection(a2.index)],
            a3.loc[annot3_.intersection(a3.index)]]


@_panel
def panel_intron_cdf_Liver():
    """Notebook cell 144 (In[301]), with `df` from cell 143 (In[299])."""
    df = get_boxplot_input(
        load_intron_coverage_groups(),
        ['snmf_and_gencode', 'snmf_only', 'gencode_only', 'no_intron_retention'])

    fig, ax = plt.subplots(figsize=(3, 3))
    colores = ['tab:purple', 'tab:blue', 'tab:red', 'tab:gray'][::-1]
    labels_write = ['Torino & Gencode', 'Torino', 'Gencode', 'Random intron'][::-1]
    for i, label in enumerate(['snmf_and_gencode', 'snmf_only',
                               'gencode_only', 'no_intron_retention'][::-1]):
        df_ = df.loc[df.label == label]
        ax.plot(np.array(df_.ir_percent.sort_values()), np.linspace(0, 1, len(df_)),
                c=colores[i], label=f'{labels_write[i]} ({len(df_):,})',
                linewidth=3)
    ax.set_xticks([0, np.sqrt(0.1), np.sqrt(0.5), np.sqrt(1)],
                  ['0', '0.1', '0.5', '1'], rotation=0)
    ax.set_ylabel('cumulative\ndistribution')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('intron coverage')
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 0.5),
              title='Retained introns\nin liver-expressed genes',
              title_fontsize=7, fontsize=7)
    return fig


@_panel
def panel_PCA_all():
    """Notebook cell 174 (In[70]). PCA over every row of the EL table.

    The heaviest panel here: the EL table is 1.5 GB as float32 and the PCA
    copies and imputes it, so this needs roughly 6-8 GB. It will not run on a
    login node -- use sinteractive.
    """
    return plot_pca_scatter_EL(load_human_EL(EL_FLOAT32),
                               load_human_metadata(), gtex_colors2)


@_panel
def panel_PCA_introns():
    """Notebook cell 175 (In[71]). PCA over the retained-intron transcripts."""
    human_EL = load_human_EL(EL_FLOAT32)
    idx = pd.Index(load_IR_transcripts().transcript.tolist()).intersection(human_EL.index)
    return plot_pca_scatter_EL(human_EL.loc[idx], load_human_metadata(), gtex_colors2)


PANELS = {
    'intron_metaplot_quintiles': panel_intron_metaplot_quintiles,
    'intron_metaplot_quintiles_vastdb': panel_intron_metaplot_quintiles_vastdb,
    'intron_metaplot_Liver': panel_intron_metaplot_Liver,
    'intron_barplot': panel_intron_barplot,
    'intron_cdf_Liver': panel_intron_cdf_Liver,
    'PCA_all': panel_PCA_all,
    'PCA_introns': panel_PCA_introns,
}
