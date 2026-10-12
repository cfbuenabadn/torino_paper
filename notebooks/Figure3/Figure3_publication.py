"""
Figure3_publication.py — assembled, publication-ready Figure 3.

Reproduces notebooks/Figure3/Figure3.png in code, keeping the reference layout
but putting every panel on an explicit millimetre plan so rows, columns and
baselines line up.

Layout method
-------------
The page is planned in millimetres from a top-left origin.  `rect(x, y, w, h)`
converts a mm box to the figure-fraction rectangle matplotlib wants, so every
later "make b 10% shorter" request is an edit to one constant.  PAGE_H is
*derived* from the vertical plan rather than typed, so it reports the truth if
the content stops fitting.

Run from the analysis/ directory.
"""

# %% CELL: imports
import os
import pickle
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

FIG_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure3'
if FIG_DIR not in sys.path:
    sys.path.append(FIG_DIR)

mpl.rcParams.update({
    'pdf.fonttype':      42,      # embed TrueType, not Type 3
    'ps.fonttype':       42,
    # 'none' keeps text as real <text> elements with ONE x for the whole string,
    # so Inkscape can edit them.  matplotlib's default ('path') outlines every
    # glyph, and its PDF writer emits one Tj per character for mathtext — which
    # is what makes edited labels pile up in the same spot.
    'svg.fonttype':      'none',
    'font.family':       'sans-serif',
    'font.sans-serif':   ['DejaVu Sans'],
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  1.6,
    'ytick.major.size':  1.6,
    'savefig.dpi':       400,
})

# %% CELL: page plan
# ----------------------------------------------------------------------------
# All geometry in millimetres, top-left origin.  Nature Genetics double column
# is 180 mm; PAGE_W is the reference dimension every column is derived from.
# ----------------------------------------------------------------------------
PAGE_W   = 180.0
MARGIN_L = 2.0
MARGIN_R = 2.0
MARGIN_T = 1.0
MARGIN_B = 1.5

BAND_GAP = 3.0                      # vertical space between panel bands

# Six stacked bands.  A|B share the first; every gene panel then gets the full
# page width, which is what buys the gene plots their size.
BAND_AB_H = 34.0                    # A + B  (fixed: chart height is a choice)
BAND_C_H  = 34.0                    # C      (fixed: 6 rows x cell height)

# One band per gene panel (D = PHC1, E = SYNGR1, F = VRK3), each sized by its
# contents in plan_page(); these are placeholders until it runs.
BAND_D_H = 61.0
BAND_E_H = 61.0
BAND_F_H = 34.0

BAND_AB_Y = MARGIN_T
BAND_C_Y  = BAND_AB_Y + BAND_AB_H + BAND_GAP
BAND_D_Y  = BAND_C_Y + BAND_C_H + BAND_GAP
BAND_E_Y  = BAND_D_Y + BAND_D_H + BAND_GAP
BAND_F_Y  = BAND_E_Y + BAND_E_H + BAND_GAP

PAGE_H = BAND_F_Y + BAND_F_H + MARGIN_B        # derived, never typed

CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

# Column splits within bands (mm, absolute x)
A_X0, A_X1 = MARGIN_L, 60.0
B_X0, B_X1 = 62.0, PAGE_W - MARGIN_R
# D..G each span the full content width
GENE_X0, GENE_X1 = MARGIN_L, PAGE_W - MARGIN_R

# Typography — one scale for the whole page (Nature Genetics floor is 5 pt)
FS_PANEL  = 9.0     # bold panel letters
FS_TITLE  = 7.0     # panel titles
FS_SUB    = 6.0     # sub-panel / species titles
FS_TICK   = 5.0     # tick labels
FS_LABEL  = 5.0     # axis labels
FS_ANNOT  = 5.0     # in-plot annotation
FS_SMALL  = 4.4     # the densest tick columns (panel E tissue names)

LW_THIN   = 0.5
LW_TRACK  = 0.5


def rect(x, y, w, h):
    """mm box (top-left origin) -> [left, bottom, width, height] figure fractions."""
    return [x / PAGE_W, 1.0 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H]


def add_ax(fig, x, y, w, h, **kw):
    return fig.add_axes(rect(x, y, w, h), **kw)


def panel_letter(fig, letter, x, y):
    fig.text(x / PAGE_W, 1.0 - y / PAGE_H, letter,
             fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')


def fig_text(fig, x, y, s, **kw):
    return fig.text(x / PAGE_W, 1.0 - y / PAGE_H, s, **kw)


# Titles that mix roman and italic are drawn as TWO text objects rather than
# mathtext.  Mathtext emits one Tj per character in the PDF and outlines part of
# the run in the SVG, which is exactly what makes a label impossible to edit in
# Inkscape — new characters inherit the per-glyph positions and pile up.
_COMPOSITE_TITLES = []
COMPOSITE_GAP = 0.9        # mm between the roman and italic runs


def two_part_title(fig, x_center, y, roman, italic, **kw):
    """Roman text followed by an italic run, centred on x_center as a whole."""
    t1 = fig.text(x_center / PAGE_W, 1.0 - y / PAGE_H, roman,
                  ha='left', va='top', **kw)
    t2 = fig.text(x_center / PAGE_W, 1.0 - y / PAGE_H, italic,
                  ha='left', va='top', fontstyle='italic', **kw)
    _COMPOSITE_TITLES.append((t1, t2, x_center))
    return t1, t2


def _place_composite_titles(fig):
    """Centre each two-part title once the renderer can measure the runs."""
    if not _COMPOSITE_TITLES:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for t1, t2, xc in _COMPOSITE_TITLES:
        w1 = t1.get_window_extent(r).transformed(inv).width * PAGE_W
        w2 = t2.get_window_extent(r).transformed(inv).width * PAGE_W
        x0 = xc - (w1 + w2 + COMPOSITE_GAP) / 2
        t1.set_x(x0 / PAGE_W)
        t2.set_x((x0 + w1 + COMPOSITE_GAP) / PAGE_W)


# %% CELL: palettes
# Tissue colours (shared by every panel that codes tissue by colour)
TISSUE_COLORS = {
    'Brain':                 '#EEEE00',
    'Cerebellum':            '#EEEE00',
    'Heart':                 '#B452CD',
    'Liver':                 '#CDB79E',
    'Testis':                '#A6A6A6',
    'Muscle':                '#7A67EE',
    'Muscle_Skeletal':       '#7A67EE',
    'Breast_Mammary_Tissue': '#00CDCD',
    'Mammary_Gland':         '#00CDCD',
    'Whole_Blood':           '#FF00FF',
    'Blood':                 '#FF00FF',
    'Lung':                  '#9ACD32',
    'Skin':                  '#3A5FCD',
    'Adipose_Subcutaneous':  '#FFA54F',
    'Subcutaneous_Adipose':  '#FFA54F',
    'Kidney':                '#FFB6C1',
    'Ovary':                 '#8B1C62',
    'Colon':                 '#CD9B1D',
}
DEFAULT_TISSUE_COLOR = '#CCCCCC'

# Panel A uses a slightly different heart/blood mapping (as in the source notebook)
BARPLOT_COLORS = dict(TISSUE_COLORS, Heart='#7A378B', Kidney='#CDB79E',
                      Colon='#CDB79E', Ovary='#FFB6C1', Pancreas='#CD9B1D',
                      Artery_Aorta='#8B1C62')

# Panel B stack.  Cerebellum is orange here so it separates from Brain — the
# exploratory notebook drew both '#EEEE00' and they were indistinguishable.
STACK_COLORS = {
    'Brain':                  '#EEEE00',
    'Brain & Cerebellum':     '#CCCC00',
    'Cerebellum':             '#FFCC00',
    'Heart':                  '#B452CD',
    'Liver':                  '#CDB79E',
    'Testis':                 '#A6A6A6',
    'Multiple tissues':       '#D62728',
}
STACK_ROWS = ['Brain only', 'Brain & Cerebellum only', 'Cerebellum only',
              'Heart only', 'Liver only', 'Testis only', 'Multiple tissues']
STACK_LABELS = ['Brain', 'Brain & Cerebellum', 'Cerebellum', 'Heart',
                'Liver', 'Testis', 'Multiple tissues']

# Coverage-track colours for the gene panels
TRACK_COLORS = {
    'brain': '#EEEE00', 'cerebellum': '#EEEE00', 'heart': '#B452CD',
    'liver': '#CDB79E', 'testis': '#A6A6A6', 'mammary': '#00CDCD',
}
TRACK_LABELS = {
    'brain': 'Brain', 'cerebellum': 'Cerebellum', 'heart': 'Heart',
    'liver': 'Liver', 'testis': 'Testis', 'mammary': 'Mammary tissue',
}

# Exon boxes are drawn 15% shorter than the row and centred on it, so adjacent
# isoform rows read as separate; the intron line is correspondingly heavier.
EXON_H    = 0.85         # exon height as a fraction of the isoform row
INTRON_LW = 1.2          # intron line width (was 0.6)

ISO_RED  = '#DC143C'     # crimson — tissue-specific isoform
ISO_NAVY = '#000080'     # navy    — non-specific isoform
ISO_GREY = '#808080'

# Some cardoso species carry a literal 'Brain,Cerebellum' tissue.  The coverage
# tracks already fold it into cerebellum ("'Cereb' in label"), and the per-species
# colour lists in the source have no slot for it — leaving it separate shifts every
# colour after it and drops the last tissue.  Group it the same way here.
TISSUE_MERGE = {'Brain,Cerebellum': 'Cerebellum'}

# Highlight bands over the loading panels, keyed by tissue group.
HL_ALPHA = 0.28
HL_GROUPS = {
    'brain':   ('#EEEE00', lambda t: (t.startswith('Brain') or
                                      t in ('Cerebellum', 'CNS', 'Hypothalamus'))),
    'heart':   ('#B452CD', lambda t: t.startswith('Heart')),
    'mammary': ('#00CDCD', lambda t: 'Mammary' in t or 'Breast' in t),
}
HL_DEFAULT = ('brain', 'heart')
HL_MAMMARY = ('brain', 'heart', 'mammary')      # SYNGR1, which is a mammary story


def tissue_groups(panel):
    """Merged, sorted tissue groups for one loadings panel, with the per-species
    colour list from the source aligned to them positionally."""
    EL = panel['el_annot']
    idx = EL.index.intersection(pd.Index(panel['sublist'])).intersection(panel['meta'].index)
    ids = panel['meta'].loc[idx, 'tissue_id'].replace(TISSUE_MERGE)
    tissues = sorted(ids.unique())
    colors = list(panel['tissue_colors'])
    if len(colors) < len(tissues):              # never expected — fail loudly
        raise ValueError(f"{panel['species']}: {len(tissues)} tissues but "
                         f'{len(colors)} colours in the source list: {tissues}')
    return tissues, colors[:len(tissues)], ids


def _darken(c, factor=0.72):
    import colorsys
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0, l * factor), s)


# %% CELL: data
DATA_DIR = os.path.join(FIG_DIR, 'figure_data')


@lru_cache(maxsize=2)
def load_figure_data(data_dir=DATA_DIR):
    def _load(name):
        with open(os.path.join(data_dir, f'{name}.pkl'), 'rb') as f:
            return pickle.load(f)
    return {
        'ssmd_barplot_df': _load('ssmd_barplot_df'),
        'df_table':        _load('df_table'),
        'df_table_mouse':  _load('df_table_mouse'),
        'heatmaps':        _load('heatmaps'),
        'panels':          _load('figure_panels'),
    }


# Display names: the pickles carry raw table keys, the figure wants readable text
TISSUE_DISPLAY = {
    'Adipose_Subcutaneous':  'Adipose Subcutaneous',
    'Artery_Aorta':          'Artery Aorta',
    'Breast_Mammary_Tissue': 'Breast Mammary Tissue',
    'Muscle_Skeletal':       'Muscle Skeletal',
    'Whole_Blood':           'Whole Blood',
    'Mammary_Gland':         'Mammary gland',
    'Subcutaneous_Adipose':  'Adipose',
}
SPECIES_DISPLAY = {'Norway rat - BN/NHsdMcwi': 'Rat'}

# Panel A's bars are ~3.5 mm apart, so its tick labels are shortened past what
# TISSUE_DISPLAY uses elsewhere.
BARPLOT_LABELS = {
    'Muscle_Skeletal':       'Muscle',
    'Whole_Blood':           'Blood',
    'Adipose_Subcutaneous':  'Adipose',
    'Artery_Aorta':          'Aorta',
    'Breast_Mammary_Tissue': 'Mammary',
}

# Short labels for the loading-panel x axis (GTEx tissue ids are far too long)
LOADING_LABELS = {
    'Adipose_Subcutaneous':               'Adipose',
    'Artery_Aorta':                       'Aorta',
    'Brain_Anterior_cingulate_cortex_BA24': 'Brain BA24',
    'Brain_Cerebellar_Hemisphere':        'Cerebellum',
    'Brain_Cortex':                       'Cortex',
    'Brain_Frontal_Cortex_BA9':           'Front. Ctx',
    'Brain_Putamen_basal_ganglia':        'Putamen',
    'Breast_Mammary_Tissue':              'Mammary',
    'Heart_Atrial_Appendage':             'Heart Atr.',
    'Heart_Left_Ventricle':               'Heart Ven.',
    'Muscle_Skeletal':                    'Muscle',
    'Skin_Not_Sun_Exposed_Suprapubic':    'Skin',
    'Skin_Sun_Exposed_Lower_leg':         'Skin (Sun)',
    'Whole_Blood':                        'Blood',
    'Mammary_Gland':                      'Mammary',
    'Subcutaneous_Adipose':               'Adipose',
    'Hypothalamus':                       'Hypothal.',
    'CNS':                                'CNS',
}


def tidy(name, table=TISSUE_DISPLAY):
    return table.get(name, name.replace('_', ' '))


# %% CELL: panel A — tissue-enriched isoforms in human
def draw_panel_A(fig, df, x0, y0, w, h):
    """Bar chart of significant genes per tissue (Human_significant_SSMD)."""
    ax = add_ax(fig, x0, y0, w, h)
    n = len(df)
    for i, (tissue, row) in enumerate(df.iterrows()):
        c = BARPLOT_COLORS.get(tissue, '#888888')
        ax.bar(i, int(row['n_sig']), color=c, width=0.78, linewidth=0)
        r, g, b = mpl.colors.to_rgb(c)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(i, 250, f"{int(row['n_sig']):,}", ha='center', va='bottom',
                rotation=90, fontsize=FS_ANNOT,
                color='black' if lum > 0.55 else 'white')

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xticks(range(n))
    ax.set_xticklabels([tidy(t, BARPLOT_LABELS) for t in df.index],
                       rotation=45, ha='right', fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK, pad=1)
    ax.tick_params(axis='x', pad=1)
    ax.set_ylabel('Number of\nsignificant genes', fontsize=FS_LABEL, labelpad=2)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


# %% CELL: panel B — cumulative sharing of tissue-specific isoforms
HUMAN_TOTALS = np.array([150, 259, 1236, 5836])
MOUSE_TOTALS = np.array([149, 256, 424, 1503, 3641])
HUMAN_COLS   = ['All animals', 'Mammals', 'Primates', 'Human only']
HUMAN_TICKS  = ['All', 'Mammals', 'Primates', 'Human']
MOUSE_COLS   = ['All animals', 'Mammals', 'Gilres', 'Rodents', 'Mouse only']
MOUSE_TICKS  = ['All', 'Mammals', 'Gilres', 'Rodents', 'Mouse']


def _stack_bands(df, cols):
    """Cumulative lower/upper edges per stacked tissue category."""
    base = np.zeros(len(cols))
    for row, label in zip(STACK_ROWS, STACK_LABELS):
        top = base + np.cumsum([df.loc[row, c] for c in cols])
        yield label, base.copy(), top
        base = top


def draw_panel_B_block(fig, df, cols, ticks, totals, x0, y0, w, h,
                       gap=7.5, tick_l=9.0, tick_r=9.0):
    """One species block: absolute counts (left) + percentage (right).

    Both axes carry outward tick labels and a y label, so a gutter is reserved
    on each side; the plotted widths are what is left over, split evenly.
    """
    aw = (w - gap - tick_l - tick_r) / 2
    ax_l = add_ax(fig, x0 + tick_l, y0, aw, h)
    ax_r = add_ax(fig, x0 + tick_l + aw + gap, y0, aw, h)
    xs = np.arange(len(cols))

    for label, lo, hi in _stack_bands(df, cols):
        ax_l.fill_between(xs, lo, hi, color=STACK_COLORS[label], linewidth=0)
        ax_r.fill_between(xs, lo / totals, hi / totals,
                          color=STACK_COLORS[label], label=label, linewidth=0)

    for i, (x, tot) in enumerate(zip(xs, totals)):
        # first/last totals sit on a spine — align them inward so they stay on-page
        ha = 'left' if i == 0 else ('right' if i == len(xs) - 1 else 'center')
        ax_l.text(x, tot, f'{tot:,}', fontsize=FS_SMALL, va='bottom', ha=ha)

    for ax in (ax_l, ax_r):
        ax.set_xlim(0, len(cols) - 1)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=FS_SMALL)
        ax.tick_params(labelsize=FS_TICK, pad=1)

    # Set the limit first, then prune the ticks to it: an auto-located tick past
    # the top of the view is still drawn, and lands outside the page.
    top = totals[-1] * 1.16
    ax_l.set_ylim(0, top)
    ax_l.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax_l.set_yticks([t for t in ax_l.get_yticks() if 0 <= t <= top])
    ax_l.set_ylabel('Genes with tissue-specific\nisoforms shared across species',
                    fontsize=FS_LABEL, labelpad=2)
    ax_l.spines[['top', 'right']].set_visible(False)

    ax_r.set_ylim(0, 1)
    ax_r.yaxis.set_label_position('right')
    ax_r.yaxis.tick_right()
    ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax_r.set_ylabel('Percentage of genes with\ntissue-specific isoforms',
                    fontsize=FS_LABEL, labelpad=2, rotation=270, va='bottom')
    ax_r.spines[['top', 'left']].set_visible(False)
    return ax_l, ax_r


def draw_panel_B_legend(fig, x0, y0, w, ncol=4, row_h=2.3, sw=2.8, sh=1.5):
    """
    One key for both blocks of panel B, laid out in `ncol` columns so it is wide
    and short. Stacking order, top of the stack first.

    Note this keys the *conservation categories* of panel B, not tissue samples —
    which is why its Cerebellum swatch is orange while the shared tissue key
    (panel C onward) uses the yellow the source assigns to cerebellum samples.
    """
    order = STACK_LABELS[::-1]
    per_col = int(np.ceil(len(order) / ncol))
    col_w = w / ncol
    for i, label in enumerate(order):
        c, r = divmod(i, per_col)
        x, y = x0 + c * col_w, y0 + r * row_h
        fig.patches.append(Rectangle(
            (x / PAGE_W, 1 - (y + sh) / PAGE_H), sw / PAGE_W, sh / PAGE_H,
            transform=fig.transFigure, facecolor=STACK_COLORS[label],
            edgecolor='none', zorder=5))
        fig_text(fig, x + sw + 0.8, y + sh / 2, label,
                 fontsize=FS_ANNOT, va='center', ha='left')
    return int(np.ceil(len(order) / ncol)) * row_h


# %% CELL: panel C — cross-species conservation heatmaps
# Blocks outlined on each heatmap: (row tissues, column tissues, colour key)
C_BLOCKS = [
    (['Brain', 'Cerebellum'],        ['Brain', 'Cerebellum'], 'Brain'),
    (['Heart', 'Muscle_Skeletal'],   ['Heart', 'Muscle'],     'Heart'),
    (['Liver'],                      ['Liver'],               'Liver'),
    (['Testis'],                     ['Testis'],              'Testis'),
]
# Cattle sits last: it is the one species whose columns differ (Muscle, no
# Cerebellum), so putting it at the end keeps the odd labelling out of the middle.
C_ORDER = ['human_macaque', 'human_rabbit', 'human_mouse',
           'human_rat', 'human_chicken', 'human_cattle']
C_VMIN, C_VMAX = 1.0, 3.0


def draw_panel_C_heatmap(fig, hm, x0, y0, cell_w, cell_h, strip=1.1,
                         show_rowlabels=False, show_collabels=True):
    """One species heatmap: colour strips, enrichment cells, counts, key blocks."""
    rows, cols = hm['tissue_list1'], hm['tissue_list2']
    w, h = cell_w * len(cols), cell_h * len(rows)

    # column colour strip (above) and row colour strip (left)
    ax_c = add_ax(fig, x0, y0, w, strip)
    ax_c.set_xlim(0, len(cols)); ax_c.set_ylim(0, 1); ax_c.axis('off')
    for j, t in enumerate(cols):
        ax_c.add_patch(Rectangle((j, 0), 1, 1,
                                 color=TISSUE_COLORS.get(t, DEFAULT_TISSUE_COLOR)))

    ax_r = add_ax(fig, x0 - strip - 0.25, y0 + strip + 0.25, strip, h)
    ax_r.set_ylim(len(rows), 0); ax_r.set_xlim(0, 1); ax_r.axis('off')
    for i, t in enumerate(rows):
        ax_r.add_patch(Rectangle((0, i), 1, 1,
                                 color=TISSUE_COLORS.get(t, DEFAULT_TISSUE_COLOR)))

    ax = add_ax(fig, x0, y0 + strip + 0.25, w, h)
    enr = hm['enrich_df'].values.astype(float)
    cnt = hm['match_df'].values
    ax.imshow(enr, cmap='Blues', vmin=C_VMIN, vmax=C_VMAX, aspect='auto',
              extent=[0, len(cols), len(rows), 0], interpolation='nearest')

    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            e = enr[i, j]
            ax.text(j + 0.5, i + 0.5, str(int(cnt[i, j])),
                    ha='center', va='center', fontsize=FS_ANNOT,
                    color='white' if (not np.isnan(e) and e >= C_VMAX * 0.75) else 'black')

    # key blocks around the tissue-matched cells
    for row_ts, col_ts, key in C_BLOCKS:
        ri = [rows.index(t) for t in row_ts if t in rows]
        ci = [cols.index(t) for t in col_ts if t in cols]
        if not ri or not ci:
            continue
        ax.add_patch(Rectangle((min(ci), min(ri)), len(ci), len(ri),
                               fill=False, edgecolor=TISSUE_COLORS[key],
                               linewidth=1.2, zorder=4))

    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels([tidy(t) for t in cols] if show_collabels else [],
                       rotation=45, ha='right', fontsize=FS_SMALL)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([tidy(t) for t in rows] if show_rowlabels else [],
                       fontsize=FS_TICK)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(SPECIES_DISPLAY.get(hm['sp2'], hm['sp2']),
                 fontsize=FS_SUB, pad=2)
    return ax


# %% CELL: gene structure blocks
# A coverage track is ~45 mm wide, i.e. ~700 px at export dpi, but the arrays
# carry one point per base — up to 60x oversampled.  Block-max decimation keeps
# the peak envelope exactly while cutting the vertex count, and lets the traces
# stay *vector*: with rasterized=True each of the ~100 track artists made the
# mixed-mode renderer allocate a full-page raster buffer at savefig.dpi, which
# pushed the export to ~3.9 GB and got it OOM-killed.
COVERAGE_MAX_PTS = 3000


def _decimate(x, y, max_pts=COVERAGE_MAX_PTS):
    """Block-max downsample a coverage trace, preserving peak heights."""
    n = len(x)
    if n <= max_pts:
        return np.asarray(x), np.asarray(y)
    k = int(np.ceil(n / max_pts))
    m = (n // k) * k
    xb = np.asarray(x[:m]).reshape(-1, k).mean(axis=1)
    yb = np.asarray(y[:m]).reshape(-1, k).max(axis=1)
    if m < n:
        xb = np.append(xb, np.mean(x[m:]))
        yb = np.append(yb, np.max(y[m:]))
    return xb, yb


def _iso_rows(panel, n_iso, colors=None, gene=None):
    """
    Isoforms to draw and their colours.

    With `gene`, the pair comes from Figure3_helpers.ISOFORM_PICKS — the two
    isoforms chosen by eye from the structure plots — rather than the first two
    of iso_order, which are simply the two highest-SSMD isoforms and were often
    not the ones of interest.
    """
    if gene is not None:
        from Figure3_helpers import picked_isoforms
        order, cols = picked_isoforms(panel, gene)
        return order[:n_iso], cols[:n_iso]
    order = panel['iso_order'][:n_iso]
    if colors is None:
        colors = [ISO_RED] + [ISO_NAVY] * (len(order) - 1)
    return order, colors[:len(order)]


def draw_gene_block(fig, panel, x0, y0, w, n_iso=2, iso_colors=None,
                    track_h=2.3, iso_h=1.15, gap=0.25, star_iso=None,
                    label_species=True, fs_species=None, label_tracks=True,
                    gene=None):
    """
    One species block: coverage tracks stacked over isoform structures.

    `star_iso` marks the last exon of that isoform with an asterisk (used for the
    alternative-last-exon calls in SYNGR1 and VRK3).

    `label_tracks` names each coverage track.  Brain and cerebellum share a
    colour, so the names cannot be replaced by the shared tissue key — instead
    the caller labels only the first block of each distinct track set, and the
    name sits on a white plate so it never strikes through the trace.

    Returns the block height.
    """
    fs_species = fs_species or FS_SUB
    tracks = panel['show_tracks']
    xlim = panel['xlim']
    x, y = panel['x'], panel['y']
    order, colors = _iso_rows(panel, n_iso, iso_colors, gene=gene)

    y_cur = y0
    for k, tk in enumerate(tracks):
        ax = add_ax(fig, x0, y_cur, w, track_h)
        c = TRACK_COLORS[tk]
        xd, yd = _decimate(x, y[tk])
        ax.plot(xd, yd, c=c, lw=LW_TRACK)
        ax.fill_between(xd, 0, yd, color=c, alpha=0.45, lw=0)
        ax.set_xlim(xlim)
        ax.set_ylim(0, max(np.max(y[tk]) * 1.05, 1e-9))
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
        if label_tracks:
            ax.text(0.0, 0.55, TRACK_LABELS[tk], transform=ax.transAxes,
                    fontsize=FS_ANNOT, va='bottom', ha='left',
                    color=_darken(c), zorder=6,
                    bbox=dict(facecolor='white', alpha=0.75, pad=0.5,
                              edgecolor='none'))
        if k == 0 and label_species:
            ax.set_title(panel['species'], fontsize=fs_species, pad=1.0)
        y_cur += track_h

    y_cur += gap
    gene_exons = panel['iso'].loc[panel['iso'].gene_id == panel['gene']]
    strand = panel['strand']
    for iso, c in zip(order, colors):
        ax = add_ax(fig, x0, y_cur, w, iso_h)
        _draw_isoform(ax, gene_exons.loc[gene_exons.transcript_id == iso],
                      c, xlim, strand,
                      star=(star_iso is not None and iso == order[star_iso]))
        y_cur += iso_h
    return y_cur - y0


def _draw_isoform(ax, exons, color, xlim, strand, star=False):
    """Exon boxes joined by an intron line, flipped for minus-strand genes."""
    x1, x2 = xlim
    segs = []
    for _, row in exons.iterrows():
        s, e = int(row.start), int(row.end)
        if strand == '-':
            s, e = x2 - e + x1, x2 - s + x1
        segs.append((s, e))
    segs.sort()
    if segs:
        ax.plot([segs[0][0], segs[-1][1]], [0.5, 0.5], c=color,
                lw=INTRON_LW, zorder=1)
        lo, hi = (1 - EXON_H) / 2, (1 + EXON_H) / 2
        for s, e in segs:
            ax.fill_between([s, e], [lo, lo], [hi, hi], color=color,
                            lw=0, zorder=2)
        if star:
            # exons are already flipped into 5'->3' display order, so the
            # alternative last exon is the rightmost one on either strand
            s, e = segs[-1]
            ax.text((s + e) / 2, 0.5, '*', ha='center', va='center',
                    fontsize=FS_SUB, color='black', zorder=4)
    ax.set_xlim(xlim); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)


def gene_block_height(panel, n_iso=2, track_h=2.3, iso_h=1.15, gap=0.25):
    return len(panel['show_tracks']) * track_h + gap + \
        min(n_iso, len(panel['iso_order'])) * iso_h


# %% CELL: loadings blocks
def draw_loading_block(fig, panel, x0, y0, w, row_h=3.0, gap=0.35,
                       swatch_h=1.0, label_h=None, n_rows=3,
                       highlight=HL_DEFAULT, show_labels=True, row_labels=None,
                       label_colors=None, title=True, ylabel=None, gene=None):
    """
    Median (10-90 %) isoform loadings per tissue, in three stacked rows:
    tissue-specific isoform, non-specific isoform, all remaining isoforms.

    Swatch colours come from the panel's own per-species list (the one the source
    `plot_factor_loadings` uses), aligned positionally to the sorted tissues.
    `highlight` names the tissue groups to band; see HL_GROUPS.
    Returns the block height in mm.
    """
    EL = panel['el_annot']
    tissues, tissue_colors, ids = tissue_groups(panel)
    samples = panel['meta'].loc[ids.index].copy()
    samples['tissue_id'] = ids
    if gene is not None:
        from Figure3_helpers import picked_isoforms
        factors = picked_isoforms(panel, gene)[0]
    else:
        factors = panel['iso_order']
    K, N = len(factors), len(tissues)
    n_ax = min(K, n_rows)

    axes = []
    y_cur = y0
    for i in range(n_ax):
        ax = add_ax(fig, x0, y_cur, w, row_h)
        axes.append(ax)
        y_cur += row_h + gap
    y_cur -= gap

    # highlight bands behind the points
    for group in (highlight or ()):
        color, matches = HL_GROUPS[group]
        for j, t in enumerate(tissues):
            if matches(t):
                for ax in axes:
                    ax.axvspan(j - 0.5, j + 0.5, color=color, alpha=HL_ALPHA,
                               lw=0, zorder=0)

    ceiling = 0.0
    for i, factor in enumerate(factors):
        ax_i = min(i, n_ax - 1)
        if i >= n_ax - 1:                     # "other isoforms" row: jitter within slot
            n_other = max(K - (n_ax - 1), 1)
            offs = (np.arange(n_other) / (2 * n_other)) - 0.25
            dx = offs[i - (n_ax - 1)]
            ceiling = max(ceiling, float(EL.loc[:, factor].max()))
            top = ceiling
            color = ISO_GREY
        else:
            dx = 0.0
            top = float(EL.loc[:, factor].max())
            color = [ISO_RED, ISO_NAVY][i] if i < 2 else ISO_GREY

        for j, tissue in enumerate(tissues):
            vals = EL.loc[samples.index[samples.tissue_id == tissue], factor].dropna()
            if len(vals) == 0:
                continue
            med = np.median(vals)
            q1, q9 = np.quantile(vals, 0.1), np.quantile(vals, 0.9)
            axes[ax_i].plot([j + dx, j + dx], [q1, q9], c=color, lw=0.5, zorder=3)
            axes[ax_i].scatter([j + dx], [med], c=color, s=2.2, lw=0, zorder=4)
        axes[ax_i].set_ylim(-0.06, top + 0.06)

    for i, ax in enumerate(axes):
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=FS_SMALL, pad=0.8)
        ax.locator_params(axis='y', nbins=2)
        ax.spines[['top', 'right']].set_visible(False)
        if row_labels and i < len(row_labels):
            ax.text(0.985, 0.93, row_labels[i], transform=ax.transAxes,
                    ha='right', va='top', fontsize=FS_ANNOT, zorder=6,
                    color=(label_colors or [ISO_RED, ISO_NAVY, 'black'])[i],
                    bbox=dict(facecolor='white', alpha=0.7, pad=0.6,
                              edgecolor='none'))

    if title:
        axes[0].set_title(panel['species'], fontsize=FS_SUB, pad=1.5)
    if ylabel:
        axes[len(axes) // 2].set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=2)

    # tissue colour swatches + rotated labels
    y_cur += 0.3
    ax_s = add_ax(fig, x0, y_cur, w, swatch_h)
    ax_s.set_xlim(-0.5, N - 0.5); ax_s.set_ylim(0, 1); ax_s.axis('off')
    for j, c in enumerate(tissue_colors):
        ax_s.add_patch(Rectangle((j - 0.42, 0), 0.84, 1, color=c, lw=0))
    y_cur += swatch_h
    if show_labels:
        for j, t in enumerate(tissues):
            ax_s.text(j, -0.35, tidy(t, LOADING_LABELS), rotation=90,
                      ha='center', va='top', fontsize=FS_SMALL,
                      transform=ax_s.get_xaxis_transform())
    return y_cur - y0


# %% CELL: assemble the figure
# Each gene panel is a grid of species cells.  A cell pairs the gene structure
# with that species' loadings block immediately to its right, so the species name
# is written once instead of twice.  The two structure columns together take
# STRUCT_FRAC of the page; the loadings share what is left.
#
# Cattle is last in every ordering: it is the species whose tissue set differs
# (no cerebellum, plus mammary/CNS/hypothalamus), so trailing it keeps the odd
# labelling out of the middle of the grid.
GENE_PANELS = [
    dict(gene='PHC1', letter='d',
         title='Shared brain-specific isoform in',
         species=['Human', 'Macaque', 'Rabbit', 'Mouse', 'Rat', 'Chicken', 'Cattle'],
         star_iso=None, highlight=HL_DEFAULT,
         key=[('Brain/Cerebellum-specific isoform', ISO_RED),
              ('Non-tissue-specific isoform', ISO_NAVY),
              ('Other isoforms', ISO_GREY)]),
    dict(gene='SYNGR1', letter='e',
         title='Shared brain-specific isoform in',
         species=['Human', 'Macaque', 'Mouse', 'Rat', 'Rabbit', 'Chicken', 'Cattle'],
         star_iso=0, highlight=HL_MAMMARY,
         key=[('Top brain isoform', ISO_RED),
              ('Top heart isoform', ISO_NAVY),
              ('Other isoforms', ISO_GREY)]),
    dict(gene='VRK3', letter='f',
         title='Primate exclusive alternative last exon in',
         species=['Human', 'Macaque', 'Rat', 'Cattle'],
         star_iso=0, highlight=HL_DEFAULT,
         key=[('Top cerebellum isoform', ISO_RED),
              ('Top heart isoform', ISO_NAVY),
              ('Other isoforms', ISO_GREY)]),
]

GENE_COLS   = 2       # species cells per row
STRUCT_FRAC = 0.60    # 3/5 of the content width goes to the two structure columns
CELL_GAP    = 3.0     # between the two cells in a row
INNER_GAP   = 4.0     # structure -> loadings inside a cell

TRACK_H = 2.4          # one coverage track
ISO_H   = 1.25         # one isoform row
BLK_GAP = 0.3          # tracks -> isoforms
SPECIES_GAP = 2.0      # between rows of cells
SPECIES_TITLE_H = 1.6  # headroom a species title needs
TITLE_H = 4.0          # panel title strip
KEY_H   = 2.6          # the isoform/loadings key line under a gene panel title

LOAD_ROW_H = 2.7       # one loadings row
SLOT_TICKS = 3.4       # gutter for a loadings block's y tick labels
E_LABEL_H  = 9.5       # rotated tissue names under a loadings block

# Tissue names under the loadings blocks:
#   'none'  — swatch strip only.  The highlight bands already mark the tissues the
#             figure makes claims about, and panels A and C name the rest.
#   'first' — spell them out on the first block of each distinct tissue set.
#             Correct but expensive: it adds ~E_LABEL_H to three rows per gene
#             panel, roughly +70 mm over the whole figure.
#   'all'   — every block.
LOAD_LABEL_MODE = 'none'

LOAD_BLK_H = 3 * LOAD_ROW_H + 2 * 0.35 + 0.3 + 1.0


def _n_slots(panels, species):
    return len(tissue_groups(panels[species])[0])


def _blk_h(panel, n_iso):
    return gene_block_height(panel, n_iso, TRACK_H, ISO_H, BLK_GAP)


def _first_of_each_set(panels, species_list, key):
    """Species that are the first to show a given track/tissue set — the only
    ones that need the set spelled out."""
    seen, out = set(), []
    for sp in species_list:
        k = tuple(key(panels[sp]))
        if k not in seen:
            seen.add(k)
            out.append(sp)
    return set(out)


def _grid_rows(species_list, ncols=GENE_COLS):
    return [species_list[i:i + ncols] for i in range(0, len(species_list), ncols)]


def _n_iso_for(gene, species):
    # ISOFORM_PICKS names exactly two isoforms per species, so the old
    # three-isoform special case for VRK3 in human no longer applies.
    return 2


def _cell_widths():
    """(cell_w, struct_w, load_w) in mm for one species cell."""
    content = GENE_X1 - GENE_X0
    cell_w = (content - CELL_GAP) / GENE_COLS
    struct_w = STRUCT_FRAC * content / GENE_COLS
    load_w = cell_w - INNER_GAP - struct_w
    return cell_w, struct_w, load_w


def _labelled_loads(panels, gene, species):
    if LOAD_LABEL_MODE == 'all':
        return set(species)
    if LOAD_LABEL_MODE == 'first':
        return _first_of_each_set(panels[gene], species,
                                  lambda p: tissue_groups(p)[0])
    return set()


def _gene_band_h(panels, cfg):
    """Title + key + rows, each row sized by its tallest cell."""
    gene, species = cfg['gene'], cfg['species']
    lab = _labelled_loads(panels, gene, species)
    h = TITLE_H + KEY_H
    for row in _grid_rows(species):
        h += max(max(_blk_h(panels[gene][sp], _n_iso_for(gene, sp)),
                     LOAD_BLK_H + (E_LABEL_H if sp in lab else 0))
                 for sp in row) + SPECIES_TITLE_H + SPECIES_GAP
    return h


def plan_page(data):
    """
    Size every content-driven band from the panels it holds, then derive PAGE_H.
    Called before any drawing, so every rect() sees the final page height.
    """
    global BAND_D_H, BAND_E_H, BAND_F_H
    global BAND_D_Y, BAND_E_Y, BAND_F_Y, PAGE_H
    P = data['panels']

    BAND_D_H, BAND_E_H, BAND_F_H = (_gene_band_h(P, c) for c in GENE_PANELS)
    BAND_D_Y = BAND_C_Y + BAND_C_H + BAND_GAP
    BAND_E_Y = BAND_D_Y + BAND_D_H + BAND_GAP
    BAND_F_Y = BAND_E_Y + BAND_E_H + BAND_GAP
    PAGE_H = BAND_F_Y + BAND_F_H + MARGIN_B
    return {}


def draw_key_line(fig, items, y, x_center=None, item_w=44.0, sw=2.8, sh=1.3):
    """A one-line key: swatch + label, repeated across, centred on the page."""
    x_center = PAGE_W / 2 if x_center is None else x_center
    x0 = x_center - len(items) * item_w / 2
    for i, (label, color) in enumerate(items):
        x = x0 + i * item_w
        fig.patches.append(Rectangle(
            (x / PAGE_W, 1 - (y + sh) / PAGE_H), sw / PAGE_W, sh / PAGE_H,
            transform=fig.transFigure, facecolor=color, edgecolor='none', zorder=5))
        fig_text(fig, x + sw + 0.8, y + sh / 2, label, fontsize=FS_ANNOT,
                 va='center', ha='left')


def draw_gene_panel(fig, panels, cfg, y_band):
    """One gene panel: title, key line, then rows of paired species cells."""
    gene, species = cfg['gene'], cfg['species']
    cell_w, struct_w, load_w = _cell_widths()
    track_lab = _first_of_each_set(panels[gene], species,
                                   lambda p: p['show_tracks'])
    load_lab = _labelled_loads(panels, gene, species)

    panel_letter(fig, cfg['letter'], MARGIN_L, y_band)
    two_part_title(fig, PAGE_W / 2, y_band + 0.2, cfg['title'], cfg['gene'],
                   fontsize=FS_TITLE)
    draw_key_line(fig, cfg['key'], y_band + TITLE_H - 0.6)

    y = y_band + TITLE_H + KEY_H
    for row in _grid_rows(species):
        row_h = max(max(_blk_h(panels[gene][sp], _n_iso_for(gene, sp)),
                        LOAD_BLK_H + (E_LABEL_H if sp in load_lab else 0))
                    for sp in row)
        for c, sp in enumerate(row):
            cx = GENE_X0 + c * (cell_w + CELL_GAP)
            panel = panels[gene][sp]

            # species name centred over the whole cell, written once
            fig_text(fig, cx + cell_w / 2, y + SPECIES_TITLE_H - 0.3, sp,
                     fontsize=FS_SUB, ha='center', va='bottom')

            n_iso = _n_iso_for(gene, sp)
            draw_gene_block(
                fig, panel, cx, y + SPECIES_TITLE_H, struct_w, n_iso=n_iso,
                iso_colors=([ISO_RED] + [ISO_NAVY] * 2) if n_iso == 3 else None,
                track_h=TRACK_H, iso_h=ISO_H, gap=BLK_GAP,
                star_iso=cfg['star_iso'], label_species=False,
                label_tracks=(sp in track_lab), gene=gene)

            draw_loading_block(
                fig, panel, cx + struct_w + INNER_GAP + SLOT_TICKS,
                y + SPECIES_TITLE_H, load_w - SLOT_TICKS,
                row_h=LOAD_ROW_H, n_rows=3, highlight=cfg['highlight'],
                show_labels=(sp in load_lab), title=False, gene=gene)
        y += row_h + SPECIES_TITLE_H + SPECIES_GAP


def build_figure(data=None):
    data = load_figure_data() if data is None else data
    panels = data['panels']
    _COMPOSITE_TITLES.clear()
    plan_page(data)
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    fig.patch.set_facecolor('white')

    # ---------------- Panel A ----------------
    panel_letter(fig, 'a', MARGIN_L, BAND_AB_Y)
    fig_text(fig, (A_X0 + A_X1) / 2 + 4, BAND_AB_Y + 0.4,
             'Tissue-enriched isoforms in human',
             fontsize=FS_TITLE, ha='center', va='top')
    draw_panel_A(fig, data['ssmd_barplot_df'],
                 A_X0 + 11.0, BAND_AB_Y + 3.4, A_X1 - A_X0 - 12.0, 18.0)

    # ---------------- Panel B ----------------
    panel_letter(fig, 'b', B_X0 - 1.5, BAND_AB_Y)
    b_gap = 6.0
    b_w = (B_X1 - B_X0 - b_gap) / 2
    for i, (title, df, cols, ticks, tot) in enumerate([
        ('Tissue-specific isoform sharing in human',
         data['df_table'], HUMAN_COLS, HUMAN_TICKS, HUMAN_TOTALS),
        ('Tissue-specific isoform sharing in mouse',
         data['df_table_mouse'], MOUSE_COLS, MOUSE_TICKS, MOUSE_TOTALS),
    ]):
        bx = B_X0 + i * (b_w + b_gap)
        fig_text(fig, bx + b_w / 2, BAND_AB_Y + 0.4, title,
                 fontsize=FS_TITLE, ha='center', va='top')
        draw_panel_B_block(fig, df, cols, ticks, tot,
                           bx, BAND_AB_Y + 3.4, b_w, 18.0)
    leg_w = 78.0
    draw_panel_B_legend(fig, (B_X0 + B_X1) / 2 - leg_w / 2,
                        BAND_AB_Y + 28.0, leg_w)

    # ---------------- Panel C ----------------
    panel_letter(fig, 'c', MARGIN_L, BAND_C_Y)
    fig_text(fig, PAGE_W / 2, BAND_C_Y + 0.2,
             'Genes with shared tissue-specific isoforms across animal species',
             fontsize=FS_TITLE, ha='center', va='top')

    c_x0, c_right = 21.5, 167.0
    n_hm = len(C_ORDER)
    hm_gap = 3.0
    hm_w = (c_right - c_x0 - (n_hm - 1) * hm_gap) / n_hm
    cell_w = hm_w / 5.0                       # every heatmap has 5 columns
    cell_h = 3.6
    c_top = BAND_C_Y + 5.4
    # Five of the six share a column set; Cattle differs (Muscle, no Cerebellum)
    # and is last.  Label those two, not all six.
    for k, key in enumerate(C_ORDER):
        draw_panel_C_heatmap(fig, data['heatmaps'][key],
                             c_x0 + k * (hm_w + hm_gap), c_top,
                             cell_w, cell_h, show_rowlabels=(k == 0),
                             show_collabels=key in ('human_macaque', 'human_cattle'))
    fig_text(fig, 6.0, c_top + 1.1 + 6 * cell_h / 2, 'Human',
             fontsize=FS_LABEL, ha='center', va='center', rotation=90)

    cax = add_ax(fig, c_right + 2.5, c_top + 1.5, 1.5, 6 * cell_h * 0.62)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(C_VMIN, C_VMAX), cmap='Blues'),
                      cax=cax, ticks=[1, 2, 3])
    cb.set_label('Enrichment', fontsize=FS_LABEL, labelpad=1)
    cb.ax.tick_params(labelsize=FS_TICK, pad=1)
    cb.outline.set_linewidth(LW_THIN)

    # ---------------- Panels D, E, F — one per gene ----------------
    for cfg, y_band in zip(GENE_PANELS, (BAND_D_Y, BAND_E_Y, BAND_F_Y)):
        draw_gene_panel(fig, panels, cfg, y_band)

    _place_composite_titles(fig)
    return fig


# Resolve the derived band heights at import, so anything that reads PAGE_W /
# PAGE_H without calling build_figure() (the layout checker does exactly this)
# sees the real page rather than the placeholder defaults.
try:
    plan_page(load_figure_data())
except (FileNotFoundError, OSError):        # data not built yet
    pass


# %% CELL: render
data = load_figure_data()
fig = build_figure(data)
print(f'page: {PAGE_W:.1f} x {PAGE_H:.1f} mm  (bands: AB {BAND_AB_H:.1f}, '
      f'C {BAND_C_H:.1f}, D {BAND_D_H:.1f}, E {BAND_E_H:.1f}, '
      f'F {BAND_F_H:.1f})')
_cw, _sw, _lw = _cell_widths()
print(f'gene cell: {_cw:.1f} mm = structure {_sw:.1f} + gap {INNER_GAP} '
      f'+ loadings {_lw:.1f}   '
      f'(structures are {200 * _sw / (GENE_X1 - GENE_X0):.0f}% of content width)')

fig.savefig(os.path.join(FIG_DIR, 'Figure3_publication.png'),
            dpi=400, facecolor='white')
fig.savefig(os.path.join(FIG_DIR, 'Figure3_publication.pdf'),
            facecolor='white')
# open THIS one in Inkscape: real <text> elements, one x per string
fig.savefig(os.path.join(FIG_DIR, 'Figure3_publication.svg'),
            facecolor='white')
print('saved Figure3_publication.png / .pdf / .svg')
fig
