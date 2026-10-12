"""
SFigure4_plot_helpers.py — drawing only. Every function takes plot-ready data
from SFigure4_helpers and does no loading or processing.
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Set3, as in the exploratory notebooks. The stack is drawn largest-first, so
# each later bar overpaints the previous one and the visible band for a class is
# the difference between consecutive cumulative sums.
#   colores[4]  any colocalisation
#   colores[3]  + a classified RNA-processing event
#   colores[2]  single class: cassette exon
#   colores[1]  single class: 3' UTR
#   colores[0]  single class: intron retention
#   colores[8]  grey background = 100%
COLORS = sns.color_palette('Set3')

CLASSIFIED = ['ir', 'utr', 'cassette', 'ir_utr', 'ir_cassette', 'cassette_utr',
              'multiple']


def _fractions(df):
    """The five cumulative fractions each per-trait stack is built from."""
    return {
        'any': df.percentage_coloc,
        'classified': df[CLASSIFIED].sum(axis=1) / df.counter,
        'cassette': (df.ir + df.utr + df.cassette) / df.counter,
        'utr': (df.ir + df.utr) / df.counter,
        'ir': df.ir / df.counter,
    }


def plot_colocs(panel, figsize=(10, 3), ylim_top=0.42, annot_size=6,
                label_size=6):
    """
    Supplementary Figure 4 — per GWAS trait, the percentage of tested regions
    that colocalise with a torino isoform, split by RNA-processing class.

    Source: Figure_4.ipynb cell 93, with the three corrections documented in
    SFigure4_helpers. The broken upper axis (95-100%) is kept: it exists so the
    grey 100% reference and the classified-event fraction stay visible without
    compressing the 0-40% range where the data lives.
    """
    df = panel['trait_selected_df']
    names = panel['name_list']
    r = df.shape[0]
    frac = _fractions(df)

    fig, axes_ = plt.subplots(
        nrows=2, ncols=2, figsize=figsize,
        gridspec_kw={'height_ratios': [1.4, 10], 'hspace': 0.08,
                     'width_ratios': [10, 0.15], 'wspace': 0.03})

    # ---- top strip: the 95-100% end of the same bars ----------------------
    top, top_all = axes_[0, 0], axes_[0, 1]
    top.bar(range(r), [1] * r, color=COLORS[8], alpha=0.3)
    top.bar(range(r), frac['classified'], color=COLORS[3])
    top_all.bar([0], [1], color=COLORS[8], alpha=0.3)
    for ax in (top, top_all):
        ax.set_ylim([0.95, 1])
        ax.set_xticks([])
        ax.set_yticks([0.95, 1.0])
        ax.set_yticklabels(['95', '100'], size=label_size)
    top.set_xlim((-1, r - 0.5))
    top.spines[['right', 'top', 'bottom']].set_visible(False)
    top_all.spines[['left', 'top', 'bottom']].set_visible(False)
    top_all.yaxis.tick_right()

    # n of testable loci over every bar, including the All-traits column
    for i, n in enumerate(df.counter):
        top.text(i - 0.25, 0.999, str(int(n)), rotation=90, size=annot_size,
                 horizontalalignment='left', verticalalignment='top')
    top_all.text(-0.25, 0.999, str(panel['total_loci']), rotation=90,
                 size=annot_size, horizontalalignment='left',
                 verticalalignment='top')
    # the All-traits column pools all 68 tested traits, so its n is larger than
    # the regions the 65 drawn bars cover
    top_all.set_xlim(top_all.get_xlim())

    # ---- main panel -------------------------------------------------------
    ax, ax_all = axes_[1, 0], axes_[1, 1]
    ax.bar(range(r), [1] * r, color=COLORS[8], alpha=0.3)
    for key, c in [('any', COLORS[4]), ('classified', COLORS[3]),
                   ('cassette', COLORS[2]), ('utr', COLORS[1]), ('ir', COLORS[0])]:
        ax.bar(range(r), frac[key], color=c)

    # "All traits" is a pooled percentage over every tested trait, the three
    # with zero colocalized regions included — zero is a result, not a gap — so
    # it runs over more regions (panel['total_loci']) than the drawn bars cover.
    pooled = panel['pooled']
    ax_all.bar([0], [1], color=COLORS[8], alpha=0.3)
    for key, c in [('any', COLORS[4]), ('classified', COLORS[3]),
                   ('cassette', COLORS[2]), ('utr', COLORS[1]), ('ir', COLORS[0])]:
        ax_all.bar([0], [pooled[key]], color=c)
    ax_all.text(0, pooled['any'] + 0.012, f"{panel['total_percent']:.1f}%",
                ha='center', va='bottom', size=annot_size + 0.5, fontweight='bold')

    ax.set_xlim((-1, r - 0.5))
    for a in (ax, ax_all):
        a.set_ylim((0, ylim_top))
        a.set_yticks([0, 0.1, 0.2, 0.3, 0.4], ['0', '10', '20', '30', '40'],
                     size=label_size)
    ax.spines[['right', 'top']].set_visible(False)
    ax_all.spines[['left', 'top']].set_visible(False)
    ax_all.yaxis.tick_right()

    ax.set_ylabel('Percentage of GWAS regions')
    ax.set_xticks(range(r), names, rotation=45, ha='right',
                  rotation_mode='anchor', size=label_size)
    ax_all.set_xticks([0], [f"All traits (n={panel['n_tested_traits']})"],
                      rotation=45, ha='right', rotation_mode='anchor',
                      size=label_size)
    return fig


def legend_entries(panel=None, counts=False):
    """
    The bands of plot_colocs, bottom of the bar upward, with what each contains.

    The stack is painted largest-first, so the band a reader sees for a colour
    is the gap between consecutive cumulative sums -- not the cumulative value
    itself. Working outward from zero:

        COLORS[0]  ir                              intron retention alone
        COLORS[1]  ir+utr        minus ir          3' UTR alone
        COLORS[2]  +cassette     minus ir+utr      cassette exon alone
        COLORS[3]  classified    minus the above   two or more of those classes
        COLORS[4]  percentage_coloc minus classified
                                                   colocalized, but the event
                                                   was 'other' or an efficiency
                                                   factor, so no class applies
        COLORS[8]  1.0           minus coloc       tested, did not colocalize

    `counts` adds the totals over the plotted traits to each label.
    """
    rows = [('Intron retention', COLORS[0], ['ir']),
            ("3' UTR", COLORS[1], ['utr']),
            ('Cassette exon', COLORS[2], ['cassette']),
            ('Two or more classes', COLORS[3],
             ['ir_utr', 'ir_cassette', 'cassette_utr', 'multiple']),
            ('Colocalized, no class assigned', COLORS[4],
             ['other', 'efficiency_factor']),
            ('Tested, no colocalization', COLORS[8], None)]
    out = []
    for label, colour, cols in rows:
        if counts and panel is not None:
            d = panel['trait_selected_df']
            n = (int(d.counter.sum() - d.coloc_counter.sum()) if cols is None
                 else int(d[cols].sum().sum()))
            label = f'{label} ({n:,})'
        out.append((label, colour))
    return out


def plot_legend(panel=None, counts=False, figsize=(3.4, 1.9), size=8,
                title=None, ncol=1):
    """
    The colour key as a figure of its own, for placing beside plot_colocs by
    hand. Pass `panel` with counts=True to label each band with its total.
    """
    import matplotlib.patches as mpatches
    entries = legend_entries(panel, counts)
    fig, ax = plt.subplots(figsize=figsize)
    ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='none', label=l)
                       for l, c in entries],
              loc='center', frameon=False, fontsize=size, ncol=ncol,
              title=title, title_fontsize=size + 1,
              handlelength=1.4, handleheight=1.1, labelspacing=0.55,
              borderpad=0.2, handletextpad=0.7)
    ax.axis('off')
    fig.tight_layout()
    return fig
