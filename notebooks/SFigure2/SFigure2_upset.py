"""The full isoform upset, as a standalone supplementary figure.

Every isoform appears exactly once, unlike the upsets in Figure 2, which drop
the splicing-efficiency factors before plotting and then drop whatever lands in
no category. Here those two groups are categories in their own right.

Categories
----------
Annotated                 matches a reference transcript or intron chain, or
                          carries a GENCODE reference
Retained Intron           RI:
alt. Splicing             CE / SE / MXE / alt.ss
alt. first/last exon      alt_5exon / alt_3exon -- kept separate from
                          alt. Splicing, not folded into it
alt. 3'UTR                UTR3
alt. 5'UTR                UTR5
Splicing efficiency       flagged efficiency_factor AND not annotated -- exclusive
Uncategorized             none of the above -- exclusive

The two exclusive categories take precedence: an isoform flagged as a splicing
efficiency factor is counted only there, never also as alt. Splicing. The one
exception is annotation -- an annotated efficiency factor is treated as
annotated and keeps its ordinary categories, matching what Figure 2 counts, so
the Annotated bar agrees between the two figures.

GENE_SUBSET applies, so importing this from SFigure2_corrected restricts it to
the 16,073-gene set exactly as the panels do. Nothing here writes.
"""

import os
import sys

from SFigure2_intron_retention import (      # noqa: F401  -- shared setup
    ANALYSIS_DIR, np, pd, plt, sns, _cached, _panel, free, _tag, load_annot,
    GENE_SUBSET,
)
import SFigure2_intron_retention as _R

# Order is the plot order, and the two exclusive categories come last.
UPSET_CATEGORIES = ['Annotated', 'Retained Intron', 'alt. Splicing',
                    'alt. first/last exon', "alt. 3'UTR", "alt. 5'UTR",
                    'Splicing efficiency', 'Uncategorized']
EXCLUSIVE = ['Splicing efficiency', 'Uncategorized']


def compute_upset_full():
    """One row per isoform, one boolean per category.

    The shared categories are read off appris_ref; then the exclusive ones are
    applied last and clear everything else on those rows.
    """
    def build():
        annot = load_annot()
        ref = annot.appris_ref.astype(str)
        df = pd.DataFrame({
            'Annotated': (ref.apply(lambda x: ('ref_transcript' in x)
                                    or ('intron_chain' in x))
                          | (annot.gencode_ref != '.')),
            'Retained Intron': ref.apply(lambda x: 'RI:' in x),
            'alt. Splicing': ref.apply(
                lambda x: ('CE' in x) or ('SE' in x) or ('MXE' in x)
                or ('alt.ss' in x)),
            'alt. first/last exon': ref.apply(
                lambda x: ('alt_5exon' in x) or ('alt_3exon' in x)),
            "alt. 3'UTR": ref.apply(lambda x: 'UTR3' in x),
            "alt. 5'UTR": ref.apply(lambda x: 'UTR5' in x),
        }, index=annot.index)

        # Splicing efficiency is exclusive -- an isoform flagged as one is not
        # also counted as alt. Splicing or anything else -- EXCEPT when it is
        # annotated. An annotated efficiency factor is treated as annotated and
        # keeps its ordinary categories, which is what Figure 2 does: its filter
        # is (~efficiency_factor) | (efficiency_factor & is_annot), so those
        # isoforms are in its Annotated bar. Both figures therefore report the
        # same Annotated count, and Splicing efficiency here equals the number
        # Figure 2 excludes.
        eff = annot.efficiency_factor.astype(bool)
        eff_only = eff & ~df['Annotated']
        df.loc[eff_only, :] = False
        df['Splicing efficiency'] = eff_only
        # Whatever still has no category is uncategorized, also exclusively.
        df['Uncategorized'] = ~df.any(axis=1)
        return df[UPSET_CATEGORIES]
    return _cached(f'upset_full:{_tag()}', build)


def upset_counts():
    """Per category: isoforms, and the genes they occur in.

    Also the completeness checks and the Figure 2 reconciliation. Gene counts
    do not sum to the gene total the way isoform counts sum to the isoform
    total: one gene commonly has isoforms in several categories.
    """
    df = compute_upset_full()
    annot = load_annot()
    gene = annot.gene.reindex(df.index)
    n = {c: int(df[c].sum()) for c in df.columns}
    n_genes = {c: int(gene[df[c]].nunique()) for c in df.columns}
    n['isoforms'] = len(df)
    n_genes['isoforms'] = int(gene.nunique())
    n['in a single category'] = int((df.sum(axis=1) == 1).sum())
    n['in no category'] = int((df.sum(axis=1) == 0).sum())

    # Figure 2's upset keeps an efficiency factor when it is annotated -- its
    # filter is (~efficiency_factor) | (efficiency_factor & is_annot) -- so
    # those isoforms appear in its Annotated bar. Here Splicing efficiency is
    # exclusive and claims them instead, which is the whole difference between
    # the two Annotated counts. Reported so the figures reconcile.
    annot = load_annot()
    ref = annot.appris_ref.astype(str)
    is_annot = (ref.apply(lambda x: ('ref_transcript' in x)
                          or ('intron_chain' in x))
                | (annot.gencode_ref != '.'))
    eff = annot.efficiency_factor.astype(bool)
    n['annotated, also a splicing-efficiency factor'] = int((is_annot & eff).sum())
    n['annotated as Figure 2 counts it'] = int(is_annot.sum())
    n['efficiency factors Figure 2 excludes'] = int((eff & ~is_annot).sum())
    n_genes['annotated, also a splicing-efficiency factor'] = int(
        annot.loc[(is_annot & eff).values].gene.nunique())
    n_genes['annotated as Figure 2 counts it'] = int(
        annot.loc[is_annot.values].gene.nunique())
    n_genes['efficiency factors Figure 2 excludes'] = int(
        annot.loc[(eff & ~is_annot).values].gene.nunique())
    return n, n_genes


@_panel
def panel_upset_full(min_subset_size=200, figsize=(9.5, 6.0)):
    """The full upset, drawn with upsetplot.

    `min_subset_size` hides the long tail of tiny intersections; every isoform
    is still counted in the category totals on the left.
    """
    from upsetplot import UpSet
    df = compute_upset_full()
    data = df.copy()
    data['count'] = 1
    series = data.groupby(UPSET_CATEGORIES)['count'].sum()

    fig = plt.figure(figsize=figsize)
    # show_counts is left off: this upsetplot against this matplotlib raises
    # "only 0-dimensional arrays can be converted to Python scalars" while
    # drawing the bar labels. The counts are written on afterwards instead.
    up = UpSet(series, subset_size='sum', min_subset_size=min_subset_size,
               sort_by='cardinality', sort_categories_by='cardinality',
               show_counts=False, element_size=None, facecolor='0.25')
    axes = up.plot(fig=fig)

    ax_bars = axes['intersections']
    for patch in ax_bars.patches:
        h = patch.get_height()
        if h > 0:
            ax_bars.text(patch.get_x() + patch.get_width() / 2, h,
                         f'{int(h):,}', ha='center', va='bottom', fontsize=6,
                         rotation=90)
    ax_bars.set_ylim(0, ax_bars.get_ylim()[1] * 1.25)
    fig.suptitle(f'Torino isoforms by category '
                 f'({len(df):,} isoforms, '
                 f'{"corrected gene set" if _R.GENE_SUBSET is not None else "all genes"})',
                 fontsize=11, y=1.0)
    return fig


def upset_table():
    """Per category: isoforms and genes, for both gene sets."""
    keep = _R.GENE_SUBSET
    try:
        _R.GENE_SUBSET = None
        free()
        a, ag = upset_counts()
        _R.GENE_SUBSET = _R.load_select_genes()
        free()
        c, cg = upset_counts()
    finally:
        _R.GENE_SUBSET = keep
        free()

    keys = (['isoforms'] + UPSET_CATEGORIES
            + ['in a single category', 'in no category',
               'annotated, also a splicing-efficiency factor',
               'annotated as Figure 2 counts it',
               'efficiency factors Figure 2 excludes'])
    rows = ['**Isoform categories (full upset)**', '',
            '| | all genes: isoforms | genes | corrected: isoforms | genes |',
            '|---|---:|---:|---:|---:|']
    for k in keys:
        cells = []
        for n, g in ((a, ag), (c, cg)):
            v = n[k]
            pct = (f' ({100 * v / n["isoforms"]:.1f}%)' if k != 'isoforms' else '')
            cells.append(f'{v:,}{pct}')
            cells.append(f'{g[k]:,}' if k in g else '')
        rows.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    rows += ['', '> Gene counts do not sum to the gene total: one gene usually '
                 'has isoforms in several categories.']
    return '\n'.join(rows)


PANELS = {'upset_full': panel_upset_full}
