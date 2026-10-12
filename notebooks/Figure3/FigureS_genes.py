"""
FigureS_genes.py — multi-gene, human-only supplementary figures.

The FigureS_<GENE> figures put one SPECIES per cell for a single gene.  These
transpose that: one GENE per cell, all in human, so several examples can be
compared side by side in the species where the coverage is deepest.

  SFigure_genes_brain        CLTB, ENPP2, REEP6
  SFigure_genes_EPN1_SERP2   EPN1, SERP2

Every drawing primitive is imported from FigureS.py rather than copied, so the
blocks are literally the same code that draws the cross-species figures; only
the cell loop differs.  Compositions live in Figure3_helpers.MULTIGENE_FIGURES.

Run from torino_paper/analysis/.
"""

# %% CELL: imports
import os
import pickle
import sys

FIG_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure3'
for _p in (FIG_DIR, '/project2/yangili1/cfbuenabadn/torino_paper/analysis'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must be set before FigureS is imported: it suppresses that module's own
# render loop, which would otherwise redraw all 27 cross-species figures.
os.environ['FIGURES_IMPORT_ONLY'] = '1'

import numpy as np
from matplotlib import pyplot as plt

import FigureS as FS
import Figure3_helpers as _F3H
from Figure3_helpers import (MULTIGENE_FIGURES, MULTIGENE_PICKS,
                             SUPP_GENE_FIGURES)

PAGE_W = FS.PAGE_W

# Rendered figures go in their own directory: at 31 genes x 3 formats they
# otherwise bury the scripts and pickles in this folder.
OUT_DIR = os.path.join(FIG_DIR, 'supplementary_gene_figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Isoform picks for these figures only
#
# Navy is the isoform with the highest median loading in HEART rather than the
# second-ranked isoform in the gene's own claim tissue.  `resolve_picks`
# excludes the red isoform when resolving 'max_heart', so when the top-heart
# isoform IS the red one it falls through to the next highest — the rule asked
# for, already implemented for PHC1/Mouse.
#
# The override is applied to the shared table, the same way FigureS.py
# overrides VRK3: each notebook runs in its own kernel, so the cross-species
# FigureS_<GENE> figures keep the picks they were published with.
#
# The 'max_heart' sentinel itself cannot be used here: it matches the tissue id
# literally, and human panels carry GTEx ids (Heart_Atrial_Appendage,
# Heart_Left_Ventricle) rather than a bare 'Heart', so it resolves to None.
# The index is therefore computed explicitly, on the SAME tissue the Heart
# loadings column displays — ROLE_TISSUE['Human']['Heart'] — so a reader can
# look at that column and see the navy isoform is the highest one in it.
MULTIGENE_GENES = sorted({g for spec in MULTIGENE_FIGURES.values()
                          for g in spec['genes']})


def role_medians(panel, role):
    """Median loading per iso_order entry in `role`'s tissue for this species."""
    tissue = _F3H.role_tissue(panel, role)
    ids = _F3H.merged_tissue_ids(panel)
    samples = ids.index[ids == tissue]
    EL = panel['el_annot']
    meds = [float(np.median(EL.loc[samples, f].dropna()))
            if len(EL.loc[samples, f].dropna()) else -np.inf
            for f in panel['iso_order']]
    return np.array(meds), tissue, len(samples)


def last_exon_len(panel, transcript_id):
    """
    Length of the 3'-most exon: lowest coordinate on the minus strand, highest
    on the plus strand.  Defined by coordinate rather than by the exon_N label,
    which is not reliably in transcription order.
    """
    iso = panel['iso']
    r = iso[iso.transcript_id == transcript_id]
    if not len(r):
        return -1
    row = r.loc[r.start.idxmin()] if panel['strand'] == '-' else r.loc[r.end.idxmax()]
    return int(row.end - row.start)


def unique_exons(panel, tid_a, tid_b):
    """
    Exons of `tid_a` with no overlap at all in `tid_b` — i.e. cassette exons
    rather than exons whose boundaries merely shift.
    """
    iso = panel['iso']
    a = iso[iso.transcript_id == tid_a][['start', 'end']].values
    b = iso[iso.transcript_id == tid_b][['start', 'end']].values
    return [(s, e) for s, e in a
            if not any(s < e2 and s2 < e for s2, e2 in b)]


def _resolve(rule, panel, exclude=()):
    """'first' -> iso_order[0]; 'max_<role>' -> highest median loading there."""
    if rule == 'first':
        return 0
    role = rule[4:].capitalize()
    meds, _, _ = role_medians(panel, role)
    return int(next(i for i in np.argsort(-meds) if i not in exclude))


def apply_picks(key, panels):
    """
    Resolve red and navy for one figure and write them into the shared
    ISOFORM_PICKS table.

    Navy excludes red, so the rule falls through to the next highest when the
    top isoform in the navy tissue is already the red one — the behaviour asked
    for, and the same one `resolve_picks` applies to its 'max_<role>' sentinel.

    The override is applied at run time the way FigureS.py overrides VRK3: each
    notebook has its own kernel, so the cross-species figures keep their own
    picks.  It must also run AFTER `import FigureS`, whose module body sets
    'max_heart' for every gene in its own list — CLTB and SH3GLB1 are in both.
    """
    spec = MULTIGENE_FIGURES[key]
    red_rule  = spec.get('red',  MULTIGENE_PICKS['red'])
    navy_rule = spec.get('navy', MULTIGENE_PICKS['navy'])
    per_gene  = spec.get('navy_by_gene', {})

    rows = []
    for gene in spec['genes']:
        panel = panels[gene]['Human']
        this_navy_rule = per_gene.get(gene, navy_rule)

        a = _resolve(red_rule, panel)
        b = _resolve(this_navy_rule, panel, exclude=(a,))
        navy_role = this_navy_rule[4:].capitalize()
        meds, tissue, n = role_medians(panel, navy_role)
        top = int(np.argmax(meds))
        swapped = False

        if spec.get('order') == 'cassette_exon':
            # Red is the isoform carrying an exon the other skips entirely.
            ta, tb = panel['iso_order'][a], panel['iso_order'][b]
            na = len(unique_exons(panel, ta, tb))
            nb = len(unique_exons(panel, tb, ta))
            if nb > na:
                a, b, swapped = b, a, True

        if spec.get('order') == 'long_last_exon':
            # Expression chooses WHICH two isoforms; last-exon length decides
            # which of them is red, so red is always the long-3' UTR one.
            la, lb = (last_exon_len(panel, panel['iso_order'][i]) for i in (a, b))
            if lb > la:
                a, b, swapped = b, a, True
        red, navy = a, b

        _F3H.ISOFORM_PICKS[(gene, 'Human')] = (red + 1, navy + 1)
        rows.append(dict(gene=gene, iso=panel['iso_order'], red=red, navy=navy,
                         top=top, fell_through=(top == red), tissue=tissue, n=n,
                         meds=meds, red_rule=red_rule, navy_rule=this_navy_rule,
                         swapped=swapped,
                         red_len=last_exon_len(panel, panel['iso_order'][red]),
                         navy_len=last_exon_len(panel, panel['iso_order'][navy])))
    return rows


# %% CELL: data
def load_gene_panels():
    """
    The cross-species pickles plus the human-only one.  CLTB comes from
    supp_panels.pkl so this figure and FigureS_CLTB draw the same isoforms.
    """
    panels = dict(FS.load_figure_data()['panels'])
    path = os.path.join(FIG_DIR, 'figure_data', 'supp_panels_genes.pkl')
    with open(path, 'rb') as fh:
        panels.update(pickle.load(fh))
    return panels


# %% CELL: page plan
# One gene per row, each spanning the full content width — the whole point of
# dropping the species axis is that the structure block gets twice the width it
# has in the cross-species figures.
TITLE_BAND  = 5.0        # figure title
GENE_LABEL_H = 3.0       # panel letter + italic gene name
ROW_GAP     = 4.0

# The key is centred inside its band, so a band shorter than the items needs
# overflows UPWARD into the last gene's isoform rows.  Derive it instead.
KEY_PITCH   = 4.2        # must match draw_isoform_key's default
KEY_PAD     = 2.0


# What red and navy actually are, given the rule that chose them.  The inherited
# "Tissue-specific / Non-specific" wording stopped being true once navy became
# the top-heart isoform rather than simply a second, unremarkable one.
RULE_LABEL = {
    'first':          'Tissue-specific isoform',
    'max_brain':      'Top brain isoform',
    'max_heart':      'Top heart isoform',
    'max_cerebellum': 'Top cerebellum isoform',
}

# When the pair is ordered by last-exon length the colours mean something else
# entirely, and the key has to say so rather than name the tissue that merely
# chose which two isoforms to draw.
ORDER_LABEL = {
    'long_last_exon': ("Long 3\u2032 UTR isoform", "Short 3\u2032 UTR isoform"),
    'cassette_exon':  ('Exon-included isoform', 'Exon-skipped isoform'),
}


def key_items(genes, panels, red_rule='first', navy_rule='max_heart',
              order=None):
    """Key entries: the grey row only when some gene has unpicked isoforms."""
    if order in ORDER_LABEL:
        red_lab, navy_lab = ORDER_LABEL[order]
    else:
        red_lab, navy_lab = RULE_LABEL[red_rule], RULE_LABEL[navy_rule]
    items = [(red_lab, FS.ISO_RED), (navy_lab, FS.ISO_NAVY)]
    if any(len(panels[g]['Human']['iso_order'])
           > FS._n_iso_for(g, 'Human', panels[g]['Human']) for g in genes):
        items.append(('Other isoforms', FS.ISO_GREY))
    return items


def key_band_h(items):
    return len(items) * KEY_PITCH + KEY_PAD

PANEL_LETTERS = 'abcdefgh'


def _row_height(panel, cfg, first):
    """Block height for one gene row, mirroring FigureS._row_heights."""
    n_iso = FS._n_iso_for(cfg['gene'], 'Human', panel)
    struct = FS._blk_h(panel, n_iso, len(FS._tracks_for(panel, cfg)))
    load = FS.load_blk_h(n_iso + 1) + (FS.ROLE_LABEL_H if first else 0)
    return max(struct, load) + FS.STAT_H


def _rules(key):
    spec = MULTIGENE_FIGURES[key]
    return (spec.get('red', MULTIGENE_PICKS['red']),
            spec.get('navy', MULTIGENE_PICKS['navy']),
            spec.get('order'))


def row_gap(key):
    """Per-figure row gap; six gene rows at the default overshoot a page."""
    return MULTIGENE_FIGURES[key].get('row_gap', ROW_GAP)


def plan(genes, panels, key):
    """Per-gene configs and row heights, plus the derived page height."""
    cfgs, heights = [], []
    for i, gene in enumerate(genes):
        cfg = FS.gene_config(gene)
        cfg['gene'] = gene
        cfgs.append(cfg)
        heights.append(_row_height(panels[gene]['Human'], cfg, first=(i == 0)))

    page_h = (FS.MARGIN_T + TITLE_BAND
              + sum(GENE_LABEL_H + h + row_gap(key) for h in heights)
              + key_band_h(key_items(genes, panels, *_rules(key)))
              + FS.MARGIN_B)
    return cfgs, heights, page_h


# %% CELL: draw one figure
def draw_multigene_figure(key, panels):
    """One figure: a stack of single-gene, human-only cells."""
    spec = MULTIGENE_FIGURES[key]
    genes = spec['genes']
    groups = spec.get('groups') or [[g] for g in genes]
    letter_of = {grp[0]: PANEL_LETTERS[i] for i, grp in enumerate(groups)}
    cfgs, heights, page_h = plan(genes, panels, key)

    # FigureS's primitives read these module globals for their geometry.
    FS.PAGE_H = page_h
    FS.MAX_ROLES = max(len(c['roles']) for c in cfgs)
    FS._COMPOSITE_TITLES.clear()

    fig = plt.figure(figsize=(PAGE_W / 25.4, page_h / 25.4))
    fig.patch.set_facecolor('white')

    FS.fig_text(fig, PAGE_W / 2, FS.MARGIN_T, spec['title'],
                fontsize=FS.FS_TITLE, ha='center', va='top')

    cell_w = FS.GENE_X1 - FS.GENE_X0
    load_w = FS.load_block_w(FS.MAX_ROLES)
    struct_w = cell_w - FS.INNER_GAP - load_w

    y = FS.MARGIN_T + TITLE_BAND
    for i, (gene, cfg, row_h) in enumerate(zip(genes, cfgs, heights)):
        panel = panels[gene]['Human']
        first = (i == 0)
        n_iso = FS._n_iso_for(gene, 'Human', panel)

        # A letter marks the start of each group; a figure with no groups
        # declared falls back to one letter per gene, as before.
        if gene in letter_of:
            FS.panel_letter(fig, letter_of[gene], FS.MARGIN_L, y)
        FS.fig_text(fig, FS.MARGIN_L + 4.5, y, gene, fontsize=FS.FS_TITLE,
                    fontstyle='italic', ha='left', va='top')
        # The claim tissue sets the isoform ranking, so it is named, not implied.
        FS.fig_text(fig, PAGE_W - FS.MARGIN_R, y,
                    f'{cfg["roles"][0] if False else panel["tissue"]}-ranked',
                    fontsize=FS.FS_TICK, color='0.45', ha='right', va='top')

        y_top = y + GENE_LABEL_H + FS.STAT_H
        FS.draw_gene_block(
            fig, panel, FS.GENE_X0, y_top, struct_w, n_iso=n_iso,
            track_h=FS.TRACK_H, iso_h=FS.ISO_H, gap=FS.BLK_GAP,
            star_iso=cfg['star_iso'], star_end=cfg.get('star_end', 'last'),
            label_species=False, label_tracks=True,
            tracks=FS._tracks_for(panel, cfg), gene=gene)

        slots, ids = FS.build_slots(panel, cfg['roles'])
        FS.draw_loading_block(
            fig, panel, FS.GENE_X0 + struct_w + FS.INNER_GAP + FS.SLOT_TICKS,
            y_top, load_w - FS.SLOT_TICKS, slots, ids,
            row_h=FS.LOAD_ROW_H, gap=FS.LOAD_GAP, swatch_h=FS.SWATCH_H,
            show_labels=first, gene=gene, stats=panel.get('ssmd_stats'),
            ylabel='Loadings' if first else None,
            # The cross-species figures label the third loadings row in place,
            # where the label runs across its own data — the loadings block is
            # only ~21 mm wide.  Here the grey row is named in the key instead.
            row_labels=None,
            label_colors=[FS.ISO_RED, FS.ISO_MID, FS.ISO_NAVY, FS.ISO_GREY])

        y += GENE_LABEL_H + row_h + row_gap(key)

    # One key for the page, centred under the last gene.
    items = key_items(genes, panels, *_rules(key))
    FS.draw_isoform_key(fig, PAGE_W / 2 - 24.0, y - row_gap(key) + KEY_PAD / 2,
                        48.0, key_band_h(items), items=items, note=None)

    FS._place_composite_titles(fig)
    return fig


# %% CELL: render
panels = load_gene_panels()

# Picks are per figure, so they are applied immediately before that figure is
# drawn: a gene in two figures under different rules would otherwise be drawn
# with whichever rule was written last.
PICKS = {}

figs = {}
for _key in MULTIGENE_FIGURES:
    _missing = [g for g in MULTIGENE_FIGURES[_key]['genes']
                if g not in panels or 'Human' not in panels.get(g, {})]
    if _missing:
        print(f'{_key:12s} SKIPPED — no human panel for {", ".join(_missing)}')
        continue
    PICKS[_key] = apply_picks(_key, panels)
    figs[_key] = draw_multigene_figure(_key, panels)
    _stem = f'SFigure_genes_{_key}'
    for _ext, _kw in (('png', dict(dpi=400)), ('pdf', {}), ('svg', {})):
        figs[_key].savefig(os.path.join(OUT_DIR, f'{_stem}.{_ext}'),
                           facecolor='white', **_kw)
    print(f'{_stem:28s} {", ".join(MULTIGENE_FIGURES[_key]["genes"]):24s} '
          f'{PAGE_W:.0f} x {figs[_key].get_size_inches()[1] * 25.4:.1f} mm '
          f'-> {_stem}.png/.pdf/.svg')
print(f'\n{len(figs)} figures written')


# %% CELL: which isoforms the rules picked
# Printed so the choice is checkable: the red isoform, the top isoform in the
# navy tissue BEFORE exclusion, and the navy actually drawn.  The last two
# differ exactly when the rule fell through because the top one was red.
for _key, _rows in PICKS.items():
    _spec = MULTIGENE_FIGURES[_key]
    print(f'\n=== {_key} ===')
    print(f'  red = {_rows[0]["red_rule"]}, navy = {_rows[0]["navy_rule"]} '
          f'(ranked on {_rows[0]["tissue"]}, n = {_rows[0]["n"]})')
    print(f'  {"gene":8s} {"red":22s} {"top in navy tissue":22s} '
          f'{"navy drawn":22s} fell through?')
    for _r in _rows:
        _o = _r['iso']
        print(f'  {_r["gene"]:8s} {_o[_r["red"]]:22s} {_o[_r["top"]]:22s} '
              f'{_o[_r["navy"]]:22s} {"yes" if _r["fell_through"] else ""}')

print('\nmedian loading in the navy tissue, per isoform:')
for _key, _rows in PICKS.items():
    for _r in _rows:
        print(f'  {_r["gene"]:8s} ' + '  '.join(
            f'{_o}={_m:.3f}' for _o, _m in zip(_r['iso'], _r['meds'])))
