"""
FigureS_bundle.py — gene x species bundles.

The FigureS_<GENE> figures are one gene across species; FigureS_genes.py is
several genes in one species.  This is the grid: rows are genes, columns are
species, so a 3' UTR contrast can be compared across both axes at once.

  SFigure_bundle_UTR3brain
      a  SH3GLB1, TSC22D2, TXNL1   — long brain 3' UTR in more than one species
      b  AGO1, CABLES1             — the long brain 3' UTR is human-only

Red is the long-3' UTR isoform preferred in brain; navy the shorter one
preferred in testis.  Both are resolved from the data per cell.

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

os.environ['FIGURES_IMPORT_ONLY'] = '1'       # suppress FigureS's render loop

import numpy as np
from matplotlib import pyplot as plt

import FigureS as FS
import Figure3_helpers as _F3H

PAGE_W  = FS.PAGE_W
OUT_DIR = os.path.join(FIG_DIR, 'supplementary_gene_figures')
os.makedirs(OUT_DIR, exist_ok=True)


# %% CELL: bundle definition
BUNDLES = {
    'UTR3brain': dict(
        species=['Human', 'Macaque', 'Mouse'],
        groups=[['SH3GLB1', 'TSC22D2', 'TXNL1'],
                ['AGO1', 'CABLES1']],
        # Heart is drawn as the control: the claim is that the long 3' UTR is
        # BRAIN-specific, which only reads if a second somatic tissue is there
        # to be compared against.
        tracks=['brain', 'heart', 'testis'],
        roles=['Brain', 'Heart', 'Testis'],
        title="Long 3′ UTR isoforms preferred in brain, short ones in testis",
    ),
}
PANEL_LETTERS = 'abcdefgh'


# %% CELL: data
def load_panels():
    out = {}
    for name in ('supp_panels', 'supp_panels_extra', 'supp_panels_genes'):
        path = os.path.join(FIG_DIR, 'figure_data', f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                out.update(pickle.load(fh))
    return out


# %% CELL: picks
def last_exon(panel, tid):
    """Length of the 3'-most exon, strand-aware."""
    r = panel['iso']
    r = r[r.transcript_id == tid]
    if not len(r):
        return -1
    row = r.loc[r.start.idxmin()] if panel['strand'] == '-' else r.loc[r.end.idxmax()]
    return int(row.end - row.start)


def _medians(panel, role):
    tis = _F3H.role_tissue(panel, role)
    ids = _F3H.merged_tissue_ids(panel)
    smp = ids.index[ids == tis] if tis else []
    EL = panel['el_annot']
    return np.array([
        float(np.median(EL.loc[smp, t].dropna()))
        if len(smp) and len(EL.loc[smp, t].dropna()) else 0.0
        for t in panel['iso_order']])


def utr_picks(panel, min_use=0.04, short_ratio=0.7):
    """
    Red  = the brain-preferred isoform with the longest 3' UTR among those the
           brain actually uses.
    Navy = the testis-preferred isoform with a SHORTER last exon than red.

    Restricting red to isoforms used in brain matters: several of these genes
    annotate an even longer 3' UTR on an isoform brain never uses, which would
    make the contrast look stronger than it is.
    """
    order = panel['iso_order']
    brain, testis = _medians(panel, 'Brain'), _medians(panel, 'Testis')
    utr = np.array([last_exon(panel, t) for t in order])

    used = [i for i in range(len(order)) if brain[i] >= min_use]
    if not used:
        used = [int(np.argmax(brain))]
    # Of the brain-used isoforms take the longest 3' UTR, but treat UTRs within
    # 10% as equivalent and prefer the one brain actually uses: mouse TSC22D2
    # has isoforms at 6715 and 6709 bp with brain loadings of 0.06 and 0.40,
    # and the 6 bp difference should not decide which is drawn.
    max_u = max(utr[i] for i in used)
    longest = [i for i in used if utr[i] >= 0.9 * max_u]
    red = max(longest, key=lambda i: brain[i])

    # "Shorter" has to mean meaningfully shorter.  Macaque TSC22D2 has isoforms
    # at 1562 and 1442 bp: picking the latter as the short form shows no
    # contrast at all, while the real short isoform sits at 671 bp.
    shorter = [i for i in range(len(order))
               if i != red and utr[i] < short_ratio * utr[red]]
    if not shorter:
        shorter = [i for i in range(len(order)) if i != red and utr[i] < utr[red]]
    if shorter:
        navy = max(shorter, key=lambda i: testis[i])
        no_short = False
    else:
        # No isoform has a shorter 3' UTR than the brain-used one, so this cell
        # cannot show the contrast.  Take the shortest available rather than the
        # most testis-enriched, which would otherwise draw a LONGER navy than
        # red and invert the figure's own key.
        others = [i for i in range(len(order)) if i != red]
        navy = min(others, key=lambda i: utr[i]) if others else red
        no_short = True
    return red, navy, dict(utr_red=int(utr[red]), utr_navy=int(utr[navy]),
                           brain_red=brain[red], testis_red=testis[red],
                           brain_navy=brain[navy], testis_navy=testis[navy],
                           no_short=no_short)


def apply_picks(key, panels):
    spec = BUNDLES[key]
    rows = []
    for g in [x for grp in spec['groups'] for x in grp]:
        for sp in spec['species']:
            if g not in panels or sp not in panels[g]:
                continue
            p = panels[g][sp]
            red, navy, info = utr_picks(p)
            _F3H.ISOFORM_PICKS[(g, sp)] = (red + 1, navy + 1)
            rows.append(dict(gene=g, species=sp, red=p['iso_order'][red],
                             navy=p['iso_order'][navy], **info))
    return rows


# %% CELL: page plan
TITLE_BAND   = 5.0
GENE_LABEL_H = 3.0
SPECIES_TITLE_H = 1.6
ROW_GAP      = 2.5
COL_GAP      = 4.0
KEY_PITCH    = 4.2
KEY_PAD      = 2.0

PAGE = {'h': 180.0}


def cell_w(n_species):
    return (FS.GENE_X1 - FS.GENE_X0 - (n_species - 1) * COL_GAP) / n_species


def cell_h(panel, spec, first):
    n_tracks = len([t for t in spec['tracks'] if panel['track_n'].get(t, 0) > 0])
    struct = n_tracks * FS.TRACK_H + FS.BLK_GAP + 2 * FS.ISO_H
    load = FS.load_blk_h(3) + (FS.ROLE_LABEL_H if first else 0)
    return max(struct, load) + FS.STAT_H


def plan(key, panels):
    spec = BUNDLES[key]
    genes = [g for grp in spec['groups'] for g in grp]
    heights = []
    for i, g in enumerate(genes):
        h = max(cell_h(panels[g][sp], spec, first=(i == 0))
                for sp in spec['species'] if sp in panels[g])
        heights.append(h)
    key_h = 2 * KEY_PITCH + KEY_PAD
    page_h = (FS.MARGIN_T + TITLE_BAND
              + sum(GENE_LABEL_H + SPECIES_TITLE_H + h + ROW_GAP for h in heights)
              + key_h + FS.MARGIN_B)
    return genes, heights, page_h, key_h


# %% CELL: draw
def draw_bundle(key, panels):
    spec = BUNDLES[key]
    genes, heights, page_h, key_h = plan(key, panels)
    letter_of = {grp[0]: PANEL_LETTERS[i] for i, grp in enumerate(spec['groups'])}

    FS.PAGE_H = page_h
    FS.MAX_ROLES = len(spec['roles'])
    FS._COMPOSITE_TITLES.clear()
    fig = plt.figure(figsize=(PAGE_W / 25.4, page_h / 25.4))
    fig.patch.set_facecolor('white')

    FS.fig_text(fig, PAGE_W / 2, FS.MARGIN_T, spec['title'],
                fontsize=FS.FS_TITLE, ha='center', va='top')

    n_sp = len(spec['species'])
    cw = cell_w(n_sp)
    load_w = FS.load_block_w(len(spec['roles']))
    struct_w = cw - FS.INNER_GAP - load_w

    y = FS.MARGIN_T + TITLE_BAND
    for i, g in enumerate(genes):
        first_row = (i == 0)
        if g in letter_of:
            FS.panel_letter(fig, letter_of[g], FS.MARGIN_L, y)
        FS.fig_text(fig, FS.MARGIN_L + 4.5, y, g, fontsize=FS.FS_TITLE,
                    fontstyle='italic', ha='left', va='top')

        y_top = y + GENE_LABEL_H + SPECIES_TITLE_H + FS.STAT_H
        for j, sp in enumerate(spec['species']):
            if sp not in panels[g]:
                continue
            p = panels[g][sp]
            cx = FS.GENE_X0 + j * (cw + COL_GAP)
            FS.fig_text(fig, cx + cw / 2, y + GENE_LABEL_H + SPECIES_TITLE_H - 0.3,
                        sp, fontsize=FS.FS_SUB, ha='center', va='bottom')

            tracks = [t for t in spec['tracks'] if p['track_n'].get(t, 0) > 0]
            FS.draw_gene_block(
                fig, p, cx, y_top, struct_w, n_iso=2,
                track_h=FS.TRACK_H, iso_h=FS.ISO_H, gap=FS.BLK_GAP,
                star_iso=None, label_species=False, label_tracks=True,
                tracks=tracks, gene=g)

            slots, ids = FS.build_slots(p, spec['roles'])
            FS.draw_loading_block(
                fig, p, cx + struct_w + FS.INNER_GAP + FS.SLOT_TICKS,
                y_top, load_w - FS.SLOT_TICKS, slots, ids,
                row_h=FS.LOAD_ROW_H, gap=FS.LOAD_GAP, swatch_h=FS.SWATCH_H,
                show_labels=(first_row and j == 0), gene=g,
                stats=p.get('ssmd_stats'),
                ylabel='Loadings' if (first_row and j == 0) else None,
                label_colors=[FS.ISO_RED, FS.ISO_MID, FS.ISO_NAVY, FS.ISO_GREY])

        y += GENE_LABEL_H + SPECIES_TITLE_H + heights[i] + ROW_GAP

    FS.draw_isoform_key(
        fig, PAGE_W / 2 - 26.0, y - ROW_GAP + KEY_PAD / 2, 52.0, key_h,
        items=[("Long 3′ UTR isoform", FS.ISO_RED),
               ("Short 3′ UTR isoform", FS.ISO_NAVY)], note=None)

    FS._place_composite_titles(fig)
    return fig


# %% CELL: render
panels = load_panels()

figs, PICKS = {}, {}
for _key in BUNDLES:
    PICKS[_key] = apply_picks(_key, panels)
    figs[_key] = draw_bundle(_key, panels)
    _stem = f'SFigure_bundle_{_key}'
    for _ext, _kw in (('png', dict(dpi=400)), ('pdf', {}), ('svg', {})):
        figs[_key].savefig(os.path.join(OUT_DIR, f'{_stem}.{_ext}'),
                           facecolor='white', **_kw)
    print(f'{_stem:28s} {PAGE_W:.0f} x '
          f'{figs[_key].get_size_inches()[1] * 25.4:.1f} mm -> {_stem}.png/.pdf/.svg')

# %% CELL: what the rule picked
for _key, _rows in PICKS.items():
    print(f'\n=== {_key} ===')
    print(f'  {"gene":9s} {"species":8s} {"red (long)":14s} {"UTR":>6s} '
          f'{"brain":>6s} {"testis":>6s}  {"navy (short)":14s} {"UTR":>6s} '
          f'{"brain":>6s} {"testis":>6s}')
    for _r in _rows:
        _flag = ''
        if _r['no_short']:
            _flag = '   <- no shorter-UTR isoform; cell shows no contrast'
        elif _r['utr_red'] < 1.5 * _r['utr_navy']:
            _flag = '   <- UTRs nearly equal'
        print(f'  {_r["gene"]:9s} {_r["species"]:8s} '
              f'{_r["red"].split(".")[1]:14s} {_r["utr_red"]:6d} '
              f'{_r["brain_red"]:6.2f} {_r["testis_red"]:6.2f}  '
              f'{_r["navy"].split(".")[1]:14s} {_r["utr_navy"]:6d} '
              f'{_r["brain_navy"]:6.2f} {_r["testis_navy"]:6.2f}{_flag}')
