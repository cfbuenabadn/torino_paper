"""
prepare_supp_gene_panels.py — human-only panels for the multi-gene figures
(Figure3_helpers.SUPP_GENE_FIGURES).

These figures put one GENE per cell instead of one species, so only the human
panel is needed and the prep is a fraction of the cost of the cross-species
sets.  CLTB is not prepared here: it already has a human panel in
supp_panels.pkl, ranked on Brain, and reusing it keeps the same isoforms in
this figure and in FigureS_CLTB.

Run from torino_paper/analysis/.
"""

import os
import pickle
import sys

FIG_DIR  = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure3'
ANALYSIS = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
for _p in (FIG_DIR, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Figure3_helpers as F

OUT = os.path.join(FIG_DIR, 'figure_data', 'supp_panels_genes.pkl')


def main():
    genes = list(F.SUPP_GENE_FIGURES)
    print(f'{len(genes)} genes, human only: {", ".join(genes)}', flush=True)

    meta = F.load_metadata()
    species_res = F.load_ssmd_results(['Human'])

    out = {}
    for gene in genes:
        cfg = F.SUPP_GENE_FIGURES[gene]
        panel = F.prepare_gene_panel(
            'Human', gene, cfg['tissue'],
            meta['species_metadata']['Human'],
            meta['species_annotation']['Human'],
            species_res['Human'], tissue_keys=F.SUPP_TRACKS)
        if panel is None:
            print(f'  [skip] {gene}: no panel')
            continue
        panel['show_tracks'] = [t for t in F.SUPP_TRACKS
                                if panel['track_n'].get(t, 0) > 0]
        out[gene] = {'Human': panel}

    # CLTB's panel (reused from supp_panels.pkl) carries SSMD stats and so
    # draws significance stars; without this the new genes would sit in the
    # same figure with no stars at all, which reads as "not significant".
    F.attach_ssmd_stats(out)

    for gene in list(out):
        cfg = F.SUPP_GENE_FIGURES[gene]
        panel = out[gene]['Human']
        print(f'{gene:7s} tissue={cfg["tissue"]:11s} '
              f'tracks={panel["show_tracks"]}  '
              f'stats={sorted(panel.get("ssmd_stats", {}))}', flush=True)

    with open(OUT, 'wb') as fh:
        pickle.dump(out, fh)
    print(f'\nwrote {OUT}  ({len(out)} genes, '
          f'{os.path.getsize(OUT) / 1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    main()
