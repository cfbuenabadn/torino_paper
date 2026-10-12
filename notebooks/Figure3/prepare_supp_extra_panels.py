"""
prepare_supp_human_panels.py — coverage/structure/loadings panels for the
additional cross-species example genes (Figure3_helpers.SUPP_EXTRA_FIGURES).

Same machinery as the conserved supplementary examples, written to its own
pickle so supp_panels.pkl is left alone.

Run from torino_paper/analysis/ — the SSMD tables are addressed relatively.
"""

import os
import pickle
import sys

FIG_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure3'
ANALYSIS = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
for _p in (FIG_DIR, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Figure3_helpers as F

OUT = os.path.join(FIG_DIR, 'figure_data', 'supp_panels_extra.pkl')


def main():
    genes = list(F.SUPP_EXTRA_FIGURES)
    print(f'{len(genes)} genes: {", ".join(genes)}', flush=True)

    print('building the orthology table ...', flush=True)
    ortho_table, _ = F.build_ortho2()
    print('loading metadata and annotation ...', flush=True)
    meta = F.load_metadata()
    print('loading SSMD results ...', flush=True)
    species_res = F.load_ssmd_results()

    panels = F.prepare_supp_panels(
        genes=genes,
        ortho_table=ortho_table,
        species_metadata=meta['species_metadata'],
        species_annotation=meta['species_annotation'],
        species_res=species_res)

    with open(OUT, 'wb') as fh:
        pickle.dump(panels, fh)
    total = sum(len(v) for v in panels.values())
    print(f'\nwrote {OUT}  ({total} panels across {len(panels)} genes, '
          f'{os.path.getsize(OUT) / 1e6:.0f} MB)', flush=True)
    print('PREPDONE', flush=True)


if __name__ == '__main__':
    main()
