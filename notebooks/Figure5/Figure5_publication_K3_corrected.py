"""
Figure5_publication_K3_corrected.py — the K=3 figure with panel C's gene labels
taken from the K=3 colocalisations.

Everything is drawn by Figure5_publication; this notebook only selects the
`k3_corrected` version and reports where its labels come from.
"""

# %% CELL: imports
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

FIG5_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure5'
ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'
os.chdir(ANALYSIS_DIR)
if FIG5_DIR not in sys.path:
    sys.path.append(FIG5_DIR)

import Figure5_helpers as H
import Figure5_publication as P

d = P.load_panels()
print('version:', P.VERSIONS['k3_corrected'])

# %% CELL: labels
# Where panel C's labels come from now.
#
# Before: `iso_coloc_genes` — the 65 genes with a colocalised torino sQTL factor
# in the **genome-wide K=5** table, intersected with the heatmap, giving 9
# labels. The heatmap itself is K=3, so the two did not match.
#
# Now: the gao_k3 colocboost summary with the feature flags from
# Fig5_numbers.ipynb (the cell marked `### THIS TABLE`), reproduced in
# H.build_k3_coloc_table(). Labelled = gene_id with ir_ad_factor True; red =
# those also with eQTL True. Ensembl ids are translated through the gao_k3
# annotation merged with genes.tab.gz.
tbl = d['k3_coloc_table']
sel = tbl.loc[tbl.ir_ad_factor]
sym = H.load_k3_gene_symbols()
genes = list(dict.fromkeys(sel.gene_id))
red = set(sel.loc[sel.eQTL].gene_id)

print(f'ir_ad_factor genes : {len(genes)}   (red, i.e. eQTL too: {len(red)})')
print(pd.DataFrame({'gene': [sym.get(g, g) for g in genes],
                    'gene_id': genes,
                    'eQTL (red)': [g in red for g in genes]}).to_string(index=False))

# How many of them the heatmap can actually show: panel C is the IR-only
# progression bed, so a gene without an IR-only factor has no row to label.
in_panel = set(d['ir_panel_k3_colocK3']['order_df'].gene)
drawn = [g for g in genes if g in in_panel]
print(f'\nrows present in panel C : {len(drawn)} of {len(genes)}  '
      f'-> {[sym.get(g, g) for g in drawn]}')
print(f'absent (no IR-only factor): '
      f'{[sym.get(g, g) for g in genes if g not in in_panel]}')

# %% CELL: figure
fig = P.save(P.build_figure(d, version='k3_corrected'),
             'Figure5_publication_K3_corrected')
print(f'page: {P.PAGE_W:.1f} x {P.PAGE_H:.1f} mm  ->  '
      f'Figure5_publication_K3_corrected.png / .pdf / .svg')
fig
