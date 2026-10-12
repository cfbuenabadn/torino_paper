"""
SFigure4.py — Supplementary Figure 4: the corrected `colocs` plot.

Run from the analysis/ directory (the helpers use '../code/...' paths).
"""

# %% CELL: imports
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

SFIG4_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/SFigure4'
ANALYSIS_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/analysis'

os.chdir(ANALYSIS_DIR)
if SFIG4_DIR not in sys.path:
    sys.path.append(SFIG4_DIR)

from SFigure4_helpers import run_all, MIN_REGIONS
import SFigure4_plot_helpers as PH

matplotlib.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
})

data_dir = os.path.join(SFIG4_DIR, 'figure_data')

# %% CELL: load
# Everything is cheap (two table reads and one pass over 140k rows), so the
# pipeline is run rather than reloaded. To reload instead:
#   def L(n):
#       with open(os.path.join(data_dir, f'{n}.pkl'), 'rb') as f:
#           return pickle.load(f)
#   filtering_trace, trait_df, colocs_panel = L('filtering_trace'), L('trait_df'), L('colocs_panel')
data = run_all(data_dir)

filtering_trace = data['filtering_trace']
trait_df = data['trait_df']
colocs_panel = data['colocs_panel']

# %% CELL: trace
# Every filtering step from the GWAS trait pool that code/rules/common.py
# defines down to the traits and loci that reach the figure.
print(filtering_trace.to_string(index=False))
print()
print(f"distinct GWAS lead SNPs submitted                 : "
      f"{filtering_trace.attrs['lead_snps']}")
print(f"colocalized regions across the plotted traits     : "
      f"{filtering_trace.attrs['colocalized_regions']}")
print("   ^ the 1740 hard-coded over the 'All traits' bar in the exploratory"
      "\n     notebooks, and the number of events in the Figure 4 upset plot"
      "\n     (Figure4/figure_data/upset_data.pkl, which has exactly 1740 rows).")
print()
print(f"traits tested   : {colocs_panel['n_tested_traits']}  "
      f"(>= {MIN_REGIONS} tested regions; 3 of them colocalized nothing)")
print(f"traits plotted  : {colocs_panel['trait_selected_df'].shape[0]}  "
      f"(the 65 with at least one colocalized region get a bar)")
print(f"tested regions  : {colocs_panel['total_loci']}   "
      f"(denominator of the 'All traits' column, all 68 traits)")
print(f"colocalized     : {colocs_panel['total_coloc']}   "
      f"= {colocs_panel['total_percent']:.1f}%")

# %% CELL: figure
fig = PH.plot_colocs(colocs_panel)
for ext in ('pdf', 'png', 'svg'):
    kw = {'dpi': 500} if ext == 'png' else {}
    fig.savefig(os.path.join(SFIG4_DIR, f'SFigure4.{ext}'),
                bbox_inches='tight', **kw)
fig

# %% CELL: legend
# The colour key, as a figure of its own so it can be placed beside the panel by
# hand. The bands are the gaps between consecutive cumulative sums of the stack,
# so each colour means a single class, a combination, or no class at all --
# legend_entries() documents the arithmetic.
for _lab, _col in PH.legend_entries(colocs_panel, counts=True):
    print(f'   {_lab}')

fig_legend = PH.plot_legend(colocs_panel, counts=True,
                            title='Colocalized GWAS loci')
for _ext in ('pdf', 'png', 'svg'):
    fig_legend.savefig(os.path.join(SFIG4_DIR, f'SFigure4_legend.{_ext}'),
                       dpi=400 if _ext == 'png' else None, bbox_inches='tight')
print('\nsaved SFigure4_legend.pdf / .png / .svg')
fig_legend

# %% CELL: funnel_md
# The filtering funnel as markdown, ready to paste into the supplement.
from IPython.display import Markdown, display

funnel_md = filtering_trace.rename(columns={
    'step': 'Step', 'traits': 'Traits', 'GWAS_regions': 'GWAS regions',
    'dropped': 'What drops here'}).to_markdown(index=False)
funnel_md = (f"**Traits and GWAS regions at every filtering step**\n\n{funnel_md}\n\n"
             f"Colocalized regions across the {colocs_panel['n_tested_traits']} tested "
             f"traits: **{filtering_trace.attrs['colocalized_regions']}** "
             f"({colocs_panel['total_percent']:.1f}% of "
             f"{colocs_panel['total_loci']}), which is also the number of events "
             f"in the Figure 4 upset plot.")

with open(os.path.join(SFIG4_DIR, 'SFigure4_filtering_trace.md'), 'w') as fh:
    fh.write(funnel_md + '\n')
display(Markdown(funnel_md))

# %% CELL: trait_table
# Per-trait table over all 85 submitted traits, with a pooled Total row.
# `tested` is tested_regions >= MIN_REGIONS, the rule that keeps 68 traits.
trait_table = data['trait_table']

trait_table.to_csv(os.path.join(SFIG4_DIR, 'SFigure4_trait_table.tsv'),
                   sep='\t', index=False)
table_md = trait_table.rename(columns={
    'trait': 'Trait', 'accession': 'Accession',
    'total_leadSnps': 'Total leadSNPs', 'tested_regions': 'Tested regions',
    'tested': 'Tested', 'colocs': 'Colocs',
    'percent_coloc': '% colocalized'}).to_markdown(index=False)
with open(os.path.join(SFIG4_DIR, 'SFigure4_trait_table.md'), 'w') as fh:
    fh.write(table_md + '\n')

print(f'{len(trait_table) - 1} traits + Total row  ->  '
      f'SFigure4_trait_table.tsv / .md')
display(Markdown(table_md))
