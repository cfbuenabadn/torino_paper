"""
Figure5_publication_K3_corrected_relaxed.py — the corrected K=3 figure on the
relaxed IR bed.

Same as Figure5_publication_K3_corrected, except panel C is built from
AD_IR_only_plots.relaxed.bed.gz: a torino exon only has to fall within the body
of the gene rather than within the span of the APPRIS-principal transcript.

Everything is drawn by Figure5_publication; this notebook only selects the
`k3_corrected_relaxed` version and reports what the relaxation buys.
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
import Figure5_plot_helpers as PH
import Figure5_publication as P

d = P.load_panels()
print('version:', P.VERSIONS['k3_corrected_relaxed'])

# %% CELL: relaxed
# Panel C, in two steps away from the strict bed.
#
# 1. Relaxed containment. The strict bed tests each torino exon with `f=1`
#    against the span of the APPRIS-principal transcript; when that transcript
#    is a short isoform the exons outside it are discarded and the gene can end
#    up with no IR interval at all. MRPL43 is the case: a 1.2 kb APPRIS span
#    against a 17.8 kb torino isoform. The relaxed bed contains the gene body
#    instead, 2,940 -> 3,225 genes.
#
# 2. Significance. Neither bed filtered on the logistic regression at all --
#    the gene set was an annotation filter, 3,908 genes carrying an IR-only
#    factor of which only 1,082 reach FDR 0.1. The bed this figure uses applies
#    that cut, and panel A's significance cut moves to 0.1 to match.
strict = d['ir_panel_k3_colocK3']
relaxed = d['ir_panel_k3_relaxed_colocK3']
drawn = d[P.VERSIONS['k3_corrected_relaxed']['ir']]
sym = H.load_k3_gene_symbols()

for tag, panel in [('strict        ', strict),
                   ('relaxed       ', relaxed),
                   ('relaxed + FDR ', drawn)]:
    names = sorted(panel['gene_names'].get(g, g) for g in panel['gene_label'])
    print(f'{tag} {len(panel["order_df"]):5d} genes | labelled {len(names)}: {names}')
print('\n^ the last row is what this figure draws')

# Panel A, at the same FDR.
b = d[P.VERSIONS['k3_corrected_relaxed']['beta']]
print(f"\npanel A at FDR {b['fdr']}:")
print(f"   No IR  {b['si_pos_ratio'] * 100:5.1f}% increasing in AD  (n={b['total_si']})")
print(f"   IR     {b['ir_pos_ratio'] * 100:5.1f}% increasing in AD  (n={b['total_ir']})")

# Of the 18 ir_ad_factor genes, which the heatmap can still show.
tbl = d['k3_coloc_table']
g18 = list(dict.fromkeys(tbl.loc[tbl.ir_ad_factor].gene_id))
absent = [sym.get(g, g) for g in g18 if g not in set(drawn['order_df'].gene)]
print(f'\nof the 18 colocalisation genes, {len(g18) - len(absent)} are labelled; '
      f'{len(absent)} are absent:')
print(f'   {absent}')

# %% CELL: figure
fig = P.save(P.build_figure(d, version='k3_corrected_relaxed'),
             'Figure5_publication_K3_corrected_relaxed')
print(f'page: {P.PAGE_W:.1f} x {P.PAGE_H:.1f} mm  ->  '
      f'Figure5_publication_K3_corrected_relaxed.png / .pdf / .svg')

# %% CELL: utr3
# Supplementary: UTR3_changes, rebuilt from the new 3' UTR logistic regression.
#
# The usage matrix is the terminal-exon grouping of AD_UTR.ipynb (factors merged
# by 3' UTR position, efficiency factors removed at isoform selection, loadings
# renormalised per sample to sum to 1 within each UTR start and then collapsed
# per length rank). What changed is the regression on top: it now uses the
# AD_K3_LR covariate set
#   tissue_id, batch, msex, race_string, spanish, age_group, log10PMI, RIN
# rather than the notebook's tissue_id/batch/msex/race.
#
# This panel is K=5 end to end -- snmf5_EL, the gao isoform annotation, and the
# genome-wide K=5 colocalisation table for the row labels. That is a different
# coloc file from panel C's gao_k3 colocboost summary.
# Genes enter at FDR <= 0.1, the cut panels A and C use. The wider covariate
# set costs power against the notebook's four covariates -- CISD1's leading UTR
# rank moves from FDR 0.026 to 0.090 on a near-identical beta (0.146 -> 0.140) --
# so 0.05 would have dropped genes for want of precision rather than effect.
utr3 = d['utr3_panel_newlr']
res = utr3['res']
print(f"source           : {utr3['source']}")
print(f"regression rows  : {len(res)}  "
      f"(FDR<=0.05: {int((res.pval_fdr <= 0.05).sum())}, "
      f"FDR<=0.1: {int((res.pval_fdr <= 0.1).sum())})")
print(f"gene threshold   : FDR <= {utr3['pval_thresh']}")
print(f"matrix           : {utr3['el_means'].shape[0]} genes x "
      f"{utr3['el_means'].shape[1]} condition-tissue groups")
print(f"labelled (K=5 coloc): {utr3['label_names']}")

_cisd1 = res.loc[res.index.str.startswith('ENSG00000122873')]
print('\nCISD1, the gene panels G/H show:')
print(_cisd1[['beta', 'pval', 'pval_fdr']].to_string(
    float_format=lambda v: f'{v:.4g}'))

g = PH.plot_utr3_changes(utr3)
for _ext in ('pdf', 'png', 'svg'):
    g.savefig(os.path.join(FIG5_DIR, f'UTR3_changes.{_ext}'),
              dpi=400 if _ext == 'png' else None, bbox_inches='tight')
print('\nsaved UTR3_changes.pdf / .png / .svg')

# %% CELL: stats
# The boxplot fits, printed rather than drawn: the two-line note on each panel
# needed 26% headroom and squashed the boxes, so F and H carry no annotation and
# these are added to the figure by hand.
print(P.report_regressions(fig).to_string(index=False))

fig


# %% CELL: utr3_numbers
# The 3' UTR counts, old covariates against new.
#
# "old" is the cached notebook regression (tissue, batch, sex, race) at FDR 0.05;
# "new" is scripts/prepare_ad_utr3_logreg.py (the AD_K3_LR covariate set) at the
# FDR 0.1 this figure uses. utr3_direction_summary does the counting: a gene
# contributes a direction only when its most significant UTR rank is the
# shortest or the longest, and a negative effective_beta is a shift toward the
# short 3' UTR.
import pickle
from IPython.display import Markdown, display

with open(os.path.join(FIG5_DIR, 'figure_data', 'utr_res.pkl'), 'rb') as fh:
    res_old = pickle.load(fh)
res_new = utr3['res']

counts = pd.DataFrame([
    H.utr3_direction_summary(res_old, 0.05, 'old: notebook covariates'),
    H.utr3_direction_summary(res_new, 0.10, 'new: AD_K3_LR covariates'),
]).set_index('analysis')
print(counts.T.to_string())

rows = [
    ('genes with >=2 polyA sites', 'genes_2plus_polyA', '{:,}'),
    ("3' UTR groups",              'utr_sites',         '{:,}'),
    ('significant sites',          'sig_sites',         '{:,}'),
    ('... in genes',               'sig_genes',         '{:,}'),
    ('genes with a usable shift',  'genes_with_shift',  '{:,}'),
]
md = ["| | old (notebook covariates, FDR 0.05) | new (AD_K3_LR covariates, FDR 0.1) |",
      "|---|---|---|"]
for name, key, fmt in rows:
    # the frame holds mixed types, so the counts come back as floats
    md.append(f"| {name} | {fmt.format(int(counts.iloc[0][key]))} | "
              f"{fmt.format(int(counts.iloc[1][key]))} |")
for name, n, pct in [("&rarr; short 3' UTR", 'short', 'short_pct'),
                     ("&rarr; long 3' UTR", 'long', 'long_pct')]:
    md.append(f"| {name} | {int(counts.iloc[0][n]):,} "
              f"(**{counts.iloc[0][pct]:.0f}%**) | "
              f"{int(counts.iloc[1][n]):,} (**{counts.iloc[1][pct]:.0f}%**) |")
utr3_counts_md = '\n'.join(md)
display(Markdown(utr3_counts_md))

# %% CELL: utr3_sweep
# Which covariate costs the asymmetry. Each row refits all 4,709 UTR groups, so
# this cell takes a couple of minutes; FDR 0.05 throughout, so the rows differ
# only in the adjustment.
BASE = ['tissue_id', 'batch', 'msex', 'race']
FULL = ['tissue_id', 'batch', 'msex', 'race_string', 'spanish',
        'age_group', 'log10PMI', 'RIN']

sweep = H.utr3_covariate_sweep({
    'notebook 4':             BASE,
    '+ RIN only':             BASE + ['RIN'],
    '+ everything except RIN': [c for c in FULL if c != 'RIN'],
    'all 8':                  FULL,
}, pval_thresh=0.05)
print(sweep[['analysis', 'n_covariates', 'sig_sites', 'genes_with_shift',
             'short', 'short_pct']].to_string(index=False))

md = ["| covariates | significant sites | short |", "|---|---|---|"]
for _, r in sweep.iterrows():
    md.append(f"| {r['analysis']} | {int(r['sig_sites']):,} | "
              f"{r['short_pct']:.0f}% |")
utr3_sweep_md = '\n'.join(md)
display(Markdown(utr3_sweep_md))

print('\nDirection is not what changes: on the genes significant in both, the '
      'shift direction agrees and the effect sizes correlate closely. What '
      'moves the split is which genes fall below significance.')
