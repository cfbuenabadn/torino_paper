"""
Figure5_examples_nb.py — the source of Figure5_examples.ipynb.

Kept on disk so the notebook can always be rebuilt from a known-good state
with nb_build.py (--tail "" --run --kernel your_env_name).

No pickles are read or written and no figure is saved: every cell goes back to
the RDS on disk, so a different gene, variant, tissue, factorisation or RDS
tree is just a different argument.
"""

# %% CELL: imports
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FIG5_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure5'
if FIG5_DIR not in sys.path:
    sys.path.append(FIG5_DIR)

import Figure5_examples as EX

print('coverage RDS :', EX.COV_RDS)
print('boxplot  RDS :', EX.BOX_RDS)
print('factors  RDS :', EX.GAO_RDS_ALL)
print('isoform bed  :', EX.ISOFORM_BED)

# %% CELL: available
# What can be plotted. Every (gene, variant, tissue) below has both a coverage
# and a boxplot RDS, so any row works for a locus panel and its boxplot.
available = EX.list_available(kind='coverage')
print(f'{len(available)} coverage RDS over {available.gene.nunique()} genes, '
      f'{available.tissue.nunique()} tissues')
print(available.groupby(['gene', 'tissue']).size().unstack(fill_value=0))

# The isoform models of a gene, and which factor each one carries — this is how
# to choose `transcripts` and `factor_list` for a gene you have not plotted yet.
EX.gene_isoforms('ENSG00000103550')

# %% CELL: panel_E
# Panel E — KNOP1 in PCC, built the long way so every piece is visible.
KNOP1 = 'ENSG00000103550'
KNOP1_SNP = 'chr16:19708463:CTCA:C'

knop1_cov = EX.load_coverage(KNOP1, KNOP1_SNP, 'PCC')
knop1_fac = EX.load_factors(KNOP1, k=5)

knop1_spec = EX.locus_spec(
    KNOP1, name='KNOP1',
    # only the isoforms carrying the two factors drawn above them
    transcripts=[f'{KNOP1}.isoform_1', f'{KNOP1}.isoform_2'],
    # the AD event: the retained intron, drawn in red
    highlight={f'{KNOP1}.isoform_2': (19706479, 19710586)},
    factor_list=['factor1', 'factor2'],
    factor_labels=['Factor 1', 'Factor 2'],
    colors=['tab:blue', 'tab:orange'])

EX.plot_locus(knop1_cov, knop1_fac, knop1_spec,
              title='AD-colocalized variant increases intron retention in '
                    '$\\it{KNOP1}$',
              snp_label=f'SNP {KNOP1_SNP.rsplit(":", 2)[0]}')
plt.show()

# %% CELL: panel_F
# Panel F — the same variant's loadings by genotype. `order` fixes the allele
# dosage direction; the colour matches factor 2's track in panel E.
knop1_box = EX.load_boxplot(KNOP1, KNOP1_SNP, 'PCC')
print('factor columns:', EX.factor_columns(knop1_box))

# regression=True fits factor ~ allele dosage and writes beta and p into the
# corner. Dosage counts steps along `order`, so here it is per C allele.
EX.plot_genotype_boxplot(
    knop1_box, f'{KNOP1}.factor_2', order=['CTCA/CTCA', 'CTCA/C', 'C/C'],
    color='tab:orange', regression=True,
    ylabel='$\\it{KNOP1}$ factor 2\nnormalized loadings\n(PCC)',
    xlabel='SNP chr16:19708463')
plt.show()

# %% CELL: panel_G
# Panel G — CISD1 in AC. Same three calls, different arguments.
CISD1 = 'ENSG00000122873'
CISD1_SNP = 'chr10:58286894:G:A'

cisd1_cov = EX.load_coverage(CISD1, CISD1_SNP, 'AC')
cisd1_fac = EX.load_factors(CISD1, k=5)

cisd1_spec = EX.locus_spec(
    CISD1, name='CISD1',
    transcripts=[f'{CISD1}.isoform_2', f'{CISD1}.isoform_4'],
    # the long 3' exon the risk allele shortens
    highlight={f'{CISD1}.isoform_2': (58287561, 58289682)},
    factor_list=['factor4', 'factor5'],
    factor_labels=['Factor 4', 'Factor 5'],
    colors=['tab:blue', 'tab:orange'])

EX.plot_locus(cisd1_cov, cisd1_fac, cisd1_spec,
              title="AD-colocalized variant is associated with shortened 3' "
                    "exon in $\\it{CISD1}$",
              snp_label='SNP chr10:58286894', snp_ha='right')
plt.show()

# %% CELL: panel_H
cisd1_box = EX.load_boxplot(CISD1, CISD1_SNP, 'AC')

EX.plot_genotype_boxplot(
    cisd1_box, f'{CISD1}.factor_4', order=['G/G', 'G/A', 'A/A'],
    color='tab:blue', regression=True, reg_loc='lower left',
    ylabel='$\\it{CISD1}$ factor 4\nnormalized loadings\n(AC)',
    xlabel='SNP chr10:58286894')
plt.show()

# %% CELL: swap
# A different gene-variant-tissue, with nothing else changed.
#
# With no `spec` at all, plot_locus draws every isoform of the gene under every
# factor of the model — the right starting point for a locus you have not
# looked at before. Narrow it with locus_spec() once you know which isoform and
# which factor carry the event.
GENE, SNP, TISSUE = 'ENSG00000103550', 'chr16:19710433:C:T', 'DLPFC'

cov = EX.load_coverage(GENE, SNP, TISSUE)
fac = EX.load_factors(GENE, k=5)

EX.plot_locus(cov, fac, title=f'{GENE} — {SNP} in {TISSUE} (all isoforms, all factors)',
              snp_label=f'SNP {SNP.rsplit(":", 2)[0]}',
              figsize=(9, 6), track_h=0.22)
plt.show()

box = EX.load_boxplot(GENE, SNP, TISSUE)
EX.plot_genotype_boxplot(box, f'{GENE}.factor_2', order=['C/C', 'C/T', 'T/T'],
                         color='tab:green', xlabel=f'SNP {SNP.rsplit(":", 2)[0]}')
plt.show()

# %% CELL: rds_path
# Any coverage_matrices RDS can be given directly with `rds_path`, for a file
# that does not follow the {gene}.{variant}.{tissue} naming. `gene` is still
# needed, to pick the isoform models and the strand.
path = EX.coverage_rds_path(CISD1, CISD1_SNP, 'PCC')
print(path, '->', os.path.exists(path))

cov_direct = EX.load_coverage(CISD1, rds_path=path)
EX.plot_locus(cov_direct, cisd1_fac, cisd1_spec,
              title=f'{CISD1} loaded straight from a path, in PCC')
plt.show()

# %% CELL: shortcut
# The four published panels, with their stored titles, colours and orders.
# Any keyword overrides one setting, so this is the same code path as above.
EX.plot_panel('E')
plt.show()

EX.plot_panel('F', order=['C/C', 'CTCA/C', 'CTCA/CTCA'])   # axis flipped
plt.show()

# %% CELL: roots
# Switching the whole RDS tree, not just the gene.
#
# `root=` takes a key of EX.RDS_ROOTS or any path; `group=` reaches into the
# extra AD/ nonAD level the *_by_AD trees carry. Nothing else about the call
# changes.
for k, v in EX.RDS_ROOTS.items():
    print(f'  {k:24s} {v}')

# Which trees actually hold this gene-variant-tissue.
print()
print(EX.compare_roots(KNOP1, KNOP1_SNP, 'PCC', kind='boxplot')[
    ['root', 'group', 'exists']].to_string(index=False))

# %% CELL: normalisation
# The same panel F under the two K=3 normalisations.
#
# These trees are the K=3 rerun: their boxplots carry factor_1..factor_3 plus an
# `AD` column, which factor_columns() knows to leave out.
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), constrained_layout=True)
for ax, root in zip(axes, ['k3_unnormalized', 'k3_qqnorm']):
    df = EX.load_boxplot(KNOP1, KNOP1_SNP, 'PCC', root=root)
    EX.plot_genotype_boxplot(df, f'{KNOP1}.factor_2',
                             order=['CTCA/CTCA', 'CTCA/C', 'C/C'],
                             color='tab:orange', ax=ax,
                             ylabel=f'{root}\nKNOP1 factor 2',
                             xlabel='SNP chr16:19708463')
plt.show()

# %% CELL: by_AD
# AD vs non-AD, for the same variant. Same call, one extra argument.
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), constrained_layout=True)
for ax, grp in zip(axes, ['AD', 'nonAD']):
    df = EX.load_boxplot(KNOP1, KNOP1_SNP, 'DLPFC', root='k3_qqnorm_by_AD', group=grp)
    EX.plot_genotype_boxplot(df, f'{KNOP1}.factor_2',
                             order=['CTCA/CTCA', 'CTCA/C', 'C/C'],
                             color='tab:red' if grp == 'AD' else 'tab:gray', ax=ax,
                             ylabel=f'{grp} (n={len(df)})\nKNOP1 factor 2',
                             xlabel='SNP chr16:19708463')
plt.show()

# %% CELL: k3_coverage
# The K=3 coverage trees, with the factorisation and the isoform bed switched to
# match: k=3 reads snmf_3, and H.K3_ISOFORMS is the gao_k3 bed. Mixing a K=3
# coverage with the K=5 bed gives the same xlim but shifts the exon models.
import Figure5_helpers as H

knop1_fac_k3 = EX.load_factors(KNOP1, k=3, iso_path=H.K3_ISOFORMS)
print(EX.gene_isoforms(KNOP1, iso_path=H.K3_ISOFORMS).to_string(index=False))

for root, grp in [('k3_coverage', None),
                  ('k3_coverage_by_AD', 'AD'),
                  ('k3_coverage_by_AD', 'nonAD')]:
    cov = EX.load_coverage(KNOP1, KNOP1_SNP, 'PCC', root=root, group=grp,
                           iso_path=H.K3_ISOFORMS)
    EX.plot_locus(cov, knop1_fac_k3,
                  title=f'$\\it{{KNOP1}}$ PCC — {root}' + (f' / {grp}' if grp else ''),
                  snp_label='SNP chr16:19708463', figsize=(9, 4.6))
    plt.show()

# %% CELL: pooled_by_ad
# By-AD boxplots with the three tissues pooled into each box.
#
# The boxes summarise AC + PCC + DLPFC together; the marker keeps each point
# attributable (AC '+', PCC 'x', DLPFC '^').
#
# POOLED_ROOT is the one place to change: every pooled plot below loads from it,
# and plot_by_ad_boxplot takes the frame, so swapping it here actually changes
# what is drawn. 'k3_qqnorm_by_AD' is quantile-normalised within each tissue, so
# its values are comparable across tissues; 'k3_unnormalized_by_AD' is not, and
# pooling it also mixes the three tissues' scales.
POOLED_ROOT = 'k3_qqnorm_by_AD'

knop1_pooled = EX.load_by_ad_tissues(KNOP1, KNOP1_SNP, root=POOLED_ROOT)
print(knop1_pooled.groupby(['group', 'tissue']).size().unstack(fill_value=0))
print(f"\ntree: {knop1_pooled.attrs['root']}  "
      f"range: {knop1_pooled[f'{KNOP1}.factor_2'].min():.3f} .. "
      f"{knop1_pooled[f'{KNOP1}.factor_2'].max():.3f}")

# KNOP1 — factor 2 is the retained-intron isoform, and it rises with the C allele.
# Drawn from `knop1_pooled`, so it follows POOLED_ROOT. The y-axis wording comes
# from the frame's tree, so it can never claim "normalized" for a raw loading.
# The fit runs separately within AD and non-AD. reg_covariates=['tissue']
# adjusts for the pooling, so a tissue difference in mean loading cannot show
# up as a genotype effect.
EX.plot_by_ad_boxplot(knop1_pooled, f'{KNOP1}.factor_2',
                      order=['CTCA/CTCA', 'CTCA/C', 'C/C'],
                      regression=True, reg_covariates=['tissue'],
                      ylabel=f'$\\it{{KNOP1}}$ factor 2\n'
                             f'{EX.loading_label(POOLED_ROOT)}\n(all tissues)',
                      xlabel='SNP chr16:19708463')
plt.show()

# %% CELL: pooled_cisd1
# CISD1 — note the factor numbering is not the K=5 one panel H used. In the K=3
# model isoform_1 is the short 3' form (factor_1) and isoform_2 the long one
# (factor_2 / factor_3), so the A allele raises factor_1 and lowers factor_2:
print(EX.gene_isoforms(CISD1, iso_path=H.K3_ISOFORMS).to_string(index=False))

cisd1_pooled = EX.load_by_ad_tissues(CISD1, CISD1_SNP, root=POOLED_ROOT)
print(cisd1_pooled.groupby('genotype')[EX.factor_columns(cisd1_pooled)]
      .mean().reindex(['G/G', 'G/A', 'A/A']).round(3))

EX.plot_by_ad_boxplot(cisd1_pooled, f'{CISD1}.factor_2',
                      order=['G/G', 'G/A', 'A/A'], legend_loc='lower left',
                      regression=True, reg_covariates=['tissue'],
                      ylabel=f'$\\it{{CISD1}}$ factor 2\n'
                             f'{EX.loading_label(POOLED_ROOT)}\n(all tissues)',
                      xlabel='SNP chr10:58286894')
plt.show()

# %% CELL: pooled_options
# Taking it apart: any subset of tissues, any markers, any factor.
# `load_by_ad_tissues` gives the pooled frame and `plot_by_ad_boxplot` draws it,
# so anything pandas can do to the frame fits in between.
two = EX.load_by_ad_tissues(KNOP1, KNOP1_SNP, tissues=('AC', 'PCC'),
                            root=POOLED_ROOT)
EX.plot_by_ad_boxplot(two, f'{KNOP1}.factor_2',
                      order=['CTCA/CTCA', 'CTCA/C', 'C/C'],
                      tissue_markers={'AC': 'o', 'PCC': 's'},
                      xlabel='SNP chr16:19708463')   # y label follows the tree
plt.show()

# The same panel under both trees, side by side — the check that a root swap is
# really reaching the plot. Unnormalized loadings are all positive and on the
# raw scale; qqnorm ones are centred near zero.
fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), constrained_layout=True)
for row, root in zip(axes, ['k3_unnormalized_by_AD', 'k3_qqnorm_by_AD']):
    d = EX.load_by_ad_tissues(KNOP1, KNOP1_SNP, root=root)
    EX.plot_by_ad_boxplot(d, f'{KNOP1}.factor_2', axes=row,
                          order=['CTCA/CTCA', 'CTCA/C', 'C/C'],
                          ylabel=f'{root}\n{EX.loading_label(root)}',
                          xlabel='SNP chr16:19708463')
plt.show()

# %% CELL: regression
# The same fits as numbers rather than annotations.
#
# `regression_table` runs genotype_regression per AD group (and per tissue with
# by_tissue=True), plus a pooled 'all' row — the check that a pooled effect is
# not carried by a single tissue.
tab = EX.regression_table(knop1_pooled, f'{KNOP1}.factor_2',
                          ['CTCA/CTCA', 'CTCA/C', 'C/C'],
                          covariates=['tissue'], by_tissue=True)
print(tab.drop(columns='formula').to_string(index=False))

# Any single fit on its own, e.g. the published panel F:
res = EX.genotype_regression(knop1_box, f'{KNOP1}.factor_2',
                             ['CTCA/CTCA', 'CTCA/C', 'C/C'])
print('\npanel F:', res['formula'], '->', EX.format_regression(res))

# Dosage follows `order`, so reversing it flips the sign of beta and leaves the
# p-value alone:
rev = EX.genotype_regression(knop1_box, f'{KNOP1}.factor_2',
                             ['C/C', 'CTCA/C', 'CTCA/CTCA'])
print('reversed:', EX.format_regression(rev))
