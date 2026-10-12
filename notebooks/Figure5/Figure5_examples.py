"""
Figure5_examples.py — regenerate the Figure 5 locus panels (E/G) and genotype
boxplots (F/H) straight from the RDS files, for any gene-variant combination.

Nothing here reads or writes a pickle and nothing saves a figure: every call
goes back to the RDS on disk, so swapping in a different gene, variant, tissue
or factorisation is just a different argument.

Three RDS families feed these panels:

    coverage   {COV_RDS}/{gene}.{variant}.{tissue}.coverage_matrices.rds
               allele-stratified coverage along the gene, written by the
               ad_genetics plotting pipeline
    boxplot    {BOX_RDS}/{gene}.{variant}.{tissue}.boxplot.rds
               one row per sample: genotype + each factor's loading
    factors    {GAO_RDS_ALL}/{gene}.rds
               the snmf factorisation; EF_smooth indexed by filtered_coords

plus the isoform bed (gao by default, gao_k3 optional), which supplies the
exon models drawn under the factor tracks and the gene's strand.

Run from the analysis/ directory, or anywhere — the paths are absolute.

Typical use:

    import Figure5_examples as EX

    cov = EX.load_coverage('ENSG00000103550', 'chr16:19708463:CTCA:C', 'PCC')
    fac = EX.load_factors('ENSG00000103550')
    EX.plot_locus(cov, fac, EX.locus_spec(
        'ENSG00000103550', name='KNOP1',
        transcripts=['ENSG00000103550.isoform_1', 'ENSG00000103550.isoform_2'],
        highlight={'ENSG00000103550.isoform_2': (19706479, 19710586)},
        factor_list=['factor1', 'factor2'], factor_labels=['Factor 1', 'Factor 2']))

    box = EX.load_boxplot('ENSG00000103550', 'chr16:19708463:CTCA:C', 'PCC')
    EX.plot_genotype_boxplot(box, 'ENSG00000103550.factor_2',
                             order=['CTCA/CTCA', 'CTCA/C', 'C/C'])

The drawing code is the same as the assembled figure's (Figure5_publication
draw_coverage / draw_isoform_rows / draw_boxplot, and the shared factor path in
Figure5_plot_helpers.plot_factors_wrapper), lifted out of the millimetre page
layout so each panel stands on its own.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
import seaborn as sns
from matplotlib import pyplot as plt

FIG5_DIR = os.path.dirname(os.path.abspath(__file__))
if FIG5_DIR not in sys.path:
    sys.path.append(FIG5_DIR)

import Figure5_helpers as H
import Figure5_plot_helpers as PH

# ---- where the RDS files live ----------------------------------------------
# Defaults are the directories the published panels were drawn from. Every
# loader takes `root=`, which accepts either a key of RDS_ROOTS or a path, so a
# whole alternative tree is one argument.
COV_RDS = H.COV_RDS_OLD
BOX_RDS = H.BOX_RDS
GAO_RDS_ALL = H.GAO_RDS_ALL
ISOFORM_BED = H.GAO_ISOFORMS          # H.K3_ISOFORMS for the K=3 universe

TORINO_CODE = '/project2/yangili1/cfbuenabadn/torino_paper/code'
K3_V2 = f'{TORINO_CODE}/k3_figure_202609-v2'
K5_CISD1 = f'{TORINO_CODE}/k5_CISD1'

#: Named RDS trees. The `*_by_AD` trees hold an extra AD/ and nonAD/ level —
#: pass group='AD' or group='nonAD' to reach into one of them. The k3_* trees
#: are the K=3 rerun: their boxplots carry factor_1..factor_3 (plus an `AD`
#: column), so pair them with load_factors(..., k=3) and the gao_k3 isoform bed.
RDS_ROOTS = {
    # coverage
    'coverage':            H.COV_RDS_OLD,          # published panels E / G
    'coverage_ad_genetics': H.COV_RDS,
    'k3_coverage':         f'{K3_V2}/coverage_matrices_rds',
    'k3_coverage_by_AD':   f'{K3_V2}/coverage_matrices_rds_by_AD',
    # boxplots
    'boxplot':             H.BOX_RDS,              # published panels F / H
    'k3_unnormalized':       f'{K3_V2}/boxplot_rds_unnormalized',
    'k3_unnormalized_by_AD': f'{K3_V2}/boxplot_rds_unnormalized_by_AD',
    'k3_qqnorm':             f'{K3_V2}/boxplot_rds_qqnorm',
    'k3_qqnorm_by_AD':       f'{K3_V2}/boxplot_rds_qqnorm_by_AD',
    # k5_CISD1: one gene, one variant, on the K=5 factorisation. Its filenames
    # spell the variant with underscores (chr10_58261486_A_T) where the other
    # trees use colons -- pass the variant exactly as the filename has it.
    'k5_cisd1_coverage':       f'{K5_CISD1}/coverage_matrices_rds',
    'k5_cisd1_coverage_by_AD': f'{K5_CISD1}/coverage_matrices_rds_by_AD',
    'k5_cisd1_qqnorm':         f'{K5_CISD1}/boxplot_rds_qqnorm',
    'k5_cisd1_qqnorm_by_AD':   f'{K5_CISD1}/boxplot_rds_qqnorm_by_AD',
}

#: The one gene-variant k5_CISD1 holds, as the loaders want it.
K5_CISD1_GENE = 'ENSG00000122873'
K5_CISD1_VARIANT = 'chr10_58261486_A_T'


def resolve_root(root, default=None, group=None):
    """
    Turn `root` into a directory: a key of RDS_ROOTS, or a path used as given.
    `group` ('AD' / 'nonAD') descends the extra level the *_by_AD trees have.
    """
    d = RDS_ROOTS.get(root, root) if root is not None else default
    if group is not None:
        d = os.path.join(d, group)
    return d

# ---- look ------------------------------------------------------------------
NAVY, RED = 'midnightblue', '#d62728'
GENO_COLORS = ['darkred', 'orangered', 'gold']   # ref/ref, ref/alt, alt/alt
FACTOR_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'goldenrod', 'tab:red']


# ================================================================
# Loading
# ================================================================

_ISO_CACHE = {}


def load_isoform_bed(path=None):
    """
    The isoform bed, cached per path — it is 460k rows, so re-reading it for
    every panel is the slowest thing in this module.
    """
    path = path or ISOFORM_BED
    if path not in _ISO_CACHE:
        _ISO_CACHE[path] = H.load_gao_isoform_bed(path)
    return _ISO_CACHE[path]


def coverage_rds_path(gene, variant, tissue, root=None, group=None):
    return os.path.join(resolve_root(root, COV_RDS, group),
                        f'{gene}.{variant}.{tissue}.coverage_matrices.rds')


def boxplot_rds_path(gene, variant, tissue, root=None, group=None):
    return os.path.join(resolve_root(root, BOX_RDS, group),
                        f'{gene}.{variant}.{tissue}.boxplot.rds')


def load_coverage(gene, variant=None, tissue=None, rds_path=None, root=None,
                  group=None, iso_path=None):
    """
    Allele-stratified coverage for one gene-variant-tissue.

    Give (gene, variant, tissue) to use the standard filename, or `rds_path` to
    point at any coverage_matrices RDS directly — `gene` is still needed, to
    pick the isoform models and the strand.

    `root` selects the tree (an RDS_ROOTS key or a path) and `group` picks
    'AD' / 'nonAD' inside a *_by_AD tree. Pair a k3_* root with
    load_factors(..., k=3) and iso_path=H.K3_ISOFORMS.
    """
    path = rds_path or coverage_rds_path(gene, variant, tissue, root, group)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no coverage RDS at {path}\n'
            f'  this tree has: {_describe_available(gene, "coverage", root, group)}')
    rds = H._readRDS()(path)
    iso = load_isoform_bed(iso_path)
    return {'coverage_dir': H.rds_to_coverage_dir(rds), 'gene': gene,
            'iso': iso.loc[iso.gene_id == gene].copy(), 'path': path}


def load_boxplot(gene, variant=None, tissue=None, rds_path=None, root=None,
                 group=None):
    """
    One row per sample: `genotype` plus a column per gene-factor
    ('{gene}.factor_{k}'), which is what plot_genotype_boxplot takes. The k3_*
    trees carry factor_1..factor_3 and an extra `AD` column.

    `root` selects the tree (an RDS_ROOTS key or a path) and `group` picks
    'AD' / 'nonAD' inside a *_by_AD tree.
    """
    path = rds_path or boxplot_rds_path(gene, variant, tissue, root, group)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no boxplot RDS at {path}\n'
            f'  this tree has: {_describe_available(gene, "boxplot", root, group)}')
    return H.read_dataframe_rds(path)


def load_factors(gene, k=5, rds_path=None, root=None, iso_path=None):
    """
    The snmf factorisation for one gene.

    `k` picks the model inside the RDS ('snmf_5' or 'snmf_3'). `coords` indexes
    EF_smooth and `xlim_coords` sets the default x limits — they are different
    vectors and both are needed.

    For the K=3 trees use k=3 together with iso_path=H.K3_ISOFORMS: the gao and
    gao_k3 isoform beds give the same xlim but different gene extents, and the
    isoform mirroring is anchored to the factorisation either way (see
    plot_locus), so mixing them silently shifts the exon models.
    """
    path = rds_path or os.path.join(root or GAO_RDS_ALL, f'{gene}.rds')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no factorisation RDS at {path}')
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    obj = ro.r['readRDS'](path)
    with (ro.default_converter + pandas2ri.converter).context():
        rds = ro.conversion.get_conversion().rpy2py(obj)
    key = f'snmf_{k}'
    if key not in rds.keys():
        raise KeyError(f'{path} has no {key}; it holds {list(rds.keys())}')
    iso = load_isoform_bed(iso_path)
    iso_gene = iso.loc[iso.gene_id == gene]
    if iso_gene.empty:
        raise KeyError(f'{gene} is not in the isoform bed {iso_path or ISOFORM_BED}')
    return {'EF_smooth': np.asarray(rds[key]['ebpmf']['EF_smooth']),
            'coords': list(rds['filtered_coords']),
            'xlim_coords': [int(x.split(':')[1]) for x in list(rds['coords'])],
            'gene': gene, 'strand': iso_gene.strand.iloc[0],
            'iso': iso_gene.copy(), 'k': k, 'path': path}


SUFFIX = {'coverage': '.coverage_matrices.rds', 'boxplot': '.boxplot.rds'}


def _scan(d, kind, gene, group=None):
    rows = []
    if not os.path.isdir(d):
        return rows
    for f in sorted(os.listdir(d)):
        full = os.path.join(d, f)
        if os.path.isdir(full):
            # a *_by_AD tree: recurse one level, tagging the group
            rows += _scan(full, kind, gene, group=f)
            continue
        if not f.endswith(SUFFIX[kind]):
            continue
        parts = f[:-len(SUFFIX[kind])].split('.')
        if len(parts) < 3:
            continue
        g, v, t = parts[0], '.'.join(parts[1:-1]), parts[-1]
        if gene is None or g == gene:
            rows.append({'gene': g, 'variant': v, 'tissue': t,
                         'group': group, 'file': f})
    return rows


def list_available(gene=None, kind='coverage', root=None, group=None):
    """
    The (gene, variant, tissue) combinations in an RDS tree, as a DataFrame —
    the quickest way to find what else can be plotted.

    `kind` is 'coverage' or 'boxplot'; `root` is an RDS_ROOTS key or a path.
    A *_by_AD tree is scanned one level deeper and its AD / nonAD level lands
    in the `group` column, which is None for the flat trees.
    """
    default = {'coverage': COV_RDS, 'boxplot': BOX_RDS}[kind]
    d = resolve_root(root, default, group)
    return pd.DataFrame(_scan(d, kind, gene),
                        columns=['gene', 'variant', 'tissue', 'group', 'file'])


def _describe_available(gene, kind, root, group):
    """One-line summary for the FileNotFoundError messages."""
    try:
        a = list_available(gene, kind=kind, root=root, group=group)
    except OSError as e:
        return f'<{e}>'
    if a.empty:
        return 'nothing for this gene'
    return (f'{len(a)} files, tissues {sorted(a.tissue.unique())}, '
            f'{a.variant.nunique()} variants')


def compare_roots(gene, variant, tissue, kind='boxplot', roots=None):
    """
    Which of the named trees actually hold a given gene-variant-tissue —
    the first thing to run before comparing normalisations.
    """
    keys = roots or [k for k, v in RDS_ROOTS.items()
                     if (kind == 'coverage') == ('coverage' in k)]
    rows = []
    for k in keys:
        by_ad = k.endswith('_by_AD')
        for g in (['AD', 'nonAD'] if by_ad else [None]):
            p = (coverage_rds_path if kind == 'coverage' else boxplot_rds_path)(
                gene, variant, tissue, root=k, group=g)
            rows.append({'root': k, 'group': g, 'exists': os.path.exists(p),
                         'path': p})
    return pd.DataFrame(rows)


def gene_isoforms(gene, iso_path=None):
    """The transcript ids and the factors each carries — what `spec` picks from."""
    iso = load_isoform_bed(iso_path)
    g = iso.loc[iso.gene_id == gene]
    return (g.groupby('transcript_id')
             .agg(factors=('factors', 'first'), strand=('strand', 'first'),
                  n_exons=('start', 'size'),
                  start=('start', 'min'), end=('end', 'max'))
             .reset_index())


# ================================================================
# Panel spec
# ================================================================

def locus_spec(gene, name=None, transcripts=None, highlight=None,
               factor_list=None, factor_labels=None, colors=None,
               iso_path=None):
    """
    What plot_locus draws, beyond the data itself.

    Every field has a usable default, so `locus_spec(gene)` alone gives all of
    the gene's isoforms under all of the factorisation's factors.

      name           gene label on the first isoform row (default: the gene id)
      transcripts    which isoform rows to draw, in order (default: all)
      highlight      {transcript_id: (start, end)} — one exon drawn in red
      factor_list    which factors to draw, as 'factor1', 'factor2', ...
      factor_labels  left-hand label per track (default: 'Factor N')
      colors         one colour per track
    """
    if transcripts is None:
        transcripts = sorted(gene_isoforms(gene, iso_path).transcript_id)
    if factor_labels is None and factor_list is not None:
        factor_labels = [f'Factor {f.replace("factor", "")}' for f in factor_list]
    return {'gene': gene, 'name': name or gene, 'transcripts': list(transcripts),
            'highlight': highlight or {}, 'factor_list': factor_list,
            'factor_labels': factor_labels, 'colors': colors}


# ================================================================
# Drawing — coverage, factor tracks, isoform models
# ================================================================

def _mirror(v, anchor):
    """Reflect a coordinate into the minus-strand drawing space."""
    return anchor[0] + anchor[1] - v


def draw_coverage(ax, coverage, strand, xlim, smooth=100, colors=None,
                  labels=None, legend_loc='upper right', lw=1.0, fs_lab=8,
                  fs_tick=7, fs_annot=7):
    """
    Genotype-stratified coverage, with the variant marked by a dashed line.

    Reproduces the running-mean smoothing and the minus-strand mirroring of
    GeneticsPlots.plot_coverage. Returns the variant's x position in drawing
    coordinates, so a caller can label it.

    Do not "fix" the ref/alt assignment. The R output names its elements
    ('var', 'var_pos', 'first_position', 'ref_allele', 'alt_allele', 'coords',
    'alts_coverage', 'het_coverage', 'ref_coverage') but the last two are
    swapped relative to their contents, and H.rds_to_coverage_dir reads them
    positionally, which corrects it. Checked against the loadings: for KNOP1 /
    chr16:19708463:CTCA:C in PCC the element *named* ref_coverage has the
    highest coverage over the retained intron (0.416 vs 0.226), while the
    boxplot has factor 2 rising with the C allele (-0.36 / 0.90 / 1.83), so
    that element is the alt homozygotes. Every tree checked -- the published,
    ad_genetics and k3_figure_202609-v2 ones -- has the same element order.
    """
    colors = colors or GENO_COLORS
    c = coverage['coverage_dir']
    coords = list(c['coords'])
    refs, hets, alts = (list(c['ref_coverage']), list(c['het_coverage']),
                        list(c['alt_coverage']))
    sc = np.array([np.mean(coords[i:i + smooth])
                   for i in range(len(coords))]) - smooth
    sa = [np.mean(alts[i:i + smooth]) for i in range(len(alts))]
    sh = [np.mean(hets[i:i + smooth]) for i in range(len(hets))]
    sr = [np.mean(refs[i:i + smooth]) for i in range(len(refs))]
    ref, alt = c['ref_allele'], c['alt_allele']
    var_pos = c['var_pos']
    if strand == '-':
        sa, sh, sr = sa[::-1], sh[::-1], sr[::-1]
        var_pos = sc[-1] + sc[0] - var_pos

    lab = labels or [f'{alt}/{alt}', f'{ref}/{alt}', f'{ref}/{ref}']
    ax.plot(sc, sa, label=lab[0], c=colors[2], linewidth=lw)
    ax.plot(sc, sh, label=lab[1], c=colors[1], linewidth=lw)
    ax.plot(sc, sr, label=lab[2], c=colors[0], linewidth=lw)

    top = np.nanmax([np.nanmax(sa), np.nanmax(sh), np.nanmax(sr)])
    ax.plot([var_pos, var_pos], [0, top], linestyle='--', c='black', linewidth=lw)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.02 * top, top * 1.12)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=3, integer=True, steps=[1, 2, 5, 10]))
    ax.set_ylabel('normalized counts', fontsize=fs_lab)
    ax.tick_params(labelsize=fs_tick)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.legend(frameon=False, fontsize=fs_annot, loc=legend_loc, handlelength=1.1,
              borderpad=0.1, labelspacing=0.2)
    return var_pos


def draw_isoform_rows(axes, iso_gene, transcripts, strand, anchor, xlim,
                      highlight=None, lw=1.1):
    """
    One isoform model per axes, drawing only `transcripts`.

    `anchor` mirrors minus-strand exons about the same coordinates the factor
    tracks are mirrored about, which is what keeps the two aligned; see
    plot_locus for why it comes from the factorisation and not the isoform bed.
    """
    highlight = highlight or {}
    for ax, t in zip(axes, transcripts):
        ex = iso_gene.loc[iso_gene.transcript_id == t].sort_values('start')
        hl = highlight.get(t)
        drawn = []
        for _, r in ex.iterrows():
            s, e = int(r.start), int(r.end)
            red = hl is not None and s == hl[0] and e == hl[1]
            drawn.append(((_mirror(e, anchor), _mirror(s, anchor)) if strand == '-'
                          else (s, e)) + (red,))
        if not drawn:
            ax.set_axis_off()
            continue
        lo = min(s for s, _, _ in drawn)
        hi = max(e for _, e, _ in drawn)
        ax.plot([lo, hi], [0.5, 0.5], color=NAVY, linewidth=lw,
                solid_capstyle='butt')
        for s, e, red in drawn:
            ax.fill_between([s, e], [0, 0], [1, 1], color=RED if red else NAVY,
                            linewidth=0, zorder=3)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


def plot_locus(coverage, factors, spec=None, title=None, snp_label=None,
               figsize=(9.0, 4.4), xlim=None, pad=1000,
               cov_h=1.0, track_h=0.30, row_h=0.10, row_gap=0.03,
               left=0.11, width=0.86, legend_loc='upper right', snp_ha='center',
               colors=None, geno_colors=None, geno_labels=None, smooth=100,
               lw=1.0, fs_lab=8, fs_tick=7, fs_annot=7, fs_title=9, fig=None,
               lm=False):
    """
    Panels E / G — coverage on top, the factor tracks under it, the isoform
    models at the bottom, all on one x axis.

    `coverage` and `factors` come from load_coverage() / load_factors() for the
    same gene; `spec` (locus_spec) says which isoforms and factors to draw. The
    heights are fractions of one another, normalised to fill the figure, so
    `track_h=0.5` makes the factor tracks half as tall as the coverage track.
    """
    gene = factors['gene']
    spec = spec or locus_spec(gene)
    strand = factors['strand']
    if xlim is None:
        xc = factors['xlim_coords']
        xlim = (xc[0] - pad, xc[-1] + pad)

    factor_list = spec['factor_list']
    if factor_list is None:
        factor_list = [f'factor{i + 1}' for i in range(factors['EF_smooth'].shape[1])]
    labels = spec['factor_labels'] or [
        f'Factor {f.replace("factor", "")}' for f in factor_list]
    track_colors = colors or spec['colors'] or FACTOR_COLORS[:len(factor_list)]

    n_f, n_i = len(factor_list), len(spec['transcripts'])
    heights = [cov_h] + [track_h] * n_f + [row_h] * n_i
    gaps = [0.0] * (1 + n_f) + [row_gap] * n_i
    total = sum(heights) + sum(gaps)

    fig = fig or plt.figure(figsize=figsize)
    boxes, y = [], 1.0
    for h, g in zip(heights, gaps):
        y -= (h + g) / total
        boxes.append(fig.add_axes([left, y, width, h / total]))
    ax_cov, f_axes, i_axes = boxes[0], boxes[1:1 + n_f], boxes[1 + n_f:]

    var_pos = draw_coverage(ax_cov, coverage, strand, xlim, smooth=smooth,
                            colors=geno_colors, labels=geno_labels,
                            legend_loc=legend_loc, lw=lw, fs_lab=fs_lab,
                            fs_tick=fs_tick, fs_annot=fs_annot)
    if title:
        ax_cov.set_title(title, fontsize=fs_title, pad=3)
    if snp_label:
        ax_cov.text(var_pos, ax_cov.get_ylim()[1] * 0.99, snp_label,
                    fontsize=fs_annot, ha=snp_ha, va='top')

    # lm=False: the factor tracks are the raw smoothed factors. The lm
    # correction (Figure5_plot_helpers.factor_lm) divides each factor by a
    # line fitted along the gene when that line's slope runs with the strand,
    # to flatten 3' coverage decay. It reshapes the tracks, so it is off here.
    PH.plot_factors_wrapper(factors['EF_smooth'], factors['coords'], spec['name'],
                            factor_list=factor_list, xlim=xlim, mute_y_axis=True,
                            ax=f_axes, strand=strand, colores=track_colors,
                            lwidth=lw, lm=lm)
    for ax, lab in zip(f_axes, labels):
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.008, 0.5, lab, transform=ax.transAxes, fontsize=fs_annot,
                ha='right', va='center')

    # Mirror the isoforms about the SAME coordinates the factor tracks use
    # (the factorisation's filtered_coords). Taking the anchor from the isoform
    # bed instead -- what GeneticsPlots does by default -- slides minus-strand
    # exons by however far that bed's gene extent differs: 49 bp with the gao
    # bed, 1403 bp with gao_k3, while xlim comes out identical either way.
    fc = [int(c.split(':')[1]) for c in factors['coords']]
    draw_isoform_rows(i_axes, coverage['iso'], spec['transcripts'], strand,
                      (min(fc), max(fc)), xlim, highlight=spec['highlight'])
    if i_axes:
        i_axes[0].text(-0.008, 0.5, spec['name'], transform=i_axes[0].transAxes,
                       fontsize=fs_annot, ha='right', va='center', style='italic')
    return fig


# ================================================================
# Drawing — genotype boxplot
# ================================================================

#: columns a boxplot frame carries that are not factor loadings. The k3_* trees
#: add `AD`; without excluding it, it would be picked as the default factor.
NON_FACTOR_COLS = ('genotype', 'AD', 'sample', 'specimenID', 'tissue')


def factor_columns(df):
    """
    The factor-loading columns of a boxplot frame.

    Prefers the '{gene}.factor_{k}' naming and falls back to "everything that
    is not a known metadata column", so an unfamiliar tree still works.
    """
    named = [c for c in df.columns if '.factor_' in str(c)]
    return named or [c for c in df.columns if c not in NON_FACTOR_COLS]


# ================================================================
# Genotype regression
# ================================================================

def genotype_dosage(genotypes, order):
    """
    Allele dosage 0/1/2 from the position in `order`.

    The encoding follows the order you pass, so beta is "per step along that
    order": with order=['CTCA/CTCA', 'CTCA/C', 'C/C'] it is the effect per C
    allele, and reversing the order flips the sign.
    """
    pos = {g: i for i, g in enumerate(order)}
    return pd.Series(genotypes).map(pos)


def genotype_regression(df, factor, order, covariates=None):
    """
    OLS of the loading on allele dosage: `factor ~ dosage (+ covariates)`.

    `covariates` names extra columns of `df` to adjust for — numeric ones enter
    as-is, categorical ones as dummies (drop-first). Pooling tissues is the case
    that wants covariates=['tissue']: without it, a tissue difference in mean
    loading is free to masquerade as a genotype effect.

    Returns a dict with beta, se, pval, n, r2 and the model formula. Values are
    NaN when fewer than three samples or only one genotype survive.
    """
    import statsmodels.api as sm

    sub = df[[c for c in {factor, 'genotype', *(covariates or [])}]].copy()
    sub['_dosage'] = genotype_dosage(sub['genotype'], order)
    sub = sub.dropna(subset=[factor, '_dosage'])

    formula = f'{factor} ~ dosage' + (' + ' + ' + '.join(covariates) if covariates else '')
    out = {'beta': np.nan, 'se': np.nan, 'pval': np.nan, 'n': len(sub),
           'r2': np.nan, 'formula': formula}
    if len(sub) < 3 or sub['_dosage'].nunique() < 2:
        return out

    X = pd.DataFrame({'dosage': sub['_dosage'].astype(float)}, index=sub.index)
    for c in covariates or []:
        col = sub[c]
        if pd.api.types.is_numeric_dtype(col):
            X[c] = col.astype(float)
        else:
            X = X.join(pd.get_dummies(col, prefix=c, drop_first=True).astype(float))
    X = X.loc[:, X.std() > 0]                      # drop constants (one tissue, say)
    if 'dosage' not in X.columns:
        return out

    fit = sm.OLS(sub[factor].astype(float), sm.add_constant(X, has_constant='add')).fit()
    out.update(beta=float(fit.params['dosage']), se=float(fit.bse['dosage']),
               pval=float(fit.pvalues['dosage']), r2=float(fit.rsquared))
    return out


def format_regression(res, digits=2):
    """'beta = 0.82, p = 3.4e-21' — the annotation text."""
    if not np.isfinite(res['beta']):
        return f"n = {res['n']} (too few genotypes to fit)"
    p = res['pval']
    p_txt = 'p < 1e-300' if p == 0 else f'p = {p:.1e}'
    return f"$\\beta$ = {res['beta']:.{digits}f}, {p_txt}, n = {res['n']}"


def annotate_regression(ax, res, loc='upper left', fontsize=7, color='black',
                        digits=2):
    """Write the beta / p-value into a corner of `ax`."""
    xy = {'upper left': (0.02, 0.98, 'left', 'top'),
          'upper right': (0.98, 0.98, 'right', 'top'),
          'lower left': (0.02, 0.02, 'left', 'bottom'),
          'lower right': (0.98, 0.02, 'right', 'bottom'),
          'upper center': (0.5, 0.98, 'center', 'top'),
          # outside the axes, for dense panels where no corner is free
          'above': (0.5, 1.02, 'center', 'bottom')}[loc]
    ax.text(xy[0], xy[1], format_regression(res, digits), transform=ax.transAxes,
            ha=xy[2], va=xy[3], fontsize=fontsize, color=color)
    return ax


def plot_genotype_boxplot(df, factor=None, order=None, color='tab:orange',
                          ylabel=None, xlabel=None, ax=None, figsize=(2.6, 2.4),
                          show_n=True, lw=1.0, marker_size=3.0, jitter=0.22,
                          fs_lab=8, fs_tick=7, regression=False,
                          reg_covariates=None, reg_loc='upper left',
                          reg_digits=2, annotate=False):
    """
    Panels F / H — one factor's loadings split by genotype, with the per-genotype
    sample size under each tick.

    `factor` defaults to the first factor column; `order` fixes the genotype
    order (and so the allele-dosage direction), defaulting to the order the RDS
    happens to carry.

    `regression=True` fits `factor ~ dosage` (see genotype_regression) and
    writes beta and the p-value into the corner named by `reg_loc`. The fit is
    left on the axes as `ax.regression_` for the numbers themselves.
    """
    factor = factor or factor_columns(df)[0]
    if factor not in df.columns:
        raise KeyError(f'{factor} is not a column; this frame has '
                       f'{factor_columns(df)}')
    order = list(order) if order is not None else list(df.genotype.unique())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x='genotype', y=factor, width=.55, fliersize=0,
                linewidth=lw, patch_artist=True, ax=ax, hue='genotype',
                legend=False, order=order, hue_order=order,
                boxprops=dict(edgecolor=color, facecolor='white'),
                whiskerprops=dict(color=color, linewidth=lw),
                capprops=dict(color=color, linewidth=lw),
                medianprops=dict(color=color, linewidth=lw))
    sns.stripplot(data=df, x='genotype', y=factor, c=color, alpha=0.4, ax=ax,
                  size=marker_size, jitter=jitter, order=order)

    n = df.genotype.value_counts()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f'{g}\n(n={int(n.get(g, 0))})' if show_n else g
                        for g in order], fontsize=fs_tick)
    ax.set_ylabel(ylabel if ylabel is not None else f'{factor}\nnormalized loadings',
                  fontsize=fs_lab, linespacing=1.4)
    ax.set_xlabel(xlabel if xlabel is not None else 'genotype', fontsize=fs_lab)
    ax.tick_params(labelsize=fs_tick)
    ax.spines[['top', 'right']].set_visible(False)

    ax.regression_ = None
    if regression:
        ax.regression_ = genotype_regression(df, factor, order, reg_covariates)
        # see plot_by_ad_boxplot: computed, not drawn, unless annotate=True
        if annotate:
            if str(reg_loc).startswith('upper'):
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.16 * (hi - lo))
            annotate_regression(ax, ax.regression_, loc=reg_loc,
                                fontsize=fs_tick, color=color, digits=reg_digits)
    return ax


# ================================================================
# Genotype boxplots pooled across tissues, split by AD status
# ================================================================

TISSUES = ('AC', 'PCC', 'DLPFC')

#: One marker per tissue, so pooled points stay attributable.
TISSUE_MARKERS = {'AC': '+', 'PCC': 'x', 'DLPFC': '^'}

#: Box outline per AD group.
AD_COLORS = {'AD': 'tab:red', 'nonAD': 'tab:gray'}


def snp_label(variant):
    """
    'chr10:58286894' from a variant id, whichever way it spells the fields.

    The published trees use 'chr:pos:ref:alt'; k5_CISD1 uses
    'chr_pos_ref_alt'. Both reduce to chrom and position.
    """
    parts = str(variant).replace('_', ':').split(':')
    return ':'.join(parts[:2]) if len(parts) >= 2 else str(variant)


def is_normalized(root):
    """Whether an RDS tree's loadings are on a common (quantile-normalised) scale."""
    return 'qqnorm' in str(RDS_ROOTS.get(root, root))


def loading_label(root):
    """Axis wording for a tree — never call a raw loading 'normalized'."""
    return 'normalized loadings' if is_normalized(root) else 'unnormalized loadings'


def load_by_ad_tissues(gene, variant, *, tissues=TISSUES, groups=('AD', 'nonAD'),
                       root='k3_qqnorm_by_AD'):
    """
    Every tissue's by-AD boxplot RDS for one gene-variant, in one frame.

    Adds `tissue` and `group` columns and keeps the RDS's own `genotype` and
    factor columns, so the result drops straight into plot_by_ad_boxplot.
    A missing tissue-group file is skipped with a note rather than raising:
    not every combination exists (KNOP1 has no C/C in AD DLPFC, for one).

    Pooling only makes sense for a tree that is already on a common scale.
    'k3_qqnorm_by_AD' is quantile-normalised within each tissue, so its values
    are comparable; 'k3_unnormalized_by_AD' is not, and pooling it mostly
    measures the difference in library depth between tissues.
    """
    frames = []
    for group in groups:
        for tissue in tissues:
            try:
                df = load_boxplot(gene, variant, tissue, root=root, group=group)
            except FileNotFoundError:
                print(f'  skipped: no {group}/{tissue} file for {gene} {variant}')
                continue
            frames.append(df.assign(tissue=tissue, group=group))
    if not frames:
        raise FileNotFoundError(f'no by-AD files at all for {gene} {variant}')
    out = pd.concat(frames, ignore_index=True)
    # remember the tree, so the axis label can say what the values actually are
    out.attrs['root'] = root
    if len(out.tissue.unique()) > 1 and not is_normalized(root):
        print(f'  note: {root} is not quantile-normalised, so pooling '
              f'{sorted(out.tissue.unique())} mixes their scales')
    return out


def load_pooled_tissues(gene, variant, *, tissues=TISSUES, root=None,
                        group='all samples'):
    """
    Every tissue of a **flat** (not by-AD) tree in one frame — the unsplit
    counterpart to load_by_ad_tissues.

    Adds `tissue` and a constant `group`, so the result drops straight into
    plot_by_ad_boxplot as a single panel with the per-tissue markers:

        d = load_pooled_tissues(gene, variant, root='k5_cisd1_qqnorm')
        plot_by_ad_boxplot(d, factor, groups=('all samples',), ...)

    Note this is a larger sample than the by-AD trees: those keep only samples
    with an AD/control label, while a flat tree also carries the unlabelled ones
    (R writes their `AD` as NA_integer_, which reads back as -2147483648).
    """
    frames = []
    for tissue in tissues:
        try:
            frames.append(load_boxplot(gene, variant, tissue, root=root)
                          .assign(tissue=tissue, group=group))
        except FileNotFoundError:
            print(f'  skipped: no {tissue} file for {gene} {variant}')
    if not frames:
        raise FileNotFoundError(f'no boxplot files for {gene} {variant}')
    out = pd.concat(frames, ignore_index=True)
    out.attrs['root'] = root
    if len(out.tissue.unique()) > 1 and not is_normalized(root):
        print(f'  note: {root} is not quantile-normalised, so pooling '
              f'{sorted(out.tissue.unique())} mixes their scales')
    return out


def report_regressions(axes, digits=4):
    """
    The fits behind a boxplot, as a DataFrame -- they are no longer drawn on the
    axes, so this is how the numbers come out.

    Takes whatever plot_by_ad_boxplot / plot_genotype_boxplot returned (an axes
    array, or a single axes).
    """
    import numpy as _np
    axes = _np.atleast_1d(axes)
    rows = []
    for ax in axes:
        r = getattr(ax, 'regression_', None)
        if not r:
            continue
        rows.append({'group': r.get('group', ''), 'model': r['formula'],
                     'beta': round(r['beta'], digits), 'se': round(r['se'], digits),
                     'pval': r['pval'], 'n': r['n'], 'r2': round(r['r2'], digits)})
    return pd.DataFrame(rows)


def plot_by_ad_boxplot(df, factor=None, order=None, groups=('AD', 'nonAD'),
                       tissue_markers=None, ad_colors=None, axes=None,
                       figsize=(6.6, 2.9), ylabel=None, xlabel=None,
                       show_n=True, marker_size=4.0, marker_lw=0.7, jitter=0.24,
                       alpha=0.65, lw=1.0, fs_lab=8, fs_tick=7, fs_legend=7,
                       sharey=True, legend_ax=0, legend_loc='best', seed=0,
                       regression=False, reg_covariates=None,
                       reg_loc='upper left', reg_digits=2, annotate=False):
    """
    One panel per AD group, each pooling every tissue in `df`.

    The boxes summarise all three tissues together; the strip points keep the
    tissue visible through the marker (AC '+', PCC 'x', DLPFC '^' by default),
    since seaborn cannot vary a marker by hue — each tissue is a separate
    stripplot call over the same `order`, so the categories stay aligned.

    `df` comes from load_by_ad_tissues(). Returns the axes array.

    `regression=True` fits `factor ~ dosage` separately within each AD group and
    writes beta and the p-value into that panel; each fit is left on its axes as
    `ax.regression_`. Because the tissues are pooled, pass
    reg_covariates=['tissue'] to adjust for them — otherwise a difference in
    mean loading between tissues can show up as a genotype effect.
    """
    factor = factor or factor_columns(df)[0]
    if factor not in df.columns:
        raise KeyError(f'{factor} is not a column; this frame has '
                       f'{factor_columns(df)}')
    order = list(order) if order is not None else list(df.genotype.unique())
    markers = tissue_markers or TISSUE_MARKERS
    colors = ad_colors or AD_COLORS
    groups = [g for g in groups if g in set(df.group)]

    if axes is None:
        _, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=sharey,
                               constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, group in zip(axes, groups):
        sub = df.loc[df.group == group]
        color = colors.get(group, 'tab:blue')
        sns.boxplot(data=sub, x='genotype', y=factor, width=.55, fliersize=0,
                    linewidth=lw, patch_artist=True, ax=ax, hue='genotype',
                    legend=False, order=order, hue_order=order,
                    boxprops=dict(edgecolor=color, facecolor='white'),
                    whiskerprops=dict(color=color, linewidth=lw),
                    capprops=dict(color=color, linewidth=lw),
                    medianprops=dict(color=color, linewidth=lw))
        # The points are scattered directly rather than through
        # sns.stripplot: '+' and 'x' are stroke-only markers, and seaborn's
        # facecolor/edgecolor handling drops their stroke entirely, so only the
        # filled marker (DLPFC '^') would show. Jitter is uniform about the
        # category index, which is what stripplot does anyway.
        pos = {g: i for i, g in enumerate(order)}
        rng = np.random.default_rng(seed)
        for tissue in [t for t in markers if t in set(sub.tissue)]:
            ts = sub.loc[sub.tissue == tissue]
            x = ts.genotype.map(pos).to_numpy(dtype=float)
            keep = ~np.isnan(x)
            x = x[keep] + rng.uniform(-jitter, jitter, size=int(keep.sum()))
            y = ts[factor].to_numpy(dtype=float)[keep]
            m = markers[tissue]
            fill = m in ('+', 'x', '1', '2', '3', '4', '|', '_')
            ax.scatter(x, y, marker=m, s=marker_size ** 2, alpha=alpha,
                       linewidths=marker_lw, zorder=3,
                       **({'color': color} if fill
                          else {'facecolors': 'none', 'edgecolors': color}))

        n = sub.genotype.value_counts()
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f'{g}\n(n={int(n.get(g, 0))})' if show_n else g
                            for g in order], fontsize=fs_tick)
        ax.set_xlabel(xlabel if xlabel is not None else 'genotype',
                      fontsize=fs_lab)
        ax.set_title(f'{group} (n={len(sub)})', fontsize=fs_lab, color=color)
        ax.tick_params(labelsize=fs_tick)
        ax.spines[['top', 'right']].set_visible(False)

        ax.regression_ = None
        if regression:
            ax.regression_ = genotype_regression(sub, factor, order, reg_covariates)
            ax.regression_['group'] = group
            # Not drawn by default: the annotation needs headroom above the
            # boxes, and that compression is what makes a real effect look
            # small. Print the numbers with report_regressions() instead, or
            # pass annotate=True to put them back on the axes.
            if annotate:
                if str(reg_loc).startswith('upper'):
                    lo, hi = ax.get_ylim()
                    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
                annotate_regression(ax, ax.regression_, loc=reg_loc,
                                    fontsize=fs_tick, color=color,
                                    digits=reg_digits)

    axes[0].set_ylabel(
        ylabel if ylabel is not None
        else f'{factor}\n{loading_label(df.attrs.get("root", "qqnorm"))}',
        fontsize=fs_lab, linespacing=1.4)
    for ax in axes[1:]:
        ax.set_ylabel('')

    # one tissue key, with per-tissue n
    counts = df.tissue.value_counts()
    handles = [plt.Line2D([], [], linestyle='None', marker=markers[t],
                          color='black',
                          markerfacecolor=('black' if markers[t] in
                                           ('+', 'x', '1', '2', '3', '4', '|', '_')
                                           else 'none'),
                          markeredgecolor='black', markeredgewidth=marker_lw,
                          markersize=marker_size,
                          label=f'{t} (n={int(counts.get(t, 0))})')
               for t in markers if t in set(df.tissue)]
    axes[legend_ax].legend(handles=handles, title='tissue', frameon=False,
                           fontsize=fs_legend, title_fontsize=fs_legend,
                           loc=legend_loc, handletextpad=0.4,
                           borderpad=0.1, labelspacing=0.25)
    return axes


def plot_gene_by_ad(gene, variant, factor=None, order=None, name=None, *,
                    tissues=TISSUES, root='k3_qqnorm_by_AD', **kwargs):
    """
    load_by_ad_tissues + plot_by_ad_boxplot in one call, for a gene-variant.

    `factor` defaults to the gene's factor 2; `name` is only the y-axis label,
    whose wording follows `root` — an unnormalized tree is not labelled
    'normalized'. `tissues` and `root` are keyword-only, so a third positional
    argument cannot silently land in `factor`.
    """
    df = load_by_ad_tissues(gene, variant, tissues=tissues, root=root)
    factor = factor or f'{gene}.factor_2'
    label = name or gene
    kwargs.setdefault(
        'ylabel', f'$\\it{{{label}}}$ {factor.split(".")[-1].replace("_", " ")}'
                  f'\n{loading_label(root)}\n(all tissues)')
    kwargs.setdefault('xlabel', f'SNP {snp_label(variant)}')
    return plot_by_ad_boxplot(df, factor, order=order, **kwargs)


def regression_table(df, factor, order, groups=('AD', 'nonAD'),
                     covariates=None, by_tissue=False):
    """
    The same fits as the plot annotations, as a DataFrame — for reading numbers
    off rather than out of a figure.

    Rows are the AD groups, plus one per tissue when `by_tissue` is set, plus an
    'all' row pooling everything.
    """
    rows = []
    for group in [g for g in groups if g in set(df.group)]:
        sub = df.loc[df.group == group]
        rows.append({'group': group, 'tissue': 'all (pooled)',
                     **genotype_regression(sub, factor, order, covariates)})
        if by_tissue:
            for t in sorted(sub.tissue.unique()):
                rows.append({'group': group, 'tissue': t,
                             **genotype_regression(sub.loc[sub.tissue == t],
                                                   factor, order, covariates)})
    rows.append({'group': 'all', 'tissue': 'all (pooled)',
                 **genotype_regression(df, factor, order, covariates)})
    return pd.DataFrame(rows)[['group', 'tissue', 'beta', 'se', 'pval', 'n',
                               'r2', 'formula']]


# ================================================================
# The four Figure 5 panels, as ready-made arguments
# ================================================================
#
# Panels E and G draw only the isoforms carrying the two factors above them,
# not all four the gene has; the mapping comes from the `factors` column of the
# isoform bed (see gene_isoforms()):
#
#   KNOP1  factor_1 -> isoform_1      factor_2 -> isoform_2
#   CISD1  factor_4 -> isoform_2      factor_5 -> isoform_4
#
# The highlighted exon is the AD event: the retained intron in KNOP1 isoform_2
# and the long 3' exon in CISD1 isoform_2 that the risk allele shortens.

PANELS = {
    'E': dict(gene='ENSG00000103550', name='KNOP1',
              variant='chr16:19708463:CTCA:C', tissue='PCC',
              transcripts=['ENSG00000103550.isoform_1',
                           'ENSG00000103550.isoform_2'],
              highlight={'ENSG00000103550.isoform_2': (19706479, 19710586)},
              factor_list=['factor1', 'factor2'],
              factor_labels=['Factor 1', 'Factor 2'],
              colors=['tab:blue', 'tab:orange'],
              title='AD-colocalized variant increases intron retention in '
                    '$\\it{KNOP1}$',
              snp_label='SNP chr16:19708463', legend_loc='upper right',
              snp_ha='center'),
    'G': dict(gene='ENSG00000122873', name='CISD1',
              variant='chr10:58286894:G:A', tissue='AC',
              transcripts=['ENSG00000122873.isoform_2',
                           'ENSG00000122873.isoform_4'],
              highlight={'ENSG00000122873.isoform_2': (58287561, 58289682)},
              factor_list=['factor4', 'factor5'],
              factor_labels=['Factor 4', 'Factor 5'],
              colors=['tab:blue', 'tab:orange'],
              title="AD-colocalized variant is associated with shortened 3' "
                    "exon in $\\it{CISD1}$",
              snp_label='SNP chr10:58286894', legend_loc='upper right',
              snp_ha='right'),
}

BOXPLOTS = {
    # F sits under E, H under G, and each takes the colour of the factor track
    # it belongs to: E factor 2 -> orange, G factor 4 -> blue.
    'F': dict(gene='ENSG00000103550', variant='chr16:19708463:CTCA:C',
              tissue='PCC', factor='ENSG00000103550.factor_2',
              order=['CTCA/CTCA', 'CTCA/C', 'C/C'], color='tab:orange',
              ylabel='$\\it{KNOP1}$ factor 2\nnormalized loadings\n(PCC)',
              xlabel='SNP chr16:19708463'),
    'H': dict(gene='ENSG00000122873', variant='chr10:58286894:G:A',
              tissue='AC', factor='ENSG00000122873.factor_4',
              order=['G/G', 'G/A', 'A/A'], color='tab:blue',
              ylabel='$\\it{CISD1}$ factor 4\nnormalized loadings\n(AC)',
              xlabel='SNP chr10:58286894'),
}


def plot_panel(letter, **overrides):
    """
    Draw one of the four published panels: 'E', 'F', 'G' or 'H'.

    Any keyword overrides the stored setting, so
    `plot_panel('E', tissue='AC')` redraws the KNOP1 locus in a different
    tissue, and `plot_panel('F', order=['C/C', 'CTCA/C', 'CTCA/CTCA'])` flips
    the genotype axis. This is a convenience over the general functions, not a
    separate code path.
    """
    if letter in PANELS:
        p = {**PANELS[letter], **overrides}
        cov = load_coverage(p['gene'], p['variant'], p['tissue'])
        fac = load_factors(p['gene'], k=p.get('k', 5))
        spec = locus_spec(p['gene'], name=p['name'],
                          transcripts=p['transcripts'], highlight=p['highlight'],
                          factor_list=p['factor_list'],
                          factor_labels=p['factor_labels'], colors=p['colors'])
        return plot_locus(cov, fac, spec, title=p['title'],
                          snp_label=p['snp_label'],
                          legend_loc=p['legend_loc'], snp_ha=p['snp_ha'],
                          **{k: v for k, v in overrides.items()
                             if k in ('figsize', 'xlim', 'cov_h', 'track_h',
                                      'row_h', 'row_gap', 'smooth', 'pad')})
    if letter in BOXPLOTS:
        p = {**BOXPLOTS[letter], **overrides}
        df = load_boxplot(p['gene'], p['variant'], p['tissue'])
        return plot_genotype_boxplot(df, p['factor'], order=p['order'],
                                     color=p['color'], ylabel=p['ylabel'],
                                     xlabel=p['xlabel'],
                                     **{k: v for k, v in overrides.items()
                                        if k in ('ax', 'figsize', 'show_n',
                                                 'marker_size', 'jitter')})
    raise KeyError(f"{letter!r} is not a panel; pick from "
                   f"{sorted(list(PANELS) + list(BOXPLOTS))}")
