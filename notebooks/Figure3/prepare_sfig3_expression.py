"""
prepare_sfig3_expression.py — expression vs. lineage-conservation data prep for
the Figure 3B supplementary figures.

Question: do the genes whose tissue-specific isoform is human-only / primate /
mammal / pan-vertebrate differ in (i) overall expression level and (ii) tissue
specificity?  Same for mouse.

Inputs
------
figure_data/F3B_genes_{human,mouse}.csv   gene lists per Figure 3B cell
GTEx v8 gene median TPM                   human expression
Mazin et al. log2 TPM (mm39)              mouse expression (median per tissue)

Output
------
figure_data/sfig3_expression.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, 'figure_data')
ORTHO_DIR = os.path.join(HERE, '..', '..', 'code', 'resources', 'annotation', 'orthology')

GTEX_MEDIAN = ('/project2/mstephens/cfbuenabadn/gtex-stm/code/gtex_tables/'
               'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz')
MAZIN_DIR   = ('/project/yangili1/bjf79/2024_comparativesplicing/code/'
               'MazinLeafcutterAnalysis/Expression')
MOUSE_TPM   = os.path.join(MAZIN_DIR, 'Mouse_UCSC.mm39_GencodeComprehensive46.log2tpm.tsv.gz')

# Per-gene loss-of-function constraint, pulled out of the 574 MB gnomAD table
# by the awk one-liner recorded in CONSTRAINT_SOURCE.
CONSTRAINT = os.path.join(HERE, 'figure_data', 'gnomad_v4_lof_constraint.tsv.gz')
CONSTRAINT_SOURCE = (
    "zcat code/gnomad.v4.1.2.constraint_metrics.tsv.bgz | awk -F'\\t' "
    "'NR>1 && $3 ~ /^ENST/ {print $1,$3,$4,$5,$90,$94,$111}'")

# ---------------------------------------------------------------------------
# Tissue definitions
# ---------------------------------------------------------------------------

# Exactly the groups the SSMD analysis used (analysis/Animal_numbers.ipynb).
HUMAN_TISSUE_GROUPS = {
    'Brain':                 ['Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex',
                              'Brain_Frontal_Cortex_BA9', 'Brain_Putamen_basal_ganglia'],
    'Cerebellum':            ['Brain_Cerebellar_Hemisphere'],
    'Heart':                 ['Heart_Atrial_Appendage', 'Heart_Left_Ventricle'],
    'Skin':                  ['Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg'],
    'Liver':                 ['Liver'],
    'Testis':                ['Testis'],
    'Muscle_Skeletal':       ['Muscle_Skeletal'],
    'Lung':                  ['Lung'],
    'Whole_Blood':           ['Whole_Blood'],
    'Breast_Mammary_Tissue': ['Breast_Mammary_Tissue'],
    'Adipose_Subcutaneous':  ['Adipose_Subcutaneous'],
}

# The five tissues the Figure 3B conservation call is made over.  Used for the
# matched human/mouse comparison; the full panels are reported alongside.
TISSUES_CONS = ['Brain', 'Cerebellum', 'Liver', 'Heart', 'Testis']

CONS_ORDER = {
    'human': ['Human only', 'Primates', 'Mammals', 'All animals'],
    'mouse': ['Mouse only', 'Rodents', 'Gilres', 'Mammals', 'All animals'],
}
TISSUE_ROWS = ['Brain only', 'Cerebellum only', 'Brain & Cerebellum only',
               'Liver only', 'Heart only', 'Testis only', 'Multiple tissues']


def _norm(name):
    """'Brain_Cortex' and 'Brain - Cortex' collapse to the same key."""
    return ''.join(c for c in name.lower() if c.isalnum())


# ---------------------------------------------------------------------------
# Expression summaries
# ---------------------------------------------------------------------------

def tau(x):
    """
    Yanai tissue-specificity index over a linear expression vector:
    0 = uniform across tissues, 1 = restricted to a single tissue.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    mx = x.max(axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (1 - x / mx).sum(axis=-1) / (x.shape[-1] - 1)
    return np.where(mx.ravel() > 0, out, np.nan)


def summarise(tpm):
    """Per-gene expression summaries from a genes x tissues linear TPM frame."""
    return pd.DataFrame({
        'median_tpm': tpm.median(axis=1),
        'mean_tpm':   tpm.mean(axis=1),
        'max_tpm':    tpm.max(axis=1),
        'tau':        tau(tpm.values),
        'top_tissue': tpm.columns[np.argmax(tpm.values, axis=1)],
    }, index=tpm.index)


# ---------------------------------------------------------------------------
# Human
# ---------------------------------------------------------------------------

def load_human_expression():
    print('  reading GTEx median TPM ...', flush=True)
    gct = pd.read_csv(GTEX_MEDIAN, sep='\t', skiprows=2)
    lookup = {_norm(c): c for c in gct.columns}

    groups = {}
    for group, members in HUMAN_TISSUE_GROUPS.items():
        cols = [lookup[_norm(m)] for m in members]
        # Group median approximated by the mean of its member tissue medians;
        # exact for the eight single-member groups.
        groups[group] = gct[cols].mean(axis=1)
    tpm = pd.DataFrame(groups)
    tpm.index = gct['Description'].values
    tpm.index.name = 'gene'

    # A handful of symbols are duplicated in GENCODE; keep the better-expressed copy.
    order = tpm.max(axis=1).values
    tpm = tpm.iloc[np.argsort(-order)]
    n_dup = int(tpm.index.duplicated().sum())
    tpm = tpm[~tpm.index.duplicated()]
    print(f'  {len(tpm)} genes, {len(tpm.columns)} tissue groups '
          f'({n_dup} duplicate symbols dropped)', flush=True)
    return tpm


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------

def load_mouse_expression():
    print('  reading Mazin mouse log2 TPM ...', flush=True)
    df = pd.read_csv(MOUSE_TPM, sep='\t', index_col=0)
    tissue = pd.Series([c.split('_')[1] for c in df.columns], index=df.columns)
    # Median across every sample of a tissue (all developmental stages), then
    # back to linear TPM so tau is computed on the same scale as for human.
    med = df.T.groupby(tissue).median().T
    tpm = np.power(2.0, med)
    tpm.index = [i.split('.')[0] for i in tpm.index]
    print(f'  {len(tpm)} genes, tissues: {", ".join(tpm.columns)}', flush=True)

    print('  mapping ENSMUSG -> gene symbol ...', flush=True)
    name_of = {}
    for fname, id_col, name_col in [
            ('human_mouse_orthology.txt', 'Mouse gene stable ID', 'Mouse gene name'),
            ('mouse_rat_orthology.txt',   'Gene stable ID',       'Gene name'),
            ('mouse_rabbit_orthology.txt',  'Gene stable ID',       'Gene name')]:
        path = os.path.join(ORTHO_DIR, fname)
        if not os.path.exists(path):
            continue
        t = pd.read_csv(path, sep='\t')
        if id_col not in t.columns or name_col not in t.columns:
            continue
        t = t[[id_col, name_col]].dropna()
        name_of.update(dict(zip(t[id_col], t[name_col])))

    symbols = pd.Series(tpm.index).map(name_of)
    tpm = tpm[symbols.notna().values]
    tpm.index = symbols.dropna().values
    tpm.index.name = 'gene'
    order = tpm.max(axis=1).values
    tpm = tpm.iloc[np.argsort(-order)]
    tpm = tpm[~tpm.index.duplicated()]
    print(f'  {len(tpm)} symbols resolved', flush=True)
    return tpm


# ---------------------------------------------------------------------------
# Loss-of-function constraint
# ---------------------------------------------------------------------------

def load_constraint():
    """
    One pLI / LOEUF per gene symbol: the MANE Select transcript where there is
    one, otherwise the Ensembl canonical.  pLI is the probability the gene is
    intolerant of heterozygous loss of function; LOEUF is the upper bound of
    the observed/expected LoF ratio, so LOW LOEUF means constrained.
    """
    print('  reading gnomAD v4.1.2 LoF constraint ...', flush=True)
    df = pd.read_csv(CONSTRAINT, sep='\t')
    # pandas types the flag columns as bool, so compare against the value
    # rather than the string the file literally holds.
    mane = df.mane_select.astype(str).str.lower() == 'true'
    cano = df.canonical.astype(str).str.lower() == 'true'
    df['rank'] = np.where(mane, 0, np.where(cano, 1, 2))
    df = df[df['rank'] < 2].sort_values('rank').drop_duplicates('gene')
    df = df.set_index('gene')[['pLI', 'loeuf', 'lof_oe']]
    print(f'  {len(df)} genes with constraint, '
          f'{int((df.pLI >= 0.9).sum())} at pLI >= 0.9', flush=True)
    return df


def mouse_to_human():
    """Mouse symbol -> human symbol, one-to-one orthologues only."""
    t = pd.read_csv(os.path.join(ORTHO_DIR, 'human_mouse_orthology.txt'), sep='\t')
    t = t[t['Mouse homology type'] == 'ortholog_one2one'].dropna(
        subset=['Gene name', 'Mouse gene name'])
    return dict(zip(t['Mouse gene name'], t['Gene name']))


def attach_constraint(tab, constraint, species):
    """
    gnomAD is human, so mouse genes inherit the constraint of their one-to-one
    human orthologue.  That is a statement about the orthologous gene, not
    about mouse variation, and the mouse panels are labelled as such.
    """
    if species == 'human':
        key = pd.Series(tab.index, index=tab.index)
    else:
        key = pd.Series(tab.index, index=tab.index).map(mouse_to_human())
    tab = tab.copy()
    for col in ('pLI', 'loeuf'):
        tab[col] = key.map(constraint[col])
    n = int(tab.pLI.notna().sum())
    print(f'  {n} of {len(tab)} genes carry a pLI '
          f'({100 * n / len(tab):.0f}%)', flush=True)
    return tab


def constraint_stats(tab, cons_order, shared_from=1):
    """
    Does constraint track how deeply the isoform is shared, and does it do so
    independently of expression?

    `shared_from` is the first category index counted as shared, so "shared"
    means anything beyond the reference species alone.
    """
    sub = tab[tab.pLI.notna()].copy()
    depth = sub.conservation.cat.codes.values + 1
    sub['depth']  = depth
    sub['shared'] = (sub.conservation.cat.codes >= shared_from).astype(int)

    out = {'n': len(sub), 'shared_frac': float(sub.shared.mean())}

    for col in ('pLI', 'loeuf'):
        rho, p = stats.spearmanr(sub['depth'], sub[col])
        groups = [sub.loc[sub.conservation == c, col].dropna().values
                  for c in cons_order]
        out[col] = {
            'rho': rho, 'p_rho': p,
            'kw_p': stats.kruskal(*[g for g in groups if len(g) > 1])[1],
            'mw_p': stats.mannwhitneyu(groups[0], groups[-1])[1],
            'summary': pd.DataFrame({
                'n':      [len(g) for g in groups],
                'median': [np.median(g) if len(g) else np.nan for g in groups],
                'q25':    [np.percentile(g, 25) if len(g) else np.nan for g in groups],
                'q75':    [np.percentile(g, 75) if len(g) else np.nan for g in groups],
            }, index=cons_order),
            'groups': dict(zip(cons_order, groups)),
        }

    # Fraction of highly constrained genes per category, with a binomial CI
    hi = sub.assign(hi=(sub.pLI >= 0.9).astype(int))
    frac = hi.groupby('conservation', observed=False).hi.agg(['sum', 'size'])
    frac['frac'] = frac['sum'] / frac['size']
    lo_ci, hi_ci = [], []
    for k, n in zip(frac['sum'], frac['size']):
        a, b = stats.beta.ppf([0.025, 0.975], [k + 0.5, k + 0.5], [n - k + 0.5, n - k + 0.5])
        lo_ci.append(0.0 if n == 0 else a)
        hi_ci.append(1.0 if n == 0 else b)
    frac['lo'], frac['hi'] = lo_ci, hi_ci
    out['pLI_high_frac'] = frac.reindex(cons_order)

    # Is constraint anything but a proxy for expression?  Partial correlation of
    # depth with pLI given expression, then a logistic model of being shared.
    expr = sub['log2_median_tpm_cons'].values
    out['partial_rho'] = {
        'pLI':   _partial_spearman(sub['depth'].values, sub.pLI.values, expr),
        'loeuf': _partial_spearman(sub['depth'].values, sub.loeuf.values, expr),
    }
    out['logit']       = _shared_model(sub, 'pLI')
    out['logit_loeuf'] = _shared_model(sub, 'loeuf')
    out['grid']        = _interaction_grid(sub)
    return out


def _partial_spearman(a, b, c):
    """Spearman of a and b with c ranked out of both."""
    ra, rb, rc = (stats.rankdata(x) for x in (a, b, c))
    resid = []
    for r in (ra, rb):
        beta = np.polyfit(rc, r, 1)
        resid.append(r - np.polyval(beta, rc))
    rho, p = stats.pearsonr(resid[0], resid[1])
    return {'rho': rho, 'p': p}


def _shared_model(sub, constraint_col='pLI'):
    """
    Logistic regression of "shared beyond the reference species" on expression
    and constraint, both standardised, with their interaction.  Coefficients
    are log odds ratios per standard deviation, so the two predictors are
    directly comparable.  LOEUF runs the other way from pLI: low = constrained.
    """
    z = lambda v: (v - v.mean()) / v.std()
    sub = sub[sub[constraint_col].notna()]
    inter = f'expression x {constraint_col}'
    X = pd.DataFrame({
        'expression':     z(sub['log2_median_tpm_cons']).values,
        constraint_col:   z(sub[constraint_col]).values,
    })
    X[inter] = X.expression * X[constraint_col]
    X = sm.add_constant(X)
    fit = sm.Logit(sub['shared'].values, X).fit(disp=0)
    ci = fit.conf_int()
    return {
        'params': fit.params, 'pvalues': fit.pvalues,
        'ci_lo': ci[0], 'ci_hi': ci[1],
        'n': int(fit.nobs), 'prsquared': float(fit.prsquared),
        'terms': ['expression', constraint_col, inter],
        'constraint_col': constraint_col,
    }


def _interaction_grid(sub, n_bins=5, pli_cut=0.9):
    """
    Fraction shared across expression bins, split by whether the gene is
    LoF-intolerant.  pLI is heaped at 0 and 1 — roughly half the genes sit
    exactly at zero — so quantile bins of pLI split ties arbitrarily and
    produce a spurious U shape.  The conventional pLI >= 0.9 cut is the
    honest way to use it.
    """
    qe = pd.qcut(sub['log2_median_tpm_cons'].rank(method='first'),
                 n_bins, labels=False)
    hi = sub.pLI >= pli_cut

    out = {}
    for name, mask in (('constrained', hi), ('unconstrained', ~hi)):
        g = sub[mask].groupby(qe[mask], observed=False).shared
        k, n = g.sum(), g.size()
        lo = stats.beta.ppf(0.025, k + 0.5, (n - k) + 0.5)
        up = stats.beta.ppf(0.975, k + 0.5, (n - k) + 0.5)
        out[name] = pd.DataFrame({'k': k, 'n': n, 'frac': k / n,
                                  'lo': lo, 'hi': up})
    out['expr_labels'] = sub.groupby(qe, observed=False)['log2_median_tpm_cons'].median()
    out['pli_cut'] = pli_cut
    out['n_bins'] = n_bins
    return out


# ---------------------------------------------------------------------------
# Joining expression to the Figure 3B cells
# ---------------------------------------------------------------------------

def build_gene_table(cells, tpm, cons_order, tissue_subset):
    """
    One row per gene: its deepest conservation category plus expression
    summaries over all tissues and over the five conservation tissues.
    """
    depth = {c: i + 1 for i, c in enumerate(cons_order)}

    all_sum  = summarise(tpm).add_suffix('_all')
    cons_sum = summarise(tpm[tissue_subset]).add_suffix('_cons')
    expr = all_sum.join(cons_sum)

    # A gene can be specific in several tissues at different depths; take the
    # deepest lineage it is shared to, so every gene is counted once.
    cells = cells.copy()
    cells['depth'] = cells.conservation.map(depth)
    deepest = cells.sort_values('depth').groupby('gene').last()

    tab = deepest.join(expr, how='inner')
    tab['conservation'] = pd.Categorical(tab.conservation, cons_order, ordered=True)
    tab['tissue_specificity'] = pd.Categorical(tab.tissue_specificity,
                                               TISSUE_ROWS, ordered=True)
    for col in ('median_tpm_all', 'median_tpm_cons', 'max_tpm_all', 'max_tpm_cons'):
        tab['log2_' + col] = np.log2(tab[col] + 0.1)

    n_missing = len(deepest) - len(tab)
    print(f'  {len(tab)} genes with expression '
          f'({n_missing} of {len(deepest)} unmatched), '
          f'{int((cells.groupby("gene").conservation.nunique() > 1).sum())} '
          f'genes span >1 conservation category', flush=True)
    return tab


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def trend_stats(tab, value, cons_order):
    """Spearman trend against conservation depth, plus group-level summaries."""
    depth = tab.conservation.cat.codes.values + 1
    v = tab[value].values
    ok = np.isfinite(v)
    rho, p_rho = stats.spearmanr(depth[ok], v[ok])

    groups = [tab.loc[tab.conservation == c, value].dropna().values
              for c in cons_order]
    kw_h, kw_p = stats.kruskal(*[g for g in groups if len(g) > 1])
    mw_u, mw_p = stats.mannwhitneyu(groups[0], groups[-1], alternative='two-sided')

    # Percentile bootstrap CI on rho, for the cross-species summary panel.
    rng = np.random.default_rng(0)
    d_ok, v_ok = depth[ok], v[ok]
    boot = np.empty(2000)
    for b in range(boot.size):
        idx = rng.integers(0, d_ok.size, d_ok.size)
        boot[b] = stats.spearmanr(d_ok[idx], v_ok[idx]).statistic
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        'value': value, 'rho': rho, 'p_rho': p_rho, 'rho_lo': lo, 'rho_hi': hi,
        'kw_H': kw_h, 'kw_p': kw_p, 'mw_p': mw_p,
        'n': int(ok.sum()),
        'summary': pd.DataFrame({
            'n':      [len(g) for g in groups],
            'median': [np.median(g) if len(g) else np.nan for g in groups],
            'q25':    [np.percentile(g, 25) if len(g) else np.nan for g in groups],
            'q75':    [np.percentile(g, 75) if len(g) else np.nan for g in groups],
        }, index=cons_order),
        'groups': dict(zip(cons_order, groups)),
    }


def decile_composition(tab, value, cons_order, n_bins=10):
    """Conservation-category composition across bins of expression."""
    v = tab[value]
    bins = pd.qcut(v.rank(method='first'), n_bins, labels=False)
    comp = (pd.crosstab(bins, tab.conservation)
              .reindex(columns=cons_order, fill_value=0))
    frac = comp.div(comp.sum(axis=1), axis=0)
    centres = v.groupby(bins).median()
    return {'counts': comp, 'frac': frac, 'centres': centres}


def stratified_trend(tab, value, cons_order, cells, min_n=15):
    """
    The same trend computed inside each tissue-specificity row, so the overall
    trend cannot be explained by tissue composition alone.
    """
    long = cells.merge(tab[[value]], left_on='gene', right_index=True, how='inner')
    long['conservation'] = pd.Categorical(long.conservation, cons_order, ordered=True)

    medians, counts, rhos = {}, {}, {}
    for row in TISSUE_ROWS:
        sub = long[long.tissue_specificity == row]
        med = sub.groupby('conservation', observed=False)[value].median()
        cnt = sub.groupby('conservation', observed=False)[value].size()
        # A median over one or two genes is noise; it is returned separately so
        # the panel can show the point as provisional instead of dropping it,
        # which otherwise reads as missing data.
        medians[row] = med
        counts[row]  = cnt
        ok = sub[value].notna()
        if ok.sum() > 30 and sub.loc[ok, 'conservation'].nunique() > 1:
            rhos[row] = stats.spearmanr(sub.loc[ok, 'conservation'].cat.codes,
                                        sub.loc[ok, value])
        else:
            rhos[row] = (np.nan, np.nan)
    return {'medians': pd.DataFrame(medians).T[cons_order],
            'counts':  pd.DataFrame(counts).T[cons_order],
            'min_n':   min_n,
            'rho':     rhos}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(species, csv, tpm, tissue_subset, constraint):
    cons_order = CONS_ORDER[species]
    cells = pd.read_csv(csv)
    tab = build_gene_table(cells, tpm, cons_order, tissue_subset)
    tab = attach_constraint(tab, constraint, species)

    out = {
        'species':     species,
        'cons_order':  cons_order,
        'tissues_all': list(tpm.columns),
        'tissues_cons': list(tissue_subset),
        'table':       tab,
        'cells':       cells,
    }
    for value in ('log2_median_tpm_cons', 'log2_median_tpm_all',
                  'tau_cons', 'tau_all'):
        out[value] = trend_stats(tab, value, cons_order)
    # Being called significant in more species requires detection in more
    # species, which itself tracks expression.  Re-test among genes that are
    # comfortably above the detection floor everywhere.
    floor = 5.0
    well = tab[tab.median_tpm_cons >= floor]
    out['expr_floor'] = floor
    out['n_well'] = len(well)
    for value in ('log2_median_tpm_cons', 'tau_cons'):
        out['well_' + value] = trend_stats(well, value, cons_order)

    out['decile_expr'] = decile_composition(tab, 'log2_median_tpm_cons', cons_order)
    out['decile_tau']  = decile_composition(tab, 'tau_cons', cons_order)
    out['strat_expr']  = stratified_trend(tab, 'log2_median_tpm_cons', cons_order, cells)
    out['strat_tau']   = stratified_trend(tab, 'tau_cons', cons_order, cells)
    out['constraint']  = constraint_stats(tab, cons_order)
    return out


def main():
    print('[1/4] human expression', flush=True)
    human_tpm = load_human_expression()
    print('[2/4] mouse expression', flush=True)
    mouse_tpm = load_mouse_expression()

    print('[3/4] LoF constraint', flush=True)
    constraint = load_constraint()

    print('[4/4] joining and testing', flush=True)
    mouse_cons = [t for t in TISSUES_CONS if t in mouse_tpm.columns]
    res = {
        'human': run('human', os.path.join(DATA_DIR, 'F3B_genes_human.csv'),
                     human_tpm, TISSUES_CONS, constraint),
        'mouse': run('mouse', os.path.join(DATA_DIR, 'F3B_genes_mouse.csv'),
                     mouse_tpm, mouse_cons, constraint),
    }

    path = os.path.join(DATA_DIR, 'sfig3_expression.pkl')
    with open(path, 'wb') as fh:
        pickle.dump(res, fh)
    print(f'wrote {path}', flush=True)

    for sp in ('human', 'mouse'):
        r = res[sp]
        print(f'\n===== {sp.upper()} =====')
        for value in ('log2_median_tpm_cons', 'tau_cons'):
            s = r[value]
            print(f'\n{value}:  Spearman rho={s["rho"]:+.3f} p={s["p_rho"]:.3g}   '
                  f'KW p={s["kw_p"]:.3g}   extremes MWU p={s["mw_p"]:.3g}')
            print(s['summary'].to_string(float_format=lambda x: f'{x:7.3f}'))
        print(f'\nwell-expressed only (median TPM >= {r["expr_floor"]}, n={r["n_well"]}):')
        for value in ('log2_median_tpm_cons', 'tau_cons'):
            w = r['well_' + value]
            print(f'  {value:22s} rho={w["rho"]:+.3f} p={w["p_rho"]:.3g}  '
                  f'extremes MWU p={w["mw_p"]:.3g}')
        print('\nstratified rho (expression):')
        for row, (rho, p) in r['strat_expr']['rho'].items():
            print(f'  {row:26s} rho={rho:+.3f}  p={p:.3g}')
        print('stratified rho (tau):')
        for row, (rho, p) in r['strat_tau']['rho'].items():
            print(f'  {row:26s} rho={rho:+.3f}  p={p:.3g}')

        c = r['constraint']
        print(f'\nLoF constraint (n={c["n"]}, {100*c["shared_frac"]:.1f}% shared '
              f'beyond the reference species):')
        for col in ('pLI', 'loeuf'):
            print(f'  {col:6s} rho={c[col]["rho"]:+.3f} p={c[col]["p_rho"]:.3g}   '
                  f'extremes MWU p={c[col]["mw_p"]:.3g}')
            print(c[col]['summary'].to_string(float_format=lambda x: f'{x:7.3f}'))
        print('  fraction pLI >= 0.9 per category:')
        print(c['pLI_high_frac'][['sum', 'size', 'frac']].to_string(
            float_format=lambda x: f'{x:7.3f}'))
        for col, pr in c['partial_rho'].items():
            print(f'  partial rho(depth, {col} | expression) = {pr["rho"]:+.3f} '
                  f'p={pr["p"]:.3g}')
        for key in ('logit', 'logit_loeuf'):
            lg = c[key]
            print(f'  logistic model of being shared, {lg["constraint_col"]} '
                  f'(n={lg["n"]}, pseudo-R2={lg["prsquared"]:.4f}):')
            for t in lg['terms']:
                print(f'    {t:22s} log-OR={lg["params"][t]:+.3f} '
                      f'[{lg["ci_lo"][t]:+.3f}, {lg["ci_hi"][t]:+.3f}]  '
                      f'OR={np.exp(lg["params"][t]):.3f}  p={lg["pvalues"][t]:.3g}')
        g = c['grid']
        print(f'  fraction shared by expression quintile, '
              f'pLI >= {g["pli_cut"]} vs below:')
        print(pd.DataFrame({
            'expr':          g['expr_labels'].round(2),
            'constrained':   g['constrained'].frac.round(3),
            'n_con':         g['constrained'].n,
            'unconstrained': g['unconstrained'].frac.round(3),
            'n_unc':         g['unconstrained'].n,
        }).to_string())


if __name__ == '__main__':
    main()
