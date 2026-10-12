"""
prepare_sfig2_foundce.py — data prep for the SFigure 2 panels on recovery of
annotated brain-specific cassette exons.

VastDB calls a brain-specific cassette exon in a gene; `analysis/found_CE.tsv`
records whether Torino recovered the isoform that INCLUDES the exon
(`found_exon`), the isoform that SKIPS it (`found_skip`), both, or neither.
The question here is what separates the events we recover from the ones we miss.

Expression and constraint come from the same sources as the Figure 3b
supplement, reused rather than re-derived, so the two supplements cannot
disagree about what "median TPM" means.

Output: figure_data/sfig2_foundce.pkl
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'figure_data')
F3_DIR   = os.path.join(HERE, '..', 'Figure3')
ANALYSIS = os.path.join(HERE, '..', '..', 'analysis')
if F3_DIR not in sys.path:
    sys.path.insert(0, F3_DIR)

from prepare_sfig3_expression import (        # noqa: E402  -- path set above
    load_human_expression, load_constraint, summarise, tau, TISSUES_CONS)

FOUND_CE = os.path.join(ANALYSIS, 'found_CE.tsv')

# The four outcomes, in the order the original plot used.
CATEGORIES = ['CE & flanking', 'flanking only', 'CE only', 'None']

# Brain-ish tissue groups, for the "is it expressed where the exon matters"
# covariate.  Cerebellum is kept separate from Brain by the SSMD grouping.
BRAIN_GROUPS = ['Brain', 'Cerebellum']


def load_events():
    """found_CE.tsv plus the fields derivable from it."""
    d = pd.read_csv(FOUND_CE, sep='\t')
    d['found_exon'] = d.found_exon.astype(bool)
    d['found_skip'] = d.found_skip.astype(bool)

    coord = d.exon.str.extract(r':(\d+)-(\d+)').astype(int)
    d['exon_len'] = coord[1] - coord[0]
    # Exon length modulo 3: a frame-preserving exon is a different object from
    # one whose inclusion shifts the reading frame, and the two need not be
    # recovered equally well.
    d['in_frame'] = (d.exon_len % 3 == 0)

    d['category'] = np.select(
        [d.found_exon & d.found_skip,
         ~d.found_exon & d.found_skip,
         d.found_exon & ~d.found_skip],
        CATEGORIES[:3], default=CATEGORIES[3])
    d['category'] = pd.Categorical(d.category, CATEGORIES, ordered=True)
    return d


def attach_gene_level(events, tpm, constraint):
    """Per-event expression, tissue specificity and constraint of its gene."""
    allsum  = summarise(tpm).add_suffix('_all')
    conssum = summarise(tpm[TISSUES_CONS]).add_suffix('_cons')
    expr = allsum.join(conssum)
    expr['brain_tpm'] = tpm[BRAIN_GROUPS].max(axis=1)

    out = events.join(expr, on='gene')
    for col in ('pLI', 'loeuf'):
        out[col] = out.gene.map(constraint[col])

    out['log2_median_tpm_cons'] = np.log2(out.median_tpm_cons + 0.1)
    out['log2_brain_tpm']       = np.log2(out.brain_tpm + 0.1)
    out['log2_exon_len']        = np.log2(out.exon_len)

    n_expr = int(out.median_tpm_cons.notna().sum())
    n_pli  = int(out.pLI.notna().sum())
    print(f'  {len(out)} events, {out.gene.nunique()} genes; '
          f'{n_expr} with expression, {n_pli} with constraint', flush=True)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def by_category(tab, value):
    """Group summaries plus a Kruskal-Wallis across the four outcomes."""
    groups = [tab.loc[tab.category == c, value].dropna().values for c in CATEGORIES]
    big = [g for g in groups if len(g) > 2]
    kw_p = stats.kruskal(*big)[1] if len(big) > 1 else np.nan
    return {
        'value': value, 'kw_p': kw_p,
        'groups': dict(zip(CATEGORIES, groups)),
        'summary': pd.DataFrame({
            'n':      [len(g) for g in groups],
            'median': [np.median(g) if len(g) else np.nan for g in groups],
            'q25':    [np.percentile(g, 25) if len(g) else np.nan for g in groups],
            'q75':    [np.percentile(g, 75) if len(g) else np.nan for g in groups],
        }, index=CATEGORIES),
    }


def by_outcome(tab, value, outcome='found_exon'):
    """
    The same variable split by the binary outcome that actually matters.
    With 7 and 8 events in two of the four categories, the four-way test is
    underpowered; the binary split is what the statistics should rest on.
    """
    a = tab.loc[tab[outcome], value].dropna().values
    b = tab.loc[~tab[outcome], value].dropna().values
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    # Rank-biserial correlation: a readable effect size for a rank test, and
    # the only thing that makes six variables on different scales comparable.
    rbc = 2 * u / (len(a) * len(b)) - 1

    rng = np.random.default_rng(0)
    boot = np.empty(2000)
    for i in range(boot.size):
        aa = rng.choice(a, a.size, replace=True)
        bb = rng.choice(b, b.size, replace=True)
        boot[i] = 2 * stats.mannwhitneyu(aa, bb).statistic / (a.size * b.size) - 1
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {'value': value, 'outcome': outcome,
            'found': a, 'missed': b,
            'n_found': len(a), 'n_missed': len(b),
            'median_found': float(np.median(a)) if len(a) else np.nan,
            'median_missed': float(np.median(b)) if len(b) else np.nan,
            'p': p, 'rank_biserial': rbc, 'rb_lo': lo, 'rb_hi': hi}


def recovery_model(tab, terms, outcome='found_exon'):
    """Logistic regression of recovery on standardised predictors."""
    sub = tab.dropna(subset=terms + [outcome])
    z = lambda v: (v - v.mean()) / v.std()
    X = pd.DataFrame({t: z(sub[t]).values for t in terms}, index=sub.index)
    X = sm.add_constant(X)
    fit = sm.Logit(sub[outcome].astype(int).values, X).fit(disp=0)
    ci = fit.conf_int()
    return {'params': fit.params, 'pvalues': fit.pvalues,
            'ci_lo': ci[0], 'ci_hi': ci[1], 'terms': terms,
            'n': int(fit.nobs), 'prsquared': float(fit.prsquared),
            'outcome': outcome}


def recovery_by_bin(tab, value, n_bins=4, outcome='found_exon'):
    """Recovery rate across quantile bins of `value`, with binomial CIs."""
    sub = tab.dropna(subset=[value])
    q = pd.qcut(sub[value].rank(method='first'), n_bins, labels=False)
    g = sub.groupby(q)[outcome]
    k, n = g.sum(), g.size()
    lo = stats.beta.ppf(0.025, k + 0.5, (n - k) + 0.5)
    hi = stats.beta.ppf(0.975, k + 0.5, (n - k) + 0.5)
    return pd.DataFrame({'k': k, 'n': n, 'rate': k / n, 'lo': lo, 'hi': hi,
                         'centre': sub.groupby(q)[value].median()})


def main():
    print('[1/3] events', flush=True)
    events = load_events()
    print(events.category.value_counts().reindex(CATEGORIES).to_string())
    print(f"  found_exon {events.found_exon.sum()}/{len(events)}   "
          f"found_skip {events.found_skip.sum()}/{len(events)}")

    print('[2/3] expression and constraint', flush=True)
    tpm = load_human_expression()
    constraint = load_constraint()
    tab = attach_gene_level(events, tpm, constraint)

    print('[3/3] testing', flush=True)
    values = ['log2_median_tpm_cons', 'log2_brain_tpm', 'tau_cons',
              'log2_exon_len', 'pLI', 'loeuf']
    res = {
        'table': tab,
        'categories': CATEGORIES,
        'counts': events.category.value_counts().reindex(CATEGORIES),
        'n_found_exon': int(events.found_exon.sum()),
        'n_found_skip': int(events.found_skip.sum()),
        'n_events': len(events),
        'by_category': {v: by_category(tab, v) for v in values},
        'by_outcome':  {v: by_outcome(tab, v) for v in values},
        'rate_expr':   recovery_by_bin(tab, 'log2_median_tpm_cons'),
        'rate_brain':  recovery_by_bin(tab, 'log2_brain_tpm'),
        'rate_len':    recovery_by_bin(tab, 'log2_exon_len'),
        # Brain expression is the covariate the biology predicts — a
        # brain-specific exon needs brain coverage to be seen — so it, not the
        # five-tissue median, is the one the model is built on.
        'model':       recovery_model(
            tab, ['log2_brain_tpm', 'log2_exon_len', 'tau_cons', 'loeuf']),
        'model_overall': recovery_model(
            tab, ['log2_median_tpm_cons', 'log2_exon_len', 'tau_cons', 'loeuf']),
        'model_both':  recovery_model(
            tab, ['log2_brain_tpm', 'log2_median_tpm_cons']),
    }

    path = os.path.join(DATA_DIR, 'sfig2_foundce.pkl')
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(res, fh)
    print(f'wrote {path}')

    print('\n===== by four-way category =====')
    for v in values:
        s = res['by_category'][v]
        print(f'\n{v}  (Kruskal-Wallis p = {s["kw_p"]:.3g})')
        print(s['summary'].to_string(float_format=lambda x: f'{x:8.3f}'))

    print('\n===== recovered vs missed (found_exon) =====')
    for v in values:
        s = res['by_outcome'][v]
        print(f'  {v:22s} found {s["median_found"]:8.3f} (n={s["n_found"]})  '
              f'missed {s["median_missed"]:8.3f} (n={s["n_missed"]})  '
              f'p={s["p"]:.3g}  r_rb={s["rank_biserial"]:+.3f}')

    for key, label in (('rate_brain', 'brain expression'),
                       ('rate_expr', 'five-tissue median expression')):
        print(f'\n===== recovery rate by {label} quartile =====')
        print(res[key].to_string(float_format=lambda x: f'{x:7.3f}'))

    for key in ('model', 'model_overall', 'model_both'):
        m = res[key]
        print(f'\n===== {key} (n={m["n"]}, pseudo-R2={m["prsquared"]:.4f}) =====')
        for t in m['terms']:
            print(f'  {t:22s} OR={np.exp(m["params"][t]):.3f} '
                  f'[{np.exp(m["ci_lo"][t]):.3f}, {np.exp(m["ci_hi"][t]):.3f}]  '
                  f'p={m["pvalues"][t]:.3g}')


if __name__ == '__main__':
    main()
