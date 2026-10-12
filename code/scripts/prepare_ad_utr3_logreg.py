"""
prepare_ad_utr3_logreg.py

The 3' UTR logistic regression of AD_UTR.ipynb (cells 70, 109), rerun with the
covariate set prepare_ad_torino_k3.py uses.

What the pipeline does, in order:

  1. Pick isoforms. annot_ref is the reference/intron-chain-matching set; of
     those, keep genes with more than one such isoform and drop every
     **efficiency factor**. This is where efficiency factors leave -- at isoform
     selection, before any grouping.
  2. Group by 3' UTR. For each gene, take each isoform's last exon, key it by
     the shared exon *start*, and cluster the *ends* that fall within 200 bp of
     one another (`compatible`). Each cluster is one distinct 3' UTR length;
     they are ranked shortest -> longest as rank0, rank1, ...
  3. Collapse the loadings. Within a UTR-start group, renormalise the EL rows of
     its factors **per sample** so they sum to 1, masking samples where those
     factors account for less than `min_proportion` (0.2) of the gene to NaN,
     then sum the renormalised rows of each length-rank group. One row per
     gene-rank, named '{gene}_rank{n}_{isoform}_{isoform}...'.
  4. Logistic regression of AD status on each row's usage.

Steps 1-3 are Figure5_helpers.build_el_utr, imported rather than copied so there
is one implementation; they are also the slow part, so the cached
notebooks/Figure5/figure_data/utr_el.pkl is reused when present.

K=5 throughout: the loadings are snmf5_EL.tab.gz and the isoform annotation is
the gao (K=5) set, as in AD_UTR.ipynb.

The one change from the notebook is the covariates:

    notebook   tissue_id, batch, msex, race
    here       tissue_id, batch, msex, race_string, spanish, age_group,
               log10PMI, RIN

Writes results/gao_k5/tables/UTR3.logreg.output.tab.gz.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

FIG5_DIR = '/project2/yangili1/cfbuenabadn/torino_paper/notebooks/Figure5'
EL_CACHE = os.path.join(FIG5_DIR, 'figure_data', 'utr_el.pkl')
METADATA = ('/project2/mstephens/cfbuenabadn/gtex-stm/code/gao_data/metadata/'
            'merged_metadata.tab.gz')
OUT = 'results/gao_k5/tables/UTR3.logreg.output.tab.gz'
# The usage matrix the regression is fitted on, written next to the results. It
# otherwise exists only as notebooks/Figure5/figure_data/utr_el.pkl, a cache
# that is documented as deletable -- so without this the inputs of a finished
# run could not be recovered.
OUT_USAGE = 'results/gao_k5/tables/UTR3.usage.tab.gz'

COVARIATES = ['tissue_id', 'batch', 'msex', 'race_string', 'spanish',
              'age_group', 'log10PMI', 'RIN']
MIN_PROPORTION = 0.2

# ---------------------------------------------------------------- logging ---
import logging as _logging

_LOG_NAME = os.path.splitext(os.path.basename(__file__))[0]
_log_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else \
    os.path.join('logs', f'{_LOG_NAME}.log')
try:
    _d = os.path.dirname(_log_path)
    if _d:
        os.makedirs(_d, exist_ok=True)
except OSError:
    _log_path = f'{_LOG_NAME}.log'

log = _logging.getLogger(_LOG_NAME)
log.setLevel(_logging.INFO)
if not log.handlers:
    _h = _logging.FileHandler(_log_path, mode='a')
    _h.setFormatter(_logging.Formatter('%(asctime)s  %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    log.addHandler(_h)
log.info('log file: %s', _log_path)

LOG_EVERY = 50
# -----------------------------------------------------------------------------


def logistic_regression_groups(el_result, group1_cols, group2_cols,
                               metadata=None, min_sample_frac=0.6):
    """
    AD status on usage, one row at a time -- the `min_sample_frac` version from
    AD_UTR.ipynb cell 64, with the same CLR-free log transform the notebook
    applies to the renormalised usage.
    """
    keep1 = [c for c in group1_cols if c in el_result.columns]
    keep2 = [c for c in group2_cols if c in el_result.columns]
    cols = keep1 + keep2

    X = el_result[cols].T.astype(float)
    n_total = len(X)
    y_full = pd.Series([0] * len(keep1) + [1] * len(keep2), index=cols)
    log_X = np.log(X.values + 1e-10)

    if metadata is not None:
        meta = metadata.loc[metadata.index.intersection(X.index)].reindex(X.index)
        parts = []
        for col in meta.columns:
            if pd.api.types.is_numeric_dtype(meta[col]):
                parts.append(((meta[col] - meta[col].mean()) / meta[col].std())
                             .rename(col))
            else:
                parts.append(pd.get_dummies(meta[col], prefix=col, drop_first=True))
        confounders = pd.concat(parts, axis=1).astype(float)
        zero_var = confounders.columns[confounders.var() == 0].tolist()
        if zero_var:
            log.info('dropping zero-variance confounders: %s', zero_var)
            confounders = confounders.drop(columns=zero_var)
        if confounders.empty:
            log.warning('all confounders dropped')
            confounders = None
    else:
        confounders = None

    features = el_result.index.tolist()
    records = []
    log.info('UTR3 logistic regression: starting over %d rows '
             '(%d control + %d AD samples, %d confounder columns)',
             len(features), len(keep1), len(keep2),
             0 if confounders is None else confounders.shape[1])
    _t0 = time.time()

    for i, feat in tqdm(enumerate(features)):
        if i and i % LOG_EVERY == 0:
            _el = time.time() - _t0
            log.info('  processed %d/%d rows (%.1f%%) in %.1f min -- %.2f s/row',
                     i, len(features), 100.0 * i / len(features),
                     _el / 60.0, _el / i)

        x_series = pd.Series(log_X[:, i], index=X.index)
        valid = x_series.notna()
        if confounders is not None:
            valid &= confounders.notna().all(axis=1)
        n_valid = int(valid.sum())

        if n_valid < min_sample_frac * n_total:
            records.append({'feature': feat, 'beta': np.nan, 'pval': np.nan,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'converged': False, 'n_samples': n_valid})
            continue

        x = x_series[valid].values.reshape(-1, 1)
        y = y_full[valid].values
        conf = confounders[valid].values if confounders is not None else None
        design = np.hstack([x, conf]) if conf is not None else x
        X_const = sm.add_constant(design, has_constant='add')

        try:
            result = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
            if not result.converged:
                raise ValueError('did not converge')
            ci = result.conf_int()
            records.append({'feature': feat, 'beta': result.params[1],
                            'pval': result.pvalues[1], 'ci_lower': ci[1, 0],
                            'ci_upper': ci[1, 1], 'converged': True,
                            'n_samples': n_valid})
        except Exception as e:
            v1 = x_series[valid][y_full[valid] == 0].values
            v2 = x_series[valid][y_full[valid] == 1].values
            from scipy import stats
            stat, pval = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            rbc = 1 - (2 * stat) / (len(v1) * len(v2))
            log.info('fallback to Mann-Whitney for %s (%s)', feat, e)
            records.append({'feature': feat, 'beta': rbc, 'pval': pval,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'converged': False, 'n_samples': n_valid})

    log.info('UTR3 logistic regression: finished %d rows in %.1f min',
             len(features), (time.time() - _t0) / 60.0)

    df = pd.DataFrame(records).set_index('feature')
    tested = df['pval'].notna()
    if tested.any():
        _, df.loc[tested, 'pval_fdr'], _, _ = multipletests(
            df.loc[tested, 'pval'], method='fdr_bh')
    else:
        df['pval_fdr'] = np.nan
    return df


def build_or_load_el_utr():
    """EL_utr + coord_map: the cached terminal-exon grouping, or a fresh build."""
    import pickle
    if os.path.exists(EL_CACHE):
        log.info('reusing %s', EL_CACHE)
        with open(EL_CACHE, 'rb') as f:
            c = pickle.load(f)
        return c['EL_utr'], c['coord_map'], c['ad'], c['ctrl']

    log.info('no cache at %s, rebuilding (slow: ~3850 genes)', EL_CACHE)
    if FIG5_DIR not in sys.path:
        sys.path.append(FIG5_DIR)
    from Figure5_helpers import load_utr_inputs, build_el_utr
    inputs = load_utr_inputs()                       # K=5: snmf5_EL + gao iso
    EL, metadata_all = inputs['EL'], inputs['metadata_all']
    EL_list, coord_map = build_el_utr(inputs, min_proportion=MIN_PROPORTION)
    ad = list(metadata_all.loc[(metadata_all['condition'] == 'AD') &
                               (metadata_all.specimenID.isin(EL.columns))].specimenID)
    ctrl = list(metadata_all.loc[(metadata_all['condition'] == 'no condition') &
                                 (metadata_all.specimenID.isin(EL.columns))].specimenID)
    return pd.concat(EL_list)[ad + ctrl], coord_map, ad, ctrl


EL_utr, coord_map, ad_samples, ctrl_samples = build_or_load_el_utr()
log.info('EL_utr: %d UTR rows x %d samples (%d AD, %d control)',
         EL_utr.shape[0], EL_utr.shape[1], len(ad_samples), len(ctrl_samples))

# ---- covariates, prepared as prepare_ad_torino_k3.py does -------------------
metadata = pd.read_csv(METADATA, sep='\t')
meta = metadata.copy()
meta.index = meta.specimenID

clean_numeric = pd.to_numeric(meta['age_death'].str.replace('+', '', regex=False),
                              errors='coerce')
meta['age_group'] = pd.cut(clean_numeric,
                           bins=[-float('inf'), 85, 89.9, float('inf')],
                           labels=[0, 1, 2], right=True)
meta['log10PMI'] = np.log10(meta.pmi)
meta['race_string'] = [{1: 'w', 2: 'b', 3: 'ai', 4: 'a'}.get(r, 'o')
                       for r in meta.race]

samples = [s for s in ctrl_samples + ad_samples if s in meta.index]
ctrl_samples = [s for s in ctrl_samples if s in samples]
ad_samples = [s for s in ad_samples if s in samples]
log.info('covariates: %s', COVARIATES)

res = logistic_regression_groups(EL_utr, ctrl_samples, ad_samples,
                                 meta.loc[samples, COVARIATES])
res['last_exon_coords'] = res.index.map(coord_map)
res['gene_id'] = [str(i).split('_rank')[0] for i in res.index]
res['rank'] = [int(str(i).split('_rank')[1].split('_')[0]) for i in res.index]

os.makedirs(os.path.dirname(OUT), exist_ok=True)

usage = EL_utr.copy()
usage.index.name = 'utr_group'
usage.to_csv(OUT_USAGE, sep='\t', index=True, header=True)
log.info('wrote %s: %d UTR groups x %d samples', OUT_USAGE, *usage.shape)

res.to_csv(OUT, sep='\t', index=True, header=True)
log.info('wrote %s: %d rows, %d genes', OUT, len(res), res.gene_id.nunique())
log.info('significant at FDR 5%%: %d rows', int((res.pval_fdr <= 0.05).sum()))
