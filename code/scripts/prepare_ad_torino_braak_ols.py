"""
prepare_ad_torino_braak_ols.py

An OLS of braak stage on isoform usage, in place of the logistic regression of
AD status that prepare_ad_torino_k3.py runs. Same shape of output, same
covariates, same per-factor annotation; three things differ:

  response   `braaksc` (0-6, continuous) instead of the binary AD/control
             contrast, so every sample with a braak stage is used rather than
             only cogdx 1 vs 4/5
  usage      taken from the qq-normalised bed
             results/gao_k3/tables/snmf3_EL.all.sorted.qqnorm.bed.gz
  transform  none. That bed is already quantile-normalised, so the
             `np.log(X + 1e-10)` of the logistic script is deliberately absent
             and X enters the model as it is read.

The model per feature is

    braaksc ~ usage + tissue_id + batch + msex + race_string + spanish
              + age_group + log10PMI + RIN

which mirrors the logistic one: usage is the predictor and the phenotype is the
response, so `beta` reads as braak stages per unit of usage.

Writes results/gao_k3/tables/OLS.output.tab.gz.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

EL_BED = ('/project2/yangili1/cfbuenabadn/torino_paper/code/results/gao_k3/'
          'tables/snmf3_EL.all.sorted.qqnorm.bed.gz')
METADATA = ('/project2/mstephens/cfbuenabadn/gtex-stm/code/gao_data/metadata/'
            'merged_metadata.tab.gz')
GENES_DIR = '/project2/mstephens/cfbuenabadn/gtex-stm/data/genes.tab.gz'
ANNOT = ('/project2/yangili1/cfbuenabadn/torino_paper/code/results/gao_k3/'
         'isoforms/annotated.isoforms.tab.gz')
ISOFORMS = ('/project2/yangili1/cfbuenabadn/torino_paper/code/results/gao_k3/'
            'isoforms/isoforms.bed.gz')
OUT = 'results/gao_k3/tables/OLS.output.tab.gz'

COVARIATES = ['tissue_id', 'batch', 'msex', 'race_string', 'spanish',
              'age_group', 'log10PMI', 'RIN']

# ---------------------------------------------------------------- logging ---
# As in prepare_ad_torino_k3.py: the log file is the one the snakemake rule
# declares and passes as the first argument; opened in append mode because the
# rule redirects stdout and stderr into the same file.
import logging as _logging

_LOG_NAME = os.path.splitext(os.path.basename(__file__))[0]
if len(sys.argv) > 1 and sys.argv[1]:
    _log_path = sys.argv[1]
else:
    _log_path = os.path.join('logs', f'{_LOG_NAME}.log')
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

LOG_EVERY = 50          # features between progress lines
# -----------------------------------------------------------------------------


def ols_braak(el_result, samples, braak, metadata=None, min_sample_frac=0.6):
    """
    Per feature, OLS of braak stage on usage with the covariates alongside.

    `el_result` is features x samples and is used untransformed -- it is already
    quantile-normalised. `braak` is the response, indexed by sample.

    Returns one row per feature: beta, pval, ci_lower, ci_upper, converged,
    n_samples, plus BH-FDR across the features that were tested. The column
    names match prepare_ad_torino_k3.py's output so downstream code that reads
    LR.output.tab.gz can read this unchanged.
    """
    cols = [c for c in samples if c in el_result.columns]
    X = el_result[cols].T.astype(float)            # samples x features
    n_total = len(X)
    y_full = braak.reindex(X.index).astype(float)

    # no transform: the bed is already qq-normalised
    X_values = X.values

    if metadata is not None:
        meta = metadata.loc[metadata.index.intersection(X.index)].reindex(X.index)
        meta_parts = []
        for col in meta.columns:
            if pd.api.types.is_numeric_dtype(meta[col]):
                standardized = (meta[col] - meta[col].mean()) / meta[col].std()
                meta_parts.append(standardized.rename(col))
            else:
                meta_parts.append(pd.get_dummies(meta[col], prefix=col,
                                                 drop_first=True))
        confounders = pd.concat(meta_parts, axis=1).astype(float)

        zero_var = confounders.columns[confounders.var() == 0].tolist()
        if zero_var:
            log.info('dropping zero-variance confounder columns: %s', zero_var)
            print(f"Dropping zero-variance confounder columns: {zero_var}")
            confounders = confounders.drop(columns=zero_var)

        if confounders.empty:
            log.warning('all confounder columns dropped, running without them')
            print("Warning: all confounder columns dropped, running without confounders")
            confounders = None
    else:
        confounders = None

    features = el_result.index.tolist()
    records = []

    log.info('OLS braaksc ~ usage: starting over %d features '
             '(%d samples, %d confounder columns)',
             len(features), len(cols),
             0 if confounders is None else confounders.shape[1])
    _t0 = time.time()

    for i, feat in tqdm(enumerate(features)):
        if i and i % LOG_EVERY == 0:
            _el = time.time() - _t0
            log.info('  processed %d/%d features (%.1f%%) in %.1f min '
                     '-- %.2f s/feature', i, len(features),
                     100.0 * i / len(features), _el / 60.0, _el / i)

        x_series = pd.Series(X_values[:, i], index=X.index)

        valid_mask = x_series.notna() & y_full.notna()
        if confounders is not None:
            valid_mask &= confounders.notna().all(axis=1)

        n_valid = int(valid_mask.sum())

        if n_valid < min_sample_frac * n_total:
            records.append({'feature': feat, 'beta': np.nan, 'pval': np.nan,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'converged': False, 'n_samples': n_valid})
            continue

        x = x_series[valid_mask].values.reshape(-1, 1)
        y = y_full[valid_mask].values
        conf = confounders[valid_mask].values if confounders is not None else None
        X_design = np.hstack([x, conf]) if conf is not None else x
        X_const = sm.add_constant(X_design, has_constant='add')

        try:
            result = sm.OLS(y, X_const).fit()
            ci = result.conf_int()
            records.append({'feature': feat,
                            'beta': result.params[1],
                            'pval': result.pvalues[1],
                            'ci_lower': ci[1, 0],
                            'ci_upper': ci[1, 1],
                            'converged': True,
                            'n_samples': n_valid})
        except Exception as e:
            log.warning('OLS failed for %s: %s', feat, e)
            print(f"OLS failed for {feat}: {e}")
            records.append({'feature': feat, 'beta': np.nan, 'pval': np.nan,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'converged': False, 'n_samples': n_valid})

    log.info('OLS braaksc ~ usage: finished %d/%d features in %.1f min, '
             '%d records', len(features), len(features),
             (time.time() - _t0) / 60.0, len(records))

    if not records:
        log.warning('no features were successfully tested')
        print("Warning: no features were successfully tested")
        return pd.DataFrame(columns=['beta', 'pval', 'ci_lower', 'ci_upper',
                                     'converged', 'n_samples', 'pval_fdr'])

    df = pd.DataFrame(records).set_index('feature')
    tested = df['pval'].notna()
    if tested.any():
        _, df.loc[tested, 'pval_fdr'], _, _ = multipletests(
            df.loc[tested, 'pval'], method='fdr_bh')
    else:
        df['pval_fdr'] = np.nan
    return df


# ---- inputs ----------------------------------------------------------------
log.info('reading %s', EL_BED)
EL = pd.read_csv(EL_BED, sep='\t', index_col=3)
samples = list(EL.columns[5:])          # after #Chr/start/end/gid/strand
log.info('usage table: %d features x %d samples', EL.shape[0], len(samples))

annot = pd.read_csv(ANNOT, sep='\t')
iso = pd.read_csv(ISOFORMS, sep='\t',
                  names=['chrom', 'start', 'end', 'gene_id', 'isoform',
                         'strand', 'factors', 'exon', 'notes'])
iso['transcript_id'] = iso.gene_id + '.' + iso.isoform
genes_dir = pd.read_csv(GENES_DIR, sep='\t', names=['gene', 'gene_id', 'loc'])
genes_dir['gene_id'] = genes_dir.gene_id.apply(lambda x: x.split('.')[0])
annot['gene_id'] = annot.transcript.apply(lambda x: x.split('.')[0])

efficiency_factor, ir_factor, ir_only_factor = [], [], []
for idx, row in annot.iterrows():
    appris_ref = row.appris_ref
    aver = appris_ref.split('/')
    if 'RI:' in appris_ref:
        ri_numbers = int([x for x in aver if x.startswith('RI:')][0].split(':')[1])
        appris_length = int(row.appris_transcript_length)
        ir_factor.append(True)
        if ((ri_numbers <= 2) and (appris_length >= 5)) or \
           ((ri_numbers <= 1) and (appris_length >= 3)):
            efficiency_factor.append(False)
            ir_only_factor.append(True)
        else:
            efficiency_factor.append(True)
            ir_only_factor.append(False)
    else:
        efficiency_factor.append(False)
        ir_factor.append(False)
        ir_only_factor.append(False)

annot['efficiency_factor'] = efficiency_factor
annot['IR_factor'] = ir_factor
annot['IR_factor_only'] = ir_only_factor
annot = annot.merge(genes_dir, on='gene_id')

# ---- metadata, prepared exactly as the logistic script does -----------------
metadata = pd.read_csv(METADATA, sep='\t', index_col=0)

clean_numeric = pd.to_numeric(metadata['age_death'].str.replace('+', '', regex=False),
                              errors='coerce')
metadata['age_group'] = pd.cut(clean_numeric,
                               bins=[-float('inf'), 85, 89.9, float('inf')],
                               labels=[0, 1, 2], right=True)
metadata['log10PMI'] = np.log10(metadata.pmi)

string_race = []
for r in metadata.race:
    if r == 1:
        string_race.append('w')
    elif r == 2:
        string_race.append('b')
    elif r == 3:
        string_race.append('ai')
    elif r == 4:
        string_race.append('a')
    elif r >= 5:
        string_race.append('o')
metadata['race_string'] = string_race

# every sample with usage, metadata and a braak stage -- not just cogdx 1 vs 4/5
braak_samples = [s for s in samples
                 if s in metadata.index and pd.notna(metadata.braaksc.get(s))]
log.info('%d of %d samples carry a braak stage', len(braak_samples), len(samples))
log.info('braak distribution: %s',
         metadata.loc[braak_samples].braaksc.value_counts().sort_index().to_dict())

OLS_output = ols_braak(EL.loc[EL['#Chr'] == 'chr16'].dropna(),#EL.dropna(), 
braak_samples,
                       metadata.loc[braak_samples, 'braaksc'],
                       metadata.loc[braak_samples, COVARIATES])

# ---- per-factor annotation, as in prepare_ad_torino_k3.py ------------------
df_mapped = (iso.assign(factors=iso['factors'].str.split(':'))
             .explode('factors')[['factors', 'transcript_id']]
             .drop_duplicates().reset_index(drop=True))

df_mapped_factors = (df_mapped.merge(
    annot[['transcript', 'efficiency_factor', 'IR_factor', 'IR_factor_only']],
    left_on='transcript_id', right_on='transcript', how='left')
    .drop(columns='transcript'))

df_mapped_factors['gene_factor'] = (
    df_mapped_factors.transcript_id.apply(lambda x: x.split('.')[0])
    + '.' + df_mapped_factors.factors)

OLS_output = OLS_output.merge(df_mapped_factors, left_index=True,
                              right_on='gene_factor')
OLS_output['gene_id'] = OLS_output.transcript_id.apply(lambda x: x.split('.')[0])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
OLS_output.to_csv(OUT, sep='\t', index=False, header=True)
log.info('wrote %s: %d rows, %d genes', OUT, len(OLS_output),
         OLS_output.gene_id.nunique())
log.info('significant at FDR 5%%: %d rows',
         int((OLS_output.pval_fdr <= 0.05).sum()))
