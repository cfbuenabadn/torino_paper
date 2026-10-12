import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import gzip
from pybedtools import BedTool
import os
from tqdm import tqdm
from scipy.stats import zscore
# import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

from sklearn.preprocessing import OneHotEncoder
from scipy.special import logit, expit
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm 
import pybedtools
from pybedtools import BedTool



from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests



import gseapy as gp
from gseapy import barplot, dotplot

genes_df = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/all_genes.protein_coding.bed.gz', sep='\t', 
                       names = ['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'])


import time

# ---------------------------------------------------------------- logging ---
# Progress log for the logistic regression below, which runs for hours with no
# output of its own. The handler flushes on every record, so the file can be
# tailed while the job runs. Named after this script, so the k3 and k5 runs do
# not write over each other.
import logging as _logging

import sys as _sys

_LOG_NAME = os.path.splitext(os.path.basename(__file__))[0]

# The log file is the one the snakemake rule declares: AD.smk passes {log} as
# the first argument. Run by hand with no argument, it falls back to
# logs/<script name>.log. Opened in append mode because the rule redirects the
# script's stdout and stderr into the same file -- both writers use O_APPEND, so
# the progress lines interleave with the tqdm/print output instead of
# overwriting it.
if len(_sys.argv) > 1 and _sys.argv[1]:
    _log_path = _sys.argv[1]
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

def logistic_regression_groups(el_result, group1_cols, group2_cols, metadata=None,
                                min_sample_frac=0.6):
    keep1 = [c for c in group1_cols if c in el_result.columns]
    keep2 = [c for c in group2_cols if c in el_result.columns]
    cols = keep1 + keep2

    X = el_result[cols].T.astype(float)  # no dropna
    n_total = len(X)
    y_full = pd.Series(
        [0] * sum(c in X.index for c in keep1) +
        [1] * sum(c in X.index for c in keep2),
        index=[c for c in cols if c in X.index]
    )

    # CLR transform using nanmean so NaNs don't corrupt other features
    log_X = X.values
    #log_X = np.log(X.values + 1e-10)
    # row_means = np.nanmean(log_X, axis=1, keepdims=True)
    # X_clr = log_X - row_means

    # build confounder matrix aligned to samples
    if metadata is not None:
        meta = metadata.loc[metadata.index.intersection(X.index)]
        meta = meta.reindex(X.index)

        meta_parts = []
        for col in meta.columns:
            if pd.api.types.is_numeric_dtype(meta[col]):
                standardized = (meta[col] - meta[col].mean()) / meta[col].std()
                meta_parts.append(standardized.rename(col))
            else:
                dummies = pd.get_dummies(meta[col], prefix=col, drop_first=True)
                meta_parts.append(dummies)

        confounders = pd.concat(meta_parts, axis=1).astype(float)

        zero_var = confounders.columns[confounders.var() == 0].tolist()
        if zero_var:
            print(f"Dropping zero-variance confounder columns: {zero_var}")
            confounders = confounders.drop(columns=zero_var)

        if confounders.empty:
            print("Warning: all confounder columns dropped, running without confounders")
            confounders = None
    else:
        confounders = None

    features = el_result.index.tolist()
    records = []

    log.info('logistic regression: starting over %d features '
             '(%d group1 + %d group2 samples, %d confounder columns)',
             len(features), len(keep1), len(keep2),
             0 if confounders is None else confounders.shape[1])
    _t0 = time.time()

    for i, feat in tqdm(enumerate(features)):
        if i and i % LOG_EVERY == 0:
            _el = time.time() - _t0
            log.info('  processed %d/%d features (%.1f%%) in %.1f min '
                     '-- %.2f s/feature', i, len(features),
                     100.0 * i / len(features), _el / 60.0, _el / i)
        x_series = pd.Series(log_X[:, i], index=X.index)

        # valid mask: non-NaN in this feature and all confounders
        valid_mask = x_series.notna()
        if confounders is not None:
            valid_mask &= confounders.notna().all(axis=1)

        n_valid = valid_mask.sum()

        if n_valid < min_sample_frac * n_total:
            records.append({
                'feature':   feat,
                'beta':      np.nan,
                'pval':      np.nan,
                'ci_lower':  np.nan,
                'ci_upper':  np.nan,
                'converged': False,
                'n_samples': n_valid,
            })
            continue

        x = x_series[valid_mask].values.reshape(-1, 1)
        y = y_full[valid_mask].values
        confounder_cols = confounders[valid_mask].values if confounders is not None else None

        if confounder_cols is not None:
            X_design = np.hstack([x, confounder_cols])
        else:
            X_design = x

        X_const = sm.add_constant(X_design, has_constant='add')

        try:
            result = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
            if not result.converged:
                raise ValueError('did not converge')

            conf = result.conf_int()
            records.append({
                'feature':   feat,
                'beta':      result.params[1],
                'pval':      result.pvalues[1],
                'ci_lower':  conf[1, 0],
                'ci_upper':  conf[1, 1],
                'converged': True,
                'n_samples': n_valid,
            })

        except Exception as e:
            print(f"Logistic regression failed for {feat}: {e}, falling back to Mann-Whitney")
            v1 = x_series[valid_mask][y_full[valid_mask] == 0].values
            v2 = x_series[valid_mask][y_full[valid_mask] == 1].values
            stat, pval = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            rbc = 1 - (2 * stat) / (len(v1) * len(v2))
            records.append({
                'feature':   feat,
                'beta':      rbc,
                'pval':      pval,
                'ci_lower':  np.nan,
                'ci_upper':  np.nan,
                'converged': False,
                'n_samples': n_valid,
            })

    log.info('logistic regression: finished %d/%d features in %.1f min, '
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
        _, df.loc[tested, 'pval_fdr'], _, _ = multipletests(df.loc[tested, 'pval'],
                                                             method='fdr_bh')
    else:
        df['pval_fdr'] = np.nan

    return df


EL = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/tables/snmf5_EL.tab.gz', sep='\t', index_col=3)

genes_df = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/all_genes.protein_coding.bed.gz', sep='\t', 
                       names = ['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'])

annot = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/results/gao/isoforms/annotated.isoforms.tab.gz', sep='\t')
iso = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/results/gao/isoforms/isoforms.bed.gz', sep='\t',
                 names = ['chrom', 'start', 'end', 'gene_id', 'isoform', 'strand', 'factors', 'exon', 'notes'])
iso['transcript_id'] = iso.gene_id + '.' + iso.isoform
genes_dir = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/data/genes.tab.gz', sep='\t',
                       names = ['gene', 'gene_id', 'loc'])
genes_dir['gene_id'] = genes_dir.gene_id.apply(lambda x: x.split('.')[0])

annot['gene_id'] = annot.transcript.apply(lambda x: x.split('.')[0])


efficiency_factor = []
ir_factor = []
ir_only_factor = []
for idx, row in annot.iterrows():
    appris_ref = row.appris_ref
    aver = appris_ref.split('/')
    if 'RI:' in appris_ref:
        ri_factor = [x for x in aver if x.startswith('RI:')][0]
        ri_numbers = int(ri_factor.split(':')[1])
        appris_length = int(row.appris_transcript_length)

        ir_factor.append(True)
        
        if ((ri_numbers <= 2) and (appris_length >= 5)) or ((ri_numbers <= 1) and (appris_length >= 3)):
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


metadata = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/gao_data/metadata/merged_metadata.tab.gz', sep='\t', index_col=0)
# metadata.index = metadata.specimenID

samples = EL.columns[5:]

## Change here to correct cogdx to 4 and 5 instead of >= 4
ad_samples = pd.Index(metadata.loc[metadata.cogdx.isin([4, 5])].index).intersection(samples)
control_samples = pd.Index(metadata.loc[metadata.cogdx.isin([1])].index).intersection(samples)

ad_control_samples = list(ad_samples) + list(control_samples)

dlpc_samples = pd.Index(metadata.loc[metadata.tissue_id == 'DLPC'].index).intersection(samples)
ac_samples = pd.Index(metadata.loc[metadata.tissue_id == 'AC'].index).intersection(samples)
pcc_samples = pd.Index(metadata.loc[metadata.tissue_id == 'PCC'].index).intersection(samples)



### Age of death to bins
clean_numeric = pd.to_numeric(metadata['age_death'].str.replace('+', '', regex=False), errors='coerce')

# 2. Define the edges of your bins (use -inf for anything below 80, inf for anything above 90)
bins = [-float('inf'), 85, 89.9, float('inf')]

# 3. Label the bins exactly how you want them to display
labels = [0, 1, 2]

# 4. Segment the data
metadata['age_group'] = pd.cut(clean_numeric, bins=bins, labels=labels, right=True)

### Prepare PMI 
metadata['log10PMI'] = np.log10(metadata.pmi)

### race from numeric to string for one-hot encoding

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




LR_output = logistic_regression_groups(EL.dropna(), control_samples, ad_samples, 
                           metadata.loc[list(control_samples) + list(ad_samples), ['tissue_id', 'batch', 'msex', 'race_string', 
                                                                                   'spanish', 'age_group', 'log10PMI', 'RIN']])


import pandas as pd

# Explode the factors column on ':' separator
df_mapped = (
    iso.assign(factors=iso['factors'].str.split(':'))
    .explode('factors')
    [['factors', 'transcript_id']]
    .drop_duplicates()
    .reset_index(drop=True)
)

df_mapped_factors = (
    df_mapped
    .merge(
        annot[['transcript', 'efficiency_factor', 'IR_factor', 'IR_factor_only']],
        left_on='transcript_id',
        right_on='transcript',
        how='left'
    )
    .drop(columns='transcript')
)


import pandas as pd
import numpy as np

factor_cols = ['factor_1', 'factor_2', 'factor_3', 'factor_4', 'factor_5']

# Build (gene_id, factor) -> lookup for each flag
base = (
    df_mapped_factors
    .assign(gene_id=df_mapped_factors['transcript_id'].str.extract(r'(ENSG\d+)')[0])
    .groupby(['gene_id', 'factors'])
)

df_mapped_factors['gene_factor'] = df_mapped_factors.transcript_id.apply(lambda x: x.split('.')[0]) + '.' + df_mapped_factors.factors


LR_output = LR_output.merge(df_mapped_factors, left_index = True, right_on='gene_factor')
LR_output['gene_id'] = LR_output.transcript_id.apply(lambda x: x.split('.')[0])

LR_output.to_csv('results/gao_k5/tables/LR.output.tab.gz', sep='\t', index=False, header=True)



