import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt
from scipy.stats import kruskal
from scipy.special import logit
import gzip

from scipy.linalg import norm
import warnings 
warnings.filterwarnings("ignore")

def analyze_dict(data_dict):
    # Find the key with the lowest value
    min_key = min(data_dict, key=data_dict.get)
    min_value = data_dict[min_key]
    
    # Find the key with the highest value
    max_key = max(data_dict, key=data_dict.get)
    max_value = data_dict[max_key]

    # Calculate the difference between the highest and lowest values
    difference = max_value - min_value

    difference = "{:.2f}".format(difference)
    
    return min_key, max_key, difference


def kw_to_row(kw_res):
    kw_stat = "{:.2f}".format(kw_res[0][0])
    kw_pval = "{:.2e}".format(kw_res[0][1])

    meds_tuple = sorted(kw_res[1].items())
    meds_list = ["{:.2f}".format(x[1]) for x in meds_tuple]

    meds_list.extend([kw_stat, kw_pval])

    return meds_list


def run_kruskalwallis(EL_factor, samples):
    BA24_idx = samples.loc[samples.tissue_id == 'Brain_Anterior_cingulate_cortex_BA24'].index
    BA24_idx = BA24_idx.intersection(EL_factor.dropna().index)
    BA24_loadings = np.array(EL_factor.dropna().loc[BA24_idx])

    BC_idx = samples.loc[samples.tissue_id == 'Brain_Cortex'].index
    BC_idx = BC_idx.intersection(EL_factor.dropna().index)
    BC_loadings = np.array(EL_factor.dropna().loc[BC_idx])

    BA9_idx = samples.loc[samples.tissue_id == 'Brain_Frontal_Cortex_BA9'].index
    BA9_idx = BA9_idx.intersection(EL_factor.dropna().index)
    BA9_loadings = np.array(EL_factor.dropna().loc[BA9_idx])

    BP_idx = samples.loc[samples.tissue_id == 'Brain_Putamen_basal_ganglia'].index
    BP_idx = BP_idx.intersection(EL_factor.dropna().index)
    BP_loadings = np.array(EL_factor.dropna().loc[BP_idx])

    HA_idx = samples.loc[samples.tissue_id == 'Heart_Atrial_Appendage'].index
    HA_idx = HA_idx.intersection(EL_factor.dropna().index)
    HA_loadings = np.array(EL_factor.dropna().loc[HA_idx])

    Li_idx = samples.loc[samples.tissue_id == 'Liver'].index
    Li_idx = Li_idx.intersection(EL_factor.dropna().index)
    Li_loadings = np.array(EL_factor.dropna().loc[Li_idx])

    Lu_idx = samples.loc[samples.tissue_id == 'Lung'].index
    Lu_idx = Lu_idx.intersection(EL_factor.dropna().index)
    Lu_loadings = np.array(EL_factor.dropna().loc[Lu_idx])

    MS_idx = samples.loc[samples.tissue_id == 'Muscle_Skeletal'].index
    MS_idx = MS_idx.intersection(EL_factor.dropna().index)
    MS_loadings = np.array(EL_factor.dropna().loc[MS_idx])

    Sk_idx = samples.loc[samples.tissue_id == 'Skin_Not_Sun_Exposed_Suprapubic'].index
    Sk_idx = Sk_idx.intersection(EL_factor.dropna().index)
    Sk_loadings = np.array(EL_factor.dropna().loc[Sk_idx])

    WB_idx = samples.loc[samples.tissue_id == 'Whole_Blood'].index
    WB_idx = WB_idx.intersection(EL_factor.dropna().index)
    WB_loadings = np.array(EL_factor.dropna().loc[WB_idx])

    res = kruskal(BA24_loadings, BC_loadings, BA9_loadings, BP_loadings, HA_loadings,
                  Li_loadings, Lu_loadings, MS_loadings, Sk_loadings, WB_loadings)

    meds = {'Brain_Anterior_cingulate_cortex_BA24':np.median(BA24_loadings), 
            'Brain_Cortex':np.median(BC_loadings), 
            'Brain_Frontal_Cortex_BA9':np.median(BA9_loadings), 
            'Brain_Putamen_basal_ganglia':np.median(BP_loadings), 
            'Heart_Atrial_Appendage':np.median(HA_loadings),
            'Liver':np.median(Li_loadings), 
            'Lung':np.median(Lu_loadings), 
            'Muscle_Skeletal':np.median(MS_loadings), 
            'Skin_Not_Sun_Exposed_Suprapubic':np.median(Sk_loadings), 
            'Whole_Blood':np.median(WB_loadings)}


    return res, meds

EL = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/filtered/snmf_10/tables/EL.bed.gz', sep='\t', index_col=3)
gtex_samples = [x for x in EL.columns if x.startswith('GTEX-')]
samples = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/config/samples.tsv', sep='\t', index_col=0)
EL = EL[[x for x in EL.columns if x.startswith('GTEX-')]]
EL['gene'] = [x.split('.')[0] for x in EL.index]

with gzip.open('results/torino/top_factors/human/GTEx.CovEig.txt.gz', 'wb') as fh:
    for gene, df in EL.loc[EL.isna().sum(axis=1) < 100].groupby('gene'):
    
        X = df[gtex_samples]
        X = X.T.dropna().T
        if X.shape[1] > 500:
            cov_X = X.cov()
            eig_X = float(np.linalg.eig(cov_X)[0][0])
    
            eig_X = "{:.2f}".format(eig_X)
    
            new_line = gene + '\t' + eig_X + '\n'
            fh.write(new_line.encode())


EL = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/filtered/snmf_10/tables/transcript.uncorrected.EL.bed.gz', sep='\t', index_col=3)
gtex_samples = [x for x in EL.columns if x.startswith('GTEX-')]
samples = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/config/samples.tsv', sep='\t', index_col=0)
EL = EL[[x for x in EL.columns if x.startswith('GTEX-')]]
EL['gene'] = [x.split('.')[0] for x in EL.index]

with gzip.open('results/torino/top_factors/human/GTEx.CovEig.transcripts.txt.gz', 'wb') as fh:
    for gene, df in EL.loc[EL.isna().sum(axis=1) < 100].groupby('gene'):
    
        X = df[gtex_samples]
        X = X.T.dropna().T
        if X.shape[1] > 500:
            cov_X = X.cov()
            eig_X = float(np.linalg.eig(cov_X)[0][0])
    
            eig_X = "{:.2f}".format(eig_X)
    
            new_line = gene + '\t' + eig_X + '\n'
            fh.write(new_line.encode())

