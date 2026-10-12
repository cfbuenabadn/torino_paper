import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import rankdata
from scipy.stats import norm
from sklearn.impute import SimpleImputer
import sys

def qqnorm(x):
    n=len(x)
    a=3.0/8.0 if n<=10 else 0.5
    return(norm.ppf( (rankdata(x)-a)/(n+1.0-2.0*a) ))

def get_PSI_table(bed, participant_list):
    bed_PSI = bed[bed.columns.intersection(participant_list)]
    bed_PSI.index = bed.pid
    bed_PSI = bed_PSI.replace('NA', np.nan).astype(float)
    # Let's do the filtering in a tissue-specific manner
    # bed_PSI = bed_PSI.loc[bed_PSI.isna().mean(axis=1) < 0.1]
    return bed_PSI

def impute_PSI(bed_PSI):
    imputer = SimpleImputer()
    imputed_df = pd.DataFrame(imputer.fit_transform(bed_PSI.T)).T
    imputed_df.columns = bed_PSI.columns 
    imputed_df.index = bed_PSI.index
    return imputed_df

def scale_PSI(imputed_df):
    df_scale = pd.DataFrame(scale(imputed_df, axis=1))
    df_scale.columns = imputed_df.columns
    df_scale.index = imputed_df.index
    return df_scale

def qqnorm_PSI(df_scale):

    df_qqnorm_list = []
    
    for sample in df_scale.columns:
        qqnorm_row = qqnorm(df_scale[sample])
        df_qqnorm_list.append(qqnorm_row)

    df_qqnorm_list = np.array(df_qqnorm_list).T
    df_qqnorm = pd.DataFrame(df_qqnorm_list)
    df_qqnorm.columns = df_scale.columns
    df_qqnorm.index = df_scale.index

    return df_qqnorm


def get_qqnorm_bed(bed, participant_list):
    
    bed6 = bed[bed.columns[:6]]
    bed_PSI = get_PSI_table(bed, participant_list)
    bed_PSI = bed_PSI.loc[bed_PSI.isna().mean(axis=1) < 0.4]
    imputed_df = impute_PSI(bed_PSI)
    df_scale = scale_PSI(imputed_df)
    df_qqnorm = qqnorm_PSI(df_scale)
    qqnorm_bed = pd.merge(bed6, df_qqnorm, left_on = 'pid', right_index=True)
    qqnorm_bed.index = qqnorm_bed.pid
    return qqnorm_bed, bed_PSI

# def qqnorm_single_tissue(input_tissue, output_tissue, participant_list, K=3):
#     ba9 = pd.read_csv(f'ebpmf_models/single_tissue/tables/{input_tissue}.EL.bed.gz', sep='\t')
#     ba9_bed, bed_PSI = get_qqnorm_bed(ba9, participant_list)
#     return ba9_bed

def qqnorm_multi_tissue(snmf_bed):
    tissue_samples = snmf_bed.columns[6:]
    all_samples =snmf_bed.columns[6:]
    multi_qqnorm, bed_PSI = get_qqnorm_bed(snmf_bed, all_samples)
    bed_cols = list(multi_qqnorm.columns[:6])

    factors_to_keep = bed_PSI.loc[(bed_PSI[tissue_samples].isna().mean(axis=1) <= 0.1) & (bed_PSI[tissue_samples].quantile(0.9, axis=1) >= 0.1)].index
    
    multi_qqnorm = multi_qqnorm.loc[factors_to_keep, bed_cols + list(tissue_samples)]
    multi_qqnorm_cols = bed_cols + ['-'.join(x.split('-')[:2]) for x in multi_qqnorm.columns[6:]]
    multi_qqnorm.columns = multi_qqnorm_cols
    return multi_qqnorm



def wrapper_multi_tissue(input_EL, output_qqnorm):
    snmf_bed = pd.read_csv(input_EL, sep='\t')
    tissue_qqnorm = qqnorm_multi_tissue(snmf_bed)
    tissue_qqnorm.to_csv(output_qqnorm, sep='\t', index=False)

if __name__ == '__main__':

    arguments = sys.argv
    input_EL = arguments[1]
    output_qqnorm = arguments[2]

    print('Normalizing data...')
    wrapper_multi_tissue(input_EL, output_qqnorm)
    print('Done!')



    