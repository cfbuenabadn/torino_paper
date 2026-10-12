import pandas as pd

snmf3 = pd.read_csv(
    '/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/tables/snmf5_EL.tab.gz',
               sep='\t')

snmf3['gid'] = snmf3.gid.apply(lambda x: x.split('.')[0])

cols_id = [x for x in snmf3.columns if x != 'R6284240-PCC']
snmf3 = snmf3[cols_id]

snmf3.to_csv('results/gao_k5/tables/snmf5_EL.all.bed.gz', sep='\t', index=False, header=True)


bed_id = snmf3.columns[:6]
sample_id = snmf3.columns[6:]

ac_cols = list(bed_id) + [x for x in sample_id if x.endswith('-AC')]
pcc_cols = list(bed_id) + [x for x in sample_id if x.endswith('-PCC')]
dlpfc_cols = list(bed_id) + [x for x in sample_id if x.endswith('-DLPFC')]

snmf3_ac = snmf3[ac_cols]
snmf3_ac.columns = [x.split('-')[0] for x in snmf3_ac.columns]


snmf3_pcc = snmf3[pcc_cols]
snmf3_pcc.columns = [x.split('-')[0] for x in snmf3_pcc.columns]

snmf3_dlpfc = snmf3[dlpfc_cols]
snmf3_dlpfc.columns = [x.split('-')[0] for x in snmf3_dlpfc.columns]

snmf3_ac.to_csv('results/gao_k5/tables/snmf5_EL.AC.bed.gz', sep='\t', index=False, header=True)
snmf3_pcc.to_csv('results/gao_k5/tables/snmf5_EL.PCC.bed.gz', sep='\t', index=False, header=True)
snmf3_dlpfc.to_csv('results/gao_k5/tables/snmf5_EL.DLPFC.bed.gz', sep='\t', index=False, header=True)



