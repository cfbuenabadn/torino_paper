import pandas as pd
import numpy as np
import os
import re

cardoso = pd.read_csv('config/cardoso_samples.tsv', sep='\t')
gtf_annotations = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/readme.txt', sep='\t')
sheepGTEx = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/FarmGTEx/sheepGTEx/metadata.txt', sep='\t')[['Sample', 'Tissue']]
cattleGTEx = pd.read_csv('config/cattleGTEx_metadata.txt', sep='\t')[['Sample', 'Tissue']]
cattleGTEx.columns = ['Sample', 'Tissue']
cattleGTEx.Tissue = cattleGTEx.Tissue.apply(lambda x: re.sub(' ', '_', x))

gene_sepecies = pd.read_csv('config/gene_species.txt', sep='\t')

tissues_to_download = ['Brain_Cerebellum', 'Pituitary', 'Pancreas', 'Spleen', 'Esophagus_Mucosa', 'Adrenal_Gland', 
                       'Thyroid', 'Ovary', 'Small_Intestine_Terminal_Ileum', 'Stomach', 'Nerve_Tibial', 'Vagina', 
                       'Prostate', 'Uterus', 'Colon_Sigmoid']

def much_more_mem_after_first_attempt(wildcards, attempt):
    if int(attempt) == 1:
        return 4000
    else:
        return 52000


# download_GTEx= pd.read_csv('config/gtex-download-40-tissues.tsv.gz', sep='\t')
# download_GTEx = download_GTEx.loc[download_GTEx.tissue_id.isin(tissues_to_download)]

# download_test = pd.read_csv('config/gtex_pancreas.txt', sep=' ', names = ['Tissue', 'IndID'])

# Load gene coordinates

cattleGenes = pd.read_csv('resources/annotation/bed/Cattle.bed.gz', sep='\t', 
                          names = ['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'])

sheepGenes = list(pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/Sheep/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)

macaqueGenes = list(pd.read_csv('resources/annotation/Macaque/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)
mouseGenes = list(pd.read_csv('resources/annotation/Mouse/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)
ratGenes = list(pd.read_csv('resources/annotation/Rat/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)
rabbitGenes = list(pd.read_csv('resources/annotation/Rabbit/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)
chickenGenes = list(pd.read_csv('resources/annotation/Chicken/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)

cattleGene_names = []
for idx, row in cattleGenes.iterrows():
    if row.gene_name == 'ensembl':
        cattleGene_names.append(row.gene_id)
    else:
        cattleGene_names.append(row.gene_name)
        
cattle_tissues = ['Ovary', 'Subcutaneous_Adipose', 'Embryo', 'Eye', 'Testis', 'Aorta', 'Hypothalamus', 'Abomasum', 
                  'Soft_Palate', 'Lung', 'Liver', 'CNS', 'Spleen', 'Sperm', 'Placenta', 'Nose', 'Colon', 'Cumulus', 
                  'Vas_Deferens', 'Tonsil', 'Monocyte', 'Follicle', 'Oviduct', 'Pituitary', 'Ileum', 'Cell_Line', 'Oocyte',
                  'Salivary_Gland', 'Lymphocyte', 'Corpus_Luteum', 'Kidney', 'Gall_Bladder', 'Granulosa', 'Duodenum', 
                  'Adipose', 'Rumen', 'Neutrophil', 'Skin', 'Hoof', 'Macrophage', 'Intervertebral', 'Bone_Marrow', 
                  'Small_Intestine', 'Heart', 'Ear', 'Uterus', 'Milk', 'Mammary_Gland', 'Muscle', 'Epididymis', 'Lymph_Node', 
                  'Jejunum', 'Visceral_Fat', 'Leukocyte', 'Vagina', 'Adrenal', 'Reticulum', 'Cartilage', 'Blood', 'Large_Intestine']

sheep_tissues = list(sheepGTEx.Tissue.unique())

selected_genes = pd.read_csv('config/human.protein_coding.bed.gz',#'/project2/mstephens/cfbuenabadn/gtex-stm/data/protein_coding_genes.bed.gz',
                                     sep='\t')

        # if (gene == 'NPIPA9.ENSG00000183889') or (gene == 'NPIPA9.ENSG00000233024'):

humanGenes = list(selected_genes.gene_name)
# for idx, row in selected_genes.iterrows():
#     if row.gene_symbol == 'NPIPA9':
#         gene_symbol = 'NPIPA9.' + row.gene
#     else:
#         gene_symbol = row.gene_symbol
#     human_genes.append(gene_symbol)

gtex_samples = pd.read_csv('config/gtex_samples.tsv', sep='\t')
gtex_tissues = sorted(gtex_samples.tissue_id.unique())

tissue_sub_list = sorted(['Brain_Anterior_cingulate_cortex_BA24',
                      'Brain_Cortex',
                      'Brain_Frontal_Cortex_BA9',
                      'Brain_Putamen_basal_ganglia',
                      'Skin_Not_Sun_Exposed_Suprapubic',
                      'Liver',
                      'Lung',
                      'Heart_Atrial_Appendage',
                      'Muscle_Skeletal',
                      'Whole_Blood'])


def GetColocTsvFormattedString(string):
    """
    return a snakemake input function that returns formatted string with wildcard values that match colocs_df based on ColocName index wildcard
    """
    def F(wildcards):
        return string.format(**colocs_df.loc[wildcards.ColocName].to_dict())
    return F


colocs_df = pd.read_csv("/project2/mstephens/cfbuenabadn/gtex-stm/code/config/ColocRunWildcards.tsv", index_col=0, comment='#', sep='\t', keep_default_na=False)
colocs_gwas = colocs_df.loc[colocs_df['FeatureCoordinatesRedefinedFor']=='ForGWASColoc']


def GetMolPhenotypesToColoc(wildcards):
    ProvidedMolPhenotypeList = colocs_df.loc[wildcards.ColocName]['MolPhenotypesToColoc']
    if ProvidedMolPhenotypeList == '':
        return ' '.join(PhenotypesToColoc)
    else:
        return ProvidedMolPhenotypeList



gwas_annot = pd.read_csv('/project2/mstephens/cfbuenabadn/gtex-stm/code/config/gwas_annot_map.csv')

gwas_traits = [x.rstrip() for x in open('/project2/mstephens/cfbuenabadn/gtex-stm/code/config/gwas_features.txt').readlines()]

gwas_traits_bjf79 = pd.read_csv("/project2/mstephens/cfbuenabadn/gtex-stm/code/config/bjf79_gwas_table.tsv", index_col='gwas', sep='\t', 
                                usecols=['gwas', 'trait', 'FTPPath', "SummaryStatsLocalFilepath", 'ProcessingMethod', "Continuous"])



gwas_traits_for_coloc = [x for x in list(gwas_annot.trait) if x in gwas_traits]
gwas_traits_for_coloc += list(gwas_traits_bjf79.index)


fh = open('../data/test_genes.txt')
test_genes = [x.rstrip() for x in fh.readlines()]
fh.close()