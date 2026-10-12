import numpy as np
import pandas as pd
import gzip
import re

import sys

arguments = sys.argv
species = arguments[1]

exons = pd.read_csv(f'resources/annotation/{species}/bed_counts/first_exons.sorted.bed.gz', sep='\t')
genes = pd.read_csv(f'resources/annotation/{species}/bed_counts/genes.sorted.bed.gz', sep='\t')

if species == 'Cattle':

    metadata = pd.read_csv('config/cattleGTEx_metadata.txt', sep='\t')[['Sample', 'Tissue']]
    metadata.columns = ['Sample', 'Tissue']
    metadata.Tissue = cattleGTEx.Tissue.apply(lambda x: re.sub(' ', '_', x))

    tissues = ['Blood', 'Liver', 'Muscle', 'Mammary_Gland', 'Subcutaneous_Adipose', 'Testis', 
                          'CNS', 'Hypothalamus', 'Colon', 'Lung', 'Skin', 'Heart']

    metadata = metadata.loc[metadata.Tissue.isin(tissues)]

elif species == 'Sheep':
    metadata = pd.read_csv('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/FarmGTEx/sheepGTEx/metadata.txt', sep='\t')[['Sample', 'Tissue']]
    tissues = list(metadata.Tissue.unique())


Genes = pd.read_csv(f"resources/annotation/{species}/bed/genes.protein_coding.bed.gz", sep='\t')

exons['length'] = exons.end - exons.start
gene_exons_length = exons.groupby('gene_name').length.sum()
        

first = True
for tissue in tissues:
    print(tissue)
    if first:
        gene_coverage = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/gene.coverage.txt.gz', 
                                    sep='\t', index_col=0)
        gene_coverage.columns = [f'{tissue}.{sample}' for sample in gene_coverage.columns]
        
        first_exon_coverage = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/first_exon.coverage.txt.gz', 
                                    sep='\t', index_col=0)
        first_exon_coverage.columns = [f'{tissue}.{sample}' for sample in first_exon_coverage.columns]
        
        gene_counts = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/gene.counts.txt.gz', 
                                    sep='\t', index_col=0)
        gene_counts.columns = [f'{tissue}.{sample}' for sample in gene_counts.columns]
        
        first_exon_counts = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/first_exon.counts.txt.gz', 
                                    sep='\t', index_col=0)
        first_exon_counts.columns = [f'{tissue}.{sample}' for sample in first_exon_counts.columns]
        
        
        first = False
    else:
        new_gene_coverage = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/gene.coverage.txt.gz', 
                                        sep='\t', index_col=0)
        new_gene_coverage.columns = [f'{tissue}.{sample}' for sample in new_gene_coverage.columns]
        gene_coverage = gene_coverage.merge(new_gene_coverage, left_index=True, right_index=True)
        
        new_first_exon_coverage = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/first_exon.coverage.txt.gz', 
                                        sep='\t', index_col=0)
        new_first_exon_coverage.columns = [f'{tissue}.{sample}' for sample in new_first_exon_coverage.columns]
        first_exon_coverage = first_exon_coverage.merge(new_first_exon_coverage, left_index=True, right_index=True)
        
        
        new_gene_counts = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/gene.counts.txt.gz', 
                                        sep='\t', index_col=0)
        new_gene_counts.columns = [f'{tissue}.{sample}' for sample in new_gene_counts.columns]
        gene_counts = gene_counts.merge(new_gene_counts, left_index=True, right_index=True)
        
        new_first_exon_counts = pd.read_csv(f'resources/annotation/{species}/tables/{tissue}/first_exon.counts.txt.gz', 
                                        sep='\t', index_col=0)
        new_first_exon_counts.columns = [f'{tissue}.{sample}' for sample in new_first_exon_counts.columns]
        first_exon_counts = first_exon_counts.merge(new_first_exon_counts, left_index=True, right_index=True)


first_exon_coverage.to_csv(f'resources/metadata/{species}/first_exon.coverage.tab.gz', 
                           sep='\t', index=True, header=True)

gene_coverage.to_csv(f'resources/metadata/{species}/gene.coverage.tab.gz', 
                           sep='\t', index=True, header=True)

first_exon_counts.to_csv(f'resources/metadata/{species}/first_exon.counts.tab.gz', 
                           sep='\t', index=True, header=True)

gene_counts.to_csv(f'resources/metadata/{species}/gene.counts.tab.gz', 
                           sep='\t', index=True, header=True)