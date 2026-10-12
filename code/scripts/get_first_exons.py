import numpy as np
import pandas as pd
import sys

arguments = sys.argv
species = arguments[1]

exons = pd.read_csv(f'resources/annotation/{species}/bed/exons.all.bed.gz', sep='\t')
genes = pd.read_csv(f'resources/annotation/{species}/bed/genes.protein_coding.bed.gz', sep='\t')

def getFirstExon(df):
    
    gene_id = df.gene_id.iloc[0]
    
    if df.gene_name.iloc[0] == 'ensembl':
        gene_name = df.gene_id.iloc[0]
    else:
        gene_name = df.gene_name.iloc[0]
        
    strand = df.strand.iloc[0]
        
    if df.strand.iloc[0] == '+':
        first_exon = df.iloc[0]
    else:
        first_exon = df.iloc[-1]
        
    row = [first_exon['#chrom'], int(first_exon.start), int(first_exon.end), gene_id, gene_name, strand]
    
    return row

first_exon_list = []

for gene, df in exons.groupby('gene_id'):
    
    first_exon_list.append(getFirstExon(df))
    
    

gene_names = []
for idx, row in genes.iterrows():
    if row.gene_name == 'ensembl':
        gene_names.append(row.gene_id)
    else:
        gene_names.append(row.gene_name)
        
genes['gene_name'] = gene_names

genes['gene_length'] = genes.end - genes.start



exon_gene_names = []
for idx, row in exons.iterrows():
    if row.gene_name == 'ensembl':
        exon_gene_names.append(row.gene_id)
    else:
        exon_gene_names.append(row.gene_name)
        
exons['gene_name'] = exon_gene_names

exons['length'] = exons.end - exons.start
gene_exons_length = exons.groupby('gene_name').length.sum()
genes = genes.merge(gene_exons_length, how='left', left_on='gene_name', right_index = True)


first_exons = pd.DataFrame(first_exon_list, 
                           columns = ['#chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand']
                          ).sort_values(['#chrom', 'start', 'end'])

first_exons['length'] = first_exons.end - first_exons.start

if species == 'Cattle':
    first_exons['#chrom'] = [x[3:] for x in first_exons['#chrom']]
    genes['#chrom'] = [x[3:] for x in genes['#chrom']]


first_exons.to_csv(f'resources/annotation/{species}/bed_counts/first_exons.bed', sep='\t', index=False, header=True)
genes.to_csv(f'resources/annotation/{species}/bed_counts/genes.bed', sep='\t', index=False, header=True)