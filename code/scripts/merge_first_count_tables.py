import numpy as np
import pandas as pd
import re
import logging
import sys

arguments = sys.argv
species = arguments[1]
tissue = arguments[2]

# Configure the logger
logging.basicConfig(filename=f'logs/meta/{tissue}.{species}.first_exons.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log messages throughout the script

if species == 'Cattle':
    metadata = pd.read_csv('config/cattleGTEx_metadata.txt', sep='\t')[['Sample', 'Tissue']]
    metadata.columns = ['Sample', 'Tissue']
    metadata.Tissue = metadata.Tissue.apply(lambda x: re.sub(' ', '_', x))

elif species == 'Sheep':
    metadata = pd.read_csv('resources/FarmGTEx/sheepGTEx/metadata.txt', sep='\t')[['Sample', 'Tissue']]


gene_names = list(pd.read_csv(f'../code/resources/annotation/{species}/bed/genes.protein_coding.bed.gz', sep='\t').gene_name)

gene_counts_list = {}
gene_coverage_list = {}

first_exon_counts_list = {}
first_exon_coverage_list = {}

first = True

metadata = metadata.loc[metadata.Tissue == tissue]

logging.info(f'Starting iterations. Total samples: {str(metadata.shape[0])}')

for idx, row in metadata.iterrows():
    tissue = row.Tissue
    sample = row.Sample
    sample_counts = pd.read_csv(f'resources/annotation/{species}/bed_counts/sample_counts/{tissue}/{sample}.first_exons.bed.gz', sep='\t',
                            names = ['#chrom', 'start', 'end', 'counts', '#chrom_2', 'start_2', 'end_2', 
                                     'gene_id', 'gene_name', 'strand', 'exon_length', 'overlap'])

    if first:
        first_exon_length = sample_counts.loc[sample_counts.gene_name.isin(gene_names)].groupby('gene_name').exon_length.first()
        first = False

    first_exon_counts = sample_counts.loc[sample_counts.gene_name.isin(gene_names)].groupby('gene_name').counts.sum()
    del sample_counts
    first_exon_coverage = first_exon_counts/first_exon_length

    first_exon_counts_list.update({sample:first_exon_counts})
    first_exon_coverage_list.update({sample:first_exon_coverage})

    logging.info(f'first exon: {tissue}, and {sample}')

first_exon_counts_table = pd.DataFrame(first_exon_counts_list)
first_exon_coverage_table = pd.DataFrame(first_exon_coverage_list)

logging.info('Done. Preparing dataframes. First exon.')

first_exon_counts_table.to_csv(f'resources/annotation/{species}/tables/{tissue}/first_exon.counts.txt.gz', sep='\t', header=True, index=True)
first_exon_coverage_table.to_csv(f'resources/annotation/{species}/tables/{tissue}/first_exon.coverage.txt.gz', sep='\t', header=True, index=True)

del first_exon_counts_table
del first_exon_coverage_table

first = True

for idx, row in metadata.iterrows():

    tissue = row.Tissue
    sample = row.Sample
        
    sample_counts_genes = pd.read_csv(f'resources/annotation/{species}/bed_counts/sample_counts/{tissue}/{sample}.genes.bed.gz', sep='\t',
                            names = ['#chrom', 'start', 'end', 'counts', '#chrom_2', 'start_2', 'end_2', 
                                     'gene_id', 'gene_name', 'strand', 'gene_length', 'exome_length', 'overlap'])
    
    if first:
        gene_length = sample_counts_genes.loc[sample_counts_genes.gene_name.isin(gene_names)].groupby('gene_name').exome_length.first()
        first = False

    
    gene_counts = sample_counts_genes.loc[sample_counts_genes.gene_name.isin(gene_names)].groupby('gene_name').counts.sum()
    del sample_counts_genes
    gene_coverage = gene_counts/gene_length
    
    gene_counts_list.update({sample:gene_counts})
    gene_coverage_list.update({sample:gene_coverage})
    
    
    
    logging.info(f'gene: {tissue}, and {sample}')
    
logging.info('Done. Preparing dataframes.')
    
gene_counts_table = pd.DataFrame(gene_counts_list)
gene_counts_coverage = pd.DataFrame(gene_coverage_list)

logging.info('Saving dataframes.')

gene_counts_table.to_csv(f'resources/annotation/{species}/tables/{tissue}/gene.counts.txt.gz', sep='\t', header=True, index=True)
gene_counts_coverage.to_csv(f'resources/annotation/{species}/tables/{tissue}/gene.coverage.txt.gz', sep='\t', header=True, index=True)

del gene_counts_table
del gene_counts_coverage

logging.info('Finished!')