import pandas as pd
import pysam
import argparse 
import os

import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
# %run load_human.py
import warnings
# from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

from tqdm import tqdm 
from pybedtools import BedTool


import sys
sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
from torino_isoforms import *
from GetLongReadsTable import *

if __name__ == '__main__':
    gtf_bed = pd.read_csv('../code/resources/annotation/Human/bed/exons.apris_principal.bed.gz', sep='\t')
    gtf_bed.columns = ['chrom', 'start', 'end', 'gene_ensembl', 'gene_id', 'strand', 'exon_id', 'transcript_id', 'factors']
    gtf_bed['notes'] = ['.']*gtf_bed.shape[0]

    arguments = sys.argv
    i = int(arguments[1])
    n = int(arguments[2])
    # correction_tag = str(arguments[5])
    
    genes = sorted(gtf_bed.gene_id.unique())
    total_genes = len(genes)

    genes_per_slice = int(total_genes/n)

    slice_start = (i-1)*genes_per_slice
    if i == n:
        slice_end = total_genes + 1
    else:
        slice_end = i*genes_per_slice

    selected_genes = genes[slice_start:slice_end]




    bam_file_list = [x for x in os.listdir('../code/resources/GTEx/ONT/bam/') if x.endswith('.bam')]
    
    isoform_table = get_fractions_per_sample_list_lean(gtf_bed, 'SRSF3', bam_file_list)



    gene_list = selected_genes
    import gzip
    with gzip.open(f'results/LongReads/tmp/long_read_gencode.counts.{str(i)}.tab.gz', 'wb') as fh:
        # with gzip.open(f'results/LongReads/tmp/long_read_gencode.missing.{str(i)}.tab.gz', 'wb') as fh2:
        fh_header = ('\t'.join(isoform_table.columns) + '\n').encode()
        fh.write(fh_header)

        # fh2_header = ('\t'.join(missing_isoforms_table.columns) + '\n').encode()
        # fh2.write(fh2_header)

        for gene in gene_list:
            
            # print(gene)
            gene_length = gtf_bed.loc[gtf_bed.gene_id == gene].end.max() - gtf_bed.loc[gtf_bed.gene_id == gene].start.min()
            # if gene_length > 200000:
            #     continue
            try:
                isoform_table = get_fractions_per_sample_list_lean(gtf_bed, gene, bam_file_list)
                for idx, row in isoform_table.astype(str).iterrows():
                    row_to_write = ('\t'.join(list(row)) + '\n').encode()
                    fh.write(row_to_write)

                # for idx, row in missing_isoforms_table.astype(str).iterrows():
                #     row_to_write = ('\t'.join(list(row)) + '\n').encode()
                #     fh2.write(row_to_write)
                    
            except:
                continue
