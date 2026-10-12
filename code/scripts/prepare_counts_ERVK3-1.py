import pandas as pd
import gzip
import os
import numpy as np
import tabix
import re
import sys

def process_and_write(gene, species='Human'):

    ERVK31_coords = "chr19:58304871-58317341"

    chrom = "chr19"
    start = 58304871
    end = 58317341

    strand = '+'

    metadata = pd.read_csv('config/gtex_samples.tsv', sep='\t')
    metadata.columns = ['Sample', 'participant', 'Tissue', 'sex', 'bed_file']
    output_name = f'resources/count_matrices_ervk31/Human/{gene}.csv.gz'
        
    file_count = 1
    with gzip.open(output_name, 'wt', compresslevel=6) as fh:
        for idx, row in metadata.iterrows():

            tissue = row.Tissue
                
            file_name = row.bed_file
            coord_str, counts_str = process_sample(tissue, file_name, chrom, start, end)
            if file_count == 1:
                fh.write(coord_str)
            fh.write(counts_str)
            file_count += 1



def process_sample(tissue, file_name, chrom, start, end):
    sample_name = file_name.split('/')[-1].split('.')[0]
    chrom_for_tabix = chrom
    # print(file_name, chrom_for_tabix, start, end)
    gene_df = tabix_dataframe(file_name, chrom_for_tabix, start, end)
    coord_str, counts_str = get_counts_string(gene_df, sample_name, tissue, chrom, start, end)
    return coord_str, counts_str


def tabix_dataframe(file_name, chrom, start, end):

    tb = tabix.open(file_name)

    records = tb.query(chrom, start, end)
    gene_df = pd.DataFrame(records, columns = ['chrom', 'start', 'end', 'counts'])
    return gene_df


def get_counts_string(gene_df, sample_name, tissue, chrom, start, end):
    coord_list = ['Sample_ID']
    sample_counts = [sample_name + '.' + tissue]
    for i, (index, row) in enumerate(gene_df.iterrows()):
        start_ = int(row.start)
        end_ = int(row.end)
        if i == 0:
            start_ = start
        if i == len(gene_df) - 1:
            end_ = end

        segment_length = end_ - start_
        segment_counts = int(float(row.counts))
        segment_to_add = [str(segment_counts)]*segment_length
        sample_counts += segment_to_add
        coord_list += list(range(start_, end_))

    if not chrom.startswith('chr'):
        chrom = 'chr' + chrom
    coord_list = [chrom + ':' + str(x) if (x != 'Sample_ID') else x for x in coord_list]
    coord_str = ','.join(coord_list)+'\n'
    counts_str = ','.join(sample_counts)+'\n'
    
    return coord_str, counts_str


if __name__ == '__main__':

    # arguments = sys.argv
    species = "Human"#arguments[1]
    gene = "ERVK3-1" #arguments[2]
    process_and_write(gene, species)

