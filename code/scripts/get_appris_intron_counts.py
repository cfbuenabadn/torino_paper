import numpy as np
import pandas as pd
import tabix
import gzip 
import sys

def process_intron_line(intron, sample_file_list):
    row = intron.decode().rstrip().split('\t')
    chrom = row[0]
    start = int(row[1])
    end = int(row[2])
    gene = row

    length = end - start

    intron_id = row[3]
    strand = row[5]

    try:
        counts_row = [chrom, row[1], row[2], intron_id, str(length), strand]
        for sample_bed in list(sample_file_list):
            counts_row.append(get_intron_counts(chrom, start, end, sample_bed))

        counts_line = ('\t'.join(counts_row) + '\n').encode()
        
    except:
        counts_line = ''.encode()

    return counts_line

def get_intron_counts(chrom, start, end, sample_file):
    
    tb = tabix.open(sample_file)
    sample_counts = tb.query(chrom, start, end)

    counts = 0
    total_bp = 0
    for bp in sample_counts:
        counts += int(bp[3])
        total_bp += 1

    return str(counts)

if __name__ == '__main__':
    arguments = sys.argv
    chunk_number = int(arguments[1])
    total_chunks = int(arguments[2])
    
    gtex_files = pd.read_csv('config/gtex_samples.tsv', sep='\t')
    
    samples_bed = []
    samples_id = []
    for idx, df in gtex_files.groupby(['tissue_id', 'sex']):
        samples = sorted(list(df.bed_file))[:5]
        samples_bed += [str(x) for x in samples]
        samples_id += [x.split('/')[-1].split('.')[0] for x in samples]


    total_rows = 174635  

    del gtex_files
    
    chunk_size = (total_rows + total_chunks - 1) // total_chunks  
    
    start_idx = (chunk_number - 1) * chunk_size
    end_idx = min(chunk_number * chunk_size, total_rows)

    with gzip.open('results/CoveragePlots/bed_files/introns/gencode.appris_introns.sorted.bed.gz', 'rb') as fh:
        with gzip.open(f'results/CoveragePlots/bed_files/introns/tmp/gencode.appris_introns.{str(chunk_number)}.bed.gz', 'wb') as fh_out:
            if chunk_number == 1:
                first_row = ['#chrom', 'start', 'end', 'intron', 'length', 'strand'] + samples_id
                first_line = ('\t'.join(first_row) + '\n').encode()
                fh_out.write(first_line)

            print('Getting started...')
            for i, intron in enumerate(fh):
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break

                print(i)
                bed_line = process_intron_line(intron, samples_bed)
                fh_out.write(bed_line)







        