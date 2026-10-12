import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
import warnings
from scipy.stats import pearsonr, spearmanr
import itertools
warnings.filterwarnings("ignore")
import gzip

def resample_array(arr, target_length):
    """
    Resample a 1D numpy array to a fixed target length.
    If the array is longer, values are averaged into bins.
    If shorter, values are interpolated linearly.
    """
    arr = np.asarray(arr, dtype=float)
    original_length = len(arr)
    
    if original_length == target_length:
        return arr

    # Create normalized positions for original and target arrays
    original_positions = np.linspace(0, 1, original_length)
    target_positions = np.linspace(0, 1, target_length)

    # Interpolate (this handles both upsampling and mild downsampling smoothly)
    resampled = np.interp(target_positions, original_positions, arr)

    return resampled


# def get_utr(human_iso, isoform):
#     gene, iso = isoform.split('.')
#     iso_df = human_iso.loc[human_iso.transcript_id == isoform]
#     strand = iso_df.strand.iloc[0]
#     # print(strand)
#     if strand == '+':
#         last_exon_row = iso_df.iloc[-1]
#         last_start = last_exon_row.start
#         last_end = last_exon_row.end

#         iso_df_exon = human_iso.loc[(human_iso.start == last_start) & (human_iso.gene_id == gene)]

#         if len(iso_df_exon) > 1:
#             iso_df_exon_end = np.array(iso_df_exon.loc[iso_df_exon.transcript_id != isoform].end)

#             dist_3prime = last_end - iso_df_exon_end
#             return dist_3prime

#     else:
#         last_exon_row = iso_df.iloc[0]
#         last_start = last_exon_row.start
#         last_end = last_exon_row.end


#         iso_df_exon = human_iso.loc[(human_iso.end == last_end) & (human_iso.gene_id == gene)]

#         # print(iso_df_exon)

#         if len(iso_df_exon) > 1:
#             iso_df_exon_start = np.array(iso_df_exon.loc[iso_df_exon.transcript_id != isoform].start)
#             dist_3prime = iso_df_exon_start - last_start
            
#             return dist_3prime

#     return None

def get_top_isoform(iso_ssmd, gene, tissue):
    top_iso = iso_ssmd[tissue].loc[iso_ssmd[tissue].gene == gene].sort_values('ssmd').iloc[-1].pid
    return top_iso


# for gene, df in annot.groupby('gene'):

def get_utr_pool(gene, annot):
    df = annot.loc[annot.gene == gene]
    transcript_utr_pool = []
    transcript_chain_pool = []
    for idx, row in df.iterrows():
        
        transcript = row.transcript
        appris_ref = row.appris_ref
        gencode_ref = row.gencode_ref

        
        if ('UTR3' in appris_ref) and ((('intron_chain' in appris_ref) or ('ref_transcript' in appris_ref)) or (('intron_chain' in gencode_ref) or ('ref_transcript' in gencode_ref))):
            transcript_utr_pool.append(transcript)
            
        if (('intron_chain' in appris_ref) or ('ref_transcript' in appris_ref)) or (('intron_chain' in gencode_ref) or ('ref_transcript' in gencode_ref)):
            transcript_chain_pool.append(transcript)
        

    return transcript_utr_pool, transcript_chain_pool      
    # if (len(transcript_utr_pool) >= 1) and (len(transcript_chain_pool) >= 2):
    #     print(gene)
    #     break

def get_UTR_coords(human_iso, isoform, transcript_chain_pool):

    gene, iso = isoform.split('.')
    iso_df = human_iso.loc[human_iso.transcript_id == isoform]
    strand = iso_df.strand.iloc[0]

    pool_df = human_iso.loc[human_iso.transcript_id.isin(transcript_chain_pool)]
    # print(strand)
    if strand == '+':
        last_exon_row = iso_df.iloc[-1]
        last_start = last_exon_row.start
        last_end = last_exon_row.end
        last_chrom = last_exon_row.chrom

        iso_df_exon = pool_df.loc[(pool_df.start == last_start)]

        if len(iso_df_exon) > 1:
            utr_last = iso_df_exon.end.max()
            utr_first = iso_df_exon.end.min()

            if (utr_last - utr_first) >= 200:
                return last_chrom, last_start, utr_first, utr_last, strand

            # dist_3prime = last_end - iso_df_exon_end
            # return dist_3prime

    else:
        last_exon_row = iso_df.iloc[0]
        last_start = last_exon_row.start
        last_end = last_exon_row.end
        last_chrom = last_exon_row.chrom

        iso_df_exon = pool_df.loc[(pool_df.end == last_end)]


        # iso_df_exon = pool_df.loc[(pool_df.end == last_end) & (pool_df.gene_id == gene)]

        # print(iso_df_exon)

        if len(iso_df_exon) > 1:

            utr_last = iso_df_exon.start.min()
            utr_first = iso_df_exon.start.max()

            if (utr_first - utr_last) >= 200:
                return last_chrom, last_end, utr_first, utr_last, strand

                
            # iso_df_exon_start = np.array(iso_df_exon.loc[iso_df_exon.transcript_id != isoform].start)
            # dist_3prime = iso_df_exon_start - last_start
            
            # return dist_3prime

    return None

def get_coords(chrom, last_start, utr3_1, utr3_2, strand, coords):
    x = np.arange(len(coords))
    if strand == '+':
        
        coord_bp = np.array([int(coord.split(':')[1]) for coord in coords])
        intron_bp = x[(coord_bp < last_start) & (coord_bp >= (last_start - 100))]
        exon_bp = x[(coord_bp <= utr3_1) & (coord_bp >= last_start)]
        utr_bp = x[(coord_bp <= utr3_2) & (coord_bp > utr3_1)]
        after_bp = x[(coord_bp > utr3_2) & (coord_bp <= (utr3_2 + 100))]

    else:
        coord_bp = np.array([int(coord.split(':')[1]) for coord in coords])
        intron_bp = x[(coord_bp > last_start) & (coord_bp <= (last_start + 100))]
        exon_bp = x[(coord_bp <= last_start) & (coord_bp >= utr3_1)]
        utr_bp = x[(coord_bp <= utr3_1) & (coord_bp > utr3_2)]
        after_bp = x[(coord_bp < utr3_2) & (coord_bp >= (utr3_2 - 100))]

    return intron_bp, exon_bp, utr_bp, after_bp

def process_gene(gene, annot, human_iso):
    gene_dict = {}
    transcript_utr_pool, transcript_chain_pool = get_utr_pool(gene, annot)
    if (len(transcript_utr_pool) >= 1) and (len(transcript_chain_pool) >= 2):
        isoform = transcript_utr_pool[0]
        utr_coords = get_UTR_coords(human_iso, isoform, transcript_chain_pool)
        if utr_coords is not None:
            chrom = utr_coords[0]
            last_start = utr_coords[1] 
            utr_first = utr_coords[2] 
            utr_last = utr_coords[3]
            strand = utr_coords[4]
            tissue_coverage_dict = {}
            with gzip.open(f'../code/resources/count_matrices_filtered/Human/{gene}.csv.gz', 'rb') as fh:
                coords = fh.readline().decode().rstrip().split(',')[1:]
                intron_bp, exon_bp, utr_bp, after_bp = get_coords(chrom, last_start, utr_first, utr_last, strand, coords)
                all_coords = sorted(list(intron_bp) + list(exon_bp) + list(utr_bp) + list(after_bp))
                
                for linea in fh:
                    row = linea.decode().rstrip().split(',')
                    counts = np.array(row[1:])[all_coords]
                    counts_line = [int(x) for x in counts]
                    total_counts = np.sum(counts_line)
                    total_len = len(counts_line)
                    if total_counts >= 1000:
                        all_counts_line = np.array([int(x) for x in row[1:]])
                        intron_rpkm = 1e9*(all_counts_line[intron_bp]/(total_counts*total_len))
                        exon_rpkm = 1e9*(all_counts_line[exon_bp]/(total_counts*total_len))
                        utr_rpkm = 1e9*(all_counts_line[utr_bp]/(total_counts*total_len))
                        after_rpkm = 1e9*(all_counts_line[after_bp]/(total_counts*total_len))
                        tissue = row[0].split('.')[1]

                        intron_rpkm = np.convolve(intron_rpkm/total_counts, np.ones(20, dtype=int), 'valid')
                        exon_rpkm = np.convolve(exon_rpkm/total_counts, np.ones(20, dtype=int), 'valid')
                        utr_rpkm = np.convolve(utr_rpkm/total_counts, np.ones(20, dtype=int), 'valid')
                        after_rpkm = np.convolve(after_rpkm/total_counts, np.ones(20, dtype=int), 'valid')

                        if strand == '+':
                            resampled_intron = list(resample_array(intron_rpkm, 25))
                            resampled_exon = list(resample_array(exon_rpkm, 100))
                            resampled_utr = list(resample_array(utr_rpkm, 100))
                            resampled_after = list(resample_array(after_rpkm, 25))
                        else:
                            resampled_intron = list(resample_array(intron_rpkm, 25))[::-1]
                            resampled_exon = list(resample_array(exon_rpkm, 100))[::-1]
                            resampled_utr = list(resample_array(utr_rpkm, 100))[::-1]
                            resampled_after = list(resample_array(after_rpkm, 25))[::-1]
                            
                        total_rpkm = np.array(resampled_intron + resampled_exon + resampled_utr + resampled_after)
                        
                        if tissue in tissue_coverage_dict.keys():
                            tissue_coverage_dict[tissue]['intron_rpkm'] += intron_rpkm
                            tissue_coverage_dict[tissue]['exon_rpkm'] += exon_rpkm
                            tissue_coverage_dict[tissue]['utr_rpkm'] += utr_rpkm
                            tissue_coverage_dict[tissue]['after_rpkm'] += after_rpkm
                            tissue_coverage_dict[tissue]['total_rpkm'] += total_rpkm
                        else:
                            tissue_coverage_dict.update({tissue:{
                                'intron_rpkm':intron_rpkm,
                                'exon_rpkm':exon_rpkm,
                                'utr_rpkm':utr_rpkm,
                                'after_rpkm':after_rpkm,
                                'total_rpkm':total_rpkm,
                                'strand':strand
                            }})

        gene_dict = tissue_coverage_dict

    return gene_dict


if __name__ == '__main__':

    arguments = sys.argv
    
    i = int(arguments[1])
    n = int(arguments[2])

   
    
    annot = pd.read_csv('results/torino/Human/isoforms/annotated.isoforms.tab.gz', sep='\t')
    annot['gene'] = annot.transcript.apply(lambda x: x.split('.')[0])
    
    human_iso = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                        names = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon', 'notes'])

    human_iso['transcript_id'] = human_iso.gene_id + '.' + human_iso.transcript_id

    efficiency_factor = []
    for idx, row in annot.iterrows():
        appris_ref = row.appris_ref
        aver = appris_ref.split('/')
        if 'RI:' in appris_ref:
            ri_factor = [x for x in aver if x.startswith('RI:')][0]
            ri_numbers = int(ri_factor.split(':')[1])
            appris_length = int(row.appris_transcript_length)
            
            if ((ri_numbers <= 2) and (appris_length >= 5)) or ((ri_numbers <= 1) and (appris_length >= 3)):
                efficiency_factor.append(False)
            else:
                efficiency_factor.append(True)
        else:
            efficiency_factor.append(False)
    
    annot['efficiency_factor'] = efficiency_factor

    

    genes = sorted(list(annot.gene.unique()))
    total_genes = len(genes)

    genes_per_slice = int(np.ceil(total_genes/n))

    slice_start = (i-1)*genes_per_slice
    if i == n:
        slice_end = total_genes + 1
    else:
        slice_end = i*genes_per_slice

    selected_genes = genes[slice_start:slice_end]

    with open(f'results/metaplots/tmp/metaplot_utr3_{str(i)}.csv', 'a') as fh:

        for gene in selected_genes:
            try:
                tissue_coverage_dict = process_gene(gene, annot, human_iso)
            except:
                tissue_coverage_dict = {}
        
            for tissue in tissue_coverage_dict.keys():
            
                total_rpkm = tissue_coverage_dict[tissue]['total_rpkm']
                norm_rpkm = total_rpkm/np.quantile(total_rpkm, 0.99)
                out_line = gene + ',' + tissue + ',' + ','.join([str(x) for x in norm_rpkm]) + '\n'
                fh.write(out_line)
        