import numpy as np
import pandas as pd
#import rpy2
import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri
#import rpy2.robjects.packages as rpackages
#ebpmf = rpackages.importr('ebpmf.alpha')
import os

from matplotlib import pyplot as plt
import seaborn as sns
import tabix
from pybedtools import BedTool
import sys

print('custom-made modules')

sys.path.append('/project2/mstephens/cfbuenabadn/gtex-stm/code/scripts')
from get_isoforms import *
from prepare_counts import *
# from sNMF_plots import *
# from tqdm import tqdm
# import pingouin as pg

print('finished loading modules')

def load_ebpmf_gene(gene, species, junc_file):
    readRDS = ro.r['readRDS']
    if species in ['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']:
        df = readRDS(f'results/torino/factorization/{species}/{gene}.rds')
    else:
        df = readRDS(f'results/torino/factorization_filtered/{species}/{gene}.rds')
    with (ro.default_converter + pandas2ri.converter).context():
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(df)

    coords = list(pd_from_r_df['torino_K10']['coords'])[1:-1]#pd_from_r_df['torino_K10']['coords']
#     print(coords)
#     print(junc_file)
    junctions_bed = run_tabix_on_junc(junc_file, coords)

    output = {'rds':pd_from_r_df, 'junctions_bed':junctions_bed}

    return output

def get_junctions_bed(coords, junc_file, gene_id, species, junc_thres=None):
    """
    Function: run_tabix
    
    Description:
    This function retrieves junction coordinates from a specified genomic region 
    using Tabix, and filters the junctions for a specific gene ID. It then formats 
    the retrieved junctions into a pandas DataFrame and converts them to a 
    BedTool object.
    
    Input:
    - coords: A list of genomic coordinates in the format 'chromosome:start-end'.
    - junc_file: A file path to a Tabix-indexed file containing junction information.
    - gene_id: A string representing the gene ID to filter the junctions.
    
    Output:
    - junctions_bed: A BedTool object containing junction coordinates for the specified gene ID, 
                     or an empty DataFrame if no junctions are found.
    """
    junctions_bed = run_tabix_on_junc(junc_file, coords)
    if junctions_bed.shape[0] == 0:
        junctions_bed = BedTool.from_dataframe(junctions_bed)
        return junctions_bed

    if species == 'Human':
        junctions_bed[3] = [x.split('.')[0] for x in junctions_bed[3]]

    junctions_bed_counts = junctions_bed.loc[junctions_bed[3] == gene_id, junctions_bed.columns[4:]].astype(int).quantile(0.999,axis=1)

    if junc_thres is None:
        junc_thres = junctions_bed_counts.max()/200
    
    # Filter junctions for the specified gene ID
    junctions_bed = junctions_bed.loc[:, range(4)]
    junctions_bed.columns = ['chrom', 'start', 'end', 'gene']

    print(junctions_bed)

    print('Diego')

    print(junctions_bed_counts)

    junctions_bed = junctions_bed[junctions_bed_counts >= junc_thres]
    
    # Return empty DataFrame if no junctions are found for the gene ID
    if junctions_bed.shape[0] == 0:
        junctions_bed = BedTool.from_dataframe(junctions_bed)
        return junctions_bed
    
    # Convert start and end positions to integers
    junctions_bed.start = junctions_bed.start.astype(int)
    junctions_bed.end = junctions_bed.end.astype(int)
    
    # Generate unique junction names
    junctions_bed['junc_names'] = ['junction_' + str(i) for i in range(1, junctions_bed.shape[0]+1)]
    
    # Rearrange DataFrame columns
    junctions_bed = junctions_bed[['chrom', 'start', 'end', 'junc_names']]
    
    # Convert DataFrame to BedTool object
    junctions_bed = BedTool.from_dataframe(junctions_bed)
    
    return junctions_bed


def write_gtf(merged_isoforms, gene, strand):
    K = len(merged_isoforms)
    isoforms = [f'isoform_{str(i+1)}' for i in range(K)]
    gene_start = np.nan
    gene_end = np.nan
    chrom = merged_isoforms['isoform_1']['df'].chrom.iloc[0]

    gtf_lines = []

    gene_description = f'gene_id "{gene}"'
    
    for isoform in isoforms:
        isoform_name = gene + '.' + isoform
        df = merged_isoforms[isoform]['df']
        factors_list = merged_isoforms[isoform]['factors']
        factors_description = ':'.join(factors_list)
        iso_start = df.start.astype(int).min()
        iso_end = df.end.astype(int).max()

        gene_start = np.nanmin([gene_start, iso_start])
        gene_end = np.nanmax([gene_end, iso_end])

        iso_start = str(iso_start)
        iso_end = str(iso_end)

        isoform_description = gene_description + f'; transcript_id "{isoform_name}"; factors_id "{factors_description}"'


        isoform_row = [chrom, 'Torino', 'transcript', iso_start, iso_end, '.', strand, '.', isoform_description]
        isoform_row = '\t'.join(isoform_row)

        gtf_lines.append(isoform_row)

        exon_count = 1
        for idx, row in df.iterrows():
            exon_start = str(int(row.start)) #str(row.start.astype(int))
            exon_end = str(int(row.end)) #str(row.end.astype(int))
            exon_name = f'exon_{str(exon_count)}'
            exon_description = isoform_description + f'; exon_id "{exon_name}"'
            exon_row = [chrom, 'Torino', 'exon', exon_start, exon_end, '.', strand, '.', exon_description]
            exon_row = '\t'.join(exon_row)
            gtf_lines.append(exon_row)
            exon_count += 1

    gene_start = str(gene_start)
    gene_end = str(gene_end)

    gene_line = [chrom, 'Torino', 'gene', str(int(float(gene_start))), str(int(float(gene_end))), '.', strand, '.', gene_description]
    gene_line = '\t'.join(gene_line)

    gtf_lines = [gene_line] + gtf_lines

    gene_gtf = '\n'.join(gtf_lines) + '\n'
    
    return gene_gtf 


def merge_EL(EL, merged_isoforms):
    K = EL.shape[1]
    EL = pd.DataFrame(EL)
    EL.columns = [f'factor_{str(i+1)}' for i in range(K)]
    EL_merged = pd.DataFrame()
    K_ = len(merged_isoforms)

    for i in range (K_):
        iso_name = f'isoform_{str(i+1)}'
        factors_in_isoform = merged_isoforms[iso_name]['factors']
        EL_sum = list(EL[factors_in_isoform].sum(axis=1))
        EL_merged[iso_name] = EL_sum
    
    return EL_merged

def process_gene(gene, gtf_fh, correct_bias, junc_file, species, strand, gene_id = '.'):#, eta_fh, pvals_fh, H_fh, kw_pvals_fh):
    rds = load_ebpmf_gene(gene, species, junc_file)
    if 'strand' in rds['rds'].keys():
        strand = rds['rds']['strand'][0]
        if strand == 'plus':
            strand = '+'
        else:
            strand = '-'
#     else:
    
        
    #junc_file = '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'
    coords = list(rds['rds']['torino_K10']['coords'])[1:-1]
    coords_all = list(rds['rds']['coords_all'])
#     coords = rds['rds'][f'ebpmf_{str(K)}']['coords']
    junctions_bed = get_junctions_bed(coords, junc_file, gene_id, species)
#     coords_all = list(rds['rds']['coords'])
#     coords = list(rds['rds'][f'ebpmf_{str(K)}']['coords'])
    isoforms = get_isoforms(rds['rds']['torino_K10']['EF_smooth'], junctions_bed, coords, coords_all, binary_fraction=0.5, 
                            correct_bias = correct_bias, print_gene = gene)
    merged_isoforms = merge_isoforms(isoforms)

    gene_gtf = write_gtf(merged_isoforms, gene, strand)
    gtf_fh.write(gene_gtf)


if __name__ == '__main__':

    arguments = sys.argv
    
    i = int(arguments[1])
    n = int(arguments[2])
    correction_tag = str(arguments[3])
    species = arguments[4]
    junc_file = arguments[5]
    species = arguments[6]
    
    if correction_tag == '3prime_corrected':
        run_lm = True
    elif correction_tag == 'uncorrected':
        run_lm = False
    else:
        raise Exception('Argument for correcting 3 prime bias should be "3prime_corrected" or "uncorrected"')

    print('Reading list of genes.')
    if species in ['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']:
        genes = os.listdir(f'results/torino/factorization/{species}')
    else:
        genes = os.listdir(f'results/torino/factorization_filtered/{species}')
        
    genes = [x.split('.')[0] for x in genes]
    print('total genes:')
    print(len(genes))
    print('')
#     samples = pd.read_csv('config/Cattle.filtered_samples_strict.txt', sep='\t', index_col=0)
#     print('loaded samples files')

    bed_table = pd.read_csv(f'resources/annotation/{species}/bed/genes.protein_coding.bed.gz', sep='\t')#, 
                            # names = ["#chrom", "start", "end", "gene_id", "gene_name", "strand"])
    
    new_gene_name = []
    for idx, row in bed_table.iterrows():
        if row.gene_name == 'ensembl':
            new_gene_name.append(row.gene_id)
        else:
            new_gene_name.append(row.gene_name)
            
    bed_table['gene_name'] = new_gene_name

    total_genes = len(genes)

    genes_per_slice = int(total_genes/n)

    slice_start = (i-1)*genes_per_slice
    if i == n:
        slice_end = total_genes + 1
    else:
        slice_end = i*genes_per_slice

    selected_genes = genes[slice_start:slice_end]

    with open(f'results/torino/{species}/isoforms/{correction_tag}_{str(i)}.gtf', 'w') as gtf_fh:
        counter = 1
        print('start processing gene')
        selected_genes = list(selected_genes)

        #if str(i) == '94':
        #    gene = 'SRSF3'
        #    gene_id = 'ENSG00000112081'
        #    strand = bed_table.loc[bed_table.gene_name == gene].strand.iloc[0]
        #    print(strand)
        #    process_gene(gene, gtf_fh, correct_bias = run_lm, junc_file=junc_file, species=species, strand = strand, gene_id = gene_id)#, eta_fh, pvals_fh, H_fh, 

        #else:
            
        
        for gene in selected_genes:
            if gene in genes:
                print(gene)
                if counter % 100 == 0:
                    print(counter)
                counter += 1
                try:
                    if species == 'Human':
                        gene_id = bed_table.loc[bed_table.gene_name == gene].gene_id.iloc[0]
                    else:
                        gene_id = '.'
                    strand = bed_table.loc[bed_table.gene_name == gene].strand.iloc[0]
                    print(strand)
                    process_gene(gene, gtf_fh, correct_bias = run_lm, junc_file=junc_file, species=species, strand = strand, gene_id=gene_id)#, eta_fh, pvals_fh, H_fh, kw_pvals_fh)
                except:
                    continue
                                    
    print('done!')





    # results/torino/factorization_filtered
    # f'{rds_dir}/{species}/{gene}.rds'
