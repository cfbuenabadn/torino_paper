import numpy as np
import pandas as pd
import os
import sys

sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
from torino_isoforms import *


def get_junctions_bed_gao(coords, junc_file, gene_id, junc_thres=None):
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

    junctions_bed[3] = [x.split('.')[0] for x in junctions_bed[3]]

    junctions_bed = junctions_bed.loc[junctions_bed[3] == gene_id]
    junctions_bed_counts = junctions_bed.loc[junctions_bed[3] == gene_id, junctions_bed.columns[4:]].astype(int).quantile(0.999,axis=1)

    if junc_thres is None:
        junc_thres = junctions_bed_counts.max()/200

    # print(junc_thres)
    
    # Filter junctions for the specified gene ID
    junctions_bed = junctions_bed.loc[:, range(4)]
    junctions_bed.columns = ['chrom', 'start', 'end', 'gene']

    junctions_bed['junc_counts'] = list(junctions_bed_counts)

    
    junctions_bed = junctions_bed[junctions_bed_counts >= junc_thres]

    # Convert start and end positions to integers
    junctions_bed.start = junctions_bed.start.astype(int)
    junctions_bed.end = junctions_bed.end.astype(int)

    ##### Make sure juncs are within gene body
    coords_start = int(coords[0].split(':')[1])
    coords_end = int(coords[-1].split(':')[1])
    junctions_bed = junctions_bed.loc[(junctions_bed.start >= coords_start) & (junctions_bed.end <= coords_end)]
    
    # Return empty DataFrame if no junctions are found for the gene ID
    if junctions_bed.shape[0] == 0:
        junctions_bed = BedTool.from_dataframe(junctions_bed)
        return junctions_bed

    # Generate unique junction names
    junctions_bed['junc_names'] = ['junction_' + str(i) for i in range(1, junctions_bed.shape[0]+1)]
    
    # Rearrange DataFrame columns
    junctions_bed = junctions_bed[['chrom', 'start', 'end', 'junc_names', 'junc_counts']]
    
    # Convert DataFrame to BedTool object
    junctions_bed = BedTool.from_dataframe(junctions_bed)
    
    return junctions_bed
    
    
    

def load_ebpmf_gene_gao(gene_id):
    readRDS = ro.r['readRDS']
    df = readRDS(f'/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/RDS_all/{gene_id}.rds')
    with (ro.default_converter + pandas2ri.converter).context():
        pd_from_r_df = ro.conversion.get_conversion().rpy2py(df)

    junc_file = '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'
    coords = pd_from_r_df['coords']
    #junctions_bed = run_tabix_on_junc(junc_file, coords, gene_id)
    #print('juncs_shape:')
    #print(junctions_bed.shape)
    output = {'rds':pd_from_r_df#, 
              #'junctions_bed':junctions_bed
             }

    return output

def get_unmerged_isoforms_gao(gene, junc_file, strand, gene_id = '.'):
    rds = load_ebpmf_gene_gao(gene)

    if 'snmf_3' not in rds['rds'].keys():
        return None

    if type(rds['rds'][f'snmf_3']) is NULLType:
        return None
        
    #junc_file = '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'
    coords = list(rds['rds']['snmf_3']['coords'])#[1:-1]
    coords_all = list(rds['rds']['coords'])

    junctions_bed = get_junctions_bed_gao(coords, junc_file, gene_id)

    isoforms = get_isoforms_junc_driven(rds['rds'][f'snmf_3']['ebpmf']['EF_smooth'], junctions_bed, coords, coords_all, strand, gene)
    return isoforms

def process_isoforms_gao(gene, genes_file, junc_file):
    strand = genes_file.loc[genes_file.gene_id == gene].strand.iloc[0]
    gene_id = genes_file.loc[genes_file.gene_id == gene].gene_id.iloc[0]
        
    isoforms = get_unmerged_isoforms_gao(gene, junc_file, gene_id = gene_id, strand = strand)
    if isoforms is None:
        return None, None
    merged_isoforms = pd.concat(factors_to_isoforms(isoforms, merge_factors_into_isoforms(isoforms, strand)))

    return merged_isoforms, isoforms

if __name__ == '__main__':

    arguments = sys.argv
    
    chunk = int(arguments[1])
    junc_file = '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'

    genes = os.listdir('/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/RDS_all/')
    genes = [x.split('.')[0] for x in genes if x.startswith('ENSG')]
    

    print('Reading list of genes.')
    bed_table = pd.read_csv(
        '/project2/mstephens/cfbuenabadn/gtex-stm/code/Annotations/gencode.v44.primary_assembly.protein_coding.genes.bed.gz', 
                            sep='\t')

    bed_table['gene_name'] = bed_table.gene_id
    genes = [x for x in genes if x in list(bed_table.gene_id)]



    n = 2000

    total_genes = len(genes)

    genes_per_slice = int(np.ceil(total_genes/2000))

    slice_start = (chunk-1)*genes_per_slice
    if chunk == 2000:
        slice_end = total_genes + 1
    else:
        slice_end = chunk*genes_per_slice

    selected_genes = genes[slice_start:slice_end]

    genes = selected_genes
    
    bed_table = bed_table.loc[bed_table.gene_id.isin(genes)]

    print('total genes:')
    print(len(genes))
    print('')

    
    with open(f'results/gao_k3/isoforms/tmp/factors_{chunk}.bed', 'w') as factors_fh:
        with open(f'results/gao_k3/isoforms/tmp/isoforms_{chunk}.bed', 'w') as isoforms_fh:
            with open(f'results/gao_k3/isoforms/tmp/missing_genes_{chunk}.txt', 'w') as missing_fh:
                counter = 1
                print('start processing gene')
                
                for gene in genes:

                    rds_file = f'/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/RDS_all/{gene}.rds'

                    if not os.path.isfile(rds_file):
                        missing_fh.write(gene + '\tMissing RDS (Torino not run).\n')
                    else:
                        
                        try:
                            merged_isoforms, factor_isoforms = process_isoforms_gao(gene, bed_table, junc_file)
    
                            if merged_isoforms is None:
                                missing_fh.write(gene + '\tNot enough coverage to run Torino.\n')
    
                            else:
    
                                for idx, row in merged_isoforms.iterrows():
                                    linea = '\t'.join([str(x) for x in list(row)]) + '\n'
                                    isoforms_fh.write(linea)
    
                                for key, factor_iso in factor_isoforms.items():
                                    factor_iso['factor_id'] = [key]*len(factor_iso)
                                    for idx, row in factor_iso.iterrows():
                                        linea = '\t'.join([str(x) for x in list(row)]) + '\n'
                                        factors_fh.write(linea)
                                    
                        except:
                            
                            continue
                                            
    print('done!')


