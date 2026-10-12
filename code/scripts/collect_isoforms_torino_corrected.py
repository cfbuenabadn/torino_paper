import numpy as np
import pandas as pd
import os
import sys

sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
from torino_isoforms import *

if __name__ == '__main__':

    arguments = sys.argv
    
    i = int(arguments[1])
    n = int(arguments[2])
    species = arguments[3]
    junc_file = arguments[4]
    species = arguments[5]
    

    print('Reading list of genes.')
    if species == 'Human':
        bed_table = pd.read_csv('config/human.protein_coding.bed.gz', sep='\t')
    else:
        bed_table = pd.read_csv(f'resources/annotation/{species}/bed/genes.protein_coding.bed.gz', sep='\t')

    genes = sorted(list(bed_table.gene_name))

    
    #genes = [x.split('.')[0] for x in genes]
    print('total genes:')
    print(len(genes))
    print('')
#     samples = pd.read_csv('config/Cattle.filtered_samples_strict.txt', sep='\t', index_col=0)
#     print('loaded samples files')

    # bed_table = pd.read_csv(f'resources/annotation/{species}/bed/genes.protein_coding.bed.gz', sep='\t')#, 
    #                         # names = ["#chrom", "start", "end", "gene_id", "gene_name", "strand"])
    
    # new_gene_name = []
    # for idx, row in bed_table.iterrows():
    #     if row.gene_name == 'ensembl':
    #         new_gene_name.append(row.gene_id)
    #     else:
    #         new_gene_name.append(row.gene_name)
            
    # bed_table['gene_name'] = new_gene_name

    total_genes = len(genes)

    genes_per_slice = int(np.ceil(total_genes/n))

    slice_start = (i-1)*genes_per_slice
    if i == n:
        slice_end = total_genes + 1
    else:
        slice_end = i*genes_per_slice

    selected_genes = genes[slice_start:slice_end]

    with open(f'results/torino/{species}/corrected_isoforms/tmp/factors_{i}.bed', 'w') as factors_fh:
        with open(f'results/torino/{species}/corrected_isoforms/tmp/isoforms_{i}.bed', 'w') as isoforms_fh:
            with open(f'results/torino/{species}/corrected_isoforms/tmp/missing_genes_{i}.txt', 'w') as missing_fh:
                counter = 1
                print('start processing gene')
                selected_genes = list(selected_genes)
                
                for gene in selected_genes:
                    if gene in genes:
                        print(gene)
                        if counter % 100 == 0:
                            print(counter)
                        counter += 1

                        if species in ['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']:
                            # rds_file = f'results/torino/factorization_extended/{species}/{gene}.rds'
                            rds_file = f'results/torino/factorization/{species}/{gene}.rds'
                        else:
                            rds_file = f'results/torino/factorization_filtered/{species}/{gene}.rds'

                        if not os.path.isfile(rds_file):
                            missing_fh.write(gene + '\tMissing RDS (Torino not run).\n')
                        else:
                            
                            try:
                                merged_isoforms, factor_isoforms = process_isoforms(gene, species, bed_table, 
                                                                                    junc_file, use_low_juncs=True)
    
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


