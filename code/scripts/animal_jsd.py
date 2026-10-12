import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import sys

def get_jensenshannon_samples(ssmd_1, ssmd_2, sample_1, sample_2, species_1, species_2, orthologues):
    if species_1 == species_2:
        x = ssmd_1[[sample_1, sample_2]].dropna()
        y1 = np.array(x[sample_1])
        y2 = np.array(x[sample_2])

    else:
        x1 = ssmd_1[sample_1].dropna()
        x2 = ssmd_2[sample_2].dropna()

        genes_1_ = x1.index
        genes_2_ = x2.index

        ortho_x = orthologues.loc[orthologues[species_1].isin(genes_1_) & orthologues[species_2].isin(genes_2_), 
            [species_1, species_2]].dropna()
        
        
        genes_1 = pd.Index(ortho_x[species_1])
        genes_2 = pd.Index(ortho_x[species_2])
        

        # print(x1.head())
        y1 = np.array(x1.loc[genes_1])
        y2 = np.array(x2.loc[genes_2])

    jsd = jensenshannon(y1, y2)

    return jsd

        
if __name__ == '__main__':
    
    arguments = sys.argv
    chunk = int(arguments[1])
    output = arguments[2]
    
    human_ssmd_indv = pd.read_csv('../analysis/human_ssmd_indv.tsv', sep='\t', index_col=0)
    macaque_ssmd_indv = pd.read_csv('../analysis/macaque_ssmd_indv.tsv', sep='\t', index_col=0)
    rabbit_ssmd_indv = pd.read_csv('../analysis/rabbit_ssmd_indv.tsv', sep='\t', index_col=0)
    mouse_ssmd_indv = pd.read_csv('../analysis/mouse_ssmd_indv.tsv', sep='\t', index_col=0)
    rat_ssmd_indv = pd.read_csv('../analysis/rat_ssmd_indv.tsv', sep='\t', index_col=0)
    chicken_ssmd_indv = pd.read_csv('../analysis/chicken_ssmd_indv.tsv', sep='\t', index_col=0)
    cattle_ssmd_indv = pd.read_csv('../analysis/cattle_ssmd_indv.tsv', sep='\t', index_col=0)
    orthologues = pd.read_csv('../analysis/orthologues.all.tsv', sep='\t')


    human_ssmd = pd.read_csv('../analysis/human_ssmd.tsv', sep='\t', index_col=0)
    macaque_ssmd = pd.read_csv('../analysis/macaque_ssmd.tsv', sep='\t', index_col=0)
    rabbit_ssmd = pd.read_csv('../analysis/rabbit_ssmd.tsv', sep='\t', index_col=0)
    mouse_ssmd = pd.read_csv('../analysis/mouse_ssmd.tsv', sep='\t', index_col=0)
    rat_ssmd = pd.read_csv('../analysis/rat_ssmd.tsv', sep='\t', index_col=0)
    chicken_ssmd = pd.read_csv('../analysis/chicken_ssmd.tsv', sep='\t', index_col=0)
    cattle_ssmd = pd.read_csv('../analysis/cattle_ssmd.tsv', sep='\t', index_col=0)
    orthologues = pd.read_csv('../analysis/orthologues.all.tsv', sep='\t')


    human_select_genes = human_ssmd.loc[human_ssmd.max(axis=1) >= 2].index.intersection(human_ssmd_indv.index)
    macaque_select_genes = macaque_ssmd.loc[macaque_ssmd.max(axis=1) >= 2].index.intersection(macaque_ssmd_indv.index)
    mouse_select_genes = mouse_ssmd.loc[mouse_ssmd.max(axis=1) >= 2].index.intersection(mouse_ssmd_indv.index)
    rat_select_genes = rat_ssmd.loc[rat_ssmd.max(axis=1) >= 2].index.intersection(rat_ssmd_indv.index)
    cattle_select_genes = cattle_ssmd.loc[cattle_ssmd.max(axis=1) >= 2].index.intersection(cattle_ssmd_indv.index)
    chicken_select_genes = chicken_ssmd.loc[chicken_ssmd.max(axis=1) >= 2].index.intersection(chicken_ssmd_indv.index)
    rabbit_select_genes = rabbit_ssmd.loc[rabbit_ssmd.max(axis=1) >= 2].index.intersection(rabbit_ssmd_indv.index)
    
    ssmd_dir = {'Human':human_ssmd_indv.loc[human_select_genes],
                'Macaque':macaque_ssmd_indv.loc[macaque_select_genes],
                'Mouse':mouse_ssmd_indv.loc[mouse_select_genes],
                'Rat':rat_ssmd_indv.loc[rat_select_genes],
                'Rabbit':rabbit_ssmd_indv.loc[rabbit_select_genes],
                'Cattle':cattle_ssmd_indv.loc[cattle_select_genes],
                'Chicken':chicken_ssmd_indv.loc[chicken_select_genes],
               }
    
    human_samples = human_ssmd_indv.columns
    macaque_samples = macaque_ssmd_indv.columns
    rabbit_samples = rabbit_ssmd_indv.columns
    mouse_samples = mouse_ssmd_indv.columns
    rat_samples = rat_ssmd_indv.columns
    chicken_samples = chicken_ssmd_indv.columns
    cattle_samples = cattle_ssmd_indv.columns
    
    lists = [human_samples, macaque_samples, mouse_samples, rat_samples, rabbit_samples, cattle_samples, chicken_samples]
    species = ['Human', 'Macaque', 'Mouse', 'Rat', 'Rabbit', 'Cattle', 'Chicken']
    pairs = [(elem, s) for lst, s in zip(lists, species) for elem in lst]

    if chunk < 400:
        chunk_pairs = pairs[(chunk-1)*(12):(chunk)*12]
    else:
        chunk_pairs = pairs[(chunk-1)*(12):]


    


    with open(output, 'w') as fh:

        if chunk == 1:
            first_row = '\t'.join(['samples'] + [i[0] for i in pairs]) + '\n'

            fh.write(first_row)
    
        for i in chunk_pairs:
            
            i_sample = i[0]
            i_species = i[1]
    
            jsd_row = [i_sample]

            print(i_sample)
        
            ssmd_1 = ssmd_dir[i_species]
            
            for j in pairs:
                j_sample = j[0]
                j_species = j[1]
                ssmd_2 = ssmd_dir[j_species]
        
                jsd = get_jensenshannon_samples(ssmd_1, ssmd_2, i_sample, j_sample, i_species, j_species, orthologues)
                jsd_row.append(str(jsd))
            jsd_out = '\t'.join(jsd_row) + '\n'
            fh.write(jsd_out)
            