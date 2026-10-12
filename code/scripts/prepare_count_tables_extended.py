import pandas as pd
import gzip
import os
import numpy as np
import tabix
import re
import sys

def process_and_write(gene, species):

    cardoso = ['Macaque', 'Rabbit', 'Rat', 'Mouse', 'Chicken']

#     gene_id = gene_dir[gene][species][0]
#     w1 = gene_dir[gene][species][1]
#     w2 = gene_dir[gene][species][2]

    if gene.startswith('chr'):
        w1 = 10000
        w2 = 10000
        coords = gene
        chrom = coords.split(':')[0]
        start = int(coords.split(':')[1].split('-')[0]) - w1
        end = int(coords.split(':')[1].split('-')[1]) + w2
        strand = coords.split(':')[2]
        
    elif species == 'Human':
        selected_genes = pd.read_csv('config/human.protein_coding.bed.gz',#'/project2/mstephens/cfbuenabadn/gtex-stm/data/protein_coding_genes.bed.gz',
                                     sep='\t')#, names = ['chrom', 'start', 'end', 'gene', 'gene_symbol', 'strand'])

        # if (gene == 'NPIPA9.ENSG00000183889') or (gene == 'NPIPA9.ENSG00000233024'):
        #     gene_sym, gene_id = gene.split('.')
        #     gene_bed = selected_genes.loc[(selected_genes.gene==gene_id) & (selected_genes.gene_symbol == gene_sym)]
        # else:
        gene_bed = selected_genes.loc[selected_genes.gene_name==gene]
        
        w1 = 50
        w2 = 50
        chrom = str(gene_bed['#chrom'].iloc[0])#.chrom.iloc[0])
        start = int(gene_bed.start.iloc[0]) - w1
        end = int(gene_bed.end.iloc[0]) + w2
        strand = str(gene_bed.strand.iloc[0])
        
        metadata = pd.read_csv('config/gtex_samples.tsv', sep='\t')
        metadata.columns = ['Sample', 'participant', 'Tissue', 'sex', 'bed_file']
        output_name = f'resources/count_matrices_extended/{species}/{gene}.csv.gz'

    else:

        selected_genes = pd.read_csv(f'resources/annotation/{species}/bed/genes.protein_coding.bed.gz', sep='\t')
        #selected_genes = pd.read_csv(f'resources/annotation/bed/{species}.bed.gz', sep='\t', #'../data/selected_genes.bed', sep='\t', 
        #                             names = ['chrom', 'start', 'end', 'gene', 'gene_symbol', 'strand'])
        
        # if (species == 'Cattle') and (gene.startswith('ENSBT')):
        #     gene_bed = selected_genes.loc[selected_genes.gene==gene]
        # elif (species == 'Human') and (gene.startswith('ENSG')):
        #     gene_bed = selected_genes.loc[selected_genes.gene==gene]
        # elif (species == 'Sheep') and (gene.startswith('ENSOARG')):
        #     gene_bed = selected_genes.loc[selected_genes.gene==gene]
        # else:
        gene_bed = selected_genes.loc[selected_genes.gene_name==gene]
        
        strand = str(gene_bed.strand.iloc[0])
        if strand == '+':
            w2 = 10000
            w1 = 1000
        else:
            
            w1 = 10000
            w2 = 1000
        chrom = str(gene_bed['#chrom'].iloc[0])
        #chrom = str(gene_bed.chrom.iloc[0])
        start = int(gene_bed.start.iloc[0]) - w1
        end = int(gene_bed.end.iloc[0]) + w2
        strand = str(gene_bed.strand.iloc[0])

        if (species == 'Macaque') and (gene == 'TSC22D4'):
            print(start)
            print(end)
            start = 42538952
            end = 42554810
        elif (species == 'Mouse') and (gene == 'Tsc22d4'):
            print(start)
            print(end)
            start = 137742992
            end = 137758991

    if species == 'Sheep':
        gtex_name = 'sheepGTEx'
    
        metadata = pd.read_csv('config/sheepGTEx_metadata.txt', sep='\t')[['Sample', 'Tissue']]
        output_name = f'resources/count_matrices_extended/{species}/{gene}.csv.gz'
    elif species == 'Cattle':
        gtex_name = 'cattleGTEx'
        metadata = pd.read_csv('config/cattleGTEx_metadata.txt', sep='\t')[['Sample', 'Tissue']]
#         tissues_20 = metadata.value_counts('Tissue').loc[metadata.value_counts('Tissue')>=40].index
#         metadata = metadata.loc[metadata.Tissue.isin(tissues_20)].groupby('Tissue').head(40).reset_index(drop=True)
        metadata.columns = ['Sample', 'Tissue']
        cattle_tissues = ['Blood', 'Liver', 'Muscle', 'Mammary_Gland', 'Subcutaneous_Adipose', 'Testis', 
                          'CNS', 'Hypothalamus', 'Colon', 'Lung', 'Skin', 'Heart']
        metadata = metadata.loc[metadata.Tissue.isin(cattle_tissues)]
        metadata['Tissue'] = metadata.Tissue.apply(lambda x: re.sub(' ', '_', x))
        output_name = f'resources/count_matrices_extended/{species}/{gene}.csv.gz'
        chrom = chrom[3:]

    elif species in cardoso:
        if strand == '+':
            read_strand = 'reverse'
        else:
            read_strand = 'forward'
    
        metadata = pd.read_csv('config/cardoso_samples.tsv', sep='\t')
        metadata = metadata.loc[metadata.Species == species, ['sampleID', 'Tissue']]
        metadata.columns = ['Sample', 'Tissue']
        output_name = f'resources/count_matrices_extended/{species}/{gene}.csv.gz'
        
    file_count = 1
    with gzip.open(output_name, 'wt', compresslevel=6) as fh:
        for idx, row in metadata.iterrows():

            tissue = row.Tissue

            if species in cardoso:

                
                if tissue == 'Brain,Cerebellum':
                    tissue = 'Cerebellum'

                file_name = f'resources/Cardoso/{species}/bed/{row.Sample}.{read_strand}5.bed.gz'
                coord_str, counts_str5 = process_sample(tissue, file_name, chrom, start, end)
                file_name = f'resources/Cardoso/{species}/bed/{row.Sample}.{read_strand}3.bed.gz'
                coord_str, counts_str3 = process_sample(tissue, file_name, chrom, start, end)
                if file_count == 1:
                    fh.write(coord_str)
    
                counts_str5 = counts_str5.split(',')
                sample_name = row.Sample + '.' + tissue #counts_str5[0]
                counts_str3 = counts_str3.split(',')
                counts5 = np.array([int(x) for x in counts_str5[1:]])
                counts3 = np.array([int(x) for x in counts_str3[1:]])
                counts = counts5 + counts3
                counts_str = [sample_name] + [str(x) for x in counts]
                counts_str = ','.join(counts_str) + '\n'
                file_count += 1
                fh.write(counts_str)
                file_count += 1
                
            elif species == 'Human':
                file_name = row.bed_file
                coord_str, counts_str = process_sample(tissue, file_name, chrom, start, end)
                if file_count == 1:
                    fh.write(coord_str)
                fh.write(counts_str)
                file_count += 1

            else:
                file_name = f'resources/FarmGTEx/{gtex_name}/bed/{tissue}/{row.Sample}.bed.gz'

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

    arguments = sys.argv
    species = arguments[1]
    gene = arguments[2]
    process_and_write(gene, species)

