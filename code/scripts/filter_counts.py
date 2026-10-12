import numpy as np
import pandas as pd
import sys
import tabix
from pybedtools import BedTool
import gzip
import csv

def get_gene_overlaps(chrom, start, end, gene, species):

    if species == 'Human':
        tb_exons_appris = tabix.open('resources/annotation/Human/bed/exons.apris_principal.bed.gz')
        tb_exons_all = tabix.open('resources/annotation/Human/bed/exons.all.bed.gz')
        # tb_exons_appris = tabix.open('Annotations/gencode.v44.primary_assembly.appris_principal_exons.bed.gz')
        # tb_exons_all = tabix.open('Annotations/gencode.v44.primary_assembly.exons.bed.gz')

    else:
        tb_exons_appris = tabix.open(f'resources/annotation/{species}/bed/exons.protein_coding.bed.gz')
        tb_exons_all = tabix.open(f'resources/annotation/{species}/bed/exons.all.bed.gz')
    
    print(chrom, start, end)
    
    exons_appris_region = tb_exons_appris.query(chrom, start, end)
    exons_appris_region = pd.DataFrame(exons_appris_region, 
                                        columns = ['#chrom', 'start', 'end', 'gene_id', 'gene_name', 
                                                   'strand', 'gene_type', 'transcript_id', 'transcript_type'])
    
    if (species == 'Cattle') & (gene.startswith('ENSBTAG')):
        exons_appris_gene = BedTool.from_dataframe(exons_appris_region.loc[exons_appris_region.gene_id == gene])
    # elif (species == 'Human') and (gene in ('NPIPA9.ENSG00000183889', 'NPIPA9.ENSG00000233024')):
    #     gene_symbol, gene_id = gene.split('.')
    #     exons_appris_gene = BedTool.from_dataframe(exons_appris_region.loc[
    #                                                (exons_appris_region.gene_name == gene_symbol) & (exons_appris_region.gene_id == gene_id)
    #                                                ])
    # else:
    exons_appris_gene = BedTool.from_dataframe(exons_appris_region.loc[exons_appris_region.gene_name == gene])

    exons_appris_gene = exons_appris_gene.merge()
    
    exons_all_region = tb_exons_all.query(chrom, start, end)
    exons_all_region = pd.DataFrame(exons_all_region, 
                                        columns = ['#chrom', 'start', 'end', 'gene_id', 'gene_name', 
                                                   'strand', 'gene_type', 'transcript_id', 'transcript_type'])
    
    if (species == 'Cattle') & (gene.startswith('ENSBTAG')):
        overlapping_exons_gene = exons_all_region.loc[exons_all_region.gene_id != gene].copy()
    elif species == 'Human':
        
        with open('config/human.expressed_genes.txt', 'r') as fh:
            expressed_genes = [x.rstrip() for x in fh.readlines()]
        overlapping_exons_gene = exons_all_region.loc[
            (exons_all_region.gene_name != gene) & (exons_all_region.gene_id.isin(expressed_genes))
            ].copy()
    else:
        overlapping_exons_gene = exons_all_region.loc[exons_all_region.gene_name != gene].copy()
    
    if len(overlapping_exons_gene) == 0:
        return []
    overlapping_exons_gene.start = overlapping_exons_gene.start.astype(int)
    overlapping_exons_gene.end = overlapping_exons_gene.end.astype(int)
    overlapping_exons_gene.start -= 25
    overlapping_exons_gene.end += 25
    overlapping_exons_gene_bed = BedTool.from_dataframe(overlapping_exons_gene)

    overlapping_exons_gene_bed = overlapping_exons_gene_bed.merge()

    overlapping_exons_gene_df = overlapping_exons_gene_bed.to_dataframe()

    overlapping_exons_exons_df = exons_appris_gene.intersect(overlapping_exons_gene_bed).to_dataframe()

    exon_appris_gene_df = exons_appris_gene.to_dataframe()

    
    total_exon_appris_gene_length = (exon_appris_gene_df.end - exon_appris_gene_df.start).sum()

    total_exons_gene_overlap = (overlapping_exons_gene_df.end - overlapping_exons_gene_df.start).sum()
    if len(overlapping_exons_exons_df) == 0:
        total_exon_exon_overlap = 0
    else:
        total_exon_exon_overlap = (overlapping_exons_exons_df.end - overlapping_exons_exons_df.start).sum()

    total_gene_length = end - start
    

    gene_overlap_percent = total_exons_gene_overlap/total_gene_length
    exon_overlap_percent = total_exon_exon_overlap/total_exon_appris_gene_length

    if (exon_overlap_percent < 0.33):
        remove_region = []

        for idx, df in overlapping_exons_gene_df.iterrows():
            bp_array = np.arange(df.start, df.end)
            regions = [df.chrom + ':' + str(x) for x in bp_array]
            remove_region.extend(regions)
            
        return remove_region
    else:
        raise Exception('Found too many overlaps on gene.')


def read_table(gene, species):
    first_line = True
    suma_array = 0
    
    
    samples_list = []
    
    with gzip.open(f'resources/count_matrices/{species}/{gene}.csv.gz', 'rt') as csvfile:
        # csvfile = gzip.GzipFile(gzip_fd)
        reader = csv.reader(csvfile)
        for column in reader:
            samples_list.append(column[0])
            #print(column)
            if first_line:
                coords = column[1:]
                first_line = False
                continue
            else:
                position = column[0]
                counts = np.array(column[1:]).astype(int)
                suma_array += counts
                
    return suma_array, coords


def filter_table(gene, species, coords, selected_coords):
    selected_array = np.array([x in selected_coords for x in coords])

    first_line = True
    
    with gzip.open(f'resources/count_matrices/{species}/{gene}.csv.gz', 'rt') as csvfile:

        reader = csv.reader(csvfile)
        
        with gzip.open(f'resources/count_matrices_filtered/{species}/{gene}.csv.gz', 'wb') as outfile:

            for column in reader:
                sample = column[0]
                counts = np.array(column[1:])
                assert len(counts) == len(selected_array)
                counts_selected = list(counts[selected_array])
    
                new_row = [sample] + counts_selected
                new_row = ','.join(new_row) + '\n'
                new_row_encoded = new_row.encode('UTF-8')
                outfile.write(new_row_encoded)

    print('done')
                

def get_long_introns(df_expressed, coords):
    coords = pd.Index(coords)
    consecutive_count = 0
    start_idx = None
    result = []
    
    # Iterate through the series
    for idx, value in df_expressed.items():
        if not value:
            # If it's True and it's the start of a sequence, record the start index
            if consecutive_count == 0:
                start_idx = idx
            consecutive_count += 1
        else:
            # If it's False, record the count and reset consecutive count
            if consecutive_count > 0:
                result.append({'idx': start_idx, 'cum_true': consecutive_count})
                consecutive_count = 0
    
    # If the sequence ends with True, record the final count
    if consecutive_count > 0:
        result.append({'idx': start_idx, 'cum_true': consecutive_count})
    
    # Create a DataFrame from the result list
    result_df = pd.DataFrame(result).set_index('idx')

    long_introns = result_df.loc[result_df.cum_true > 500]

    coord_remove_list = []
    
    for idx, length in long_introns.iterrows():
        chrom, coord = idx.split(':')
        coord = int(coord)
        length = int(length.iloc[0])
        coord_ = coord + length
    
        coord_remove = [chrom + ':' + str(i) for i in range(coord + 25, coord_ - 25)]
    
        coord_remove_list.extend(coord_remove)

    coord_remove_list = pd.Index(coord_remove_list)
    selected_coords = coords.difference(coord_remove_list)

    return selected_coords, coord_remove_list, long_introns


def main():
    arguments = sys.argv
    gene = arguments[1]
    species = arguments[2]

    print('read table')

    suma_array, coords = read_table(gene, species)
    coords = pd.Index(coords)
    
    chrom, start = coords[0].split(':')
#     if species == 'Cattle':
#         chrom = chrom[3:]
    start = int(start)
    end = int(coords[-1].split(':')[1])

    selected_coords = coords
    coord_remove_list = pd.Index([])
    
    overlap_bp = get_gene_overlaps(chrom, start, end, gene, species)

    overlap_bp = pd.Index(overlap_bp).intersection(coords)


    remove_no_counts = len(coord_remove_list)

    coord_remove_list = coord_remove_list.union(overlap_bp)

    selected_coords = selected_coords.difference(coord_remove_list)
    
    print(len(selected_coords))

    print('select coords')

    filter_table(gene, species, coords, selected_coords)


    selected_array = np.array([x in selected_coords for x in coords])
    removed_array = np.array([x in coord_remove_list for x in coords])

    total_X = np.sum(suma_array)
    filtered_X = np.sum(suma_array[selected_array])

    X_length = len(suma_array)
    overlap_length = len(overlap_bp)
    new_length = len(selected_coords)
    removed_sections = 0 #long_introns.shape[0]
    removed_sum = np.sum(suma_array[removed_array])
    if not any(removed_array):
        print('nothing was removed')
        removed_max = 0
    else:
        removed_max = np.max(suma_array[removed_array])
    filtered_sum = np.sum(suma_array[selected_array])
    filtered_max = np.max(suma_array[selected_array])

    print('creating stats file')

    with open(f'resources/filter_stats/{species}/{gene}.stats', 'w') as fh:
        fh.write('total_length\t' + str(X_length) + '\n')
        fh.write('no_count_length\t' + str(remove_no_counts) + '\n')
        fh.write('overlap_length\t' + str(overlap_length) + '\n')
        fh.write('filtered_length\t' + str(new_length) + '\n')
        fh.write('total_counts\t' + str(total_X) + '\n')
        fh.write('filtered_counts\t' + str(filtered_X) + '\n')
        fh.write('max_counts_in_bp\t' + str(filtered_max) + '\n')
        fh.write('max_total_counts_in_bp\t' + str(filtered_sum) + '\n')
        fh.write('max_counts_in_removed_bp\t' + str(removed_max) + '\n')
        fh.write('max_total_counts_in_removed_bp\t' + str(removed_sum) + '\n')
        


if __name__ == '__main__':
    main()
    

