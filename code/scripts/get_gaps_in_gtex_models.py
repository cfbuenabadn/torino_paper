import os
import gzip

genes = [x.split('.')[0] for x in os.listdir('resources/count_matrices_filtered/Human/')]

def get_gaps(coords, gene):
    chrom, start = coords[0].split(':')
    gaps = []
    gap_counter = 1
    for i in range(len(coords)-1):
        chrom, pos = coords[i].split(':')
        chrom, next_pos = coords[i+1].split(':')
        pos = int(pos)
        next_pos = int(next_pos)
        if (next_pos - pos) > 1:
            gap = [chrom, str(pos), str(next_pos), f'{gene}:{str(gap_counter)}\n']
            gap = '\t'.join(gap)
            gap_counter += 1
            
            gaps.append(gap)
    return gaps

def get_gaps_per_gene(gene):
    with gzip.open(f'resources/count_matrices_filtered/Human/{gene}.csv.gz', 'rb') as fh_:
        line = fh_.readline()
        
    coords = line.decode().split(',')[1:]
    gaps = get_gaps(line.decode().split(',')[1:], gene)
    return gaps

with gzip.open('results/CoveragePlots/other_files/GTEx_overlapping_regions.bed.gz', 'wb') as fh:
    fh.write('chrom\tstart\tend\tgap\n'.encode())
    for gene in genes:
        gaps = get_gaps_per_gene(gene)
        for gap in gaps:
            fh.write(gap.encode())
    