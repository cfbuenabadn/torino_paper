import gzip 
import re
import sys
import numpy as np

arguments = sys.argv
species = arguments[1]


with gzip.open(f'results/leafcutter2/{species}/leafcutter2.junction_counts.unclassified.gz', 'rb') as fh:
    with open(f'results/leafcutter2/{species}/junctions.tab', 'w') as fh2:
        header = fh.readline().decode()
        header = re.sub('.junc ', '\t', header)
        first_line = '#chrom\tstart\tend\tgene\t' + header
        
        fh2.write(first_line)
        for line in fh:
            linea = line.decode().rstrip().split(' ')
            coords = linea[0]
            chrom, start, end, cluster = coords.split(':')
            counts = np.array([int(x) for x in linea[1:]])            
            
            if (np.sum(counts > 0) >= 10) and (np.sum(counts) >= 20):
                new_line = [chrom, start, end, '.'] + linea[1:]
                new_line = '\t'.join(new_line) + '\n'
                fh2.write(new_line)
