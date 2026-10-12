import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
import gzip
import os
import sys

arguments = sys.argv
species = arguments[1]
tissue = arguments[2]

if species == 'Cattle':
    species_directory = 'FarmGTEx/cattleGTEx'
elif species == 'Sheep':
    species_directory = 'FarmGTEx/sheepGTEx'

with open(f'resources/metadata/{species}/tissue_tables/metadata.{tissue}.txt', 'w') as fh:
    samples_list = []
    counts_list = []
    print(f'Working on {species} tissue {tissue}')
    bed_samples = [x.split('.')[0] for x in os.listdir(f'resources/{species_directory}/bed/{tissue}') if x.endswith('.bed.gz')]
    print(f'Total: {str(len(bed_samples))} samples')
    for sample in bed_samples:
        counts = 0
        total_lines = 0
        print(f'Working on resources/{species_directory}/bed/{tissue}/{sample}.bed.gz')
        with gzip.open(f'resources/{species_directory}/bed/{tissue}/{sample}.bed.gz', 'rb') as fh2:
            for line in fh2:
                if (total_lines % 1000000) == 0:
                    print(total_lines)
                line_counts = int(float(line.decode().rstrip().split('\t')[-1]))
                counts += line_counts
                total_lines += 1

        sample_row = f'{sample}\t{tissue}\t{str(counts)}\n'
        fh.write(sample_row)
        print(sample_row)
