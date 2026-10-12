import gzip
import os
import glob
import sys
from collections import defaultdict

DestinationDir = sys.argv[1].rstrip('/')
SourceDir = sys.argv[2].rstrip('/')

FilesToSplit = glob.glob(SourceDir + '/*.txt.gz')

os.makedirs(DestinationDir, exist_ok=True)

header = "phenotype\tphenotype_id\tgwas_locus\tsnp\tbeta\tbeta_se\tp\n"

for file_n in FilesToSplit:
    fdst_fn = os.path.basename(file_n)

    # Dictionary of chromosome -> open file handle
    outfiles = {}

    with gzip.open(file_n, 'rb') as fh:
        fh.readline()  # skip header
        for line in fh:
            chrom = line.decode().split('\t')[2].split('_')[0]

            if chrom not in outfiles:
                # Open new gzip file for this chromosome
                out_path = os.path.join(DestinationDir, f"{chrom}.{fdst_fn}")
                f = gzip.open(out_path, 'wb')
                f.write(header.encode())
                outfiles[chrom] = f

            outfiles[chrom].write(line)

    # Close all open handles
    for f in outfiles.values():
        f.close()
