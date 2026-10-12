import gzip
import os
import glob
import sys

DestinationDir = sys.argv[1].rstrip('/')
SourceDir = sys.argv[2].rstrip('/')

FilesToSplit = glob.glob(SourceDir + '/*.txt.gz')

chunk1 = [f'chr{str(i)}' for i in range(1, 4)]
chunk2 = [f'chr{str(i)}' for i in range(4, 9)]
chunk3 = [f'chr{str(i)}' for i in range(9, 16)]
chunk4 = [f'chr{str(i)}' for i in range(16, 23)] + ['chrX']

os.makedirs(DestinationDir, exist_ok=True)
for file_n in FilesToSplit:
    fdst_fn = os.path.basename(file_n)
    with gzip.open(DestinationDir + "/chunk1." + fdst_fn,'wb') as fdst_chunk1:
        _ = fdst_chunk1.write("phenotype\tphenotype_id\tgwas_locus\tsnp\tbeta\tbeta_se\tp\n".encode())
        with gzip.open(DestinationDir + "/chunk2." + fdst_fn,'wb') as fdst_chunk2:
            _ = fdst_chunk2.write("phenotype\tphenotype_id\tgwas_locus\tsnp\tbeta\tbeta_se\tp\n".encode())
            with gzip.open(DestinationDir + "/chunk3." + fdst_fn,'wb') as fdst_chunk3:
                _ = fdst_chunk3.write("phenotype\tphenotype_id\tgwas_locus\tsnp\tbeta\tbeta_se\tp\n".encode())
                with gzip.open(DestinationDir + "/chunk4." + fdst_fn,'wb') as fdst_chunk4:
                    _ = fdst_chunk4.write("phenotype\tphenotype_id\tgwas_locus\tsnp\tbeta\tbeta_se\tp\n".encode())

                    with gzip.open(file_n, 'rb') as fh:
                        fh.readline()
                        for line in fh:
                            chrom = line.decode().split('\t')[2].split('_')[0]
                            if chrom in chunk1:
                                fdst_chunk1.write(line)
                            elif chrom in chunk2:
                                fdst_chunk2.write(line)
                            elif chrom in chunk3:
                                fdst_chunk3.write(line)
                            elif chrom in chunk4:
                                fdst_chunk4.write(line)
                    
                    
