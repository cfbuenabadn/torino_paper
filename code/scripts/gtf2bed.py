import gzip
import pandas as pd
import sys

if __name__ == '__main__':
    arguments = sys.argv
    input_gtf = arguments[1]
    output_bed = arguments[2]
    with open(input_gtf, 'r') as fh:
        with gzip.open(output_bed, 'wb') as fh2:
            header = ['#chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon']
            header = ('\t'.join(header) + '\n').encode()
            fh2.write(header)
            for line in fh:
                row = line.rstrip().split('\t')
                if row[2] == "exon":
                    chrom = row[0]
                    start = row[3]
                    end = row[4]
                    strand = row[6]
                    tags = row[8].split('"')
                    gene_id = tags[1]
                    transcript_id = tags[3]
                    factors = tags[5]
                    exon = tags[7]
        
                    out_row = [chrom, start, end, gene_id, transcript_id, strand, factors, exon]
                    out_line = ('\t'.join(out_row) + '\n').encode()
                    fh2.write(out_line)
                
# zcat snmf_exons.bed.gz | bedtools sort -i - | bgzip -c > snmf.exons.sorted.bed.gz