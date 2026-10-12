import gtfparse
import polars as pl
import pandas as pd
import gzip

def get_tags(x):
    row_tags = []
    tag_list = x.split(',')
    if 'basic' in tag_list:
        row_tags.append('basic')
    else:
        row_tags.append('')
    if 'Ensembl_canonical' in tag_list:
        row_tags.append('Ensembl_canonical')
    else:
        row_tags.append('')
    if 'MANE_Select' in tag_list:
        row_tags.append('MANE_Select')
    else:
        row_tags.append('')
    if ',appris_' in x:
        appris_tags = [y for y in tag_list if y[:7] == 'appris_']
        row_tags.extend(appris_tags)
    else:
        row_tags.append('')
    return row_tags


def process_row(row):
    chrom = str(row.seqname)
    start = str(int(float(row.start)))
    end = str(int(float(row.end)))
    gene_id = str(row.gene_id).split('.')[0]
    gene_name = str(row.gene_name)
    transcript_id = str(row.transcript_id).split('.')[0]
    strand = str(row.strand)

    transcript_type = str(row.transcript_type)

    exon_id = str(row.exon_id).split('.')[0]
    transcript_support = str(row.transcript_support_level)
    if transcript_support in ['NA', '']:
        transcript_support = '0'
    else:
        transcript_support = str(int(float(transcript_support)))
    
    tags = row.tag
    row_tags = get_tags(tags)
    row_tags = '\t'.join(row_tags)

    bed_row = [chrom, start, end, gene_id, gene_name, transcript_id, strand, exon_id, transcript_support, row_tags, transcript_type]
    bed_line = '\t'.join(bed_row) + '\n'#.encode()
    return bed_line

if __name__ == '__main__':

    gtf_file = 'resources/annotation/Mouse/gtf/gencode.vM36.annotation.gtf.gz'
    gtf_df = gtfparse.read_gtf(gtf_file)
    
    cols = gtf_df.columns
    gtf_df = pd.DataFrame(gtf_df.filter(pl.col('feature') == 'exon'))
    gtf_df.columns = cols
    
    gtf_exons = gtf_df.loc[(gtf_df.feature == 'exon') & (gtf_df.gene_type == 'protein_coding')]
    
    with open('resources/annotation/Mouse/bed/gencode.exons_annotation.bed', 'w') as fh:
        header = ['#chrom', 'start', 'end', 'gene_id', 'gene_name', 'transcript_id', 'strand', 'exon_id', 
                  'transcript_support_level', 'basic', 'Ensembl_canonical', 'MANE_Select', 'appris', 'transcript_type']
        header = ('\t'.join(header) + '\n')
        fh.write(header)
        for idx, row in gtf_exons.iterrows():
            print(idx)
            exon_line = process_row(row)
            if exon_line[:3] == 'chr':
                exon_line = exon_line
                fh.write(exon_line)
            
