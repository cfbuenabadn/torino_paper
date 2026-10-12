import numpy as np
import pandas as pd
from pybedtools import BedTool
import gzip
import gc 

print('starting')

print('loading Torino')

torino_exons_file = 'results/torino/Human/isoforms/isoforms.bed.gz'
torino_exons_df = pd.read_csv(torino_exons_file, sep='\t', names = ['chrom', 'start', 'end', 'gene_id', 
                                                 'transcript_id', 'strand', 'factors', 'exon_id', 'notes'])

torino_exons_df['transcript_id'] = torino_exons_df.gene_id + '.' + torino_exons_df.transcript_id
torino_exons_df['gene_name'] = torino_exons_df.gene_id


print('loading gencode')
gencode_exons_file = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz'
gencode_exons_df = pd.read_csv(gencode_exons_file, sep='\t', 
                            names = ['chrom', 'start', 'end', 'gene_id', 'gene_name',
                                                           'transcript_id', 'strand', 'exon_id', 'transcript_support_level',
                                                           'basic', 'Ensembl_canonical', 'MANE_Select', 'appris', 'transcript_type'])

gencode_exons_df = gencode_exons_df.loc[~gencode_exons_df.chrom.isin(['chrY', 'chrM'])]


torino_introns_df = pd.read_csv('results/torino/annotations/torino_introns.bed.gz', sep='\t')
gencode_introns_df = pd.read_csv('results/torino/annotations/gencode_introns.bed.gz', sep='\t')

torino_introns_bed = BedTool.from_dataframe(torino_introns_df.loc[torino_introns_df.end >= torino_introns_df.start]).sort()
gencode_introns_bed = BedTool.from_dataframe(gencode_introns_df.loc[gencode_introns_df.end >= gencode_introns_df.start]).sort()


annot_columns = ['chrom', 'start', 'end', 'gene_name', 'transcript_id', 'strand']

gencode_exons_df = gencode_exons_df.loc[(gencode_exons_df.end >= gencode_exons_df.start), annot_columns].drop_duplicates()
gencode_exons_bed = BedTool.from_dataframe(gencode_exons_df).sort()

torino_exons_df = torino_exons_df.loc[(torino_exons_df.end >= torino_exons_df.start), annot_columns].drop_duplicates()
torino_exons_bed = BedTool.from_dataframe(torino_exons_df).sort()

print('loading vastdb')

vastdb_exons_df = pd.read_csv('resources/vastdb/EVENT_INFO-hg38.tab.gz', sep='\t')
vastdb_exons_df = vastdb_exons_df.dropna()
vastdb_exons_cols = list(vastdb_exons_df.columns)
vastdb_exons_df['chrom'] = vastdb_exons_df.CO_A.apply(lambda x: x.split(':')[0])
vastdb_exons_df['start'] = vastdb_exons_df.CO_A.apply(lambda x: x.split(':')[1].split('-')[0])
vastdb_exons_df['end'] = vastdb_exons_df.CO_A.apply(lambda x: x.split('-')[1])
vastdb_exons_cols = ['chrom', 'start', 'end'] + vastdb_exons_cols
vastdb_exons_df = vastdb_exons_df[vastdb_exons_cols]

vastdb_exons_df = vastdb_exons_df.loc[(~vastdb_exons_df.COMPLEX.isin(['MIC-S', 'MIC_S', 'IR', 'Alt3', 'Alt5', 'MIC-M'])) & (vastdb_exons_df.EVENT.apply(lambda x: x.startswith('HsaEX')))]
vastdb_exons_df = vastdb_exons_df.loc[(~ vastdb_exons_df.COMPLEX.apply(lambda x: (x.startswith('ME') or (x == 'A_NA') or (x == 'ANN') or x == 'NA*')))]
vastdb_exons_df = vastdb_exons_df[[x for x in vastdb_exons_df.columns if x not in ['Seq_C1', 'Seq_A', 'Seq_C2']]]
vastdb_exons_df['strand'] = vastdb_exons_df.REF_CO.apply(lambda x: x.split(':')[-1])

intron_start_list = []
intron_end_list = []
for idx, row in vastdb_exons_df.iterrows():
    strand = row.strand
    if strand == '+':
        intron_start = int(row.CO_C1.split('-')[-1])
        intron_end = int(row.CO_C2.split('-')[0].split(':')[1])
    else:
        intron_start = int(row.CO_C2.split('-')[-1])
        intron_end = int(row.CO_C1.split('-')[0].split(':')[1])
        
    intron_start_list.append(intron_start)
    intron_end_list.append(intron_end)

vastdb_introns_df = pd.DataFrame()
vastdb_introns_df['chrom'] = list(vastdb_exons_df.chrom)
vastdb_introns_df['start'] = intron_start_list
vastdb_introns_df['end'] = intron_end_list
vastdb_introns_df['GENE'] = list(vastdb_exons_df.GENE)
vastdb_introns_df['EVENT'] = list(vastdb_exons_df.EVENT)
vastdb_introns_df['strand'] = list(vastdb_exons_df.strand)

vastdb_introns_bed = BedTool.from_dataframe(vastdb_introns_df).sort()
vastdb_exons_bed = BedTool.from_dataframe(vastdb_exons_df).sort()


print('find overlaps')


# introns_columns = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand']
introns_columns = ['chrom', 'start', 'end', 'gene_name', 'transcript_id', 'strand']


gencode_vastdb_exon_overlaps = gencode_exons_bed.intersect(vastdb_exons_bed, f=1, F=1, wo=True).to_dataframe(
    names = annot_columns + ['vastdb_' + x for x in list(vastdb_exons_df.columns)] + ['overlap']
)

gencode_vastdb_intron_overlaps = gencode_introns_bed.intersect(vastdb_introns_bed, f=1, F=1, wo=True).to_dataframe(
    names = introns_columns + ['vastdb_' + x for x in list(vastdb_introns_df.columns)] + ['overlap']
)

gencode_vastdb_exons_captured = pd.Index(gencode_vastdb_exon_overlaps.vastdb_EVENT.unique())
gencode_vastdb_introns_captured = pd.Index(gencode_vastdb_intron_overlaps.vastdb_EVENT.unique())

torino_vastdb_exon_overlaps = torino_exons_bed.intersect(vastdb_exons_bed, f=1, F=1, wo=True).to_dataframe(
    names = annot_columns + ['vastdb_' + x for x in list(vastdb_exons_df.columns)] + ['overlap']
)

torino_vastdb_intron_overlaps = torino_introns_bed.intersect(vastdb_introns_bed, f=1, F=1, wo=True).to_dataframe(
    names = introns_columns + ['vastdb_' + x for x in list(vastdb_introns_df.columns)] + ['overlap']
)

torino_vastdb_exons_captured = pd.Index(torino_vastdb_exon_overlaps.vastdb_EVENT.unique())
#del torino_vastdb_exons_captured

torino_vastdb_introns_captured = pd.Index(torino_vastdb_intron_overlaps.vastdb_EVENT.unique())


# def get_internal_exons(exons_df):
#     internal_exon_chrom_list = []
#     internal_exon_start_list = []
#     internal_exon_end_list = []
#     internal_exon_gene_list = []
#     internal_exon_transcript_list = []
#     internal_exon_strand_list = []
#     internal_exon_intron_start_list = []
#     internal_exon_intron_end_list = []
#     internal_exon_flanking1_start_list = []
#     internal_exon_flanking2_end_list = []
    
#     for transcript, df in tqdm(exons_df.groupby('transcript_id'), leave=True, position=0):
#         iso_len = df.shape[0]
#         if iso_len >= 3:
#             internal_exon_chrom = df.chrom.iloc[0]
#             internal_exon_gene = df.gene_id.iloc[0]
#             internal_exon_transcript = transcript
#             internal_exon_strand = df.strand.iloc[0]
#             for i in range(1, iso_len-1):
                
#                 internal_exon_start = df.iloc[i].start
#                 internal_exon_end = df.iloc[i].end
        
        
#                 internal_exon_intron_start = df.iloc[i-1].end
#                 internal_exon_intron_end = df.iloc[i+1].start

#                 internal_exon_flanking1_start = df.iloc[i-1].start
#                 internal_exon_flanking2_end = df.iloc[i+1].end
            
    
#                 internal_exon_chrom_list.append(internal_exon_chrom)
#                 internal_exon_start_list.append(internal_exon_start)
#                 internal_exon_end_list.append(internal_exon_end)
#                 internal_exon_gene_list.append(internal_exon_gene)
#                 internal_exon_transcript_list.append(internal_exon_transcript)
#                 internal_exon_strand_list.append(internal_exon_strand)
#                 internal_exon_intron_start_list.append(internal_exon_intron_start)
#                 internal_exon_intron_end_list.append(internal_exon_intron_end)

#                 internal_exon_flanking1_start_list.append(internal_exon_flanking1_start)
#                 internal_exon_flanking2_end_list.append(internal_exon_flanking2_end)
    
#     internal_exon_df = pd.DataFrame()
#     internal_exon_df['chrom'] = internal_exon_chrom_list
#     internal_exon_df['start'] = internal_exon_start_list
#     internal_exon_df['end'] = internal_exon_end_list
#     internal_exon_df['gene'] = internal_exon_gene_list
#     internal_exon_df['transcript'] = internal_exon_transcript_list
#     internal_exon_df['strand'] = internal_exon_strand_list
#     internal_exon_df['intron_start'] = internal_exon_intron_start_list
#     internal_exon_df['intron_end'] = internal_exon_intron_end_list

#     internal_exon_df['flanking1_start'] = internal_exon_flanking1_start_list
#     internal_exon_df['flanking2_end'] = internal_exon_flanking2_end_list


#     exon_coordinates = internal_exon_df.chrom + ':' + internal_exon_df.start.astype(str) + '-' + internal_exon_df.end.astype(str)
#     internal_exon_df['exon_coordinates'] = exon_coordinates
#     exon_name = internal_exon_df.exon_coordinates + ':' + internal_exon_df.gene
#     internal_exon_df['exon_id'] = exon_name

#     return internal_exon_df

# torino_internal_exons_df = get_internal_exons(torino_exons_df)
# gencode_internal_exons_df = get_internal_exons(gencode_exons_df)

# torino_internal_exons_df = torino_internal_exons_df[[x for x in torino_internal_exons_df.columns if x != 'transcript']].drop_duplicates().reset_index(drop=True)
# gencode_internal_exons_df = gencode_internal_exons_df[[x for x in gencode_internal_exons_df.columns if x != 'transcript']].drop_duplicates().reset_index(drop=True)

# torino_internal_exons_df.to_csv('torino_internal_exons.bed.gz', sep='\t', header=True, index=False)
# gencode_internal_exons_df.to_csv('gencode_internal_exons.bed.gz', sep='\t', header=True, index=False)

print('that was a lot of memory')

torino_internal_exons_df = pd.read_csv('results/torino/annotations/torino_internal_exons_w_transcript.bed.gz', sep='\t')
gencode_internal_exons_df = pd.read_csv('results/torino/annotations/gencode_internal_exons_w_transcript.bed.gz', sep='\t')

torino_internal_exons_df['transcripts'] = torino_internal_exons_df.groupby(['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end']).transcripts.transform(lambda x: '|'.join(sorted(x)))
gencode_internal_exons_df['transcripts'] = gencode_internal_exons_df.groupby(['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end']).transcripts.transform(lambda x: '|'.join(sorted(x)))

torino_internal_exons_df_flanking1 = pd.DataFrame(torino_internal_exons_df.groupby(['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end', 'transcripts']).flanking1_start.max().astype(int))
torino_internal_exons_df_flanking2 = pd.DataFrame(torino_internal_exons_df.groupby(['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end', 'transcripts']).flanking2_end.min().astype(int))
torino_internal_exons_df = torino_internal_exons_df_flanking1.merge(torino_internal_exons_df_flanking2, left_index=True, right_index=True).reset_index()

del torino_internal_exons_df_flanking1
del torino_internal_exons_df_flanking2

gencode_internal_exons_df_flanking1 = pd.DataFrame(gencode_internal_exons_df.groupby(['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end', 'transcripts']).flanking1_start.max().astype(int))
gencode_internal_exons_df_flanking2 = pd.DataFrame(gencode_internal_exons_df.groupby(['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end', 'transcripts']).flanking2_end.min().astype(int))
gencode_internal_exons_df = gencode_internal_exons_df_flanking1.merge(gencode_internal_exons_df_flanking2, left_index=True, right_index=True).reset_index()

del gencode_internal_exons_df_flanking1
del gencode_internal_exons_df_flanking2

torino_internal_exons_bed = BedTool.from_dataframe(torino_internal_exons_df).sort()
torino_col_names = list(torino_internal_exons_df.columns)
del torino_internal_exons_df

gencode_internal_exons_bed = BedTool.from_dataframe(gencode_internal_exons_df).sort()
gencode_col_names = list(gencode_internal_exons_df.columns)
del gencode_internal_exons_df

del torino_exons_df
del gencode_exons_df

del torino_introns_df
del gencode_introns_df

intron_cols = [f'{x}_intron' for x in ['chrom', 'start', 'end', 'gene_name', 'transcript_id', 'strand']]
torino_col_names += intron_cols #[f'{x}_intron' for x in torino_introns_bed.to_dataframe().columns]
torino_col_names += ['overlaps']

gencode_col_names += intron_cols
gencode_col_names += ['overlaps']

torino_cassette_exons = torino_internal_exons_bed.intersect(torino_introns_bed, f=1, wo=True).to_dataframe(names = torino_col_names)

# torino_cassette_exons = torino_internal_exons_intron_intersect.loc[torino_internal_exons_intron_intersect.overlaps > 0]
torino_cassette_exons = torino_cassette_exons.loc[
(torino_cassette_exons.intron_start == torino_cassette_exons.start_intron) & (torino_cassette_exons.intron_end == torino_cassette_exons.end_intron)
][['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end', 'flanking1_start', 'flanking2_end', 'transcripts']].drop_duplicates()

gencode_cassette_exons = gencode_internal_exons_bed.intersect(gencode_introns_bed, f=1, wo=True).to_dataframe(
    names = gencode_col_names)

# gencode_cassette_exons = gencode_internal_exons_intron_intersect.loc[gencode_internal_exons_intron_intersect.overlaps > 0]
gencode_cassette_exons = gencode_cassette_exons.loc[
(gencode_cassette_exons.intron_start == gencode_cassette_exons.start_intron) & (gencode_cassette_exons.intron_end == gencode_cassette_exons.end_intron)
][['chrom', 'start', 'end', 'gene', 'exon_coordinates', 'strand', 'intron_start', 'intron_end', 'flanking1_start', 'flanking2_end', 'transcripts']].drop_duplicates()

gencode_exon_name = []
current_gene = ''
for idx, row in gencode_cassette_exons.iterrows():
    gene_ = row.gene
    if gene_ != current_gene:
        gene_counter = 1
        current_gene = gene_
    exon_name = current_gene + ':' + str(gene_counter)
    gencode_exon_name.append(exon_name)
    gene_counter += 1

gencode_cassette_exons['exon_name'] = gencode_exon_name

gencode_cassette_exons_bed = BedTool.from_dataframe(gencode_cassette_exons)


gencode_cassette_exons_columns = gencode_cassette_exons.columns
torino_cassette_exons_columns = torino_cassette_exons.columns

torino_gencode_exons_captured = torino_internal_exons_bed.intersect(gencode_cassette_exons_bed, f=1, F=1, wo=True).to_dataframe(
    names = torino_col_names[:11] + [f'{x}_gencode' for x in gencode_cassette_exons_columns] + ['overlaps']
)

print('midway')

torino_gencode_exons_captured = torino_gencode_exons_captured.loc[(torino_gencode_exons_captured.intron_start.astype(int) == torino_gencode_exons_captured.intron_start_gencode.astype(int)) & (torino_gencode_exons_captured.intron_end.astype(int) == torino_gencode_exons_captured.intron_end_gencode.astype(int))]

torino_gencode_cassette = pd.Index(torino_cassette_exons.exon_coordinates).intersection(pd.Index(torino_gencode_exons_captured.exon_coordinates))

gencode_cassette_introns_bed = BedTool.from_dataframe(gencode_cassette_exons[['chrom', 'intron_start', 'intron_end', 'exon_name', 'start', 
                                                                              'end', 'flanking1_start', 'flanking2_end', 'transcripts']])

del gencode_cassette_exons

torino_gencode_introns_captured = torino_introns_bed.intersect(gencode_cassette_introns_bed, f=1, F=1, wo=True).to_dataframe(
    names = ['chrom', 'start', 'end', 'gene', 'transcript_id', 'strand', 'chrom_gencode', 'intron_start_gencode', 
             'intron_end_gencode', 'exon_name_gencode', 'start_gencode', 'end_gencode', 'flanking1_start_gencode', 'flanking2_end_gencode', 'transcripts', 'overlaps']
)

del torino_introns_bed

torino_gencode_introns_captured = torino_gencode_introns_captured[['chrom', 'start', 'end', 'gene', 'strand', 'chrom_gencode', 'intron_start_gencode', 
             'intron_end_gencode', 'exon_name_gencode', 'start_gencode', 'end_gencode', 'flanking1_start_gencode', 'flanking2_end_gencode', 'transcripts', 'overlaps']].drop_duplicates()

# Captured only Gencode exon
torino_gencode_exons_captured.loc[~torino_gencode_exons_captured.exon_coordinates.isin(torino_gencode_cassette)]

# Captured cassette exon
torino_gencode_exons_captured.loc[torino_gencode_exons_captured.exon_coordinates.isin(torino_gencode_cassette)]

# Captured intron only
torino_gencode_introns_captured.loc[~torino_gencode_introns_captured.exon_name_gencode.isin(torino_gencode_exons_captured.exon_name_gencode)]

print('gencode')

with open('results/CoveragePlots/bed_files/cassette_exons/gencode/exon_only.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/gencode/exon_only.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/gencode/exon_only.flanking2.bed', 'w') as fh2:
            for idx, row in torino_gencode_exons_captured.loc[~torino_gencode_exons_captured.exon_coordinates.isin(torino_gencode_cassette)].iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.exon_name_gencode, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start_gencode), str(row.intron_start_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end_gencode), str(row.flanking2_end_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)
    
with open('results/CoveragePlots/bed_files/cassette_exons/gencode/cassette_exon.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/gencode/cassette_exon.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/gencode/cassette_exon.flanking2.bed', 'w') as fh2:
            for idx, row in torino_gencode_exons_captured.loc[torino_gencode_exons_captured.exon_coordinates.isin(torino_gencode_cassette)].iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.exon_name_gencode, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start_gencode), str(row.intron_start_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end_gencode), str(row.flanking2_end_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)


gtex_gaps_df = pd.read_csv('results/CoveragePlots/other_files/GTEx_overlapping_regions.bed.gz', sep='\t')
gtex_gaps_bed = BedTool.from_dataframe(gtex_gaps_df)

del gtex_gaps_df

introns_captured_cols = torino_gencode_introns_captured.columns


introns_only_bed = BedTool.from_dataframe(torino_gencode_introns_captured.loc[~torino_gencode_introns_captured.exon_name_gencode.isin(torino_gencode_exons_captured.exon_name_gencode)])
introns_only_filtered = introns_only_bed.intersect(gtex_gaps_bed, v=True).intersect(torino_exons_bed, v=True).to_dataframe(names = introns_captured_cols)
# introns_only_filtered.iterrows()
del torino_gencode_introns_captured

with open('results/CoveragePlots/bed_files/cassette_exons/gencode/intron_only.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/gencode/intron_only.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/gencode/intron_only.flanking2.bed', 'w') as fh2:
            for idx, row in introns_only_filtered.iterrows(): #torino_gencode_introns_captured.loc[~torino_gencode_introns_captured.exon_name_gencode.isin(torino_gencode_exons_captured.exon_name_gencode)].iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start_gencode), str(row.end_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start_gencode), str(row.intron_start_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end_gencode), str(row.flanking2_end_gencode), row.exon_name_gencode, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)

print('vastdb')


with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/exon_only.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/exon_only.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/exon_only.flanking2.bed', 'w') as fh2:
            for idx, row in vastdb_exons_df.loc[vastdb_exons_df.EVENT.isin(torino_vastdb_exons_captured.difference(torino_vastdb_introns_captured))].iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.EVENT]) + '\n'
                if strand == '+':
                    flanking1_start = row.CO_C1.split(':')[1].split('-')[0]
                    flanking1_end = row.CO_C1.split(':')[1].split('-')[1]
                    flanking2_start = row.CO_C2.split(':')[1].split('-')[0]
                    flanking2_end = row.CO_C2.split(':')[1].split('-')[1]
                if strand == '-':
                    flanking1_start = row.CO_C2.split(':')[1].split('-')[0]
                    flanking1_end = row.CO_C2.split(':')[1].split('-')[1]
                    flanking2_start = row.CO_C1.split(':')[1].split('-')[0]
                    flanking2_end = row.CO_C1.split(':')[1].split('-')[1]
                    
                flanking1_line = '\t'.join([row.chrom, flanking1_start, flanking1_end, row.EVENT]) + '\n'
                flanking2_line = '\t'.join([row.chrom, flanking2_start, flanking2_end, row.EVENT]) + '\n'
                
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)


print('vastdb1')
with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/cassette_exon.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/cassette_exon.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/cassette_exon.flanking2.bed', 'w') as fh2:
            for idx, row in vastdb_exons_df.loc[vastdb_exons_df.EVENT.isin(torino_vastdb_exons_captured.intersection(torino_vastdb_introns_captured))].iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.EVENT]) + '\n'
                if strand == '+':
                    flanking1_start = row.CO_C1.split(':')[1].split('-')[0]
                    flanking1_end = row.CO_C1.split(':')[1].split('-')[1]
                    flanking2_start = row.CO_C2.split(':')[1].split('-')[0]
                    flanking2_end = row.CO_C2.split(':')[1].split('-')[1]
                if strand == '-':
                    flanking1_start = row.CO_C2.split(':')[1].split('-')[0]
                    flanking1_end = row.CO_C2.split(':')[1].split('-')[1]
                    flanking2_start = row.CO_C1.split(':')[1].split('-')[0]
                    flanking2_end = row.CO_C1.split(':')[1].split('-')[1]
                    
                flanking1_line = '\t'.join([row.chrom, flanking1_start, flanking1_end, row.EVENT]) + '\n'
                flanking2_line = '\t'.join([row.chrom, flanking2_start, flanking2_end, row.EVENT]) + '\n'

                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)


introns_captured_cols = vastdb_exons_df.columns
introns_only_bed = BedTool.from_dataframe(vastdb_exons_df.loc[vastdb_exons_df.EVENT.isin(torino_vastdb_introns_captured.difference(torino_vastdb_exons_captured))])
introns_only_filtered = introns_only_bed.intersect(gtex_gaps_bed, v=True).intersect(torino_exons_bed, v=True).to_dataframe(names = introns_captured_cols)
# introns_only_filtered.iterrows()
print('vastdb2')
with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/intron_only.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/intron_only.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/vastdb/intron_only.flanking2.bed', 'w') as fh2:
            for idx, row in introns_only_filtered.iterrows(): #vastdb_exons_df.loc[vastdb_exons_df.EVENT.isin(torino_vastdb_introns_captured.difference(torino_vastdb_exons_captured))].iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.EVENT]) + '\n'
                if strand == '+':
                    flanking1_start = row.CO_C1.split(':')[1].split('-')[0]
                    flanking1_end = row.CO_C1.split(':')[1].split('-')[1]
                    flanking2_start = row.CO_C2.split(':')[1].split('-')[0]
                    flanking2_end = row.CO_C2.split(':')[1].split('-')[1]
                if strand == '-':
                    flanking1_start = row.CO_C2.split(':')[1].split('-')[0]
                    flanking1_end = row.CO_C2.split(':')[1].split('-')[1]
                    flanking2_start = row.CO_C1.split(':')[1].split('-')[0]
                    flanking2_end = row.CO_C1.split(':')[1].split('-')[1]
                    
                flanking1_line = '\t'.join([row.chrom, flanking1_start, flanking1_end, row.EVENT]) + '\n'
                flanking2_line = '\t'.join([row.chrom, flanking2_start, flanking2_end, row.EVENT]) + '\n'

                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)


print('finished that part')

del introns_only_filtered
del torino_vastdb_exons_captured
del torino_vastdb_introns_captured
#del torino_gencode_exons_captured

exon_name_list = []
current_gene = ''
for idx, row in torino_cassette_exons.iterrows():
    gene_ = row.gene
    if gene_ != current_gene:
        gene_counter = 1
        current_gene = gene_
    gene_name = current_gene + ':' + str(gene_counter)
    exon_name_list.append(gene_name)



torino_cassette_exons['exon_name'] = exon_name_list

torino_cassette_exons.to_csv('results/torino/Human/isoforms/cassette_exons.bed.gz', sep='\t', header=True, index=False)

torino_gencode_cassette_exons = torino_cassette_exons.loc[torino_cassette_exons.exon_coordinates.isin(torino_gencode_exons_captured.exon_coordinates)]


torino_cassette_exons_bed = BedTool.from_dataframe(torino_cassette_exons)

torino_vastdb_cassette_exons = torino_cassette_exons_bed.intersect(vastdb_exons_bed, f=1, F=1, wo=True).to_dataframe(
    names = list(torino_cassette_exons.columns) + [f'{x}_vastdb' for x in vastdb_exons_df.columns] + ['overlaps']
)

del vastdb_exons_df
del torino_gencode_exons_captured

# intron_start_vastdb = []
# intron_end_vastdb = []
# for idx, row in torino_vastdb_cassette_exons.iterrows():
#     if row.strand_vastdb == '+':
#         intron_start = row.CO_C1_vastdb
#         intron_end = row.CO_C2_vastdb
#     else:
#         intron_start = row.CO_C2_vastdb
#         intron_end = row.CO_C1_vastdb
#     intron_start = int(intron_start.split('-')[-1])
#     intron_end = int(intron_end.split(':')[-1].split('-')[0])

#     intron_start_vastdb.append(intron_start)
#     intron_end_vastdb.append(intron_end)

# torino_vastdb_cassette_exons['intron_start_vastdb'] = intron_start_vastdb
# torino_vastdb_cassette_exons['intron_end_vastdb'] = intron_end_vastdb

# torino_vastdb_cassette_exons = torino_vastdb_cassette_exons.loc[
# (torino_vastdb_cassette_exons.intron_start == torino_vastdb_cassette_exons.intron_start_vastdb) & (torino_vastdb_cassette_exons.intron_end == torino_vastdb_cassette_exons.intron_end_vastdb)
# ]

def confirm_junctions(cassette_exons, junctions_bed):
    
    I1 = cassette_exons[['chrom', 'intron_start', 'start', 'gene', 'exon_coordinates']].copy()
    I1.columns = ['chrom', 'start', 'end', 'gene', 'exon_coordinates']
    I1.start += 1
    I1.end -= 1
    
    I2 = cassette_exons[['chrom', 'end', 'intron_end', 'gene', 'exon_coordinates']].copy()
    I2.columns = ['chrom', 'start', 'end', 'gene', 'exon_coordinates']
    I2.start += 1
    I2.end -= 1
    
    SE = cassette_exons[['chrom', 'intron_start', 'intron_end', 'gene', 'exon_coordinates']].copy()
    SE.columns = ['chrom', 'start', 'end', 'gene', 'exon_coordinates']
    SE.start += 1
    SE.end -= 1
    
    I1_ = BedTool.from_dataframe(I1).intersect(junctions_bed, f=1, F=1, wao=True).to_dataframe(names = list(I1.columns) + [f'{x}_junc' for x in junctions.columns] + ['overlaps'])
    I2_ = BedTool.from_dataframe(I2).intersect(junctions_bed, f=1, F=1, wao=True).to_dataframe(names = list(I2.columns) + [f'{x}_junc' for x in junctions.columns] + ['overlaps'])
    SE_ = BedTool.from_dataframe(SE).intersect(junctions_bed, f=1, F=1, wao=True).to_dataframe(names = list(SE.columns) + [f'{x}_junc' for x in junctions.columns] + ['overlaps'])
    
    I1_confirmed = pd.Index(I1_.loc[I1_.overlaps > 0].exon_coordinates.unique())
    I2_confirmed = pd.Index(I2_.loc[I2_.overlaps > 0].exon_coordinates.unique())
    SE_confirmed = pd.Index(SE_.loc[SE_.overlaps > 0].exon_coordinates.unique())

    return I1_confirmed, I2_confirmed, SE_confirmed


import pandas as pd
from pybedtools import BedTool
import gc

def confirm_junctions_optimized(cassette_exons, junctions_bed, chunk_size=10000):
    """
    Memory-optimized version that processes data in chunks and avoids unnecessary copies.
    
    Args:
        cassette_exons: DataFrame with exon information
        junctions_bed: BedTool object with junction information  
        chunk_size: Number of rows to process at once (adjust based on available memory)
    
    Returns:
        Tuple of confirmed junction indices (I1_confirmed, I2_confirmed, SE_confirmed)
    """
    
    # Get total number of rows for chunking
    total_rows = len(cassette_exons)
    
    # Initialize sets to collect confirmed coordinates (more memory efficient than Index)
    I1_confirmed = set()
    I2_confirmed = set()
    SE_confirmed = set()
    
    # Process data in chunks
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = cassette_exons.iloc[start_idx:end_idx]
        
        # Process I1 (first intron) for this chunk
        I1_chunk = create_interval_dataframe(
            chunk, 
            ['chrom', 'intron_start', 'start', 'gene', 'exon_coordinates'],
            start_offset=1, end_offset=-1
        )
        
        confirmed_coords = process_bedtool_intersection(I1_chunk, junctions_bed)
        I1_confirmed.update(confirmed_coords)
        
        # Process I2 (second intron) for this chunk
        I2_chunk = create_interval_dataframe(
            chunk,
            ['chrom', 'end', 'intron_end', 'gene', 'exon_coordinates'], 
            start_offset=1, end_offset=-1
        )
        
        confirmed_coords = process_bedtool_intersection(I2_chunk, junctions_bed)
        I2_confirmed.update(confirmed_coords)
        
        # Process SE (skipped exon) for this chunk
        SE_chunk = create_interval_dataframe(
            chunk,
            ['chrom', 'intron_start', 'intron_end', 'gene', 'exon_coordinates'],
            start_offset=1, end_offset=-1
        )
        
        confirmed_coords = process_bedtool_intersection(SE_chunk, junctions_bed)
        SE_confirmed.update(confirmed_coords)
        
        # Force garbage collection after each chunk
        gc.collect()
        
        # Optional: print progress
        if (start_idx // chunk_size + 1) % 10 == 0:
            print(f"Processed {end_idx}/{total_rows} rows")
    
    # Convert sets back to pandas Index for compatibility
    return (pd.Index(list(I1_confirmed)), 
            pd.Index(list(I2_confirmed)), 
            pd.Index(list(SE_confirmed)))

def create_interval_dataframe(df, columns, start_offset=0, end_offset=0):
    """
    Create interval DataFrame without making unnecessary copies.
    """
    # Create new DataFrame with only needed columns
    result = df[columns].copy()
    result.columns = ['chrom', 'start', 'end', 'gene', 'exon_coordinates']
    
    # Apply offsets in-place
    if start_offset != 0:
        result['start'] += start_offset
    if end_offset != 0:  
        result['end'] += end_offset
        
    return result

def process_bedtool_intersection(interval_df, junctions_bed):
    """
    Process BedTool intersection and return confirmed coordinates.
    """
    try:
        # Create BedTool from DataFrame
        bed_obj = BedTool.from_dataframe(interval_df)
        
        # Perform intersection
        intersected = bed_obj.intersect(junctions_bed, f=1, F=1, wao=True)
        
        # Convert to DataFrame with minimal column set
        result_df = intersected.to_dataframe(usecols=[4, -1])  # Only exon_coordinates and overlap columns
        result_df.columns = ['exon_coordinates', 'overlaps']
        
        # Get confirmed coordinates
        confirmed = result_df.loc[result_df['overlaps'] > 0, 'exon_coordinates'].unique()
        
        # Clean up BedTool objects
        bed_obj.delete_temporary_files()
        intersected.delete_temporary_files()
        
        return set(confirmed)
        
    except Exception as e:
        print(f"Error in intersection: {e}")
        return set()

# Alternative: Even more memory-efficient version using iterative processing
def confirm_junctions_minimal_memory(cassette_exons, junctions_bed, batch_size=1000):
    """
    Ultra-low memory version that processes one row at a time or in very small batches.
    Use this if the chunked version still uses too much memory.
    """
    I1_confirmed = set()
    I2_confirmed = set() 
    SE_confirmed = set()
    
    for idx, row in cassette_exons.iterrows():
        # Process I1
        I1_interval = create_single_interval(row, 'intron_start', 'start', offset_start=1, offset_end=-1)
        if check_junction_overlap(I1_interval, junctions_bed):
            I1_confirmed.add(row['exon_coordinates'])
            
        # Process I2  
        I2_interval = create_single_interval(row, 'end', 'intron_end', offset_start=1, offset_end=-1)
        if check_junction_overlap(I2_interval, junctions_bed):
            I2_confirmed.add(row['exon_coordinates'])
            
        # Process SE
        SE_interval = create_single_interval(row, 'intron_start', 'intron_end', offset_start=1, offset_end=-1)  
        if check_junction_overlap(SE_interval, junctions_bed):
            SE_confirmed.add(row['exon_coordinates'])
            
        # Periodic cleanup
        if idx % batch_size == 0:
            gc.collect()
            
    return (pd.Index(list(I1_confirmed)),
            pd.Index(list(I2_confirmed)), 
            pd.Index(list(SE_confirmed)))

def create_single_interval(row, start_col, end_col, offset_start=0, offset_end=0):
    """Create a single interval string for bedtools."""
    start = row[start_col] + offset_start
    end = row[end_col] + offset_end
    return f"{row['chrom']}\t{start}\t{end}\t{row['gene']}\t{row['exon_coordinates']}"

def check_junction_overlap(interval_string, junctions_bed):
    """Check if a single interval overlaps with junctions."""
    try:
        # Create temporary BedTool from string
        temp_bed = BedTool(interval_string, from_string=True)
        intersect_result = temp_bed.intersect(junctions_bed, f=1, F=1, wao=True)
        
        # Check if there's any overlap > 0
        for line in intersect_result:
            if int(line.fields[-1]) > 0:  # overlap column
                temp_bed.delete_temporary_files()
                intersect_result.delete_temporary_files() 
                return True
                
        temp_bed.delete_temporary_files()
        intersect_result.delete_temporary_files()
        return False
        
    except Exception:
        return False

# Additional utility function for monitoring memory usage
def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Usage example:
"""
# Try the chunked version first
I1_conf, I2_conf, SE_conf = confirm_junctions_optimized(cassette_exons, junctions_bed, chunk_size=5000)

# If still too much memory, try the minimal memory version  
I1_conf, I2_conf, SE_conf = confirm_junctions_minimal_memory(cassette_exons, junctions_bed, batch_size=500)
"""

print('loading juncs')
junctions = pd.read_csv('resources/annotation/Human/bed/gtex_junctions.tab.gz', sep='\t')
# junctions['gene'] = junctions.gene.apply(lambda x: x.split('.')[0])
junctions_bed = BedTool.from_dataframe(junctions)

print('hola de la verdad')
torino_I1, torino_I2, torino_SE = confirm_junctions(torino_cassette_exons, junctions_bed) # confirm_junctions(torino_cassette_exons, junctions_bed)
print('victory!')
torino_junctions_cassette_exons = torino_cassette_exons.loc[torino_cassette_exons.exon_coordinates.isin(torino_I1.intersection(torino_I2))]

torino_unconfirmed_cassette_exons = torino_cassette_exons.loc[~torino_cassette_exons.exon_coordinates.isin(
    pd.Index(torino_gencode_cassette_exons.exon_coordinates).union(pd.Index(torino_vastdb_cassette_exons.exon_coordinates)).union(
        pd.Index(torino_junctions_cassette_exons.exon_coordinates))
)]


print('torino')


with open('results/CoveragePlots/bed_files/cassette_exons/torino/gencode.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/torino/gencode.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/torino/gencode.flanking2.bed', 'w') as fh2:
            for idx, row in torino_gencode_cassette_exons.iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.exon_name, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start), str(row.intron_start), row.exon_name, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end), str(row.flanking2_end), row.exon_name, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)
    
with open('results/CoveragePlots/bed_files/cassette_exons/torino/vastdb.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/torino/vastdb.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/torino/vastdb.flanking2.bed', 'w') as fh2:
            for idx, row in torino_vastdb_cassette_exons.iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.exon_name, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start), str(row.intron_start), row.exon_name, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end), str(row.flanking2_end), row.exon_name, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)

with open('results/CoveragePlots/bed_files/cassette_exons/torino/junctions.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/torino/junctions.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/torino/junctions.flanking2.bed', 'w') as fh2:
            for idx, row in torino_junctions_cassette_exons.iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.exon_name, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start), str(row.intron_start), row.exon_name, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end), str(row.flanking2_end), row.exon_name, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)

with open('results/CoveragePlots/bed_files/cassette_exons/torino/unmatched.exon.bed', 'w') as fh:
    with open('results/CoveragePlots/bed_files/cassette_exons/torino/unmatched.flanking1.bed', 'w') as fh1:
        with open('results/CoveragePlots/bed_files/cassette_exons/torino/unmatched.flanking2.bed', 'w') as fh2:
            for idx, row in torino_unconfirmed_cassette_exons.iterrows():
                exon_line = '\t'.join([row.chrom, str(row.start), str(row.end), row.exon_name, row.transcripts]) + '\n'
                flanking1_line = '\t'.join([row.chrom, str(row.flanking1_start), str(row.intron_start), row.exon_name, row.transcripts]) + '\n'
                flanking2_line = '\t'.join([row.chrom, str(row.intron_end), str(row.flanking2_end), row.exon_name, row.transcripts]) + '\n'
                fh.write(exon_line)
                fh1.write(flanking1_line)
                fh2.write(flanking2_line)
