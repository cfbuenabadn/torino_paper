import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pybedtools
import gzip


def extract_retained_introns(protein_coding_bed, retained_intron_bed):
    X = pybedtools.BedTool.from_dataframe(protein_coding_bed)
    Y = pybedtools.BedTool.from_dataframe(retained_intron_bed)

    intersection = Y.intersect(X, c=True).to_dataframe()
    intersection.columns = ['chrom', 'start', 'end', 'overlaps']
    Z = pybedtools.BedTool.from_dataframe(intersection.loc[intersection.overlaps >= 2])

    if len(Z) >= 1:

        Z = Z.subtract(X).to_dataframe()
        Z.start -= 1
        Z.end += 1
        Z = Z[['chrom', 'start', 'end']]
        Z = pybedtools.BedTool.from_dataframe(Z).intersect(X, c=True).to_dataframe()
        Z = Z.loc[Z.name >= 2]
    
        if len(Z) >= 1:
            Z.start += 1
            Z.end -= 1

    else:
        Z = pd.DataFrame(columns = ['chrom', 'start', 'end'])

    return Z

def get_coverage_percentage(bed, start, end, hard_start, hard_end):

    if len(bed) == 1:
        x = np.max([((merged_intron_retention.start - start)/(end - start)).iloc[0], 0])
        y = np.min([((merged_intron_retention.end - start)/(end - start)).iloc[0], 1])
        covered = np.abs(y-x)
    else:
    
        X = ((bed.end - start)/(end - start)).iloc[:-1]
        Y = ((bed.start - start)/(end - start)).iloc[1:]
    
        covered = 0
    
        if (X.iloc[0] > 0) and (X.iloc[0] < 1):
            x = X.iloc[0]
            covered += x
    
        for i in range(len(X)-1):
            x = np.min([np.max([0, X.iloc[i+1]]), 1])
            y = np.max([np.min([1, Y.iloc[i]]), 0])
            
            covered += np.abs(y-x)
    
        if (Y.iloc[-1] < 1) and (Y.iloc[-1] > 0):
            y = Y.iloc[-1]
            covered += np.abs(1-y)

    return covered


print('begin')

gencode_exons_bed = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz'
gencode_exons = pd.read_csv(gencode_exons_bed, sep='\t', 
                            names = ['chrom', 'start', 'end', 'gene_id', 'gene_name',
                                                           'transcript_id', 'strand', 'exon_id', 'transcript_support_level',
                                                           'basic', 'Ensembl_canonical', 'MANE_Select', 'appris', 'transcript_type'])

gencode_exons = gencode_exons.loc[~gencode_exons.chrom.isin(['chrY', 'chrM'])]

torino_exons_bed = 'results/torino/Human/isoforms/isoforms.bed.gz'
torino_exons = pd.read_csv(torino_exons_bed, sep='\t', names = ['chrom', 'start', 'end', 'gene_id', 
                                                 'transcript_id', 'strand', 'factors', 'exon_id', 'notes'])

torino_exons['transcript_id'] = torino_exons.gene_id + '.' + torino_exons.transcript_id





annotation = pd.read_csv('results/torino/Human/isoforms/annotated.isoforms.tab.gz', sep='\t')
annotation = annotation.loc[annotation.appris_transcript_length != 'appris_transcript_length']
annotation['appris_transcript_length'] = annotation.appris_transcript_length.astype(int)
annotation['gene_id'] = [x.split('.')[0] for x in annotation.transcript]

annotation['gene'] = annotation.transcript.apply(lambda x: x.split('.')[0])



efficiency_factor = []
for idx, row in annotation.iterrows():
    appris_ref = row.appris_ref
    aver = appris_ref.split('/')
    if 'RI:' in appris_ref:
        ri_factor = [x for x in aver if x.startswith('RI:')][0]
        ri_numbers = int(ri_factor.split(':')[1])
        appris_length = int(row.appris_transcript_length)
        
        if ((ri_numbers <= 2) and (appris_length >= 5)) or ((ri_numbers <= 1) and (appris_length >= 3)):
            efficiency_factor.append(False)
        else:
            efficiency_factor.append(True)
    else:
        efficiency_factor.append(False)

annotation['efficiency_factor'] = efficiency_factor



with open('results/CoveragePlots/bed_files/introns/torino.all_retained_introns.bed', 'wt') as fh:
    with open('results/CoveragePlots/bed_files/introns/torino.all_flanking_exons_1.bed', 'wt') as fh_1:
        with open('results/CoveragePlots/bed_files/introns/torino.all_flanking_exons_2.bed', 'wt') as fh_2:

            for gene, df in torino_exons.groupby('gene_id'):

                print(gene)
                gene_counter = 1
        
                annot_slice = annotation.loc[annotation.gene_id == gene]


                gencode_df = gencode_exons.loc[(gencode_exons.gene_name == gene) & (gencode_exons.transcript_type == 'protein_coding') & (~gencode_exons.appris.isna())]

                annotated_isoforms = list(gencode_df.transcript_id)

                #####
            
                retained_introns = list(annot_slice.loc[
                    annot_slice.appris_ref.apply(lambda x: 'RI:' in x) #& (~annot_slice.efficiency_factor)
                    ].transcript)


                ### Changed this a bit
                # bedtool = pybedtools.BedTool.from_dataframe(df.loc[df.transcript_id.isin(annotated_isoforms)])
                bedtool = pybedtools.BedTool.from_dataframe(gencode_df.loc[gencode_df.transcript_id.isin(annotated_isoforms)])
                ###################
            
                merged_bedtool = bedtool.merge().to_dataframe()
                if (len(annotated_isoforms) > 0) and (len(merged_bedtool) >= 2):
                    
                    start = merged_bedtool.end.min()
                    end = merged_bedtool.start.max()
                    hard_start = df.start.min()
                    hard_end = df.end.max()
                    
                    if (len(retained_introns) > 0):
                        
                        merged_intron_retention = pybedtools.BedTool.from_dataframe(df.loc[df.transcript_id.isin(retained_introns)])
                        merged_intron_retention = merged_intron_retention.merge().to_dataframe()
            
                        if (len(merged_intron_retention) >= 1):# or ((len(merged_intron_retention) == 1) and (percent < 0.5)):
            
                            strand = df.iloc[0].strand
                            transcripts = '|'.join(retained_introns)
                            
                            retained_i = extract_retained_introns(merged_bedtool, merged_intron_retention)
                            X = pybedtools.BedTool.from_dataframe(merged_bedtool)

        
                            for idx, row in retained_i.iterrows():
        
                                row_df = pd.DataFrame(row).T
                                row_df.start -= 1
                                row_df.end += 1
                                row_df = pybedtools.BedTool.from_dataframe(row_df)
        
                                flanking_exons = X.intersect(row_df, u = True).sort().to_dataframe()
        
                                assert flanking_exons.shape[0] >= 2
        
                                flanking_exon_1 = flanking_exons.iloc[0]
                                flanking_exon_2 = flanking_exons.iloc[-1]

                                intron_name = f'{gene}_torino_intron_all_{str(gene_counter)}'
        
                                line = ('\t'.join([row.chrom, str(row.start), str(row.end), intron_name, transcripts, strand]) + '\n')
                                line = line#.encode()
                                fh.write(line)
        
                                line = ('\t'.join([flanking_exon_1.chrom, str(flanking_exon_1.start), str(flanking_exon_1.end), 
                                                   intron_name, transcripts, strand]) + '\n')
                                # line = line.encode()
                                fh_1.write(line)
        
                                line = ('\t'.join([flanking_exon_2.chrom, str(flanking_exon_2.start), str(flanking_exon_2.end), 
                                                   intron_name, transcripts, strand]) + '\n')
                                # line = line.encode()
                                fh_2.write(line)
        
                                gene_counter += 1
                                

                        
