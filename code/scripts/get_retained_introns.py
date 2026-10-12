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



with gzip.open('results/CoveragePlots/bed_files/introns/torino.retained_introns.bed.gz', 'wb') as fh:
    with open('results/CoveragePlots/bed_files/introns/torino.flanking_exons_1.bed', 'wt') as fh_1:
        with open('results/CoveragePlots/bed_files/introns/torino.flanking_exons_2.bed', 'wt') as fh_2:

            for gene, df in torino_exons.groupby('gene_id'):

                print(gene)

        
                gene_counter = 1
        
                annot_slice = annotation.loc[annotation.gene_id == gene]

                ###### This was changed
                # annotated_isoforms = list(annot_slice.loc[
                #                           annot_slice.appris_ref.apply(lambda x: ('intron.chain' in x) or ('ref.transcript' in x))
                #                           ].transcript)

                ######## These ones were added

                gencode_df = gencode_exons.loc[(gencode_exons.gene_name == gene) & (gencode_exons.transcript_type == 'protein_coding') & (~gencode_exons.appris.isna())]

                annotated_isoforms = list(gencode_df.transcript_id)

                #####
            
                retained_introns = list(annot_slice.loc[
                    annot_slice.appris_ref.apply(lambda x: 'RI:' in x) & (~annot_slice.efficiency_factor)
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

                                intron_name = f'{gene}_torino_intron_{str(gene_counter)}'
        
                                line = ('\t'.join([row.chrom, str(row.start), str(row.end), intron_name, transcripts, strand]) + '\n')
                                line = line.encode()
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
                                

                        
with gzip.open('results/CoveragePlots/bed_files/introns/gencode.retained_introns.bed.gz', 'wb') as fh:
    with open('results/CoveragePlots/bed_files/introns/gencode.flanking_exons_1.bed', 'wt') as fh_1:
        with open('results/CoveragePlots/bed_files/introns/gencode.flanking_exons_2.bed', 'wt') as fh_2:

            for gene, df in gencode_exons.loc[gencode_exons.gene_name.isin(torino_exons.gene_id.unique())].groupby('gene_name'):

                print(gene)
        
                gene_counter = 1
        
                annotated_isoforms = list(df.loc[(df.transcript_type == 'protein_coding') & (~df.appris.isna())].transcript_id)
            
                retained_introns = list(df.loc[df.transcript_type == 'retained_intron'].transcript_id)
            
                bedtool = pybedtools.BedTool.from_dataframe(df.loc[df.transcript_id.isin(annotated_isoforms)])
                merged_bedtool = bedtool.merge().to_dataframe()
                if (len(annotated_isoforms) > 0) and (len(merged_bedtool) >= 2):
                
                    start = merged_bedtool.end.min()
                    end = merged_bedtool.start.max()
                    hard_start = df.start.min()
                    hard_end = df.end.max()
                    
                    if (len(retained_introns) > 0):
            
                        merged_intron_retention = pybedtools.BedTool.from_dataframe(df.loc[df.transcript_id.isin(retained_introns)])
                        merged_intron_retention = merged_intron_retention.merge().to_dataframe()
            
                        if len(merged_intron_retention) >= 1:
        
                            try:
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

                                    intron_name = f'{gene}_gencode_intron_{str(gene_counter)}'

                                    
                                    
                                    line = ('\t'.join([row.chrom, str(row.start), str(row.end), intron_name, 
                                                       transcripts, strand]) + '\n')
                                    line = line.encode()
                                    fh.write(line)

                                    start_ = int(flanking_exon_1.start)
                                    end_ = int(flanking_exon_1.end)
                                    if (end_ - start_) < 20:
                                        start_ = end_ - 20
            
                                    line = ('\t'.join([flanking_exon_1.chrom, str(start_), str(end_), 
                                                       intron_name, transcripts, strand]) + '\n')
                                    # line = line.encode()
                                    fh_1.write(line)

                                    start_ = int(flanking_exon_2.start)
                                    end_ = int(flanking_exon_2.end)
                                    if (end_ - start_) < 20:
                                        end_ = start_ + 20
            
                                    line = ('\t'.join([flanking_exon_2.chrom, str(start_), str(end_), 
                                                       intron_name, transcripts, strand]) + '\n')
                                    # line = line.encode()
                                    fh_2.write(line)
            
                                    gene_counter += 1
                
                            except:
                                continue


with gzip.open('results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed.gz', 'wb') as fh:
    with open('results/CoveragePlots/bed_files/introns/appris_flanking_exons_1.bed', 'wt') as fh_1:
        with open('results/CoveragePlots/bed_files/introns/appris_flanking_exons_2.bed', 'wt') as fh_2:



            # changing to include more appris

            appris_transcripts = gencode_exons.loc[
            (gencode_exons.transcript_type == 'protein_coding') & (~gencode_exons.appris.isna()) & gencode_exons.gene_name.isin(torino_exons.gene_id.unique())
            ]

            
            # appris_transcripts = gencode_exons.loc[
            # (gencode_exons.transcript_type == 'protein_coding') & (gencode_exons.appris=='appris_principal_1') & gencode_exons.gene_id.isin(torino_exons.gene_id.unique())
            # ]

            # This used to be transcript_id
            for gene, df in appris_transcripts.groupby('gene_name'):

                print(gene)
        
                gene_counter = 1

                ### This line was added
                df = pybedtools.BedTool.from_dataframe(df[['chrom', 'start', 'end', 'transcript_id', 'strand']]).sort().merge(
                                                      c=[4, 5], o = ['collapse', 'first']).to_dataframe(
                    names = ['chrom', 'start', 'end', 'transcript_id', 'strand'])

                if len(df) >= 2:
                    chrom = df.iloc[0].chrom
                    strand = df.iloc[0].strand
                    # gene = df.iloc[0].gene_id
                    transcript = df.iloc[0].transcript_id
                    
                    intron_pos = zip(df.end.iloc[:-1], df.start.iloc[1:])
        
                    for i, (intron_start, intron_end) in enumerate(intron_pos):
                        intron_name = f'{gene}_appris_intron_{str(gene_counter)}'
                        line = ('\t'.join([chrom, str(intron_start), str(intron_end), intron_name, transcript, strand]) + '\n').encode()
                        fh.write(line)
        
                        flanking_exon_start = int(df.iloc[i].start)
                        flanking_exon_end = int(df.iloc[i].end)
                        line = ('\t'.join([chrom, str(flanking_exon_start), str(flanking_exon_end), intron_name, transcript, strand]) + '\n')#.encode()
                        fh_1.write(line)
        
                        flanking_exon_start = int(df.iloc[i+1].start)
                        flanking_exon_end = int(df.iloc[i+1].end)
                        line = ('\t'.join([chrom, str(flanking_exon_start), str(flanking_exon_end), intron_name, transcript, strand]) + '\n')#.encode()
                        fh_2.write(line)


exons = pd.read_csv('resources/vastdb/EVENT_INFO-hg38.tab.gz', sep='\t')
exons = exons.dropna()
exons = exons.loc[exons.COMPLEX == 'IR'].copy()
exons_cols = list(exons.columns)
exons['chrom'] = exons.CO_A.apply(lambda x: x.split(':')[0])
exons['start'] = exons.CO_A.apply(lambda x: x.split(':')[1].split('-')[0]).astype(int)
exons['end'] = exons.CO_A.apply(lambda x: x.split('-')[1]).astype(int)
exons_cols = ['chrom', 'start', 'end'] + exons_cols
exons['strand'] = exons.FULL_CO.apply(lambda x: x.split(':')[-1])

exons.start -= 1
exons.end += 1

flanking_exon_1_start_list = []
flanking_exon_1_end_list = []

flanking_exon_2_start_list = []
flanking_exon_2_end_list = []

for idx, row in exons.iterrows():
    if row.strand == '+':
        flanking_exon_1 = row.CO_C1.split(':')[1]
        flanking_exon_2 = row.CO_C2.split(':')[1]
    else:
        flanking_exon_1 = row.CO_C2.split(':')[1]
        flanking_exon_2 = row.CO_C1.split(':')[1]

    flanking_exon_1_start = int(flanking_exon_1.split('-')[0])
    flanking_exon_1_end = int(flanking_exon_1.split('-')[1])

    flanking_exon_2_start = int(flanking_exon_2.split('-')[0])
    flanking_exon_2_end = int(flanking_exon_2.split('-')[1])

exons['flanking_exon_1_start'] = flanking_exon_1_start
exons['flanking_exon_1_end'] = flanking_exon_1_end
exons['flanking_exon_2_start'] = flanking_exon_2_start
exons['flanking_exon_2_end'] = flanking_exon_2_end

exons = exons[['chrom', 'start', 'end', 'GENE', 'EVENT', 'strand', 'flanking_exon_1_start', 'flanking_exon_1_end',
              'flanking_exon_2_start', 'flanking_exon_2_end']].copy()

genes = pd.read_csv('resources/annotation/Human/bed/genes.protein_coding.bed.gz', sep='\t', names = ['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'])


exons_bed = pybedtools.BedTool.from_dataframe(exons).sort()
genes_bed = pybedtools.BedTool.from_dataframe(genes).sort()

vastdb_exons = exons_bed.intersect(genes_bed, f=1, wao=True, s=True).to_dataframe(names = list(exons.columns) + [f'{x}_gene' for x in list(genes.columns)] + ['overlaps'])

vastdb_exons['intron_len'] = vastdb_exons.end - vastdb_exons.start

vastdb_exons = vastdb_exons.loc[vastdb_exons.overlaps >= vastdb_exons.intron_len]

vastdb_exons[['chrom', 'start', 'end', 'gene_id_gene', 'EVENT', 'strand']].to_csv('results/CoveragePlots/bed_files/introns/vastdb.retained_introns.bed.gz', 
                                                                   sep='\t', header=False, index=False)

vastdb_exons[['chrom', 'flanking_exon_1_start', 'flanking_exon_1_end', 'gene_id_gene', 'EVENT', 'strand']].to_csv('results/CoveragePlots/bed_files/introns/vastdb.flanking_exons_1.bed', 
                                                                   sep='\t', header=False, index=False)

vastdb_exons[['chrom', 'flanking_exon_2_start', 'flanking_exon_2_end', 'gene_id_gene', 'EVENT', 'strand']].to_csv('results/CoveragePlots/bed_files/introns/vastdb.flanking_exons_2.bed', 
                                                                   sep='\t', header=False, index=False)

