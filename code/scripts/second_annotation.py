import numpy as np
import pandas as pd
import gzip
import sys


def is_subchain(introns_a, introns_b):

    if introns_a == introns_b:
        return 'chain.match'

    intron_chain_length_a = len(introns_a)
    intron_chain_length_b = len(introns_b)

    if intron_chain_length_a > intron_chain_length_b:
        diff_chain = intron_chain_length_a - intron_chain_length_b
        # print(diff_chain)
        for i in range(diff_chain+1):
            if (introns_b == introns_a[i:intron_chain_length_b+i]):
                
                return 'subchain.b'
    elif intron_chain_length_a < intron_chain_length_b:
        diff_chain = intron_chain_length_b - intron_chain_length_a
        # print(diff_chain)
        for i in range(diff_chain+1):
            # print(introns_b[i:intron_chain_length_a+i])
            if (introns_a == introns_b[i:intron_chain_length_a+i]):
                return 'subchain.a'

    return 'no.match'


def get_intron_chain(transcript_bed):
    if pd.DataFrame(transcript_bed).shape[1] > 3:
        intron_chain = []
        for x in zip(transcript_bed.end[:-1], transcript_bed.start[1:]):
            intron_chain.append(x)
    else:
        intron_chain = None
        
    return intron_chain


def compare_transcript_v_annotation(transcript_bed, input_transcript, exons, gene):
    snmf_iso = get_intron_chain(transcript_bed)
    
    utr_annot = []
    chain_annot = []
    transcript_list = []
    
    for transcript, df in exons.loc[exons.gene_id == gene].groupby('transcript_id'):
        if df.shape[0] <= 1:
            continue
        if transcript == input_transcript:
            continue
        annot_iso = get_intron_chain(df)
        chain_match = is_subchain(snmf_iso, annot_iso)
    
        chain_annot.append(chain_match) 
        transcript_list.append(chain_match+ ':' + transcript)
    
        if df.iloc[0].strand == '+':
            if snmf_iso[-1][1] == annot_iso[-1][1]:
                snmf_last = int(transcript_bed.iloc[-1].end)
                gencode_last = int(df.iloc[-1].end)
                diff = np.abs(snmf_last - gencode_last)
                if diff <= 200:
                    utr_annot.append('annotated')
                else:
                    utr_annot.append('alt.utr')
            else:
                utr_annot.append('other')
                
        elif df.iloc[0].strand == '-':
            if snmf_iso[0][0] == annot_iso[0][0]:
                snmf_last = int(transcript_bed.iloc[0].start)
                gencode_last = int(df.iloc[0].start)
                diff = np.abs(snmf_last - gencode_last)
                if diff <= 200:
                    utr_annot.append('annotated')
                else:
                    utr_annot.append('alt.utr')
            else:
                utr_annot.append('other')

    df_ = pd.DataFrame()
    df_['chain_annot'] = chain_annot
    df_['utr_annot'] = utr_annot
    df_['transcript'] = transcript_list
    return df_
    


def get_annotation_transcript(snmf_exons, gencode_exons, transcript):
    gene = transcript.split('.')[0]
    transcript_bed = snmf_exons.loc[snmf_exons.transcript_id == transcript]
    try:
        df_gencode = compare_transcript_v_annotation(transcript_bed, transcript, gencode_exons, gene)
        df_snmf = compare_transcript_v_annotation(transcript_bed, transcript, snmf_exons, gene)
    
        # print(df_gencode)
        if ('chain.match' in list(df_gencode.chain_annot)) or ('subchain.a' in list(df_gencode.chain_annot)):
            
            if 'chain.match' in list(df_gencode.chain_annot):
                transcripts = '|'.join(list(df_gencode.loc[df_gencode.chain_annot == 'chain.match'].transcript))
            else:
                transcripts = '|'.join(list(df_gencode.loc[df_gencode.chain_annot == 'subchain.a'].transcript))
            
            # if 'annotated' in list(df_gencode.loc[df_gencode.chain_annot.isin(['chain.match', 'subchain.a'])].utr_annot):
            if 'annotated' in list(df_gencode.utr_annot):
                gencode_annot = 'annotated_chain.annotated_utr'
            else:
                gencode_annot = 'annotated_chain.alt_utr'
        else:
            transcripts = '.'
            if 'annotated' in list(df_gencode.utr_annot):
                gencode_annot = 'unannotated_chain.annotated_utr'
            else:
                gencode_annot = 'unannotated_chain.alt_utr'
    
        snmf_annot = 'no_alt'
    
        if ('chain.match' in list(df_snmf.chain_annot)) or ('subchain.a' in list(df_snmf.chain_annot)) or ('subchain.b' in list(df_snmf.chain_annot)):
            if 'alt.utr' in list(df_snmf.loc[df_snmf.chain_annot.isin(['chain.match', 'subchain.a', 'subchain.b'])].utr_annot):
                snmf_annot = 'alt.utr'
    except:
        return 'unannotated_chain.annotated_utr', 'no_alt', '.'

    return gencode_annot, snmf_annot, transcripts

def define_annotation(snmf_exons, gencode_exons, transcript, annotation):
    gencode_annot, snmf_annot, transcripts = get_annotation_transcript(snmf_exons, gencode_exons, transcript)
    gencode_chain, gencode_utr = gencode_annot.split('.')

    ri = 'RI:' in annotation.loc[annotation.transcript == transcript].appris_ref.iloc[0]
    
    if (gencode_chain == 'annotated_chain'):
        chain = 'annotated_chain'
    else:
        chain = 'unannotated_chain'

    if ri:
        if chain == 'annotated_chain':
            ir = 'annotated.intron_retention'
        elif chain == 'unannotated_chain':
            ir = 'unannotated.intron_retention'
    else:
        ir = 'no_intron_retention'
        
    if gencode_utr == 'annotated_utr':
        if snmf_annot == 'alt.utr':
            utr = 'annotated:alt.utr'
        else:
            utr = 'no_utr'
            
    else:
        if snmf_annot == 'alt.utr':
            utr = 'unannotated:alt.utr'
        else:
            utr = 'no_utr'

    return chain, ir, utr, transcripts

if __name__ == '__main__':

    arguments = sys.argv
    gencode_exons_file = arguments[1]
    snmf_exons_file = arguments[2]
    annotation_file = arguments[3]
    output_file = arguments[4]

    # gencode_exons_bed = '/project2/mstephens/cfbuenabadn/gtex-stm/code/Annotations/gencode.v44.primary_assembly.exons.sorted.bed.gz'
    gencode_exons = pd.read_csv(gencode_exons_file, sep='\t', 
                                names = ['chrom', 'start', 'end', 'gene_id', 'gene_name',
                                                               'transcript_id', 'strand', 'exon_id', 'transcript_support_level',
                                                               'basic', 'Ensembl_canonical', 'MANE_Select', 'appris', 'transcript_type'])
    
    # snmf_exons_bed = '/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/filtered/snmf_10/tables/snmf.merged_isoforms.exons.sorted.bed.gz'
    snmf_exons = pd.read_csv(snmf_exons_file, sep='\t', names = ['chrom', 'start', 'end', 'gene_id', 
                                                     'transcript_id', 'strand', 'factors', 'exon_id', 'notes'])

    snmf_exons['transcript_id'] = snmf_exons.gene_id + '.' + snmf_exons.transcript_id

    gencode_exons['gene_id'] = gencode_exons.gene_name
    
    
    annotation = pd.read_csv(annotation_file, sep='\t')
    annotation = annotation.loc[annotation.appris_transcript_length != 'appris_transcript_length']
    annotation['appris_transcript_length'] = annotation.appris_transcript_length.astype(int)
    annotation['gene_id'] = [x.split('.')[0] for x in annotation.transcript]
    
            
    
    transcripts = list(snmf_exons.transcript_id.unique())
    
    ir_annot_list = []
    utr_annot_list = []
    chain_annot_list = []
    
    with gzip.open(output_file, 'wb') as fh:
    
        for isoform in transcripts:
            print(isoform)
            try:
                chain_annot, ir_annot, alt_utr_annot, transcripts = define_annotation(snmf_exons, gencode_exons, isoform, annotation)
            except:
                chain_annot = '.'
                ir_annot = '.'
                alt_utr_annot = '.'
                transcripts = '.'
            #chain_annot, ir_annot, alt_utr_annot = define_annotation(snmf_exons, gencode_exons, isoform, annotation)
            gene, iso = isoform.split('.')
            row = ('\t'.join([gene, isoform, chain_annot, transcripts, ir_annot, alt_utr_annot]) + '\n').encode()
            fh.write(row)
    
    print('done!')
            
        