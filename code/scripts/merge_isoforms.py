import numpy as np
import pandas as pd
import sys

def intron_chain(exons):
    """Generate the intron chain from a list of exons."""
    chain = []
    for i in range(1, len(exons)):
        chain.append((exons[i-1][1], exons[i][0]))
    return chain

def transcript_list_to_dict(transcripts):
    out_dir = {}
    for idx, df in transcripts:
        out_dir.update({idx:df})
    return out_dir


def find_largest_subset(transcripts):
    """Find the largest transcript for which each transcript is a subset."""

    transcripts_dir = transcript_list_to_dict(transcripts)
    
    result = {}
    for i, (tid_b, tb) in enumerate(transcripts):
        result[tid_b] = None
        for j, (tid_a, ta) in enumerate(transcripts):
            if (i != j) and is_subset(tb, ta):
                if result[tid_b] is None or (len(ta) > len(transcripts_dir[result[tid_b]])):
                    result[tid_b] = tid_a
    return result

def get_transcript_supersets(transcript_subset_dict):

    any_none = False
    for transcript, transcript_supraset in transcript_subset_dict.items():
        if transcript_supraset is None:
            any_none = True

    if not any_none:
        raise Exception('Superset loop.')
    
    output_dir = {}
    for transcript, transcript_supraset in transcript_subset_dict.items():
        if transcript_supraset is None:
            transcript_supraset_key = transcript
        else:
            transcript_supraset_key = transcript_supraset
            while transcript_subset_dict[transcript_supraset_key] is not None:
                transcript_supraset_key = transcript_subset_dict[transcript_supraset_key] 
            
            
        if transcript_supraset_key in output_dir.keys():
            output_dir[transcript_supraset_key].append(transcript)
        else:
            output_dir.update({transcript_supraset_key:[transcript]})

    return output_dir



def get_factor_chain(transcript_list, transcripts_dict):
    factor_chain = []
    for transcript in transcript_list:
        factor_chain.append(transcripts_dict[transcript]['factor_chain'].iloc[0])

    factor_chain = ':'.join(factor_chain)
    return factor_chain


def merge_transcripts_in_gene(gene_id, gene_df, fh):
    transcripts = list(gene_df.groupby('transcript_id'))
    transcript_subset_dict = find_largest_subset(transcripts)
    supra_transcripts = get_transcript_supersets(transcript_subset_dict)
    transcripts_dict = transcript_list_to_dict(list(gene_df.groupby('transcript_id')))
    
    supra_transcripts_list = sorted(supra_transcripts.keys())
    
    for i, supra_transcript in enumerate(supra_transcripts_list):
        idx = str(i+1)
        isoform_id = f'{gene_id}.isoform_{idx}'
    
        supra_transcript_bed = transcripts_dict[supra_transcript]
        factor_chain = get_factor_chain(supra_transcripts[supra_transcript], transcripts_dict)
    
        for j, row in supra_transcript_bed.iterrows():
            chrom = row.chrom
            start = str(row.start)
            end = str(row.end)
            strand = row.strand
            exon_id = row.exon_id
    
            new_line_list = [chrom, start, end, gene_id, isoform_id, strand, factor_chain, exon_id]
            new_line = '\t'.join(new_line_list) + '\n'
            fh.write(new_line)

def is_subset(transcript_b, transcript_a):
    """Check if transcript_b is a subset of transcript_a."""
    exons_b = list(zip(transcript_b['start'], transcript_b['end']))
    exons_a = list(zip(transcript_a['start'], transcript_a['end']))

    introns_b = intron_chain(exons_b)
    introns_a = intron_chain(exons_a)

    # Check if intron chain of B is a subset of intron chain of A
    if not all(intron in introns_a for intron in introns_b):
        return False

    # Check if all shared entries are consecutive from 3' end
    strand = transcript_b['strand'].iloc[0]

    intron_chain_length_a = len(introns_a)
    intron_chain_length_b = len(introns_b)

    if intron_chain_length_b >= 1:
        if strand == '+':
            sub_chain_a = introns_a[-intron_chain_length_b:]
        else:
            sub_chain_a = introns_a[:intron_chain_length_b]

        if introns_b != sub_chain_a:
            return False

    # Check if single exon does not span more than one exon (intron retention)
    elif intron_chain_length_a >= 2:
        if strand == '+':
            # exon starts before the end of second to last exon in a
            if exons_b[0][0] <= exons_a[-2][1]:
                return False
        else:
            #exon ends after start of second exon in a
            if exons_b[0][1] >= exons_a[1][0]:
                return False

    exon_chain_length_a = len(exons_a)
    exon_chain_length_b = len(exons_b)

    # check 5' exon
    if strand == '+':
        exon_a_5 = exons_a[-exon_chain_length_b]
        exon_a_5_start = exon_a_5[0]
        exon_a_5_length = exon_a_5[1] - exon_a_5[0]

        exon_b_5 = exons_b[0]
        exon_b_5_start = exon_b_5[0]
        exon_b_5_length = exon_b_5[1] - exon_b_5[0]

        if exon_chain_length_a == exon_chain_length_b:
            exon_diff = abs(exon_a_5_start - exon_b_5_start)
            exon_diff_percent = np.min([exon_diff/exon_a_5_length, exon_diff/exon_b_5_length])
            
        else:
            exon_diff = exon_a_5_start - exon_b_5_start
            exon_diff_percent = exon_diff/exon_a_5_length
    else:
        exon_a_5 = exons_a[exon_chain_length_b-1]
        exon_a_5_start = exon_a_5[1]
        exon_a_5_length = exon_a_5[1] - exon_a_5[0]
        

        exon_b_5 = exons_b[-1]
        exon_b_5_start = exon_b_5[1]
        exon_b_5_length = exon_b_5[1] - exon_b_5[0]

        if exon_chain_length_a == exon_chain_length_b:
            exon_diff = abs(exon_b_5_start - exon_a_5_start)
            exon_diff_percent = np.min([exon_diff/exon_a_5_length, exon_diff/exon_b_5_length])
        else:
            exon_diff = exon_b_5_start - exon_a_5_start
            exon_diff_percent = exon_diff/exon_a_5_length

    # exon 5' in chain b starts much earlier than in chain a, suggesting alternative initiation site
    if (exon_diff > 200) or (exon_diff_percent > 0.25):
        return False

    # check 3' exon
    if strand == '+':
        exon_a_3 = exons_a[-1]
        exon_b_3 = exons_b[-1]

        exon_a_3_end = exon_a_3[1]
        exon_a_3_length = exon_a_3[1] - exon_a_3[0]

        exon_b_3_end = exon_b_3[1]
        exon_b_3_length = exon_b_3[1] - exon_b_3[0]

    else:
        exon_a_3 = exons_a[0]
        exon_b_3 = exons_b[0]

        exon_a_3_end = exon_a_3[0]
        exon_a_3_length = exon_a_3[1] - exon_a_3[0]

        exon_b_3_end = exon_b_3[0]
        exon_b_3_length = exon_b_3[1] - exon_b_3[0]

    exon_diff = abs(exon_a_3_end - exon_b_3_end)
    exon_diff_percent = np.min([exon_diff/exon_a_3_length, exon_diff/exon_b_3_length])

    if (exon_diff > 200) or (exon_diff_percent > 0.25):
        return False

    return True


if __name__ == '__main__':
    arguments = sys.argv
    bed_file = arguments[1]
    output_file = arguments[2]
    
    # Load the BED file
    #bed_file = 'ebpmf_models/filtered/snmf_10/tables/snmf.3prime_corrected.exons.sorted.bed.gz'
    
    columns = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factor_chain', 'exon_id']
    df = pd.read_csv(bed_file, sep='\t', header=None, names=columns)
    
    # Group by gene_id
    genes = df.groupby('gene_id')

    with open(output_file, 'w') as fh:
        for gene_id, gene_df in genes:
            merge_transcripts_in_gene(gene_id, gene_df, fh)
