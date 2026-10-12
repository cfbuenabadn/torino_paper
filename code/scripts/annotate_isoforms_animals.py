import numpy as np
import pandas as pd
import gzip
from pybedtools import BedTool
import tabix

import sys


def get_juncs(bed):
    juncs = pd.DataFrame()
    chrom_cols = list(bed.chrom)[:-1]
    start_cols = list(bed.end.astype(int)+1)[:-1]
    end_cols = list(bed.start.astype(int)-1)[1:]
    juncs['chrom'] = chrom_cols
    juncs['start'] = start_cols
    juncs['end'] = end_cols
    return juncs

def get_donor_acceptors(juncs):
    donors = list(juncs.start)
    acceptors = list(juncs.end)
    return donors, acceptors

def same_intron_chain(ref, query):
    if (len(ref) != len(query)) or (len(query) == 1):
        return False
    else:
        ref_juncs = get_juncs(ref)
        query_juncs = get_juncs(query)
        match_test = np.array(ref_juncs == query_juncs)
        is_chain = np.all(match_test)

        if is_chain:
            return True
        else:
            return False

def alt_utr(ref, query, strand, prime = 'five'):
    if strand == '-':
        ref = ref[::-1]
        query = query[::-1]

    if prime == 'five':
        i = 0
    elif prime == 'three':
        i = -1
    else:
        raise Exception('invalid prime end')
        
    ref_start = int(ref.iloc[i].start)
    ref_end = int(ref.iloc[i].end)
    ref_exon_len = ref_end - ref_start

    ref_exon_len = np.max([ref_exon_len, 1])

    query_start = int(query.iloc[i].start)
    query_end = int(query.iloc[i].end)
    query_exon_len = query_end - query_start

    query_exon_len = np.max([query_exon_len, 1])

    ratio_ref = ref_exon_len/query_exon_len
    ratio_query = query_exon_len/ref_exon_len
    

    if ((prime == 'five') and (strand == '+')) or ((prime == 'three') and (strand == '-')):
        if ref_end != query_end:
            utr = 'alt.exon'
        elif (np.abs(ref_start - query_start) < 200) and (ratio_ref > 0.75) and (ratio_query > 0.75):
            utr = 'same'
        else:
            utr = 'alt.utr'
    else:
        if ref_start != query_start:
            utr = 'alt.exon'
        elif (np.abs(ref_end - query_end) < 200) and (ratio_ref > 0.75) and (ratio_query > 0.75):
            utr = 'same'
        else:
            utr = 'alt.utr'
    return utr

def get_donor_acceptors(juncs):
    donors = list(juncs.start)
    acceptors = list(juncs.end)
    return donors, acceptors

def has_cassette_exon(query, ref):
    try:
        query_juncs = get_juncs(query)
        query_donors, query_acceptors = get_donor_acceptors(query_juncs)
        
        ref_juncs = get_juncs(ref)
        ref_donors, ref_acceptors = get_donor_acceptors(ref_juncs)
    
        for idx, rdonor in enumerate(ref_donors):
            racceptor = ref_acceptors[idx]
            if (rdonor in query_donors) and (racceptor in query_acceptors):
                idx_donor = query_donors.index(rdonor)
                idx_acceptor = query_acceptors.index(racceptor)
        
                if (idx_acceptor - idx_donor) == 1:
                    return True
        return False
    except:
        return False

def has_multiple_cassette_exons(query, ref):
    try:
        query_juncs = get_juncs(query)
        query_donors, query_acceptors = get_donor_acceptors(query_juncs)
        
        ref_juncs = get_juncs(ref)
        ref_donors, ref_acceptors = get_donor_acceptors(ref_juncs)
    
        for idx, rdonor in enumerate(ref_donors):
            racceptor = ref_acceptors[idx]
            if (rdonor in query_donors) and (racceptor in query_acceptors):
                idx_donor = query_donors.index(rdonor)
                idx_acceptor = query_acceptors.index(racceptor)
        
                if (idx_acceptor - idx_donor) > 1:
                    return True
        return False
    except:
        return False

def has_mutually_exclusive_exons(query, ref):
    try:
        query_juncs = get_juncs(query)
        query_donors, query_acceptors = get_donor_acceptors(query_juncs)
        
        ref_juncs = get_juncs(ref)
        ref_donors, ref_acceptors = get_donor_acceptors(ref_juncs)
    
        for idx, rdonor in enumerate(ref_donors):
            racceptor = ref_acceptors[idx]
            if idx < (len(ref_donors) - 1):
                nracceptor = ref_acceptors[idx+1]
                if (rdonor in query_donors) and (nracceptor in query_acceptors):
                    idx_donor = query_donors.index(rdonor)
                    idx_acceptor = query_acceptors.index(nracceptor)
        
                    if (idx_acceptor - idx_donor) == 1:
        
                        query_exon_start = query_acceptors[idx_donor]
                        query_exon_end = query_donors[idx_acceptor]
        
                        ref_exon_start = ref_acceptors[idx]
                        ref_exon_end = ref_donors[idx+1]
        
                        if (query_exon_end < ref_exon_start) or (query_exon_start > ref_exon_end):
                            return True
        return False
    except:
        return False
    
# def has_cassette_exon(query, ref):
#     query_bed = BedTool.from_dataframe(query)
#     ref_juncs = BedTool.from_dataframe(get_juncs(ref))

#     try:
#         cassette_exon = ref_juncs.intersect(query_bed, F=1)

#     except: 
#         print('comoquieras')
#         return False

#     if len(cassette_exon) >= 1:
#         return True
#     else:
#         return False

def has_missed_exon(query, ref):
    exon_missing = has_cassette_exon(ref, query)
    multiple_exons_missing = has_multiple_cassette_exons(ref, query)
    exons_are_missing = exon_missing or multiple_exons_missing
    return exons_are_missing

def has_retained_intron(query, ref):
    query_bed = BedTool.from_dataframe(query)
    ref_juncs = BedTool.from_dataframe(get_juncs(ref))

    try:
        retained_intron = query_bed.intersect(ref_juncs, F=1)
    except:
        return 0

    return len(retained_intron)
    # if len(retained_intron) >= 1:
    #     return True
    # else:
    #     return False

def has_novel_intron(query, ref):
    novel_intron =  has_retained_intron(ref, query) >= 1
    return novel_intron
                
            
def is_intron_subset(query, ref):
    if (len(query) == 1) or (len(ref)==1):
        return False
    
    query_juncs = BedTool.from_dataframe(get_juncs(query))
    ref_juncs_bed = get_juncs(ref)
    ref_juncs = BedTool.from_dataframe(ref_juncs_bed)

    try:
        junc_intersection = query_juncs.intersect(ref_juncs, F=1, f=1).to_dataframe()

    except:
        return False
    all_juncs_contained = ((len(junc_intersection) == len(query_juncs)) and (len(ref_juncs_bed) > len(junc_intersection)))

    return all_juncs_contained

def is_intron_supraset(query, ref):
    is_supraset = is_intron_subset(ref, query)
    return is_supraset

def is_transcript_subset(query, ref, strand):

    is_chain = same_intron_chain(ref, query)
    is_subset = is_intron_subset(query, ref)

    cassette_exon = has_cassette_exon(query, ref)
    
    retained_intron = has_retained_intron(query, ref)
    novel_intron = has_novel_intron(query, ref)
    skipped_exon = has_missed_exon(query, ref)

    if (not is_chain) and (not cassette_exon) and (not (retained_intron >= 1)) and (not novel_intron) and (not skipped_exon) and is_subset:
        query_bed = BedTool.from_dataframe(query)
        ref_bed = BedTool.from_dataframe(ref)

        try:
            ref_subset = ref_bed.intersect(query_bed, wa=True).to_dataframe()
        except:
            return False

        alt_5 = alt_utr(ref_subset, query, strand, prime = 'five')
        alt_3 = alt_utr(ref_subset, query, strand, prime = 'three')

        if (alt_5 != 'alt.exon') and (alt_3 != 'alt.exon'):
            return True
            
    return False


def is_transcript_supraset(query, ref, strand):
    is_supraset = is_transcript_subset(ref, query, strand)
    return is_supraset

def is_same_transcript(query, ref, strand):
    if len(query) != len(ref):
        return False
    if len(query) == 1:
        is_chain = True
    else:
        is_chain = same_intron_chain(ref, query)
    alt_5 = alt_utr(ref, query, strand, prime = 'five')
    alt_3 = alt_utr(ref, query, strand, prime = 'three')

    if is_chain and (alt_5 == 'same') and (alt_3 == 'same'):
        return True
    else:
        return False

def annotate_transcript(query, ref, strand):
    is_same = is_same_transcript(query, ref, strand)
    if is_same:
        return 'ref_transcript'
    alt_5 = alt_utr(query, ref, strand, 'five')
    alt_3 = alt_utr(query, ref, strand, 'three')
    out = []
    if alt_5 == 'alt.utr':
        out.append('UTR5')
    elif (alt_5 == 'alt.exon') and (len(query) > 1):
        out.append('alt_5exon')
    if alt_3 == 'alt.utr':
        out.append('UTR3')
    elif (alt_3 == 'alt.exon') and (len(query) > 1):
        out.append('alt_3exon')
    is_chain = same_intron_chain(query, ref)
    if is_chain:

        out = ['intron_chain'] + out
        
        assert len(out) > 0
        out = '/'.join(out)
        return out
    else:
        cassette_exon = has_cassette_exon(query, ref)
        multiple_cassette_exons = has_multiple_cassette_exons(query, ref)
        mutually_exclusive_exons = has_mutually_exclusive_exons(query, ref)
        retained_intron = has_retained_intron(query, ref)
        novel_intron = has_novel_intron(query, ref)
        skipped_exon = has_missed_exon(query, ref)

        if cassette_exon:
            out.append('CE')
        if multiple_cassette_exons:
            out.append('MCE')
        if mutually_exclusive_exons:
            out.append('MXE')
        if retained_intron>= 1:
            out.append(f'RI:{str(retained_intron)}')
        if novel_intron:
            out.append('NI')
        if skipped_exon:
            out.append('SE')

        alt_ss_count = count_alt_ss(query, ref)
        if alt_ss_count > 0:
            alt_ss_annot = f'alt.ss.{str(alt_ss_count)}'
            out.append(alt_ss_annot)
    # if len(out) > 0:
        # out = '/'.join(out)
        # return out
    # else:
    if is_transcript_subset(query, ref, strand):
        out = ['transcript_subset'] + out
    elif is_transcript_supraset(query, ref, strand):
        out = ['transcript_supraset'] + out
    if len(out) >= 1:
        out = '/'.join(out)
        return out
    else:
        return ('other')
    

def count_alt_ss(query, ref):

    alt_ss = 0
    
    if (len(query) < 3) or (len(ref) < 3):
        return alt_ss
        
    query_inner_exons = query[1:-1]
    ref_inner_exons = ref[1:-1]

    for idx, row in query_inner_exons.iterrows():
        for idx2, row2 in ref_inner_exons.iterrows():
            if (int(row.start) == int(row2.start)) and (int(row.end) != int(row2.end)):
                alt_ss += 1
            elif (int(row.start) != int(row2.start)) and (int(row.end) == int(row2.end)):
                alt_ss += 1
    return alt_ss
        

def annotate_transcripts_with_ref(query_bed, ref_bed, species):
    
    strand = list(query_bed.strand)[0]
    assert strand == list(ref_bed.strand)[0]

    bed_cols = ['chrom', 'start', 'end']
    
    ref_transcript_id = get_best_reference(ref_bed, species)
    ref_transcript_bed = ref_bed.loc[ref_bed.transcript_id == ref_transcript_id, bed_cols].reset_index(drop=True)

    query_transcripts = sorted(query_bed.transcript_id.unique())
    gencode_transcripts = [x for x in sorted(ref_bed.transcript_id.unique()) if x != ref_transcript_id]
    appris_annotation = []
    gencode_annotation = []

    reference_transcript_len = len(ref_transcript_bed)
    query_transcript_len = []

    for query_transcript in query_transcripts:
        query_transcript_bed = query_bed.loc[query_bed.transcript_id == query_transcript, bed_cols].reset_index(drop=True)
        query_transcript_len.append(len(query_transcript_bed))
        query_appris_annotation = annotate_transcript(query_transcript_bed, ref_transcript_bed, strand)
        appris_annotation.append(query_appris_annotation)

        query_gencode_annotation = '.'
        for gencode_transcript in gencode_transcripts:
            gencode_transcript_bed = ref_bed.loc[ref_bed.transcript_id == gencode_transcript, bed_cols].reset_index(drop=True)
            query_gencode = annotate_transcript(query_transcript_bed, gencode_transcript_bed, strand)
            if query_gencode == 'ref_transcript':
                query_gencode_annotation = 'ref_transcript:' + gencode_transcript
                break
            elif 'intron_chain' not in query_gencode_annotation:
                if 'intron_chain' in query_gencode:
                    query_gencode_annotation = 'intron_chain:' + gencode_transcript
                elif 'transcript_subset' in query_gencode:
                    query_genocde_annotation = 'transcript_subset:' + gencode_transcript

        gencode_annotation.append(query_gencode_annotation)

    #appris_annotation = [x + ':' + ref_transcript_id for x in appris_annotation]

    df_out = pd.DataFrame()
    df_out['transcript'] = query_transcripts
    df_out['appris_transcript'] = [ref_transcript_id]*len(query_transcripts)
    df_out['appris_transcript_length'] = [reference_transcript_len]*len(query_transcripts)
    df_out['query_transcript_len'] = query_transcript_len
    df_out['appris_ref'] = appris_annotation
    df_out['gencode_ref'] = gencode_annotation

    return df_out

def get_best_reference(ref_exons, species):
    if species != 'Mouse':
        if np.sum(ref_exons.transcript_type == 'protein_coding') > 0:
            selected_transcript = list(ref_exons.loc[ref_exons.transcript_type == 'protein_coding'].transcript_id)[0]
            print('prot_coding')
        else:
            selected_transcript = list(ref_exons.transcript_id)[0]
        return selected_transcript
    if any(ref_exons.appris == 'appris_principal_1'):
        ref_exons = ref_exons.loc[ref_exons.appris == 'appris_principal_1']
    elif any(ref_exons.MANE_Select == 'MANE_Select'):
        ref_exons = ref_exons.loc[ref_exons.MANE_Select == 'MANE_Select']
    elif any(ref_exons.Ensembl_canonical == 'Ensembl_canonical'):
        ref_exons = ref_exons.loc[ref_exons.Ensembl_canonical == 'Ensembl_canonical']
    elif any(ref_exons.basic == 'basic'):
        ref_exons = ref_exons.loc[ref_exons.basic == 'basic']
    elif any(ref_exons.transcript_support_level.astype(int) == 1):
        ref_exons = ref_exons.loc[ref_exons.transcript_support_level.astype(int) == 1]
    selected_transcript = list(ref_exons.transcript_id)[0]
    return selected_transcript

if __name__ == '__main__':
    arguments = sys.argv
    species = arguments[1]
    gencode_exons_bed = arguments[2]
    snmf_exons_bed = arguments[3]
    # out_prefix = arguments[3]
    i = int(arguments[4])
    n = int(arguments[5])
    # correction_tag = str(arguments[5])
    # K = str(int(arguments[6]))

    # gencode_exons_bed = 'gencode.v44.primary_assembly.exons.sorted.bed.gz'

    if species == 'Mouse':
        gencode_exons = pd.read_csv(gencode_exons_bed, sep='\t', names = ['chrom', 'start', 'end', 'gene_id', 'gene_name',
                                                               'transcript_id', 'strand', 'exon_id', 'transcript_support_level',
                                                               'basic', 'Ensembl_canonical', 'MANE_Select', 'appris', 'transcript_type'])
    else:
        gencode_exons = pd.read_csv(gencode_exons_bed, sep='\t')

        gene_names = []
        for idx, row in gencode_exons.iterrows():
            if row.gene_name == 'ensembl':
                gene_names.append(row.gene_id)
            else:
                gene_names.append(row.gene_name)

        gencode_exons['gene_name'] = gene_names
        gencode_exons['chrom'] = gencode_exons['#chrom']
    
    # snmf_exons_bed = '/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/filtered/snmf_10/tables/snmf.exons.sorted.bed.gz'

    gencode_exons['end'] = gencode_exons['end'] - 1
    snmf_exons = pd.read_csv(snmf_exons_bed, sep='\t', names = ['chrom', 'start', 'end', 'gene_id', 
                                                     'transcript_id', 'strand', 'factors', 'exon_id', 'notes'])

    snmf_exons['transcript_id'] = snmf_exons.gene_id + '.' + snmf_exons.transcript_id
    genes = list(snmf_exons.gene_id.unique())
    total_genes = len(genes)

    genes_per_slice = int(np.ceil(total_genes/n))

    slice_start = (i-1)*genes_per_slice
    if i == n:
        slice_end = total_genes + 1
    else:
        slice_end = i*genes_per_slice

    selected_genes = genes[slice_start:slice_end]

    out_prefix = f'results/torino/{species}/isoforms/tmp/annotated.isoforms'
    
    annot_out_tab = out_prefix + f'_{str(i)}.tab.gz'
    
    with gzip.open(annot_out_tab, 'wb') as fh:
        write_first_line = True
        counts = 1
        
        for gene in selected_genes:

            # try:
            
            snmf_bed = snmf_exons.loc[snmf_exons.gene_id == gene]
            gencode_bed = gencode_exons.loc[gencode_exons.gene_name == gene]
            print(gene)
            try:
                annot = annotate_transcripts_with_ref(snmf_bed, gencode_bed, species)
                if write_first_line:
                    first_line = ('\t'.join(list(annot.columns)) + '\n').encode()
                    fh.write(first_line)
                    write_first_line = False
                for idx, row in annot.iterrows():
                    line = ('\t'.join(list(row.astype(str))) + '\n').encode()
                    fh.write(line)
            except:
                continue            
