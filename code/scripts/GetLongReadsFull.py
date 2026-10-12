import pandas as pd
import pysam
import argparse 
import os

import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
# %run load_human.py
import warnings
# from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

from tqdm import tqdm 
from pybedtools import BedTool


import sys
sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
from torino_isoforms import *

def getSpliceSites(entry, min_len = 70, max_len = 100000):
    '''
    Takes a read and looks for coordinates of splice junctions using
    the CIGAR tuple. 
    '''
    cigarList = entry.cigartuples
    X=entry.pos
    cummulative=0
    coordinates=[]
    first_match = False
    for (code,seqlen) in cigarList:
        if code == 7:
            if not first_match:
                transcript_start = X+cummulative
                first_match = True
            cummulative+=seqlen
            transcript_end = X+cummulative
        elif (code == 0) or (code == 2) or (code == 8):
            cummulative+=seqlen 
            
        elif code == 3:
            # skip (junction)
            five_prime_ss = str(X+cummulative)
            cummulative+=seqlen  ### add the intron length
            three_prime_ss = str(X+cummulative+1) ### 3' exon start (prior exon splice-site + intron length)
            if seqlen > max_len: ### ignore very long introns
                return None
            elif seqlen > min_len: ### only report introns longer than min length
                coordinates.append([five_prime_ss,three_prime_ss])

    cigar_dict = {'juncs':coordinates, 'transcript':(str(transcript_start), str(transcript_end))}
    return cigar_dict


def getEntrySpliceSitesBed(bamf, entry, read_name):
    '''
    Create BED file rows for a read. Each entry corresponds to 
    a splice junction.
    '''
    chrom, strand = getChromAndStrand(bamf, entry)
    cigar_dict = getSpliceSites(entry)
    if cigar_dict is None:
        return None, None

    coord = cigar_dict['juncs']

    # if len(coord) == 0:
    #     bed_juncs = []
    
    

    transcript_coords = cigar_dict['transcript']
    if len(coord) < 1:
        bed_juncs = []
        bed_isoforms = [[chrom, transcript_coords[0], transcript_coords[1], read_name + ':' + str(1), entry.qname, strand]]
        

    else:
        bed_isoforms = [[chrom, transcript_coords[0], coord[0][0], read_name + ':' + str(1), entry.qname, strand]]
        bed_juncs = [[chrom, coord[i][0], coord[i][1], 
                 read_name + ':' + str(i+1), entry.qname, strand] for i in range(len(coord))]
        
        if len(coord) >= 2:
            for i in range(len(coord)-1):
                exon_row = [chrom, coord[i][1], coord[i+1][0], read_name + ':' + str(i+2), entry.qname, strand]
                bed_isoforms.append(exon_row)
    
        last_exon = [chrom, coord[-1][1], transcript_coords[1], read_name + ':' + str(len(coord)+1), entry.qname, strand]
        bed_isoforms.append(last_exon)

    # print(bed_juncs)
    # bed_juncs = [str(x) for x in bed_juncs]
    # bed_isoforms = [str(x) for x in bed_isoforms]
    
    return bed_juncs, bed_isoforms

def bed_rows_to_str(bed_rows):
    '''
    BED rows list to str
    '''
    bed_rows_str = '\n'.join(['\t'.join(x) for x in bed_rows]) + '\n'
    return bed_rows_str



def getChromAndStrand(bamf, entry):
    '''
    Get chromosome and strand information for read
    '''
    chrom = bamf.getrname( entry.rname )
    if entry.is_reverse:
        strand = '-'
    else:
        strand = '+'
    return chrom, strand

from tqdm import tqdm
def BamFileToJuncAndIsoformBed(bam_path, output):
    '''
    Open BAM file and write BED file with all junctions
    per read. Each read can have multiple junctions. Each
    junction can appear repeated in multiple reads.
    '''
    bamf = pysam.Samfile(bam_path, "rb" )
    autosomes = ['chr' + str(i) for i in range(1, 23)]
    juncs_file = f'{output}.juncs.bed'
    isoforms_file = f'{output}.isoforms.bed'
    with open(juncs_file, 'w') as fh_juncs:
        with open(isoforms_file, 'w') as fh_isoforms:
            read_counts = 1
            for entry in tqdm(bamf.fetch(), position=0, leave=True):
                chrom = bamf.getrname( entry.rname )
                if chrom not in autosomes:
                    continue
                # if 'N' in entry.cigarstring:
                read_name = 'LongRead.' + str(read_counts)
                bed_juncs, bed_isoforms = getEntrySpliceSitesBed(bamf, entry, read_name)
                
                if not bed_juncs:
                    continue
                junc_bed_str = bed_rows_to_str(bed_juncs)
                # print(bed_isoforms)
                isoforms_bed_str = bed_rows_to_str(bed_isoforms)
                fh_juncs.write(junc_bed_str)
                fh_isoforms.write(isoforms_bed_str)
                read_counts += 1


def transcript_subset(bed1, bed2, strict = False):
    """
    Return True if transcript_bed2 is a subset of transcript_bed1
    according to:
      - intron chain containment
      - exon overlap/length rules
      - strand-aware 5'/3' tolerance (more lenient at 5' end)
    """
    bed1 = bed1.sort_values(["chrom", "start"]).reset_index(drop=True)
    bed2 = bed2.sort_values(["chrom", "start"]).reset_index(drop=True)
    # print('hola')
    # basic checks
    if not (bed1.chrom.iloc[0] == bed2.chrom.iloc[0]):
        return False
    
    strand = bed1.strand.iloc[0]
    # print(strand)
    chain1 = get_intron_chain(bed1)
    chain2 = get_intron_chain(bed2)
    if not chain2:
        if len(chain1) >= 1:
            if strict:
                return exon_compatible_read_strict(bed1, bed2)
            else:
                return exon_compatible_read(bed1, bed2)
        else:
            return False

    # print(chain1)
    # print(chain2)
    matches = find_subchain_indices(chain1, chain2)
    # print(matches)
    if not matches:
        return False
    
    # helper for exon tolerance with variable parameters
    def exon_tolerable(exon1, exon2, mode):
        s1, e1 = exon1
        s2, e2 = exon2
        len1, len2 = e1 - s1, e2 - s2
        diff_start = abs(s1 - s2)
        diff_end = abs(e1 - e2)

        if mode == "strict":
            return s1 <= s2 and e2 <= e1
        elif mode == "loose":
            return (diff_start <= 200 and diff_end <= 200 and
                    (len1 <= len2 * 1.25 or len2 <= len1 * 1.25))
        elif mode == "lax":
            return (diff_start <= 500 and diff_end <= 500 and
                    (len1 <= len2 * 1.5 or len2 <= len1 * 1.5))
        else:
            raise ValueError("mode must be 'strict', 'loose', or 'lax'")
    
    for i in matches:
        first1_idx, last1_idx = i, i + len(chain2)
        first_exon1 = (bed1.iloc[first1_idx].start, bed1.iloc[first1_idx].end)
        last_exon1  = (bed1.iloc[last1_idx].start,  bed1.iloc[last1_idx].end)
        
        first_exon2 = (bed2.iloc[0].start, bed2.iloc[0].end)
        last_exon2  = (bed2.iloc[-1].start, bed2.iloc[-1].end)
        
        # Determine if subchain starts/ends at transcript boundaries
        first_at_start = (i == 0)
        last_at_end = (last1_idx == len(bed1) - 1)
        
        # Figure out which side is 5' and which is 3' depending on strand
        if strand == "+":
            five_prime = "first"
            three_prime = "last"
        else:  # reverse for negative strand
            five_prime = "last"
            three_prime = "first"

        # print(five_prime)
        # print(first_at_start)
        # print(last_at_end)
        # print('888')
        # pick appropriate mode for first and last exon
        if (five_prime == "first" and first_at_start): # first exon in + strand match
            first_mode = "lax"
        elif (three_prime == "first" and first_at_start): # last exon in - strand match
            first_mode = "loose"
        else: # internal exon
            first_mode = "strict"
        
        if (five_prime == "last" and last_at_end): # 5' end at the - strand
            last_mode = "lax"
        elif (three_prime == "last" and last_at_end): # last exon in + strand match
            last_mode = "loose"
        else:
            last_mode = "strict"

        # apply checks
        first_ok = exon_tolerable(first_exon1, first_exon2, first_mode)
        last_ok = exon_tolerable(last_exon1, last_exon2, last_mode)
        # print(last_ok)
        if first_ok and last_ok:
            return True
    
    return False


import pysam

import pysam
def iso_list_to_factors_dict(isoform_bed, gene_id = 'gene_1'):
    iso_bed = []
    isoform_cols = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon_id', 'notes']

    for iso in isoform_bed:
        
        chrom = iso[0]
        start = iso[1]
        end = iso[2]
        exon_id = iso[3]
        transcript_id = iso[4]
        strand = iso[5]
        factors = gene_id
        notes = '.'

        out_row = [chrom, int(start), int(end), gene_id, transcript_id, strand, transcript_id, exon_id, notes]
        iso_bed.append(out_row)
            
    iso_bed = pd.DataFrame(iso_bed)
    iso_bed.columns = isoform_cols

    return iso_bed, transcript_id



def get_long_reads(bam_path, coords):
    chrom, base_pairs, strand = coords.split(':')
    start, end = base_pairs.split('-')
    start = int(start)
    end = int(end)

    isoform_cols = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon_id', 'notes']

    
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    
    autosomes = ['chr' + str(i) for i in range(1, 23)]
    
    # query by region (chromosome, start, end)
    junc_bed = []
    isoform_bed = []
    
    read_counts = 1
    
    for entry in bamfile.fetch(chrom, start, end):
        
        chrom = bamfile.getrname( entry.rname )
        if chrom not in autosomes:
            continue
        # if 'N' in entry.cigarstring:
        read_name = 'LongRead.' + str(read_counts)
        bed_juncs, bed_isoforms = getEntrySpliceSitesBed(bamfile, entry, read_name)
        
        # if not bed_juncs:
        #     continue

        if bed_isoforms is not None:
            junc_bed.append(bed_juncs)
            isoform_bed.append(bed_isoforms)
            read_counts += 1
    
    bamfile.close()

    iso_dict = {}
    for iso in isoform_bed:
        iso_bed, transcript_id = iso_list_to_factors_dict(iso)
        iso_dict.update({transcript_id:iso_bed})

    return iso_dict #long_read_beds


def get_reads_per_isoform(isoform_bed, long_read_beds, strand):
    reads = 0
    for av in long_read_beds:
        if is_same_isoform(isoform_bed, av, strand):
            reads += len(av.notes.iloc[0])

    return reads

def is_intron_chain(isoform_1, isoform_2):
    intron_chain_1 = get_intron_chain(isoform_1)
    intron_chain_2 = get_intron_chain(isoform_2)

    if (len(intron_chain_1) == 0) or (len(intron_chain_2) == 0):
        return False
    
    same_chain = is_same_chain(intron_chain_1, intron_chain_2)

    return same_chain

def get_gene_coords(isoform_bed, gene):
    start = str(isoform_bed.loc[isoform_bed.gene_id == gene].start.min() - 1000)
    end = str(isoform_bed.loc[isoform_bed.gene_id == gene].end.max() + 1000)

    strand = isoform_bed.loc[isoform_bed.gene_id == gene].strand.iloc[0]
    chrom = isoform_bed.loc[isoform_bed.gene_id == gene].chrom.iloc[0]
    coords = chrom +':' + start + '-' + end + ':' + strand
    return coords, strand

def get_reads_per_gene_isoforms(isoform_bed, gene, bam_path):
    isoform_bed = isoform_bed.loc[isoform_bed.gene_id == gene]
    coords, strand = get_gene_coords(isoform_bed, gene)
    print(coords)

    long_read_beds = get_long_reads(bam_path, coords)

    print('done!')

    isoform_list = sorted(list(isoform_bed.transcript_id.unique()))

    print(len(long_read_beds.keys()))

    for isoform in isoform_list: 

        print(isoform)
        isoform_ = isoform_bed.loc[isoform_bed.transcript_id == isoform]
        # print(isoform_.shape)
        print(get_reads_per_isoform(isoform_, long_read_beds, strand))

def get_reads_per_isoform(isoform_bed, long_read_beds, strand):
    reads = 0
    for av in long_read_beds:
        if transcript_subset(isoform_bed, av):
            reads += len(av.notes.iloc[0])

    return reads


from collections import defaultdict

def get_reads_per_gene_isoforms_v2(isoform_bed, gene, bam_path):
    """
    Count reads per isoform with hierarchy:
      1. exact matches (is_same_isoform)
      2. unique subset matches (transcript_subset)
      3. ambiguous / unmatched categories
    """
    isoform_bed = isoform_bed.loc[isoform_bed.gene_id == gene]
    coords, strand = get_gene_coords(isoform_bed, gene)
    long_read_beds = get_long_reads(bam_path, coords)
    
    isoform_list = sorted(isoform_bed.transcript_id.unique())
    isoform_dict = {iso: isoform_bed.loc[isoform_bed.transcript_id == iso]
                    for iso in isoform_list}
    
    # counters
    isoform_counts = defaultdict(int)
    ambiguous_reads = 0
    unmatched_reads = 0
    
    # Work on a copy so we can modify safely
    remaining_reads = list(long_read_beds)
    
    # === 1. Exact matches ===
    to_remove = []
    for av in remaining_reads:
        matched_iso = None
        for iso, iso_bed in isoform_dict.items():
            if is_same_isoform(iso_bed, av, strand):
                matched_iso = iso
                break  # exact matches are unique
        if matched_iso:
            isoform_counts[matched_iso] += len(av.notes.iloc[0])
            to_remove.append(av)
    
    # remove matched reads
    remaining_reads = [av for av in remaining_reads if all(av is not tr for tr in to_remove)]
    #remaining_reads = [av for av in remaining_reads if av not in to_remove]
    
    # === 2. Subset matches ===
    for av in remaining_reads:
        matching_isos = []
        for iso, iso_bed in isoform_dict.items():
            if transcript_subset(iso_bed, av):
                matching_isos.append(iso)
        
        if len(matching_isos) == 1:
            isoform_counts[matching_isos[0]] += len(av.notes.iloc[0])
        elif len(matching_isos) > 1:
            ambiguous_reads += len(av.notes.iloc[0])
        else:
            unmatched_reads += len(av.notes.iloc[0])
    
    # === 3. Report summary ===
    print(f"\nResults for gene {gene}")
    for iso in isoform_list:
        print(f"  {iso}: {isoform_counts[iso]} reads")
    print(f"  Ambiguous: {ambiguous_reads}")
    print(f"  Unmatched: {unmatched_reads}")
    
    return {
        "gene": gene,
        "isoform_counts": dict(isoform_counts),
        "ambiguous": ambiguous_reads,
        "unmatched": unmatched_reads,
    }


def find_subchain_indices(chain1, chain2):
    """Return all start indices in chain1 where chain2 matches consecutively."""
    matches = []
    for i in range(len(chain1) - len(chain2) + 1):
        if chain1[i:i+len(chain2)] == chain2:
            matches.append(i)
    return matches


def exon_subset_or_tolerable(exon1, exon2, loose=False):
    """
    exon1 = (start1, end1) from file1
    exon2 = (start2, end2) from file2
    
    loose=False  → exon2 must be subset of exon1
    loose=True   → exon2 and exon1 may differ up to 200bp or 25% in length
    """
    s1, e1 = exon1
    s2, e2 = exon2
    
    if not loose:
        # strict subset: exon2 within exon1
        return s1 <= s2 and e2 <= e1
    
    # loose tolerance
    len1, len2 = e1 - s1, e2 - s2
    diff_start = abs(s1 - s2)
    diff_end = abs(e1 - e2)
    return (
        diff_start <= 200 and diff_end <= 200 and
        (len1 <= len2 * 1.25 or len2 <= len1 * 1.25)
    )

from collections import defaultdict

def get_reads_per_gene_isoforms_v3(isoform_bed, gene, bam_path, coords, strand, print_report = False, strict = False):
    """
    Count reads per isoform with hierarchy:
      1. exact matches (is_same_isoform)
      2. unique subset matches (transcript_subset)
      3. unique intron-chain matches (is_intron_chain)
      4. ambiguous / unmatched categories
    """
    isoform_bed = isoform_bed.loc[isoform_bed.gene_id == gene]
    
    
        
    long_read_beds = get_long_reads(bam_path, coords)
    
    isoform_list = sorted(isoform_bed.transcript_id.unique())
    isoform_dict = {iso: isoform_bed.loc[isoform_bed.transcript_id == iso]
                    for iso in isoform_list}

    isoform_counts = {}#defaultdict(int)
    for iso in isoform_list:
        isoform_counts.update({iso:0})
    ambiguous_reads = 0
    unmatched_reads = 0
    
    remaining_reads = list(long_read_beds.keys())
    
    # === 1. Exact matches ===
    to_remove = []
    for read in remaining_reads:
        av = long_read_beds[read]
        matched_iso = None
        for iso, iso_bed in isoform_dict.items():
            if is_same_isoform(iso_bed, av, strand):
                matched_iso = iso
                break  # exact matches are unique
        if matched_iso:
            isoform_counts[matched_iso] += 1 #len(av.notes.iloc[0])
            to_remove.append(read)
        # else:
        #     unmatched_reads += 1

    if strict:
        unmatched_reads = len(remaining_reads) - len(to_remove)
        # unmatched_read_ids = 
        return {
            "gene": gene,
            "isoform_counts": dict(isoform_counts),
            "ambiguous": ambiguous_reads,
            "unmatched": unmatched_reads
            # "unmatched_ids":unmatched_read_ids
        }

    else:
        
        # Remove exact matches (by identity)
        remaining_reads = [read for read in remaining_reads if all(read is not tr for tr in to_remove)]
        
        # === 2. Subset matches ===
        unmatched_after_subset = []  # store those that fail subset match
        for read in remaining_reads:
            av = long_read_beds[read]
            matching_isos = []
            # for iso, iso_bed in isoform_dict.items():
            #     if False:#transcript_subset(iso_bed, av, strict = strict):
            #         matching_isos.append(iso)
            
            # if False:#len(matching_isos) == 1:
            #     isoform_counts[matching_isos[0]] += 1 #len(av.notes.iloc[0])
            # elif False:#len(matching_isos) > 1:
            #     ambiguous_reads += 1 #len(av.notes.iloc[0])
            # else:
            unmatched_after_subset.append(av)  # only these go to next step
        
        # === 3. Intron chain matches (only for previously unmatched reads) ===
        unmatched_read_ids = []
        for av in unmatched_after_subset:
            matching_isos = []
            for iso, iso_bed in isoform_dict.items():
                if is_intron_chain(iso_bed, av):
                    matching_isos.append(iso)
            
            if len(matching_isos) == 1:
                isoform_counts[matching_isos[0]] += 1 #len(av.notes.iloc[0])
            elif len(matching_isos) > 1:
                ambiguous_reads += 1 #len(av.notes.iloc[0])
            else:
                unmatched_reads += 1 #len(av.notes.iloc[0])
                read_id = av.transcript_id.iloc[0]
                unmatched_read_ids.append(read_id)
        
        # === 4. Report summary ===
        if print_report:
            print(f"\nResults for gene {gene}")
            for iso in isoform_list:
                print(f"  {iso}: {isoform_counts[iso]} reads")
            print(f"  Ambiguous: {ambiguous_reads}")
            print(f"  Unmatched: {unmatched_reads}")
        
        return {
            "gene": gene,
            "isoform_counts": dict(isoform_counts),
            "ambiguous": ambiguous_reads,
            "unmatched": unmatched_reads,
            "unmatched_ids":unmatched_read_ids
        }#, coords


def get_long_reads_filtered(bam_path, coords, allowed_reads=None):
    chrom, base_pairs, strand = coords.split(':')
    start, end = map(int, base_pairs.split('-'))

    isoform_cols = ['chrom','start','end','gene_id','transcript_id',
                    'strand','factors','exon_id','notes']

    bamfile = pysam.AlignmentFile(bam_path, "rb")
    autosomes = ['chr' + str(i) for i in range(1, 23)]

    isoform_bed = []
    junc_bed = []

    read_counts = 1

    for entry in bamfile.fetch(chrom, start, end):

        # filter by autosome
        chrom = bamfile.getrname(entry.rname)
        if chrom not in autosomes:
            continue

        # only keep spliced reads
        # if 'N' not in entry.cigarstring:
        #     continue

        # restrict to unmatched reads if provided
        if allowed_reads is not None:
            if entry.qname not in allowed_reads:
                continue

        read_name = entry.qname   # KEEP ORIGINAL QNAME!

        bed_juncs, bed_isoforms = getEntrySpliceSitesBed(bamfile, entry, read_name)
        # if not bed_juncs:
        #     continue

        if bed_isoforms is not None:

            junc_bed.append(bed_juncs)
            isoform_bed.append(bed_isoforms)
    
            read_counts += 1

    bamfile.close()

    iso_dict = {}
    for iso in isoform_bed:
        iso_bed, transcript_id = iso_list_to_factors_dict(iso)
        iso_dict[transcript_id] = iso_bed

    if len(iso_dict) < 200:
        long_read_beds = factors_to_isoforms(
            iso_dict,
            merge_factors_into_isoforms(iso_dict, strand)
        )
    else:
        print('what')
        long_read_beds = iso_dict

    return long_read_beds

def exon_compatible_read_strict(bed1, bed2):
    # print(bed2.shape)
    start_bed2 = bed2.start.iloc[0]
    end_bed2 = bed2.end.iloc[0]

    len2 = end_bed2 - start_bed2

    for idx, row in bed1.iterrows():
        len1 = row.end - row.start
        #if (row.start <= start_bed2) and (row.end >= end_bed2):# and (len2 >= (len2 * 0.75)):
        if (row.start <= start_bed2) and (row.end >= end_bed2) and (len2 >= (len1 * 0.75)):
            return True

    # if any((bed1.start >= start_bed2) & (bed1.end <= end_bed2)):
    #     return True

    # check 5' end
    diff_start = np.abs(start_bed2 - bed1.start.iloc[0])
    len1 = bed1.end.iloc[0] - bed1.start.iloc[0]

    if (diff_start <= 200) and (end_bed2 <= bed1.end.iloc[0]) and ((len1 <= (len2 * 1.25)) and (len2 <= (len1 * 1.25))):
        return True


    diff_end = np.abs(end_bed2 - bed1.end.iloc[-1])
    len1 = bed1.end.iloc[-1] - bed1.start.iloc[-1]

    if (diff_end <= 200) and (start_bed2 >= bed1.start.iloc[-1]) and ((len1 <= (len2 * 1.25)) and (len2 <= (len1 * 1.25))):
        return True

    return False
        

def exon_compatible_read(bed1, bed2):
    # print(bed2.shape)
    start_bed2 = bed2.start.iloc[0]
    end_bed2 = bed2.end.iloc[0]

    len2 = end_bed2 - start_bed2

    for idx, row in bed1.iterrows():
        if (row.start <= start_bed2) and (row.end >= end_bed2):
            return True

    # if any((bed1.start >= start_bed2) & (bed1.end <= end_bed2)):
    #     return True

    # check 5' end
    diff_start = np.abs(start_bed2 - bed1.start.iloc[0])
    len1 = bed1.end.iloc[0] - bed1.start.iloc[0]

    if (diff_start <= 200) and (end_bed2 <= bed1.end.iloc[0]) and ((len1 <= (len2 * 1.25)) or (len2 <= (len1 * 1.25))):
        return True


    diff_end = np.abs(end_bed2 - bed1.end.iloc[-1])
    len1 = bed1.end.iloc[-1] - bed1.start.iloc[-1]

    if (diff_end <= 200) and (start_bed2 >= bed1.start.iloc[-1]) and ((len1 <= (len2 * 1.25)) or (len2 <= (len1 * 1.25))):
        return True

    return False
    
    

def summarize_isoform_counts(res, gene):
    rows = []

    # for gene in genes:
    # res = get_reads_per_gene_isoforms_v3(isoform_bed, gene, bam_path)

    total = (
        sum(res["isoform_counts"].values()) +
        res["ambiguous"] +
        res["unmatched"]
    )

    # per-isoform rows
    for iso, ct in res["isoform_counts"].items():
        frac = ct / total if total > 0 else 0
        rows.append([gene, iso, ct, frac])

    # ambiguous
    if total > 0:
        
        rows.append([gene, "ambiguous", res["ambiguous"], res["ambiguous"] / total])
    
        # unmatched
        rows.append([gene, "unmatched", res["unmatched"], res["unmatched"] / total])

    else:
        rows.append([gene, "ambiguous", res["ambiguous"], 0])
    
        # unmatched
        rows.append([gene, "unmatched", res["unmatched"], 0])

    df = pd.DataFrame(rows, columns=["gene", "isoform", "counts", "fraction"])
    return df


def get_long_read_counts_per_gene(isoform_bed, gene, bam_file):
    x, coords = get_reads_per_gene_isoforms_v3(isoform_bed, gene, bam_file)
    isoform_counts = summarize_isoform_counts(x, gene)
    unmatched_isoforms = get_long_reads_filtered(bam_file, coords, allowed_reads = x['unmatched_ids'])
    total_counts = isoform_counts.counts.sum()
    if (total_counts > 0):
        missing_isoforms = get_missing_isoforms(unmatched_isoforms, total_counts, gene=gene)
    else:
        missing_isoforms = None

    return isoform_counts, missing_isoforms


def get_long_read_counts_per_gene_lean(isoform_bed, gene, bam_file, coords, strand, strict = False):
    x = get_reads_per_gene_isoforms_v3(isoform_bed, gene, bam_file, coords=coords, strand = strand, strict = strict)
    isoform_counts = summarize_isoform_counts(x, gene)

    return isoform_counts#, missing_isoforms


def get_missing_isoforms(unmatched_isoforms_list, total_counts, gene='gene_id'):
    isoform_list = []
    i = 1
    for unmatched_isoform in unmatched_isoforms_list:
        counts = len(unmatched_isoform.notes.iloc[0])
        isoform_id = f'{gene}.missing_isoform_{str(i)}'
        row = [gene, isoform_id, counts]
        i += 1
        isoform_list.append(row)

    isoform_df = pd.DataFrame(isoform_list)
    isoform_df.columns = ['gene', 'isoform', 'counts']
    # if total_counts >= 1:
    isoform_df['fraction'] = isoform_df.counts/total_counts
    # else:
        
    return isoform_df


def get_fractions_per_sample_list(isoform_bed, gene, bam_file_list):


    isoform_table = None
    missing_isoforms_table = None
    
    for bam_file in bam_file_list:
        bam_file_path = '../code/resources/GTEx/ONT/bam/' + bam_file
        bam_sample = bam_file.split('.')[0].split('_rep')[0].split('_')[-1]#bam_file.split('_')[-1].split('.')[0]
        isoform_counts, missing_isoforms = get_long_read_counts_per_gene(isoform_bed, gene, bam_file_path)
        isoform_counts.columns = ['gene', 'isoform', f'{bam_sample}.counts', f'{bam_sample}.fractions']

        if isoform_table is None:
            isoform_table = isoform_counts
            missing_isoform_table = missing_isoforms
        else:
            isoform_table = isoform_table.merge(isoform_counts, on = ['gene', 'isoform'], how='outer')
            missing_isoforms_table = pd.concat([missing_isoforms_table, missing_isoforms])

    return isoform_table, missing_isoforms_table



def get_fractions_per_sample_list_lean(isoform_bed, gene, bam_file_list, coords, strand, strict = False):


    isoform_table = None
    # missing_isoforms_table = None
    
    for bam_file in bam_file_list:
        bam_file_path = '../code/resources/GTEx/ONT/bam/' + bam_file
        bam_sample = bam_file.split('.')[0].split('_rep')[0].split('_')[-1]#bam_file.split('_')[-1].split('.')[0]
        isoform_counts = get_long_read_counts_per_gene_lean(isoform_bed, gene, bam_file_path, coords, strand, strict = strict)
        isoform_counts.columns = ['gene', 'isoform', f'{bam_sample}.counts', f'{bam_sample}.fractions']

        if isoform_table is None:
            isoform_table = isoform_counts
            # missing_isoform_table = missing_isoforms
        else:
            isoform_table = isoform_table.merge(isoform_counts, on = ['gene', 'isoform'], how='outer')
            # missing_isoforms_table = pd.concat([missing_isoforms_table, missing_isoforms])

    return isoform_table#, missing_isoforms_table




if __name__ == '__main__':
    isoform_bed = pd.read_csv('../code/results/torino/Human/isoforms/isoforms.bed.gz', sep='\t',
                         names = ['chrom', 'start', 'end', 'gene_id', 'transcript_id', 'strand', 'factors', 'exon_id', 'notes'])

    autosomes = [f'chr{str(x)}' for x in range(1, 23)]

    isoform_bed = isoform_bed.loc[isoform_bed.chrom.isin(autosomes)]


    gtf_bed = pd.read_csv('../code/resources/annotation/Human/bed/exons.all.bed.gz', sep='\t')
    gtf_bed = gtf_bed.loc[gtf_bed.gene_type == 'protein_coding']
    gtf_bed.columns = ['chrom', 'start', 'end', 'gene_ensembl', 'gene_id', 'strand', 'exon_id', 'transcript_id', 'factors']
    gtf_bed['notes'] = ['.']*gtf_bed.shape[0]

    arguments = sys.argv
    i = int(arguments[1])
    n = int(arguments[2])
    # correction_tag = str(arguments[5])
    

    # if i >= 1000:
    #     i_ = 1000
    # else:
    #     i_ = i

    
    genes = sorted(isoform_bed.gene_id.unique())
    total_genes = len(genes)

    genes_per_slice = int(np.ceil(total_genes/n))

    slice_start = (i-1)*genes_per_slice
    if i == n:
        slice_end = total_genes + 1
    else:
        slice_end = i*genes_per_slice

    selected_genes = genes[slice_start:slice_end]


    print(len(selected_genes))


    bam_file_list = [x for x in os.listdir('../code/resources/GTEx/ONT/bam/') if x.endswith('.bam')]

    coords, strand = get_gene_coords(isoform_bed, 'SRSF3')
    
    isoform_table = get_fractions_per_sample_list_lean(isoform_bed, 'SRSF3', bam_file_list, coords, strand)


    # if i >= 1000:
    #     total_genes = len(selected_genes)

    #     genes_per_slice = 10 #int(total_genes/n)
        
    #     i_ = i - 1000
    #     slice_start = (i_)*genes_per_slice
    #     n_ = 1100
    #     if i == n_:
    #         slice_end = total_genes + 1
    #     else:
    #         slice_end = (i_+1)*genes_per_slice
        
    #     selected_genes = selected_genes[slice_start:slice_end]
        

    # gene_list = selected_genes


    gene_list = selected_genes
    import gzip
    with gzip.open(f'results/LongReads/tmp/full_long_read_torino.counts.{str(i)}.tab.gz', 'wb') as fh_torino:
        with gzip.open(f'results/LongReads/tmp/full_long_read_gencode.counts.{str(i)}.tab.gz', 'wb') as fh_gencode:
            with gzip.open(f'results/LongReads/tmp/full_long_read_torino_strict.counts.{str(i)}.tab.gz', 'wb') as fh_torino_strict:
                with gzip.open(f'results/LongReads/tmp/full_long_read_gencode_strict.counts.{str(i)}.tab.gz', 'wb') as fh_gencode_strict:
                    # with gzip.open(f'results/LongReads/tmp/long_read.missing.{str(i)}.tab.gz', 'wb') as fh2:
                    fh_header = ('\t'.join(isoform_table.columns) + '\n').encode()
                    fh_torino.write(fh_header)
                    fh_gencode.write(fh_header)
                    fh_torino_strict.write(fh_header)
                    fh_gencode_strict.write(fh_header)
            
                    # fh2_header = ('\t'.join(missing_isoforms_table.columns) + '\n').encode()
                    # fh2.write(fh2_header)
            
                    for gene in gene_list:
            
                        coords, strand = get_gene_coords(isoform_bed, gene)
                        
                        print(gene)
                        gene_length = isoform_bed.loc[isoform_bed.gene_id == gene].end.max() - isoform_bed.loc[isoform_bed.gene_id == gene].start.min()
                        # if gene_length > 500000:
                        #     continue
                        # try:
                        isoform_table_torino = get_fractions_per_sample_list_lean(isoform_bed, gene, bam_file_list, coords, strand, strict = False)
                        # isoform_table, missing_isoforms_table = get_fractions_per_sample_list(isoform_bed, gene, bam_file_list)
                        for idx, row in isoform_table_torino.astype(str).iterrows():
                            row_to_write = ('\t'.join(list(row)) + '\n').encode()
                            fh_torino.write(row_to_write)
                    # except:
                    #     print('failed Torino')

                    # try:

                        isoform_table_torino = get_fractions_per_sample_list_lean(isoform_bed, gene, bam_file_list, coords, strand, strict = True)
                        # isoform_table, missing_isoforms_table = get_fractions_per_sample_list(isoform_bed, gene, bam_file_list)
                        for idx, row in isoform_table_torino.astype(str).iterrows():
                            row_to_write = ('\t'.join(list(row)) + '\n').encode()
                            fh_torino_strict.write(row_to_write)
                    # except:
                    #     print('failed Torino strict')

                    # try:
        
                        isoform_table_gencode = get_fractions_per_sample_list_lean(gtf_bed, gene, bam_file_list, coords, strand, strict = False)
                        # isoform_table, missing_isoforms_table = get_fractions_per_sample_list(isoform_bed, gene, bam_file_list)
                        for idx, row in isoform_table_gencode.astype(str).iterrows():
                            row_to_write = ('\t'.join(list(row)) + '\n').encode()
                            fh_gencode.write(row_to_write)

                    # except:
                    #     print('failed Gencode')

                    # try:

                        isoform_table_gencode = get_fractions_per_sample_list_lean(gtf_bed, gene, bam_file_list, coords, strand, strict = True)
                        # isoform_table, missing_isoforms_table = get_fractions_per_sample_list(isoform_bed, gene, bam_file_list)
                        for idx, row in isoform_table_gencode.astype(str).iterrows():
                            row_to_write = ('\t'.join(list(row)) + '\n').encode()
                            fh_gencode_strict.write(row_to_write)
        
                            
                    # except:
                    #     print('failed Gencode strict')
