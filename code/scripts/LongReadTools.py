"""
LongReadTools.py
================
Unified toolkit for long-read isoform quantification and visualization.

Merges functionality from:
  - GetLongReadsTable.py (batch quantification script)
  - Fig2_LongReads_Definitive.ipynb (interactive analysis notebook)

Requires torino_isoforms (is_same_isoform, get_intron_chain, is_same_chain,
merge_factors_into_isoforms, factors_to_isoforms).
"""

import pandas as pd
import pysam
import numpy as np
import warnings
import gzip

from collections import defaultdict

warnings.filterwarnings("ignore")

import sys
sys.path.append('/project2/yangili1/cfbuenabadn/torino/scripts')
from torino_isoforms import (
    is_same_isoform,
    get_intron_chain,
    is_same_chain,
    merge_factors_into_isoforms,
    factors_to_isoforms,
    same_terminal_exons,
    get_exon_chain,
)


# =============================================================================
# BAM reading / BED construction
# =============================================================================

def getSpliceSites(entry, min_len=70, max_len=100000):
    """Extract splice junction coordinates from a read's CIGAR tuples."""
    cigarList = entry.cigartuples
    X = entry.pos
    cummulative = 0
    coordinates = []
    first_match = False
    for (code, seqlen) in cigarList:
        if code == 7:
            if not first_match:
                transcript_start = X + cummulative
                first_match = True
            cummulative += seqlen
            transcript_end = X + cummulative
        elif (code == 0) or (code == 2) or (code == 8):
            cummulative += seqlen
        elif code == 3:
            five_prime_ss = str(X + cummulative)
            cummulative += seqlen
            three_prime_ss = str(X + cummulative + 1)
            if seqlen > max_len:
                return None
            elif seqlen > min_len:
                coordinates.append([five_prime_ss, three_prime_ss])

    cigar_dict = {
        'juncs': coordinates,
        'transcript': (str(transcript_start), str(transcript_end)),
    }
    return cigar_dict


def getChromAndStrand(bamf, entry):
    """Get chromosome and strand for a BAM entry."""
    chrom = bamf.getrname(entry.rname)
    strand = '-' if entry.is_reverse else '+'
    return chrom, strand


def getEntrySpliceSitesBed(bamf, entry, read_name):
    """Build BED rows (junctions + exons) for a single read."""
    chrom, strand = getChromAndStrand(bamf, entry)
    cigar_dict = getSpliceSites(entry)
    if cigar_dict is None:
        return None, None

    coord = cigar_dict['juncs']
    transcript_coords = cigar_dict['transcript']

    if len(coord) < 1:
        bed_juncs = []
        bed_isoforms = [[chrom, transcript_coords[0], transcript_coords[1],
                          read_name + ':1', entry.qname, strand]]
    else:
        bed_isoforms = [[chrom, transcript_coords[0], coord[0][0],
                          read_name + ':1', entry.qname, strand]]
        bed_juncs = [[chrom, coord[i][0], coord[i][1],
                       read_name + ':' + str(i + 1), entry.qname, strand]
                      for i in range(len(coord))]
        if len(coord) >= 2:
            for i in range(len(coord) - 1):
                exon_row = [chrom, coord[i][1], coord[i + 1][0],
                            read_name + ':' + str(i + 2), entry.qname, strand]
                bed_isoforms.append(exon_row)
        last_exon = [chrom, coord[-1][1], transcript_coords[1],
                     read_name + ':' + str(len(coord) + 1), entry.qname, strand]
        bed_isoforms.append(last_exon)

    return bed_juncs, bed_isoforms


def bed_rows_to_str(bed_rows):
    """Convert list-of-lists BED rows to a string."""
    return '\n'.join(['\t'.join(x) for x in bed_rows]) + '\n'


def iso_list_to_factors_dict(isoform_bed, gene_id='gene_1'):
    """Convert raw BED row lists into a DataFrame with standard columns."""
    isoform_cols = ['chrom', 'start', 'end', 'gene_id', 'transcript_id',
                    'strand', 'factors', 'exon_id', 'notes']
    iso_bed = []
    for iso in isoform_bed:
        chrom, start, end, exon_id, transcript_id, strand = iso[0], iso[1], iso[2], iso[3], iso[4], iso[5]
        out_row = [chrom, int(start), int(end), gene_id, transcript_id,
                   strand, transcript_id, exon_id, '.']
        iso_bed.append(out_row)
    iso_bed = pd.DataFrame(iso_bed, columns=isoform_cols)
    return iso_bed, transcript_id


# =============================================================================
# Read quality metrics
# =============================================================================

def get_read_quality(entry):
    """
    Extract alignment quality metrics from a BAM entry.

    Returns a dict with: read_length, aligned_fraction, mapping_quality,
    identity, n_mismatches, n_insertions, n_deletions.
    Returns None if the CIGAR is unavailable.
    """
    cigar = entry.cigartuples
    if cigar is None:
        return None

    matches = sum(l for c, l in cigar if c == 7)       # =
    mismatches = sum(l for c, l in cigar if c == 8)     # X
    m_bases = sum(l for c, l in cigar if c == 0)        # M
    insertions = sum(l for c, l in cigar if c == 1)     # I
    deletions = sum(l for c, l in cigar if c == 2)      # D
    soft_clips = sum(l for c, l in cigar if c == 4)     # S

    query_len = entry.query_length or 0
    if query_len == 0:
        return None

    aligned_len = query_len - soft_clips
    aligned_fraction = aligned_len / query_len

    # Identity: matches / (matches + mismatches + insertions + deletions)
    if matches > 0 or mismatches > 0:
        # =/X encoding (minimap2 --eqx)
        denom = matches + mismatches + insertions + deletions
        identity = matches / denom if denom > 0 else 0.0
        n_mismatches = mismatches
    elif m_bases > 0:
        # M encoding: use NM tag
        nm = entry.get_tag('NM') if entry.has_tag('NM') else 0
        n_mismatches = max(0, nm - insertions - deletions)
        inferred_matches = m_bases - n_mismatches
        denom = m_bases + insertions + deletions
        identity = inferred_matches / denom if denom > 0 else 0.0
    else:
        identity = 0.0
        n_mismatches = 0

    return {
        'read_length': query_len,
        'aligned_fraction': aligned_fraction,
        'mapping_quality': entry.mapping_quality,
        'identity': identity,
        'n_mismatches': n_mismatches,
        'n_insertions': insertions,
        'n_deletions': deletions,
    }


# =============================================================================
# BAM → isoform dict
# =============================================================================

def get_long_reads(bam_path, coords, min_read_length=200,
                   min_aligned_fraction=0.8, min_identity=0.9,
                   return_quality=False):
    """
    Fetch reads from a BAM file within *coords* and return a dict
    mapping read keys to exon-BED DataFrames.

    Reads shorter than *min_read_length*, with aligned fraction below
    *min_aligned_fraction*, or with identity below *min_identity* are
    excluded.  If *return_quality* is True, also returns a dict mapping
    read keys to quality-metric dicts.
    """
    chrom, base_pairs, strand = coords.split(':')
    start, end = map(int, base_pairs.split('-'))

    bamfile = pysam.AlignmentFile(bam_path, "rb")
    autosomes = ['chr' + str(i) for i in range(1, 23)]

    isoform_bed = []
    quality_dict = {}
    read_counts = 1

    for entry in bamfile.fetch(chrom, start, end):
        chrom_entry = bamfile.getrname(entry.rname)
        if chrom_entry not in autosomes:
            continue

        # Read quality filter
        qm = get_read_quality(entry)
        if qm is None:
            continue
        if qm['read_length'] < min_read_length:
            continue
        if qm['aligned_fraction'] < min_aligned_fraction:
            continue
        if qm['identity'] < min_identity:
            continue

        read_name = 'LongRead.' + str(read_counts)
        _, bed_isoforms = getEntrySpliceSitesBed(bamfile, entry, read_name)
        if bed_isoforms is not None:
            isoform_bed.append(bed_isoforms)
            if return_quality:
                quality_dict[entry.qname] = qm
            read_counts += 1

    bamfile.close()

    iso_dict = {}
    for iso in isoform_bed:
        iso_bed, transcript_id = iso_list_to_factors_dict(iso)
        iso_dict[transcript_id] = iso_bed

    if return_quality:
        return iso_dict, quality_dict
    return iso_dict


def get_long_reads_filtered(bam_path, coords, allowed_reads=None,
                            min_read_length=200, min_aligned_fraction=0.8,
                            min_identity=0.9):
    """
    Like get_long_reads, but optionally restricted to *allowed_reads*
    (a set/list of original qnames).  Reads are then merged into isoforms
    when the number of reads is manageable (<200).
    """
    chrom, base_pairs, strand = coords.split(':')
    start, end = map(int, base_pairs.split('-'))

    bamfile = pysam.AlignmentFile(bam_path, "rb")
    autosomes = ['chr' + str(i) for i in range(1, 23)]

    isoform_bed = []
    read_counts = 1

    for entry in bamfile.fetch(chrom, start, end):
        chrom_entry = bamfile.getrname(entry.rname)
        if chrom_entry not in autosomes:
            continue
        if allowed_reads is not None:
            if entry.qname not in allowed_reads:
                continue

        # Read quality filter
        qm = get_read_quality(entry)
        if qm is None:
            continue
        if qm['read_length'] < min_read_length:
            continue
        if qm['aligned_fraction'] < min_aligned_fraction:
            continue
        if qm['identity'] < min_identity:
            continue

        read_name = entry.qname
        _, bed_isoforms = getEntrySpliceSitesBed(bamfile, entry, read_name)
        if bed_isoforms is not None:
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
            merge_factors_into_isoforms(iso_dict, strand),
        )
    else:
        long_read_beds = iso_dict

    return long_read_beds


# =============================================================================
# Intron / exon matching helpers
# =============================================================================

def find_subchain_indices(chain1, chain2):
    """Return all start indices in *chain1* where *chain2* occurs."""
    matches = []
    for i in range(len(chain1) - len(chain2) + 1):
        if chain1[i:i + len(chain2)] == chain2:
            matches.append(i)
    return matches


def is_chain_match_tolerant(chain1, chain2, tolerance):
    """
    True if *chain1* and *chain2* have the same length and every intron
    pair satisfies abs(s1-s2) <= tolerance and abs(e1-e2) <= tolerance.
    """
    if len(chain1) != len(chain2):
        return False
    for (s1, e1), (s2, e2) in zip(chain1, chain2):
        if abs(s1 - s2) > tolerance or abs(e1 - e2) > tolerance:
            return False
    return True


def find_subchain_indices_tolerant(chain1, chain2, tolerance):
    """
    Like find_subchain_indices but allows per-coordinate tolerance.
    Returns list of start indices in *chain1* where *chain2* matches.
    """
    matches = []
    for i in range(len(chain1) - len(chain2) + 1):
        segment = chain1[i:i + len(chain2)]
        if is_chain_match_tolerant(segment, chain2, tolerance):
            matches.append(i)
    return matches


def _assign_reads_by_weights(pending_ambiguous, isoform_weights_fn):
    """
    Deterministically assign ambiguous reads to isoforms for plotting.

    Each group of ambiguous reads sharing the same set of matching isoforms
    is distributed using the largest-remainder method: compute fractional
    allocations, give each isoform floor(frac) reads, then distribute the
    remaining reads to isoforms with the largest fractional remainders.

    Parameters
    ----------
    pending_ambiguous : list of (read_key, [iso1, iso2, ...])
    isoform_weights_fn : callable(matching_isos) -> dict {iso: weight}
        Returns non-negative weights for each matching isoform.

    Returns
    -------
    dict {iso: [read_keys...]}  — reads assigned to each isoform
    """
    from collections import defaultdict
    assignments = defaultdict(list)

    # Group ambiguous reads by their matching-isoform set (order-independent)
    groups = defaultdict(list)
    for read, matching_isos in pending_ambiguous:
        key = tuple(sorted(matching_isos))
        groups[key].append(read)

    for iso_tuple, reads in groups.items():
        matching_isos = list(iso_tuple)
        weights = isoform_weights_fn(matching_isos)
        total_w = sum(weights[iso] for iso in matching_isos)

        if total_w == 0:
            # fallback to equal
            weights = {iso: 1.0 for iso in matching_isos}
            total_w = len(matching_isos)

        n = len(reads)
        # Fractional allocation
        fracs = {iso: n * weights[iso] / total_w for iso in matching_isos}
        floors = {iso: int(fracs[iso]) for iso in matching_isos}
        remainders = {iso: fracs[iso] - floors[iso] for iso in matching_isos}

        assigned = sum(floors.values())
        leftover = n - assigned

        # Distribute leftover by largest remainder
        by_remainder = sorted(matching_isos,
                              key=lambda iso: remainders[iso], reverse=True)
        final_counts = dict(floors)
        for i in range(leftover):
            final_counts[by_remainder[i]] += 1

        # Assign reads in order
        idx = 0
        for iso in matching_isos:
            for _ in range(final_counts[iso]):
                assignments[iso].append(reads[idx])
                idx += 1

    return dict(assignments)


def resolve_tolerance_ties(read_chain, candidate_isos, iso_chains):
    """
    Given a read's intron chain and multiple candidate isoforms that
    all match within tolerance, return a single winner or None (ambiguous).

    Rule (i): if exactly one candidate matches ALL introns perfectly
    (zero mismatch), it wins.
    Rule (ii): if no candidate is all-perfect, or multiple are,
    the read is ambiguous.
    """
    perfect = []
    for iso in candidate_isos:
        ref_chain = iso_chains[iso]
        if read_chain == ref_chain:
            perfect.append(iso)
    if len(perfect) == 1:
        return perfect[0]
    return None


def is_intron_chain(isoform_1, isoform_2, tolerance=0):
    """True if the two BEDs share the same full intron chain."""
    chain1 = get_intron_chain(isoform_1)
    chain2 = get_intron_chain(isoform_2)
    if (len(chain1) == 0) or (len(chain2) == 0):
        return False
    if tolerance == 0:
        return is_same_chain(chain1, chain2)
    return is_chain_match_tolerant(chain1, chain2, tolerance)


def exon_compatible_read(bed1, bed2, allow_short_reads=False):
    """
    Check whether an unspliced read (bed2, single exon) is compatible
    with one of the exons of a multi-exon isoform (bed1).

    allow_short_reads=False (default): the read must span >= 75% of
        the exon length.  Boundary checks use 'and' (both length
        ratios must hold).
    allow_short_reads=True: any contained read passes the containment
        check.  Boundary checks still use 'and'.
    """
    start_bed2 = bed2.start.iloc[0]
    end_bed2 = bed2.end.iloc[0]
    len2 = end_bed2 - start_bed2

    # Internal-exon containment
    for idx, row in bed1.iterrows():
        len1 = row.end - row.start
        if allow_short_reads:
            read_check = True
        else:
            read_check = (len2 >= (len1 * 0.75))
        if (row.start <= start_bed2) and (row.end >= end_bed2) and read_check:
            return True

    # 5' end tolerance (first exon)
    diff_start = np.abs(start_bed2 - bed1.start.iloc[0])
    len1 = bed1.end.iloc[0] - bed1.start.iloc[0]
    if ((diff_start <= 200) and (end_bed2 <= bed1.end.iloc[0])
            and (len1 <= (len2 * 1.25)) and (len2 <= (len1 * 1.25))):
        return True

    # 3' end tolerance (last exon)
    diff_end = np.abs(end_bed2 - bed1.end.iloc[-1])
    len1 = bed1.end.iloc[-1] - bed1.start.iloc[-1]
    if ((diff_end <= 200) and (start_bed2 >= bed1.start.iloc[-1])
            and (len1 <= (len2 * 1.25)) and (len2 <= (len1 * 1.25))):
        return True

    return False


def transcript_subset(bed1, bed2, allow_short_reads=False, intron_tolerance=0):
    """
    Return True if the read transcript (bed2) is a subset of the
    reference isoform (bed1), allowing for strand-aware tolerance
    at terminal exons.

    For unspliced reads matched against spliced isoforms, delegates
    to exon_compatible_read with *allow_short_reads*.

    intron_tolerance: max bp mismatch per intron coordinate (default 0).
    """
    bed1 = bed1.sort_values(["chrom", "start"]).reset_index(drop=True)
    bed2 = bed2.sort_values(["chrom", "start"]).reset_index(drop=True)

    if bed1.chrom.iloc[0] != bed2.chrom.iloc[0]:
        return False

    strand = bed1.strand.iloc[0]
    chain1 = get_intron_chain(bed1)
    chain2 = get_intron_chain(bed2)

    # Unspliced read vs spliced isoform
    if not chain2:
        if len(chain1) >= 1:
            return exon_compatible_read(bed1, bed2, allow_short_reads=allow_short_reads)
        else:
            return False

    if intron_tolerance > 0:
        matches = find_subchain_indices_tolerant(chain1, chain2, intron_tolerance)
    else:
        matches = find_subchain_indices(chain1, chain2)
    if not matches:
        return False

    def exon_tolerable(exon1, exon2, mode):
        s1, e1 = exon1
        s2, e2 = exon2
        len1, len2 = e1 - s1, e2 - s2
        diff_start = abs(s1 - s2)
        diff_end = abs(e1 - e2)
        if mode == "strict":
            return s1 <= s2 and e2 <= e1
        elif mode == "loose":
            return (diff_start <= 200 and diff_end <= 200
                    and (len1 <= len2 * 1.25 or len2 <= len1 * 1.25))
        elif mode == "lax":
            return (diff_start <= 500 and diff_end <= 500
                    and (len1 <= len2 * 1.5 or len2 <= len1 * 1.5))
        else:
            raise ValueError("mode must be 'strict', 'loose', or 'lax'")

    for i in matches:
        first1_idx = i
        last1_idx = i + len(chain2)
        first_exon1 = (bed1.iloc[first1_idx].start, bed1.iloc[first1_idx].end)
        last_exon1 = (bed1.iloc[last1_idx].start, bed1.iloc[last1_idx].end)
        first_exon2 = (bed2.iloc[0].start, bed2.iloc[0].end)
        last_exon2 = (bed2.iloc[-1].start, bed2.iloc[-1].end)

        first_at_start = (i == 0)
        last_at_end = (last1_idx == len(bed1) - 1)

        if strand == "+":
            five_prime, three_prime = "first", "last"
        else:
            five_prime, three_prime = "last", "first"

        if five_prime == "first" and first_at_start:
            first_mode = "lax"
        elif three_prime == "first" and first_at_start:
            first_mode = "loose"
        else:
            first_mode = "strict"

        if five_prime == "last" and last_at_end:
            last_mode = "lax"
        elif three_prime == "last" and last_at_end:
            last_mode = "loose"
        else:
            last_mode = "strict"

        first_ok = exon_tolerable(first_exon1, first_exon2, first_mode)
        last_ok = exon_tolerable(last_exon1, last_exon2, last_mode)
        if first_ok and last_ok:
            return True

    return False


# =============================================================================
# Gene coordinate helpers
# =============================================================================

def get_gene_coords(isoform_bed, gene):
    """Return 'chr:start-end:strand' and strand for *gene*."""
    sub = isoform_bed.loc[isoform_bed.gene_id == gene]
    start = str(sub.start.min() - 1000)
    end = str(sub.end.max() + 1000)
    strand = sub.strand.iloc[0]
    chrom = sub.chrom.iloc[0]
    coords = f'{chrom}:{start}-{end}:{strand}'
    return coords, strand


# =============================================================================
# Tier 4: partial match rescue helpers
# =============================================================================

def intron_precision_recall(read_chain, iso_chain, tolerance=0):
    """
    Compute intron-level precision and recall between a read and an isoform.

    Parameters
    ----------
    read_chain : list of (start, end) tuples
        Intron chain of the read.
    iso_chain : list of (start, end) tuples
        Intron chain of the isoform.
    tolerance : int
        Max bp mismatch per intron coordinate (same as intron_tolerance).

    Returns
    -------
    (precision, recall, n_shared)
        precision = shared / len(read_chain)  (fraction of read introns
                    explained by the isoform)
        recall    = shared / len(iso_chain)   (fraction of isoform introns
                    present in the read)
        n_shared  = number of matching introns
    """
    if not read_chain or not iso_chain:
        return (0.0, 0.0, 0)

    # For each read intron, check if any isoform intron matches (within tol).
    # Greedy left-to-right: both chains are sorted by position, so walk them.
    shared = 0
    j = 0  # pointer into iso_chain
    for r_start, r_end in read_chain:
        # Advance j past isoform introns that are clearly before this read intron
        while j < len(iso_chain) and iso_chain[j][1] < r_start - tolerance:
            j += 1
        # Check candidates starting from j
        for k in range(j, len(iso_chain)):
            i_start, i_end = iso_chain[k]
            if i_start > r_start + tolerance:
                break  # all remaining iso introns are too far right
            if (abs(r_start - i_start) <= tolerance and
                    abs(r_end - i_end) <= tolerance):
                shared += 1
                break

    precision = shared / len(read_chain)
    recall = shared / len(iso_chain)
    return (precision, recall, shared)


# =============================================================================
# Core read classification
# =============================================================================

def classify_long_reads(isoform_bed, gene, bam_path, coords=None, strand=None,
                        return_read_ids=False, allow_short_reads=True,
                        print_report=False, ambiguous_mode='keep',
                        intron_tolerance=0,
                        min_read_length=200, min_aligned_fraction=0.8,
                        min_identity=0.9, return_quality=False,
                        rescue_unmatched=False,
                        rescue_min_precision=0.8,
                        rescue_min_recall=0.5,
                        min_exon_coverage=0.75):
    """
    Classify long reads against annotated isoforms using a 3-tier hierarchy:
      1. Exact matches   (is_same_isoform)
      2. Subset matches  (transcript_subset)
      3. Intron-chain matches (is_intron_chain)
    Remaining reads are classified as ambiguous or unmatched.

    Parameters
    ----------
    isoform_bed : DataFrame
        Isoform annotation BED (all genes; will be filtered to *gene*).
    gene : str
        Gene identifier to classify.
    bam_path : str
        Path to the BAM file.
    coords : str or None
        Genomic region 'chr:start-end:strand'.  If None, computed from
        *isoform_bed* with 1 kb flanks.
    strand : str or None
        '+' or '-'.  If None, inferred from *isoform_bed*.
    return_read_ids : bool
        If True, isoform_counts values are lists of read keys and the
        function also returns the long_read_beds dict (for visualization).
        If False (default), isoform_counts values are integers/floats.
    allow_short_reads : bool
        Passed through to transcript_subset / exon_compatible_read.
    print_report : bool
        Print a summary to stdout.
    ambiguous_mode : str
        'keep'         — ambiguous reads stay in the ambiguous bucket
                         (default).
        'equal'        — split evenly (1/N) among matching isoforms.
        'proportional' — split by unambiguous isoform proportions;
                         falls back to equal when all matching isoforms
                         have zero unambiguous counts.
    intron_tolerance : int
        Maximum allowed bp mismatch per intron coordinate (start and
        end checked independently).  0 = exact matching (default).
    min_read_length : int
        Minimum read length in bp to pass quality filter.
    min_aligned_fraction : float
        Minimum fraction of read bases aligned to genome.
    min_identity : float
        Minimum alignment identity to pass quality filter.
    return_quality : bool
        If True, result dict includes 'quality_df': a DataFrame with
        per-read quality metrics and isoform assignment.
    rescue_unmatched : bool
        If True, run Tier 4 partial-match rescue on reads that remain
        unmatched after Tiers 1–3.  Default False.
    rescue_min_precision : float
        Tier 4: minimum fraction of the read's introns that must match
        an isoform.
    rescue_min_recall : float
        Tier 4: minimum fraction of the isoform's introns that must
        be present in the read.
    min_exon_coverage : float
        Minimum fraction of at least one isoform exon that a read must
        cover to be assigned to that isoform.  0.0 disables the check
        (default).  A value of 0.5 requires the read to cover at least
        50% of at least one exon, filtering out short reads that map
        to a small tail of a long terminal exon without reaching any
        informative splice junction.

    Returns
    -------
    If return_read_ids is False:
        dict with keys: gene, isoform_counts, ambiguous, unmatched,
        unmatched_ids, ambiguous_detail, and optionally quality_df.
    If return_read_ids is True:
        (long_read_beds, result_dict, coords)
        where result_dict has the same keys but isoform_counts values
        are lists (unambiguous reads only), and ambiguous/unmatched
        are also lists of read keys.
    """
    isoform_bed = isoform_bed.loc[isoform_bed.gene_id == gene]

    if coords is None or strand is None:
        coords, strand = get_gene_coords(isoform_bed, gene)

    if return_quality:
        long_read_beds, quality_dict = get_long_reads(
            bam_path, coords, min_read_length=min_read_length,
            min_aligned_fraction=min_aligned_fraction,
            min_identity=min_identity, return_quality=True)
    else:
        long_read_beds = get_long_reads(
            bam_path, coords, min_read_length=min_read_length,
            min_aligned_fraction=min_aligned_fraction,
            min_identity=min_identity)

    isoform_list = sorted(isoform_bed.transcript_id.unique())
    isoform_dict = {iso: isoform_bed.loc[isoform_bed.transcript_id == iso]
                    for iso in isoform_list}

    # Pre-compute intron chains for tolerance tie-breaking / rescue
    if intron_tolerance > 0 or rescue_unmatched:
        iso_intron_chains = {iso: get_intron_chain(isoform_dict[iso])
                             for iso in isoform_list}

    # Accumulators: read-ID lists for unambiguous assignments
    if return_read_ids:
        isoform_counts = {iso: [] for iso in isoform_list}
    else:
        isoform_counts = {iso: 0.0 for iso in isoform_list}

    # Pending ambiguous reads: list of (read_key, [matching_isos])
    pending_ambiguous = []
    unmatched_reads = [] if return_read_ids else 0
    unmatched_read_ids = []

    # Track per-read assignments for quality reporting
    if return_quality:
        read_assignments = {}

    remaining_reads = list(long_read_beds.keys())

    # Helper to assign a uniquely-matched read
    def _assign(read, iso):
        if return_read_ids:
            isoform_counts[iso].append(read)
        else:
            isoform_counts[iso] += 1
        if return_quality:
            read_assignments[read] = iso

    # Helper: check if a read covers >= min_exon_coverage of at least one
    # exon in the candidate isoform.
    def _passes_exon_coverage(read, iso):
        if min_exon_coverage <= 0:
            return True
        av = long_read_beds[read]
        read_start = int(av.start.min())
        read_end = int(av.end.max())
        iso_bed = isoform_dict[iso]
        for _, exon in iso_bed.iterrows():
            exon_start = int(exon.start)
            exon_end = int(exon.end)
            exon_len = exon_end - exon_start
            if exon_len <= 0:
                continue
            overlap = max(0, min(read_end, exon_end) - max(read_start, exon_start))
            if overlap / exon_len >= min_exon_coverage:
                return True
        return False

    # Helper to resolve multi-matches (tolerance tie-breaking + ambiguous)
    def _handle_multi(read, matching_isos):
        if intron_tolerance > 0 and len(matching_isos) > 1:
            read_chain = get_intron_chain(long_read_beds[read])
            if read_chain:
                winner = resolve_tolerance_ties(
                    read_chain, matching_isos, iso_intron_chains)
                if winner is not None:
                    _assign(read, winner)
                    return
        pending_ambiguous.append((read, list(matching_isos)))

    # =====================================================================
    # Phase 1: collect matches across 3 tiers
    # =====================================================================

    # === Tier 1. Exact matches (is_same_isoform) ===
    to_remove = []
    for read in remaining_reads:
        av = long_read_beds[read]
        matched_iso = None
        for iso, iso_bed in isoform_dict.items():
            if is_same_isoform(iso_bed, av, strand):
                matched_iso = iso
                break
        if matched_iso:
            _assign(read, matched_iso)
            to_remove.append(read)
        elif intron_tolerance > 0:
            # Try tolerant full-isoform match
            read_chain = get_intron_chain(av)
            read_exons = get_exon_chain(av)
            if read_chain:
                tolerant_matches = []
                for iso, iso_bed in isoform_dict.items():
                    iso_chain = iso_intron_chains[iso]
                    if iso_chain and is_chain_match_tolerant(
                            iso_chain, read_chain, intron_tolerance):
                        iso_exons = get_exon_chain(iso_bed)
                        if same_terminal_exons(iso_exons, read_exons, strand):
                            tolerant_matches.append(iso)
                if len(tolerant_matches) == 1:
                    _assign(read, tolerant_matches[0])
                    to_remove.append(read)
                elif len(tolerant_matches) > 1:
                    _handle_multi(read, tolerant_matches)
                    to_remove.append(read)

    remaining_reads = [r for r in remaining_reads if r not in set(to_remove)]

    # === Tier 2. Subset matches (transcript_subset) ===
    unmatched_after_subset = []
    for read in remaining_reads:
        av = long_read_beds[read]
        matching_isos = []
        for iso, iso_bed in isoform_dict.items():
            if transcript_subset(iso_bed, av, allow_short_reads=allow_short_reads,
                                 intron_tolerance=intron_tolerance):
                matching_isos.append(iso)

        matching_isos = [iso for iso in matching_isos
                         if _passes_exon_coverage(read, iso)]

        if len(matching_isos) == 1:
            _assign(read, matching_isos[0])
        elif len(matching_isos) > 1:
            _handle_multi(read, matching_isos)
        else:
            unmatched_after_subset.append(read)

    # === Tier 3. Intron chain matches (is_intron_chain) ===
    unmatched_after_tier3 = []
    for read in unmatched_after_subset:
        av = long_read_beds[read]
        matching_isos = []
        for iso, iso_bed in isoform_dict.items():
            if is_intron_chain(iso_bed, av, tolerance=intron_tolerance):
                matching_isos.append(iso)

        matching_isos = [iso for iso in matching_isos
                         if _passes_exon_coverage(read, iso)]

        if len(matching_isos) == 1:
            _assign(read, matching_isos[0])
        elif len(matching_isos) > 1:
            _handle_multi(read, matching_isos)
        else:
            unmatched_after_tier3.append(read)

    # === Tier 4. Partial match rescue (intron precision/recall) ===
    if rescue_unmatched and unmatched_after_tier3:
        # Pre-compute isoform genomic boundaries for out-of-bounds veto
        iso_bounds = {iso: (isoform_dict[iso].start.min(),
                            isoform_dict[iso].end.max())
                      for iso in isoform_list}

        still_unmatched = []
        for read in unmatched_after_tier3:
            av = long_read_beds[read]
            read_chain = get_intron_chain(av)
            if not read_chain:
                still_unmatched.append(read)
                continue

            # Score every isoform and keep those passing thresholds
            candidates = []  # (iso, total_missed, precision, recall)
            n_read_introns = len(read_chain)
            for iso in isoform_list:
                iso_chain = iso_intron_chains[iso]
                if not iso_chain:
                    continue

                # Veto: skip if any read exon is entirely outside the
                # isoform's genomic boundaries
                iso_start, iso_end = iso_bounds[iso]
                read_outside = any(
                    int(row.end) <= iso_start or int(row.start) >= iso_end
                    for _, row in av.iterrows())
                if read_outside:
                    continue

                prec, rec, n_shared = intron_precision_recall(
                    read_chain, iso_chain, tolerance=intron_tolerance)
                n_iso_introns = len(iso_chain)
                missed_by_read = n_read_introns - n_shared
                missed_by_iso = n_iso_introns - n_shared
                total_missed = missed_by_read + missed_by_iso
                passes_ratio = (prec >= rescue_min_precision and
                                rec >= rescue_min_recall)
                passes_count = (total_missed <= 1 and n_shared > 0)
                if passes_ratio or passes_count:
                    if _passes_exon_coverage(read, iso):
                        candidates.append((iso, total_missed, prec, rec))

            if len(candidates) == 0:
                still_unmatched.append(read)
            elif len(candidates) == 1:
                _assign(read, candidates[0][0])
            else:
                # Pick the best match: fewest total mismatches, then
                # highest precision+recall as tiebreaker
                candidates.sort(key=lambda x: (x[1], -(x[2] + x[3])))
                best_missed = candidates[0][1]
                best_score = candidates[0][2] + candidates[0][3]
                # Keep only candidates tied for best
                tied = [c for c in candidates
                        if c[1] == best_missed
                        and c[2] + c[3] == best_score]
                if len(tied) == 1:
                    _assign(read, tied[0][0])
                else:
                    _handle_multi(read, [c[0] for c in tied])
        unmatched_after_tier3 = still_unmatched

    # Finalize unmatched reads
    for read in unmatched_after_tier3:
        av = long_read_beds[read]
        if return_read_ids:
            unmatched_reads.append(read)
        else:
            unmatched_reads += 1
        read_id = av.transcript_id.iloc[0]
        unmatched_read_ids.append(read_id)
        if return_quality:
            read_assignments[read] = 'unmatched'

    # =====================================================================
    # Phase 2: resolve ambiguous reads
    # =====================================================================

    # --- Step 2a: disambiguate by 3' end proximity ---
    # ONT reads capture the polyA tail, so the 3' end of the read is
    # informative about which isoform it came from.  For each ambiguous
    # read, check if one candidate isoform has a uniquely closest 3' end;
    # if so, assign the read directly and remove it from the pending list.
    still_ambiguous = []
    for read, matching_isos in pending_ambiguous:
        av = long_read_beds[read]
        if strand == '+':
            read_three_prime = int(av.end.max())
        else:
            read_three_prime = int(av.start.min())

        dists = []
        for iso in matching_isos:
            iso_bed = isoform_dict[iso]
            if strand == '+':
                iso_three_prime = int(iso_bed.end.max())
            else:
                iso_three_prime = int(iso_bed.start.min())
            dists.append(abs(read_three_prime - iso_three_prime))

        min_dist = min(dists)
        closest = [iso for iso, d in zip(matching_isos, dists) if d == min_dist]
        if len(closest) == 1:
            _assign(read, closest[0])
        else:
            still_ambiguous.append((read, matching_isos))
    pending_ambiguous = still_ambiguous

    if ambiguous_mode == 'keep':
        if return_read_ids:
            ambiguous_reads = [r for r, _ in pending_ambiguous]
        else:
            ambiguous_reads = len(pending_ambiguous)
    elif ambiguous_mode == 'equal':
        ambiguous_reads = [] if return_read_ids else 0
        if return_read_ids:
            # Distribute reads deterministically (each read to exactly one iso)
            assigned = _assign_reads_by_weights(
                pending_ambiguous,
                lambda isos: {iso: 1.0 for iso in isos})
            for iso, reads_list in assigned.items():
                isoform_counts[iso].extend(reads_list)
        else:
            for read, matching_isos in pending_ambiguous:
                weight = 1.0 / len(matching_isos)
                for iso in matching_isos:
                    isoform_counts[iso] += weight
    elif ambiguous_mode == 'proportional':
        ambiguous_reads = [] if return_read_ids else 0
        # Compute unambiguous totals
        if return_read_ids:
            unambig_totals = {iso: len(isoform_counts[iso])
                              for iso in isoform_list}
        else:
            unambig_totals = {iso: isoform_counts[iso]
                              for iso in isoform_list}
        if return_read_ids:
            # Distribute reads deterministically by unambiguous proportions
            assigned = _assign_reads_by_weights(
                pending_ambiguous,
                lambda isos: {iso: unambig_totals.get(iso, 0) for iso in isos})
            for iso, reads_list in assigned.items():
                isoform_counts[iso].extend(reads_list)
        else:
            for read, matching_isos in pending_ambiguous:
                relevant = [unambig_totals.get(iso, 0) for iso in matching_isos]
                total = sum(relevant)
                if total == 0:
                    weight = 1.0 / len(matching_isos)
                    for iso in matching_isos:
                        isoform_counts[iso] += weight
                else:
                    for iso, c in zip(matching_isos, relevant):
                        isoform_counts[iso] += c / total
    else:
        raise ValueError(
            f"ambiguous_mode must be 'keep', 'equal', or 'proportional', "
            f"got '{ambiguous_mode}'")

    # Track ambiguous reads in quality assignments
    if return_quality:
        for read, _ in pending_ambiguous:
            read_assignments[read] = 'ambiguous'

    # =====================================================================
    # Report
    # =====================================================================
    if print_report:
        print(f"\nResults for gene {gene}")
        for iso in isoform_list:
            ct = len(isoform_counts[iso]) if return_read_ids else isoform_counts[iso]
            print(f"  {iso}: {ct} reads")
        amb = len(ambiguous_reads) if isinstance(ambiguous_reads, list) else ambiguous_reads
        unm = len(unmatched_reads) if isinstance(unmatched_reads, list) else unmatched_reads
        print(f"  Ambiguous: {amb}")
        print(f"  Unmatched: {unm}")

    result = {
        "gene": gene,
        "isoform_counts": dict(isoform_counts),
        "ambiguous": ambiguous_reads,
        "unmatched": unmatched_reads,
        "unmatched_ids": unmatched_read_ids,
        "ambiguous_detail": pending_ambiguous,
    }

    # Build quality DataFrame
    if return_quality:
        quality_rows = []
        for read, assignment in read_assignments.items():
            qm = quality_dict.get(read)
            if qm is not None:
                row = dict(qm)
                row['read'] = read
                row['assignment'] = assignment
                quality_rows.append(row)
        result['quality_df'] = pd.DataFrame(quality_rows)

    if return_read_ids:
        return long_read_beds, result, coords
    else:
        return result


# =============================================================================
# Summarisation helpers
# =============================================================================

def summarize_isoform_counts(res, gene):
    """
    Convert the dict returned by classify_long_reads into a tidy DataFrame
    with columns [gene, isoform, counts, fraction].

    Handles integer counts, float counts, and list-of-read-IDs in
    isoform_counts.
    """
    rows = []

    def _count(val):
        if isinstance(val, list):
            return len(val)
        return val  # int or float

    total = (sum(_count(v) for v in res["isoform_counts"].values())
             + _count(res["ambiguous"])
             + _count(res["unmatched"]))

    for iso, ct in res["isoform_counts"].items():
        c = _count(ct)
        frac = c / total if total > 0 else 0
        rows.append([gene, iso, c, frac])

    amb = _count(res["ambiguous"])
    unm = _count(res["unmatched"])
    rows.append([gene, "ambiguous", amb, amb / total if total > 0 else 0])
    rows.append([gene, "unmatched", unm, unm / total if total > 0 else 0])

    return pd.DataFrame(rows, columns=["gene", "isoform", "counts", "fraction"])


def get_missing_isoforms(unmatched_isoforms_list, total_counts, gene='gene_id'):
    """Build a DataFrame describing unmatched (novel) isoform groups."""
    isoform_list = []
    for i, unmatched_isoform in enumerate(unmatched_isoforms_list, 1):
        counts = len(unmatched_isoform.notes.iloc[0])
        isoform_id = f'{gene}.missing_isoform_{i}'
        isoform_list.append([gene, isoform_id, counts])

    isoform_df = pd.DataFrame(isoform_list, columns=['gene', 'isoform', 'counts'])
    isoform_df['fraction'] = isoform_df.counts / total_counts
    return isoform_df


# =============================================================================
# Per-gene wrappers
# =============================================================================

def get_long_read_counts_per_gene(isoform_bed, gene, bam_file,
                                  coords=None, strand=None,
                                  allow_short_reads=True,
                                  ambiguous_mode='keep',
                                  intron_tolerance=0,
                                  min_read_length=200,
                                  min_aligned_fraction=0.8,
                                  min_identity=0.9,
                                  rescue_unmatched=False,
                                  rescue_min_precision=0.8,
                                  rescue_min_recall=0.5,
                                  min_exon_coverage=0.75):
    """
    Full pipeline: classify reads, summarise counts, identify missing
    isoforms among unmatched reads.
    """
    res = classify_long_reads(
        isoform_bed, gene, bam_file,
        coords=coords, strand=strand,
        return_read_ids=False, allow_short_reads=allow_short_reads,
        ambiguous_mode=ambiguous_mode, intron_tolerance=intron_tolerance,
        min_read_length=min_read_length,
        min_aligned_fraction=min_aligned_fraction,
        min_identity=min_identity,
        rescue_unmatched=rescue_unmatched,
        rescue_min_precision=rescue_min_precision,
        rescue_min_recall=rescue_min_recall,
        min_exon_coverage=min_exon_coverage,
    )
    isoform_counts = summarize_isoform_counts(res, gene)

    if coords is None:
        coords, strand = get_gene_coords(
            isoform_bed.loc[isoform_bed.gene_id == gene], gene)

    unmatched_isoforms = get_long_reads_filtered(
        bam_file, coords, allowed_reads=res['unmatched_ids'],
        min_read_length=min_read_length,
        min_aligned_fraction=min_aligned_fraction,
        min_identity=min_identity)
    total_counts = isoform_counts.counts.sum()
    if total_counts > 0:
        missing_isoforms = get_missing_isoforms(
            unmatched_isoforms, total_counts, gene=gene)
    else:
        missing_isoforms = None

    return isoform_counts, missing_isoforms


def get_long_read_counts_per_gene_lean(isoform_bed, gene, bam_file,
                                       coords=None, strand=None,
                                       allow_short_reads=True,
                                       ambiguous_mode='keep',
                                       intron_tolerance=0,
                                       min_read_length=200,
                                       min_aligned_fraction=0.8,
                                       min_identity=0.9,
                                       rescue_unmatched=False,
                                       rescue_min_precision=0.8,
                                       rescue_min_recall=0.5,
                                       min_exon_coverage=0.75):
    """Classify + summarise without the missing-isoform step."""
    res = classify_long_reads(
        isoform_bed, gene, bam_file,
        coords=coords, strand=strand,
        return_read_ids=False, allow_short_reads=allow_short_reads,
        ambiguous_mode=ambiguous_mode, intron_tolerance=intron_tolerance,
        min_read_length=min_read_length,
        min_aligned_fraction=min_aligned_fraction,
        min_identity=min_identity,
        rescue_unmatched=rescue_unmatched,
        rescue_min_precision=rescue_min_precision,
        rescue_min_recall=rescue_min_recall,
        min_exon_coverage=min_exon_coverage,
    )
    return summarize_isoform_counts(res, gene)


# =============================================================================
# Multi-sample wrappers
# =============================================================================

def get_fractions_per_sample_list(isoform_bed, gene, bam_file_list,
                                  bam_path='../code/resources/GTEx/ONT/bam/',
                                  coords=None, strand=None,
                                  allow_short_reads=True,
                                  ambiguous_mode='keep',
                                  intron_tolerance=0,
                                  min_read_length=200,
                                  min_aligned_fraction=0.8,
                                  min_identity=0.9,
                                  rescue_unmatched=False,
                                  rescue_min_precision=0.8,
                                  rescue_min_recall=0.5,
                                  min_exon_coverage=0.75):
    """Count reads per isoform across multiple BAM files (full mode)."""
    isoform_table = None
    missing_isoforms_table = None

    for bam_file in bam_file_list:
        bam_file_path = bam_path + bam_file
        bam_sample = bam_file.split('.')[0].split('_rep')[0].split('_')[-1]
        isoform_counts, missing_isoforms = get_long_read_counts_per_gene(
            isoform_bed, gene, bam_file_path,
            coords=coords, strand=strand,
            allow_short_reads=allow_short_reads,
            ambiguous_mode=ambiguous_mode, intron_tolerance=intron_tolerance,
            min_read_length=min_read_length,
            min_aligned_fraction=min_aligned_fraction,
            min_identity=min_identity,
            rescue_unmatched=rescue_unmatched,
            rescue_min_precision=rescue_min_precision,
            rescue_min_recall=rescue_min_recall,
            min_exon_coverage=min_exon_coverage,
        )
        isoform_counts.columns = ['gene', 'isoform',
                                   f'{bam_sample}.counts', f'{bam_sample}.fractions']
        if isoform_table is None:
            isoform_table = isoform_counts
            missing_isoforms_table = missing_isoforms
        else:
            isoform_table = isoform_table.merge(
                isoform_counts, on=['gene', 'isoform'], how='outer')
            missing_isoforms_table = pd.concat(
                [missing_isoforms_table, missing_isoforms])

    return isoform_table, missing_isoforms_table


def get_fractions_per_sample_list_lean(isoform_bed, gene, bam_file_list,
                                       # bam_path='../code/resources/GTEx/ONT/bam/',
                                       coords=None, strand=None,
                                       allow_short_reads=True,
                                       ambiguous_mode='keep',
                                       intron_tolerance=0,
                                       min_read_length=200,
                                       min_aligned_fraction=0.8,
                                       min_identity=0.9,
                                       rescue_unmatched=False,
                                       rescue_min_precision=0.8,
                                       rescue_min_recall=0.5,
                                       min_exon_coverage=0.75):
    """Count reads per isoform across multiple BAM files (lean, no missing)."""
    isoform_table = None

    for bam_file in bam_file_list:
        bam_file_path = bam_file #bam_path + bam_file
        bam_sample = bam_file.split('.')[0].split('_rep')[0].split('_')[-1]
        isoform_counts = get_long_read_counts_per_gene_lean(
            isoform_bed, gene, bam_file_path,
            coords=coords, strand=strand,
            allow_short_reads=allow_short_reads,
            ambiguous_mode=ambiguous_mode, intron_tolerance=intron_tolerance,
            min_read_length=min_read_length,
            min_aligned_fraction=min_aligned_fraction,
            min_identity=min_identity,
            rescue_unmatched=rescue_unmatched,
            rescue_min_precision=rescue_min_precision,
            rescue_min_recall=rescue_min_recall,
            min_exon_coverage=min_exon_coverage,
        )
        isoform_counts.columns = ['gene', 'isoform',
                                   f'{bam_sample}.counts', f'{bam_sample}.fractions']
        if isoform_table is None:
            isoform_table = isoform_counts
        else:
            isoform_table = isoform_table.merge(
                isoform_counts, on=['gene', 'isoform'], how='outer')

    return isoform_table


def get_long_reads_multi_samples(isoform_bed, gene, bam_files, annot=None,
                                 bam_path='../code/resources/GTEx/ONT/bam/',
                                 allow_short_reads=True,
                                 ambiguous_mode='keep',
                                 intron_tolerance=0,
                                 return_unmatched=False,
                                 min_read_length=200,
                                 min_aligned_fraction=0.8,
                                 min_identity=0.9,
                                 return_quality=False,
                                 rescue_unmatched=False,
                                 rescue_min_precision=0.8,
                                 rescue_min_recall=0.5):
    """
    Classify reads across multiple BAM files, returning per-isoform
    read-ID lists and the raw long_read_beds dict (for visualization).

    Parameters
    ----------
    annot : DataFrame or None
        If provided, only isoforms where ~annot.efficiency_factor are
        included.  Must have columns 'gene', 'transcript',
        'efficiency_factor'.
    return_unmatched : bool
        If True, also collect unmatched (and, when ambiguous_mode='keep',
        ambiguous) read keys and return them as a third element.
    return_quality : bool
        If True, also return a combined quality DataFrame as an
        additional element.
    """
    if annot is not None:
        isoforms = [x.split('.')[1] for x in
                     annot.loc[(annot.gene == gene) & (~annot.efficiency_factor)].transcript.unique()]
    else:
        isoforms = [x for x in
                     isoform_bed.loc[isoform_bed.gene_id == gene].transcript_id.unique()]

    isoform_dict = {iso: [] for iso in isoforms}
    long_reads_bed = {}
    unmatched_reads = []
    quality_dfs = []

    for bam in bam_files:
        bam_file = bam_path + bam
        long_reads_bed_, reads_dict_, coords = classify_long_reads(
            isoform_bed, gene, bam_file,
            return_read_ids=True,
            allow_short_reads=allow_short_reads,
            ambiguous_mode=ambiguous_mode,
            intron_tolerance=intron_tolerance,
            min_read_length=min_read_length,
            min_aligned_fraction=min_aligned_fraction,
            min_identity=min_identity,
            return_quality=return_quality,
            rescue_unmatched=rescue_unmatched,
            rescue_min_precision=rescue_min_precision,
            rescue_min_recall=rescue_min_recall,
        )
        long_reads_bed.update(long_reads_bed_)
        for iso in isoforms:
            if iso in reads_dict_['isoform_counts']:
                isoform_dict[iso] += reads_dict_['isoform_counts'][iso]
        if return_unmatched:
            if isinstance(reads_dict_['unmatched'], list):
                unmatched_reads += reads_dict_['unmatched']
            if isinstance(reads_dict_['ambiguous'], list):
                unmatched_reads += reads_dict_['ambiguous']
        if return_quality and 'quality_df' in reads_dict_:
            quality_dfs.append(reads_dict_['quality_df'])

    combined_quality = (pd.concat(quality_dfs, ignore_index=True)
                        if quality_dfs else pd.DataFrame())

    if return_unmatched and return_quality:
        return long_reads_bed, isoform_dict, unmatched_reads, combined_quality
    if return_unmatched:
        return long_reads_bed, isoform_dict, unmatched_reads
    if return_quality:
        return long_reads_bed, isoform_dict, combined_quality
    return long_reads_bed, isoform_dict


# =============================================================================
# Plotting
# =============================================================================

def plot_isoform(isoform_df, ax, color, xlim1, xlim2, lwidth, strand='-'):
    """Draw exon blocks + intron line for one isoform on *ax*."""
    is_first = True
    for idx, row in isoform_df.iterrows():
        start = int(row.start)
        end = int(row.end)
        if strand == '-':
            end_ = xlim2 - start + xlim1
            start = xlim2 - end + xlim1
            end = end_
        if is_first:
            first = end
            is_first = False
        ax.fill_between([start, end], [0, 0], [1, 1], color=color, zorder=2)
    ax.plot([first, start], [0.5, 0.5], c=color, linewidth=lwidth)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['bottom', 'top', 'right', 'left']].set_visible(False)
    ax.set_xlim([xlim1, xlim2])


def plot_isoform_stacked(isoform_df, ax, y_base, color, xlim1, xlim2, lwidth,
                         strand='-', read_frac=1):
    """Draw one isoform on *ax* at vertical position *y_base*."""
    is_first = True
    for idx, row in isoform_df.iterrows():
        start = int(row.start)
        end = int(row.end)
        if strand == '-':
            end_ = xlim2 - start + xlim1
            start = xlim2 - end + xlim1
            end = end_
        if is_first:
            first = end
            is_first = False
        ax.fill_between([start, end],
                        [y_base - 0.4 * read_frac, y_base - 0.4 * read_frac],
                        [y_base + 0.4 * read_frac, y_base + 0.4 * read_frac],
                        color=color, zorder=2, rasterized=True)
    ax.plot([first, start], [y_base, y_base], c=color, linewidth=lwidth,
            alpha=0.25, rasterized=True)
    return ax


def group_transcripts(df):
    """Assign non-overlapping vertical groups to stacked transcripts."""
    spans = (df.groupby("transcript_id")
               .agg(start=("start", "min"), end=("end", "max"))
               .reset_index())
    spans = spans.sort_values("start").reset_index(drop=True)

    groups = []
    group_assignments = [-1] * len(spans)

    for i, row in spans.iterrows():
        ti_start, ti_end = row["start"], row["end"]
        best_group = None
        best_gap = float("inf")
        for g_idx, g in enumerate(groups):
            last_idx = g[-1]
            last_end = spans.loc[last_idx, "end"]
            if last_end <= ti_start:
                gap = ti_start - last_end
                if gap < best_gap:
                    best_gap = gap
                    best_group = g_idx
        if best_group is None:
            groups.append([i])
            group_assignments[i] = len(groups) - 1
        else:
            groups[best_group].append(i)
            group_assignments[i] = best_group

    spans["group"] = group_assignments
    df = df.merge(spans[["transcript_id", "group"]], on="transcript_id")
    return df


def make_isoform_match_bed(long_reads_bed, isoform_list, strand='+'):
    """Concatenate read BEDs for reads in *isoform_list*."""
    isoform_match_list = [long_reads_bed[iso] for iso in isoform_list]
    isoform_match_bed = pd.concat(isoform_match_list, axis=0)
    isoform_match_bed['strand'] = [strand] * isoform_match_bed.shape[0]
    return isoform_match_bed


def plot_isoforms_single_ax(annotation_exons, gene, color_list=None, lwidth=5,
                            iso_order=None, xlim=None, K=100, ax=None,
                            total_reads=100, pos=False):
    """Draw all isoforms of a gene on a single axis, stacked vertically."""
    import matplotlib.pyplot as plt

    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]
    strand = gene_exons.strand.iloc[0]

    if K >= 150:
        lwidth = 0.5
    elif K >= 80:
        lwidth = 1
    elif K >= 50:
        lwidth = 2
    elif K >= 20:
        lwidth = 3
    elif K >= 10:
        lwidth = 4

    if xlim is None:
        xlim1 = gene_exons.start.min() - 1000
        xlim2 = gene_exons.end.max() + 1000
    else:
        xlim1, xlim2 = xlim

    if color_list is None:
        color_list = ['gray'] * K

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 3))

    y_base = 0
    read_frac = 1 / total_reads

    for i, group in enumerate(sorted(gene_exons.group.unique())):
        iso_per_group = gene_exons.loc[gene_exons.group == group].transcript_id.unique()
        for j, iso in enumerate(iso_per_group):
            iso_df = gene_exons[gene_exons.transcript_id == iso]
            color = color_list[i % len(color_list)]
            ax = plot_isoform_stacked(iso_df, ax, y_base, color, xlim1, xlim2,
                                      lwidth, strand=strand, read_frac=read_frac)
        if pos:
            y_base += 1 * read_frac
        else:
            y_base -= 1 * read_frac

    ax.set_xticks([])
    ax.spines[['bottom', 'top', 'right']].set_visible(False)
    ax.set_xlim([xlim1, xlim2])
    if pos:
        ax.set_ylim([-2 / total_reads, K / total_reads])
    else:
        ax.set_ylim([-K / total_reads, 2 / total_reads])

    return ax


def plot_gene_isoforms(isoforms_dict, coordinates, color_list=None, axes=None,
                       figsize=None, lwidth=5, xlim=None, iso_order=None,
                       strand='+'):
    """Plot isoforms on separate axes (one per isoform)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if xlim is None:
        xlim1 = int(coordinates[0].split(':')[1])
        xlim2 = int(coordinates[-1].split(':')[1])
        xlim = (xlim1, xlim2)

    if color_list is None:
        color_list = sns.color_palette("Dark2")

    K = len(iso_order) if iso_order else len(isoforms_dict)
    if figsize is None:
        figsize = (20, 3)
    if axes is None:
        fig, axes = plt.subplots(K, 1, figsize=figsize)

    if K == 1:
        isoform_df = isoforms_dict['isoform_1']['df']
        plot_isoform(isoform_df, axes, color_list[0], xlim[0], xlim[1],
                     lwidth=lwidth, strand=strand)
    else:
        items = iso_order if iso_order else [f'isoform_{i+1}' for i in range(K)]
        for i, iso in enumerate(items):
            isoform_df = isoforms_dict[iso]['df']
            plot_isoform(isoform_df, axes[i], color_list[i], xlim[0], xlim[1],
                         lwidth=lwidth, strand=strand)


def plot_gene_isoforms_single(isoform_df, coordinates, color_list=None,
                              axes=None, figsize=None, lwidth=5, xlim=None,
                              iso_order=None, strand='+'):
    """Plot all isoforms of a gene on a single axis."""
    import seaborn as sns

    if xlim is None:
        xlim1 = int(coordinates[0].split(':')[1])
        xlim2 = int(coordinates[-1].split(':')[1])
        xlim = (xlim1, xlim2)

    if color_list is None:
        color_list = sns.color_palette("Dark2")

    plot_isoform(isoform_df, axes, color_list[0], xlim[0], xlim[1],
                 lwidth=lwidth, strand=strand)


def plot_isoform_annotations(annotation_exons, gene, colores=None, start=None,
                             end=None, figsize=None, lwidth=5, iso_order=None,
                             axes=None, xlim=None):
    """Plot annotated isoforms for a gene on separate axes."""
    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]
    strand = gene_exons.strand.iloc[0]

    if iso_order is None:
        isoforms = sorted(gene_exons.transcript_id.unique())
    else:
        isoforms = iso_order

    isoform_dict = {}
    for iso in isoforms:
        df = gene_exons.loc[gene_exons.transcript_id == iso].copy()
        df['transcript_id'] = iso
        isoform_dict[iso] = {'df': df}

    chrom = gene_exons.chrom.iloc[0]
    if start is None:
        start = str(gene_exons.start.min() - 1000)
    if end is None:
        end = str(gene_exons.end.max() + 1000)
    coords = [f'{chrom}:{start}', f'{chrom}:{end}']

    plot_gene_isoforms(isoform_dict, coords, color_list=colores,
                       figsize=figsize, lwidth=lwidth, axes=axes, xlim=xlim,
                       iso_order=iso_order, strand=strand)


def plot_isoform_annotations_single(annotation_exons, gene, colores=None,
                                    start=None, end=None, figsize=None,
                                    lwidth=5, iso_order=None, axes=None,
                                    xlim=None):
    """Plot annotated isoforms for a gene on a single shared axis."""
    gene_exons = annotation_exons.loc[annotation_exons.gene_id == gene]
    strand = gene_exons.strand.iloc[0]

    chrom = gene_exons.chrom.iloc[0]
    if start is None:
        start = str(gene_exons.start.min() - 1000)
    if end is None:
        end = str(gene_exons.end.max() + 1000)
    coords = [f'{chrom}:{start}', f'{chrom}:{end}']

    plot_gene_isoforms_single(annotation_exons, coords, color_list=colores,
                              figsize=figsize, lwidth=lwidth, axes=axes,
                              xlim=xlim, iso_order=iso_order, strand=strand)


def plot_isoform_annotations_and_long_reads_single_ax(
        isoform_bed, gene, isoform_id, long_reads_bed, isoform_reads,
        coords, strand, max_reads=100, figsize=(20, 8), iso_color='black',
        reads_color='gray', axes=None, total_reads=100, pos=False):
    """Plot one annotated isoform + its matching long reads."""
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=figsize,
                                 height_ratios=[1, 0.5, 10] * max_reads,
                                 gridspec_kw={'wspace': 0})

    iso_filter = ((isoform_bed.gene_id == gene)
                  & (isoform_bed.transcript_id == isoform_id))
    if pos:
        plot_isoform_annotations_single(
            isoform_bed.loc[iso_filter], gene,
            axes=axes[2], xlim=(coords[0], coords[1]), colores=[iso_color])
    else:
        plot_isoform_annotations_single(
            isoform_bed.loc[iso_filter], gene,
            axes=axes[0], xlim=(coords[0], coords[1]), colores=[iso_color])

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].spines[['bottom', 'top', 'right', 'left']].set_visible(False)

    try:
        grouped = group_transcripts(
            make_isoform_match_bed(long_reads_bed, isoform_reads, strand=strand))
        target_ax = axes[0] if pos else axes[2]
        plot_isoforms_single_ax(
            grouped, 'gene_1', xlim=(coords[0], coords[1]), K=max_reads,
            ax=target_ax, color_list=[reads_color] * max_reads,
            total_reads=total_reads, pos=pos)
    except Exception:
        pass


def _plot_unmatched_reads_single_ax(long_reads_bed, unmatched_reads, coords,
                                    strand, max_reads, reads_color, axes,
                                    total_reads, pos=False):
    """Plot unmatched reads on axes (no annotation isoform row)."""
    # Clear the annotation row (no reference isoform for unmatched)
    annot_ax = axes[2] if pos else axes[0]
    annot_ax.set_xticks([])
    annot_ax.set_yticks([])
    annot_ax.spines[['bottom', 'top', 'right', 'left']].set_visible(False)
    annot_ax.set_xlim([coords[0], coords[1]])

    # Separator
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].spines[['bottom', 'top', 'right', 'left']].set_visible(False)

    # Plot the reads
    try:
        grouped = group_transcripts(
            make_isoform_match_bed(long_reads_bed, unmatched_reads, strand=strand))
        target_ax = axes[0] if pos else axes[2]
        plot_isoforms_single_ax(
            grouped, 'gene_1', xlim=(coords[0], coords[1]), K=max_reads,
            ax=target_ax, color_list=[reads_color] * max_reads,
            total_reads=total_reads, pos=pos)
    except Exception:
        target_ax = axes[0] if pos else axes[2]
        target_ax.set_xticks([])
        target_ax.set_yticks([])
        target_ax.spines[['bottom', 'top', 'right', 'left']].set_visible(False)
        target_ax.set_xlim([coords[0], coords[1]])


def plot_gene_long_read_transcripts_comparison_single_ax(
        isoform_bed, gene, long_reads_bed1, long_reads_bed2,
        isoform_dict1, isoform_dict2, isoform_list, iso_color_list,
        reads_color1, reads_color2, figsize=(20, 12), pos=False, yticks=None,
        unmatched_reads1=None, unmatched_reads2=None,
        unmatched_color='lightgray'):
    """
    Side-by-side comparison of long reads from two conditions.

    Parameters
    ----------
    unmatched_reads1, unmatched_reads2 : list of read keys or None
        If provided, an additional column pair is appended showing unmatched
        (and ambiguous, if ambiguous_mode='keep') reads from each condition.
        Obtain these from get_long_reads_multi_samples(..., return_unmatched=True).
    unmatched_color : str
        Color for unmatched read blocks (default 'lightgray').
    """
    import matplotlib.pyplot as plt

    show_unmatched = (unmatched_reads1 is not None or unmatched_reads2 is not None)

    iso_gene = isoform_bed.loc[isoform_bed.gene_id == gene]
    gene_len = iso_gene.end.max() - iso_gene.start.min()
    if gene_len >= 50000:
        w = 5000
    elif gene_len >= 20000:
        w = 2000
    elif gene_len >= 10000:
        w = 1000
    else:
        w = 500
    coords_start = iso_gene.start.min() - w
    coords_end = iso_gene.end.max() + w
    reads_1 = 10
    reads_2 = 10
    total_reads_1 = 0
    total_reads_2 = 0
    strand = iso_gene.strand.iloc[0]

    for iso in isoform_list:
        try:
            iso_bed = group_transcripts(
                make_isoform_match_bed(long_reads_bed1, isoform_dict1[iso], strand=strand))
            reads_1 = max(reads_1, iso_bed.group.max() + 1)
            total_reads_1 += iso_bed.group.max()
        except Exception:
            pass
        try:
            iso_bed = group_transcripts(
                make_isoform_match_bed(long_reads_bed2, isoform_dict2[iso], strand=strand))
            reads_2 = max(reads_2, iso_bed.group.max() + 1)
            total_reads_2 += iso_bed.group.max()
        except Exception:
            pass

    # Also account for unmatched reads in the height calculations
    if show_unmatched:
        if unmatched_reads1:
            try:
                unm_bed = group_transcripts(
                    make_isoform_match_bed(long_reads_bed1, unmatched_reads1, strand=strand))
                reads_1 = max(reads_1, unm_bed.group.max() + 1)
                total_reads_1 += unm_bed.group.max()
            except Exception:
                pass
        if unmatched_reads2:
            try:
                unm_bed = group_transcripts(
                    make_isoform_match_bed(long_reads_bed2, unmatched_reads2, strand=strand))
                reads_2 = max(reads_2, unm_bed.group.max() + 1)
                total_reads_2 += unm_bed.group.max()
            except Exception:
                pass

    pct_1 = reads_1 / total_reads_1 if total_reads_1 > 0 else 1
    pct_2 = reads_2 / total_reads_2 if total_reads_2 > 0 else 1

    if pct_1 > pct_2:
        reads_2 = int(np.ceil(reads_2 * pct_1 / pct_2))
    else:
        reads_1 = int(np.ceil(reads_1 * pct_2 / pct_1))

    n_cols = len(isoform_list) * 2 + (2 if show_unmatched else 0)

    if pos:
        fig, axes = plt.subplots(
            3, n_cols, figsize=figsize,
            height_ratios=[50, 0.1, 1],
            gridspec_kw={'wspace': 0, 'hspace': 0})
    else:
        fig, axes = plt.subplots(
            3, n_cols, figsize=figsize,
            height_ratios=[1, 0.1, 50],
            gridspec_kw={'wspace': 0, 'hspace': 0})

    for i, iso in enumerate(isoform_list):
        iso_color = iso_color_list[i]
        plot_isoform_annotations_and_long_reads_single_ax(
            isoform_bed, gene, iso, long_reads_bed1, isoform_dict1[iso],
            (coords_start, coords_end), strand=strand, max_reads=reads_1,
            figsize=figsize, iso_color=iso_color, reads_color=reads_color1,
            axes=axes[:, i * 2], total_reads=total_reads_1, pos=pos)
        plot_isoform_annotations_and_long_reads_single_ax(
            isoform_bed, gene, iso, long_reads_bed2, isoform_dict2[iso],
            (coords_start, coords_end), strand=strand, max_reads=reads_2,
            figsize=figsize, iso_color=iso_color, reads_color=reads_color2,
            axes=axes[:, i * 2 + 1], total_reads=total_reads_2, pos=pos)

        for ax_idx in [0, 2]:
            axes[ax_idx, i * 2 + 1].set_xticks([])
            axes[ax_idx, i * 2].set_xticks([])

        if pos:
            axes[0, i * 2 + 1].set_yticks([])
            axes[0, i * 2 + 1].spines[['bottom', 'top', 'right', 'left']].set_visible(False)
            if i > 0:
                axes[0, i * 2].set_yticks([])
                axes[0, i * 2].spines[['bottom', 'top', 'right', 'left']].set_visible(False)
        else:
            axes[2, i * 2 + 1].set_yticks([])
            axes[2, i * 2 + 1].spines[['bottom', 'top', 'right', 'left']].set_visible(False)
            if i > 0:
                axes[2, i * 2].set_yticks([])
                axes[2, i * 2].spines[['bottom', 'top', 'right', 'left']].set_visible(False)

        if yticks is not None:
            target_row = 0 if pos else 2
            axes[target_row, 0].set_yticks(yticks[0], yticks[1])
            axes[target_row, 0].tick_params(axis='y', labelsize=24)
            axes[target_row, 0].set_ylabel('Percentage of reads', fontsize=24)

    # --- Unmatched column pair ---
    if show_unmatched:
        unm_col1 = len(isoform_list) * 2
        unm_col2 = unm_col1 + 1

        _plot_unmatched_reads_single_ax(
            long_reads_bed1, unmatched_reads1 or [], (coords_start, coords_end),
            strand=strand, max_reads=reads_1, reads_color=unmatched_color,
            axes=axes[:, unm_col1], total_reads=total_reads_1, pos=pos)
        _plot_unmatched_reads_single_ax(
            long_reads_bed2, unmatched_reads2 or [], (coords_start, coords_end),
            strand=strand, max_reads=reads_2, reads_color=unmatched_color,
            axes=axes[:, unm_col2], total_reads=total_reads_2, pos=pos)

        for ax_idx in [0, 2]:
            axes[ax_idx, unm_col1].set_xticks([])
            axes[ax_idx, unm_col2].set_xticks([])

        if pos:
            axes[0, unm_col2].set_yticks([])
            axes[0, unm_col2].spines[['bottom', 'top', 'right', 'left']].set_visible(False)
            axes[0, unm_col1].set_yticks([])
            axes[0, unm_col1].spines[['bottom', 'top', 'right', 'left']].set_visible(False)
        else:
            axes[2, unm_col2].set_yticks([])
            axes[2, unm_col2].spines[['bottom', 'top', 'right', 'left']].set_visible(False)
            axes[2, unm_col1].set_yticks([])
            axes[2, unm_col1].spines[['bottom', 'top', 'right', 'left']].set_visible(False)


# =============================================================================
# BAM → BED file writing (batch mode)
# =============================================================================

def BamFileToJuncAndIsoformBed(bam_path, output):
    """Write junction and isoform BED files from a BAM file."""
    from tqdm import tqdm

    bamf = pysam.Samfile(bam_path, "rb")
    autosomes = ['chr' + str(i) for i in range(1, 23)]
    juncs_file = f'{output}.juncs.bed'
    isoforms_file = f'{output}.isoforms.bed'
    with open(juncs_file, 'w') as fh_juncs:
        with open(isoforms_file, 'w') as fh_isoforms:
            read_counts = 1
            for entry in tqdm(bamf.fetch(), position=0, leave=True):
                chrom = bamf.getrname(entry.rname)
                if chrom not in autosomes:
                    continue
                read_name = 'LongRead.' + str(read_counts)
                bed_juncs, bed_isoforms = getEntrySpliceSitesBed(
                    bamf, entry, read_name)
                if not bed_juncs:
                    continue
                fh_juncs.write(bed_rows_to_str(bed_juncs))
                fh_isoforms.write(bed_rows_to_str(bed_isoforms))
                read_counts += 1
