import pandas as pd
import gzip
import sys
from collections import defaultdict

bed_file = sys.argv[1]
eff_tab_file = sys.argv[2]
out_all_gtf = sys.argv[3]
out_filtered_gtf = sys.argv[4]

# Read the BED file
# Columns: chr, start, end, gene, isoform, strand, factors, exon_label, notes
bed_df = pd.read_csv(bed_file, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'gene', 'isoform', 'strand',
                            'factors', 'exon_label', 'notes'])

# Create renamed transcript IDs: gene.isoform
bed_df['transcript_id'] = bed_df['gene'] + '.' + bed_df['isoform']

# Read efficiency table
eff_df = pd.read_csv(eff_tab_file, sep='\t')

# Get set of isoforms to keep for filtered GTF:
# present in eff_tab AND NOT efficiency factors
filtered_isoforms = set(
    eff_df.loc[eff_df['efficiency_factor'] == False, 'transcript'].values
)

# All isoforms present in eff_tab (for the full GTF we include everything from BED)
all_isoforms = set(bed_df['transcript_id'].values)


def write_gtf(df, isoform_set, outfile):
    """Write GTF for isoforms in isoform_set."""
    # Filter to relevant isoforms
    sub = df[df['transcript_id'].isin(isoform_set)].copy()

    # Group by gene to get gene-level coordinates
    gene_groups = sub.groupby('gene')
    transcript_groups = sub.groupby(['gene', 'transcript_id'])

    lines = []

    # Build gene entries
    gene_info = {}
    for gene, gdf in gene_groups:
        chrom = gdf['chrom'].iloc[0]
        strand = gdf['strand'].iloc[0]
        gene_start = gdf['start'].min()
        gene_end = gdf['end'].max()
        gene_info[gene] = (chrom, gene_start, gene_end, strand)

    # Build transcript entries and exons, sorted by genomic position
    transcript_entries = []
    for (gene, tid), tdf in transcript_groups:
        chrom = tdf['chrom'].iloc[0]
        strand = tdf['strand'].iloc[0]
        tx_start = tdf['start'].min()
        tx_end = tdf['end'].max()
        transcript_entries.append((chrom, tx_start, tx_end, strand, gene, tid, tdf))

    # Sort by chrom, start
    transcript_entries.sort(key=lambda x: (x[0], x[1]))

    # Group by gene for output, but we'll write gene-by-gene
    # First, determine gene order by their earliest transcript
    gene_order = []
    seen_genes = set()
    for entry in transcript_entries:
        gene = entry[4]
        if gene not in seen_genes:
            gene_order.append(gene)
            seen_genes.add(gene)

    for gene in gene_order:
        chrom, gene_start, gene_end, strand = gene_info[gene]
        # Gene line (GTF is 1-based)
        attr = f'gene_id "{gene}";'
        lines.append(f'{chrom}\ttorino\tgene\t{gene_start + 1}\t{gene_end}\t.\t{strand}\t.\t{attr}')

        # Get transcripts for this gene
        gene_transcripts = [e for e in transcript_entries if e[4] == gene]
        gene_transcripts.sort(key=lambda x: x[1])

        for chrom, tx_start, tx_end, strand, g, tid, tdf in gene_transcripts:
            # Transcript line
            attr = f'gene_id "{gene}"; transcript_id "{tid}";'
            lines.append(f'{chrom}\ttorino\ttranscript\t{tx_start + 1}\t{tx_end}\t.\t{strand}\t.\t{attr}')

            # Exon lines, sorted by position
            exons = tdf.sort_values('start')
            for idx, row in exons.iterrows():
                attr = f'gene_id "{gene}"; transcript_id "{tid}";'
                lines.append(f'{row["chrom"]}\ttorino\texon\t{row["start"] + 1}\t{row["end"]}\t.\t{row["strand"]}\t.\t{attr}')

    with gzip.open(outfile, 'wt') as f:
        f.write('\n'.join(lines) + '\n')


# Write full GTF (all isoforms)
write_gtf(bed_df, all_isoforms, out_all_gtf)

# Write filtered GTF (only non-efficiency-factor isoforms present in eff_tab)
write_gtf(bed_df, filtered_isoforms, out_filtered_gtf)
