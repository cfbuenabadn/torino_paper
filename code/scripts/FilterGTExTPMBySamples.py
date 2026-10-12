#!/usr/bin/env python3
"""
Filter GTEx gene TPM GCT file to keep only samples present in an isoform BED file.

Usage:
    python FilterGTExTPMBySamples.py

Reads sample IDs from columns 7+ of the BED file, then streams through the GCT file
keeping Name, Description, and matching sample columns. Writes gzipped TSV output.
"""

import gzip
import sys

BED_FILE = "results/torino/tables/Human/corrected_isoforms_filtered.merged_isoforms.EL.bed.gz"
GCT_FILE = "/project2/mstephens/cfbuenabadn/gtex-stm/code/gtex_tables/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
OUT_FILE = "resources/GTEx/data/GTEX.gene_expression.tpm.tsv.gz"


def get_bed_samples(bed_file):
    with gzip.open(bed_file, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    # columns 7+ (0-indexed: 6+) are sample IDs
    return set(header[6:])


def filter_gct(gct_file, bed_samples, out_file):
    with gzip.open(gct_file, "rt") as fin, gzip.open(out_file, "wt") as fout:
        # skip first two rows
        fin.readline()
        fin.readline()

        # third row: header
        header_line = fin.readline().rstrip("\n").split("\t")
        keep_idx = []
        for i, col in enumerate(header_line):
            if col in ("Name", "Description") or col in bed_samples:
                keep_idx.append(i)

        fout.write("\t".join(header_line[i] for i in keep_idx) + "\n")

        for line in fin:
            fields = line.rstrip("\n").split("\t")
            fout.write("\t".join(fields[i] for i in keep_idx) + "\n")


if __name__ == "__main__":
    print("Reading sample IDs from BED file...", flush=True)
    bed_samples = get_bed_samples(BED_FILE)
    print(f"  Found {len(bed_samples)} samples", flush=True)

    print("Streaming and filtering GCT file...", flush=True)
    filter_gct(GCT_FILE, bed_samples, OUT_FILE)
    print(f"Done. Output written to {OUT_FILE}", flush=True)
