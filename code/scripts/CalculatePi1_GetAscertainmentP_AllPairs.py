#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
#
######################################################################
# @author      : bjf79 (bjf79@midway2-login1.rcc.local)
# @file        : CalculatePi1_GetAscertainmentP
# @created     : Thursday Jun 23, 2022 12:15:33 CDT
#
# @description : 
######################################################################

import sys
import pysam
import pandas as pd
import re
import numpy as np
import glob
from operator import itemgetter
import vcf

# I like to script and debug with an interactive interpreter. If using
# interactive interpreter to script quick args with sys.argv, parse_args with
# hardcoded args below
if hasattr(sys, 'ps1'):
    # sys.argv = ["", "scratch/PairwisePi1Traits.1.txt.gz" ,"scratch/PairwisePi1Traits.P.1.txt.gz"] + "QTLs/QTLTools/chRNA.Expression_cheRNA/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/chRNA.Expression_eRNA/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/chRNA.Expression_lncRNA/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/chRNA.Expression_ncRNA/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/chRNA.Expression_snoRNA/NominalPassForColoc.txt.tabix.gz".split(' ')
    sys.argv = "scripts/CalculatePi1_GetAscertainmentP_AllPairs.py pi1/PairwiseTraitsToCompare/32.txt.gz pi1/PairwiseTraitsToCompare/P.32.txt.gz QTLs/QTLTools/Expression.Splicing/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/Expression.Splicing.Subset_YRI/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/chRNA.Expression.Splicing/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/MetabolicLabelled.30min/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/MetabolicLabelled.60min/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/H3K27AC/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/H3K4ME3/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/H3K4ME1/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/H3K36ME3/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/APA_Nuclear/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/APA_Total/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/polyA.Splicing/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/polyA.Splicing.Subset_YRI/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/chRNA.Splicing/NominalPassForColoc.txt.tabix.gz QTLs/QTLTools/ProCap/NominalPassForColoc.txt.tabix.gz".split(' ')
    # sys.argv = ["", "scratch/PairwisePi1Traits.1.txt.gz" ,"scratch/PairwisePi1Traits.P.1.txt.gz"] + glob.glob("QTLs/QTLTools/*/NominalPassForColoc.txt.tabix.gz")

_, f_in, f_out = sys.argv[:3]
tabix_f_in_list = sys.argv[3:]

gtex_vcf = vcf.Reader(open('/project2/yangili1/cfbuenabadn/torino_paper/code/resources/GTEx/genotype/GTEx_Analysis_2017-06-05_v8_WGS_VCF_files_GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz', 'br'))

def GetSummaryStatsFromOtherTabixFiles(row, TabixFilesDict, ColumnIndexes, gtex_vcf):
    """
    from row of df
    """
    fetch_region = row["singletrait_topvar_chr.x"], row["singletrait_topvar_pos.x"]-1, row["singletrait_topvar_pos.x"]
    try:
        tbx_fetch = TabixFilesDict[row["PC2"]]['NominalPass'].fetch(*fetch_region)
        ToReturn = pd.Series([None for i in ColumnIndexes]) 
        for line in tbx_fetch:
            var_id = row["singletrait_topvar.x"]
            if (var_id == line[9]) and row["P2"] == line[5]:
                rec_ = get_snp_record(gtex_vcf, var_id)
                maf, total_alleles = get_maf(rec_, TabixFilesDict[row["PC1"]]['Samples'])
                if (total_alleles >= 60) and (maf >= 0.05):
                    maf, total_alleles = get_maf(rec_, TabixFilesDict[row["PC2"]]['Samples'])
                    if (total_alleles >= 60) and (maf >= 0.05):
                        ToReturn = pd.Series([float(i) for i in itemgetter(*ColumnIndexes)(line)])
        return ToReturn
    except ValueError:
        return pd.Series([None for i in ColumnIndexes])

def get_maf(rec_, tissue_samples):
    hom_refs = [x.sample for x in rec_.get_hom_refs() if x.sample in tissue_samples]
    hets = [x.sample for x in rec_.get_hets() if x.sample in tissue_samples]
    hom_alts = [x.sample for x in rec_.get_hom_alts() if x.sample in tissue_samples]

    total_alleles = (len(hom_refs)*2) + (2*len(hets))+ (len(hom_alts)*2)
    total_ref = (len(hom_refs)*2) + len(hets)
    ref_freq = total_ref/total_alleles
    alt_freq = 1- ref_freq

    maf = np.min([ref_freq, alt_freq])

    return maf, total_alleles

def get_snp_record(vcf, var_id):
    chrom, position, ref, alt, _ = var_id.split('_')
    position = int(position)
    record_out = None
    for record in vcf.fetch(chrom, position-1, position+1):  
        if (record.POS == position) and (record.REF == ref) and (alt in record.ALT):
            record_out = record
    return record_out








file_suffix = tabix_f_in_list[0].split('/')[4]
# k = f_in.split('_')[1].split('.')[0]
# annotation = file_suffix.split('_')[0]
suffix_qqnorm = f'qqnorm.sorted.bed.gz'

TabixFilesDict = {re.search("results/QTLs/GTEx/(.+?)/" + file_suffix, fn).group(1):{
    'NominalPass':pysam.TabixFile(fn, parser=pysam.asTuple()),
    'Samples':pd.read_csv(f'results/QTLs/GTEx/{re.search("results/QTLs/GTEx/(.+?)/" + file_suffix, fn).group(1)}/{suffix_qqnorm}', sep='\t').columns[6:]
                                                                      } for fn in tabix_f_in_list}
print(TabixFilesDict)
# dfhead = df.head(100)
# dfhead[['trait.x.p.in.y', 'x.beta.in.y', 'x.beta_se.in.y']] = dfhead.apply(GetSummaryStatsFromOtherTabixFiles, axis=1, TabixFilesDict=TabixFilesDict, ColumnIndexes = [11, 13,14])
# dfhead['trait.x.p.in.y'] = dfhead.apply(GetAscertainmentSNP_P, axis=1, TabixFilesDict=TabixFilesDict )
# dfhead.to_csv(f_out, sep='\t', index=False)

df = pd.read_csv(f_in, sep='\t')

df[['trait.x.p.in.y', 'x.beta.in.y', 'x.beta_se.in.y']] = df.apply(GetSummaryStatsFromOtherTabixFiles, axis=1, TabixFilesDict=TabixFilesDict, ColumnIndexes = [13, 15,16], gtex_vcf=gtex_vcf)

df.to_csv(f_out, sep='\t', index=False)

