N_PermutationChunks = 100

rule SplitAndCombineSummaryStatsPerLocus:
    """
    since hyprcoloc will require reading in summary stat data in R, it will be
    more convenient if the summary stats were organized into smaller files so
    we don't have to read in huge files into R. Also, it is sort of pointless
    to attempt colocalization on molecular traits that don't have an QTL
    signal. Therefore, this script reads the QTLtools output files for multiple
    phenotypes, and writes out the necessary summary stats for all phenotypes
    to new smaller files (one file for each gwas trait or gene) if they pass
    some minimum Pvalue treshold from the QTLtools permutation test.
    """
    input:
        zip( expand(
            "results/QTLs/GTEx/{tissue}/ForGWASColoc.PermutationPass.FDR_Added.txt.gz",
            tissue=gtex_tissues,
        ),
        expand(
            "results/QTLs/GTEx/{tissue}/ForGWASColoc.NominalPass.Chunks/{{n}}.txt",
            tissue=gtex_tissues,
        )) 
    output:
        directory("hyprcoloc/LociWiseSummaryStatsInput/Chunks/ForGWASColoc/{n}")
    log:
        "logs/SplitAndCombineSummaryStatsPerGene.ForGWASColoc.{n}.log"
    resources:
        mem_mb = much_more_mem_after_first_attempt
    params:
        MinNominalP = 0.01
    shell:
        """
        python scripts/CombineAndSplitSummaryStatsForGWASColoc.py {output}/ {params.MinNominalP} {input} &> {log}
        """

rule GatherSummaryStatsFileChunks:
    input:
        expand("hyprcoloc/LociWiseSummaryStatsInput/Chunks/ForGWASColoc/{n}/", n=range(1, 1+N_PermutationChunks) )
    output:
        directory("hyprcoloc/LociWiseSummaryStatsInput/ForGWASColoc.Merged")
    log:
        "logs/coloc/GatherSummaryStatsFileChunks.log"
    resources:
        mem_mb = much_more_mem_after_first_attempt
    shell:
        """
        python scripts/MergeSummaryStatChunks.py {output} {input} &> {log}
        """

rule SplitSummaryFilesIntoChromosomeChunks:
    input:
        "hyprcoloc/LociWiseSummaryStatsInput/ForGWASColoc.Merged"
    output:
        directory("hyprcoloc/LociWiseSummaryStatsInput/ForGWASColoc")
    log:
        "logs/coloc/GatherSummaryStatsFileChunks_split.log"
    resources:
        mem_mb = much_more_mem_after_first_attempt
    shell:
        """
        python scripts/SplitFilesForColocByChrom.py {output} {input} &> {log}
        """

rule GatherHyprcolocInput:
    input:
        "hyprcoloc/LociWiseSummaryStatsInput/ForGWASColoc",



















rule create_gwascoloc_bash_scripts:
    output:
        expand("hyprcoloc/Results/ForGWASColoc/{{ColocName}}/Chunks/{n}.sh", n=range(0,config["gwas_coloc_chunks"]))
    wildcard_constraints:
        ColocName = '|'.join(colocs_gwas.index)
    params:
        PhenotypesToColoc = GetMolPhenotypesToColoc
    run:
        from itertools import cycle
        gwas_bashscript_pairs = zip(gwas_traits_for_coloc, cycle(output))
        chrom_list = [f"chr{str(i)}" for i in range(1, 23)] + ['chrX']
        for accession, out_f in gwas_bashscript_pairs:
            with open(out_f, 'a') as f:
                for chunk in chrom_list:
                    _ = f.write(f'Rscript scripts/hyprcoloc_gwas2.R hyprcoloc/LociWiseSummaryStatsInput/ForGWASColoc/{chunk}.{accession}.txt.gz /project2/mstephens/cfbuenabadn/gtex-stm/code/gwas/StatsForColoc/{accession}.standardized.txt.gz {out_f.rstrip(".sh")}.txt.gz "{params.PhenotypesToColoc}"\n')
        # If there are more output files than accession numbers, the extra
        # output files won't get made in the previous loop and snakemake will
        # complain of missing output files. as a fail safe, let's append to
        # each file in output, in effect making an empty file if a file wasn't
        # made in the for loop above
        for f in output:
            open(f, 'a').close()


rule gwas_coloc_chunk:
    input:
        bashscript = "hyprcoloc/Results/ForGWASColoc/{ColocName}/Chunks/{n}.sh",
        MolQTLSummaryStats = GetColocTsvFormattedString("hyprcoloc/LociWiseSummaryStatsInput/ForGWASColoc"),
        Gwas_summary_stats =  expand("/project2/mstephens/cfbuenabadn/gtex-stm/code/gwas/StatsForColoc/{accession}.standardized.txt.gz", accession=gwas_traits_for_coloc),
        NominalPass_tabix = "results/QTLs/GTEx/Brain_Cortex/ForGWASColoc.NominalPass.txt.tabix.gz"
        # ModifiedCondaEnvConfirmation = "hyprcoloc/hyprcoloc_installed.touchfile"
    wildcard_constraints:
        ColocName = '|'.join(colocs_gwas.index),
    output:
        "hyprcoloc/Results/ForGWASColoc/{ColocName}/Chunks/{n}.txt.gz"
    resources:
        mem_mb = 58000 #lambda wildcards, attempt: 32000 if int(attempt) == 1 else 62000
    log:
        "logs/gwas_coloc_chunk/ForGWASColoc/{ColocName}/{n}.log"
    shell:
        """
        bash {input.bashscript} &> {log}
        """



rule Gather_gwas_coloc_chunks:
    input:
        expand("hyprcoloc/Results/ForGWASColoc/{{ColocName}}/Chunks/{n}.txt.gz", n=range(0, config["gwas_coloc_chunks"]))
    output:
        "hyprcoloc/Results/ForGWASColoc/{ColocName}/results.txt.gz"
    params:
        header = "GWASLeadSnpChrom_Pos_RefAllele_AltAllele_rsID_trait\tHyprcolocIteration\tColocalizedTraits\tPosteriorColocalizationPr\tRegionalAssociationPr\tTopCandidateSNP\tProportionPosteriorPrExplainedByTopSNP\tDroppedTrait"
    wildcard_constraints:
        ColocName = '|'.join(colocs_gwas.index),
    shell:
        """
        cat <(echo "{params.header}") <(zcat {input}) | gzip - > {output}
        """


#hyprcoloc/Results/ForGWASColoc/GWAS_GTEx_01/results.txt.gz
