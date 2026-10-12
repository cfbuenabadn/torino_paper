# use tabix genewise qtltools output to ascertain p value in trait2 for top SNP determined from trait1 discoveries.
# There are so many potential trait pairs, so i am going to break this down into chunks to paralelize

NumPvalsForPi1Chunks = 40

rule CalculatePi1_GetTraitPairs_AllTraits:
    input:
        Nominal = expand("results/QTLs/GTEx/{tissue_id}/{{QTLsRun}}NominalPass.txt.tabix.gz", tissue_id = gtex_tissues),
        Permutation = expand("results/QTLs/GTEx/{tissue_id}/{{QTLsRun}}PermutationPass.txt.gz", tissue_id = gtex_tissues)
    params:
        NumChunks = NumPvalsForPi1Chunks
    output:
        expand("results/pi1/PairwiseTraitsToCompare/Pairwise.{{QTLsRun}}{n}.txt.gz", n=range(1, 1+NumPvalsForPi1Chunks))
    #conda:
    #    "../envs/r_essentials.yml"
    wildcard_constraints:
        QTLsRun = 'Grouped|'
    resources:
        mem_mb = 32000
    log:
        "logs/CalculatePi1_GetTraitPairs_AllTraits.{QTLsRun}.log"
    shell:
        """
        Rscript scripts/CalculatePi1_GetTraitPairs_AllTraits.R {params.NumChunks} results/pi1/PairwiseTraitsToCompare/Pairwise.{wildcards.QTLsRun} {input.Permutation} &> {log}
        """

#rule GatherChunks:
#    input:
#        expand("pi1/PairwiseTraitsToCompare/Pairwise.snmf_{K}.{n}.txt.gz", n=range(1, 1+NumPvalsForPi1Chunks),
#        expand("pi1/PairwiseTraitsToCompare/Pairwise.transcripts_10.{n}.txt.gz", n=range(1, 1+NumPvalsForPi1Chunks))



rule GetPvalsForPi1AllTraitPairs:
    input:
        TraitsToCompare = "results/pi1/PairwiseTraitsToCompare/Pairwise.{QTLsRun}{chunk}.txt.gz",
        tabix_QTLsOut = expand("results/QTLs/GTEx/{tissue_id}/{{QTLsRun}}NominalPass.txt.tabix.gz", tissue_id=gtex_tissues)
    output:
        "results/pi1/PairwiseTraitsToCompare/P{QTLsRun}{chunk}.txt.gz"
    log:
        "logs/GetPvalsForPi1AllTraitPairs/Ascertainment{QTLsRun}.{chunk}.log"
    wildcard_constraints:
        QTLsRun = "Grouped|"
    resources:
        mem_mb = 32000
    shell:
        """
        python scripts/CalculatePi1_GetAscertainmentP_AllPairs.py {input.TraitsToCompare} {output} {input.tabix_QTLsOut} &> {log}
        """

rule GatherPvalsForPi1AllTraitPairs:
    input:
        expand("results/pi1/PairwiseTraitsToCompare/P{{QTLsRun}}{chunk}.txt.gz", chunk=range(1, NumPvalsForPi1Chunks+1))
    output:
        "results/pi1/input_tables/PairwisePi1Traits{QTLsRun}P.all.txt.gz"
    wildcard_constraints:
        QTLsRun = "Grouped|"
    shell:
        """
        cat <(zcat {input[0]} | head -1) <(zcat {input} | grep -v -P '^PC1\\t') | gzip - > {output}
        """

rule CalculatePi1Table:
    input:
        "results/pi1/input_tables/PairwisePi1Traits{QTLsRun}P.all.txt.gz"
    output:
        "results/pi1/pi1_tables/Pi1{QTLsRun}.txt"
    log:
        "logs/pi1_tables/Pi1{QTLsRun}.log"
    wildcard_constraints:
        QTLsRun = "Grouped|"
    resources:
        mem_mb = 32000
    shell:
        """
        Rscript scripts/CalculatePi1Statistics.R {input} {output} &> {log}
        """

               
              