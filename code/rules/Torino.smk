rule TorinoFactorization:
    input:
        'resources/count_matrices{filter_level}/{species}/{gene}.csv.gz'
    output:
        "results/torino/factorization{filter_level}/{species}/{gene}.rds"
    log:
        "logs/torino/factorization{filter_level}/{species}/{gene}.log"
    resources:
        mem_mb = 82000
    params:
        min_counts = 100
    wildcard_constraints:
        filter_level = '|_filtered|_extended'
    shell:
        """
        module load gsl
        module load R/4.1.0
        module load gcc/13.2.0
        {config[Rscript]} scripts/run_torino_factorization.R {input} {params} {wildcards.species} {output} &> {log}
        """

rule TorinoERVK31:
    input:
        '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/count_matrices_ervk31/Human/ERVK3-1.csv.gz'
    output:
        "/project2/yangili1/cfbuenabadn/torino_paper/code/results/ervk31/ervk31.rds"
    log:
        "logs/ervk31.log"
    resources:
        mem_mb = 48000
    shell:
        """
        module load gsl
        module load R/4.1.0
        module load gcc/13.2.0
        {config[Rscript]} scripts/run_torino_ervk31.R &> {log}
        """
        

rule collect_human_tables:
    input:
        expand('resources/count_matrices/Human/{gene}.csv.gz', gene=humanGenes)

rule collect_torino_human:
    input:
        expand("results/torino/factorization_filtered/Human/{gene}.rds", gene=humanGenes)


rule collectCattle_Torino:
    input:
        #expand('resources/count_matrices_filtered/Cattle/{gene}.csv.gz', gene=cattleGene_names)
        expand("results/torino/factorization_filtered/Cattle/{gene}.rds", gene=cattleGene_names)


rule collectCardoso_torino:
    input:
        #expand("results/torino/factorization/Macaque/{gene}.rds", gene=macaqueGenes),
        #expand("results/torino/factorization/Mouse/{gene}.rds", gene=mouseGenes),
        #expand("results/torino/factorization/Rat/{gene}.rds", gene=ratGenes),
        expand("results/torino/factorization/Chicken/{gene}.rds", gene=chickenGenes),
        #expand("results/torino/factorization/Rabbit/{gene}.rds", gene=rabbitGenes),


rule collectCardoso_torino_extended:
    input:
        expand("results/torino/factorization_extended/Macaque/{gene}.rds", gene=macaqueGenes),
        expand("results/torino/factorization_extended/Mouse/{gene}.rds", gene=mouseGenes),
        expand("results/torino/factorization_extended/Rat/{gene}.rds", gene=ratGenes),
        expand("results/torino/factorization_extended/Chicken/{gene}.rds", gene=chickenGenes),
        expand("results/torino/factorization_extended/Rabbit/{gene}.rds", gene=rabbitGenes),


def getParamsFilter(wildcards):
    if wildcards.filter_level == '':
        return '_unfiltered'
    else:
        return wildcards.filter_level

rule collect_loadings:
    output:
        'results/torino/tables/{species}{filter_level}/EL.K10.bed.gz',
    log:
        'logs/collect_EL/{species}{filter_level}/EL.log'
    resources:
        mem_mb = 48000
    wildcard_constraints:
        filter_level = '|_filtered|_extended',
        species = 'Cattle|Human|Sheep|Macaque|Rat|Mouse|Rabbit|Chicken|Opossum'
    params:
        getParamsFilter
    shell:
        """
        module load gsl
        module load R/4.1.0
        module load gcc/13.2.0
        ({config[Rscript]} scripts/collect_loadings.R {wildcards.species} {params}) &> {log}
        """
        

n_slice = 100


rule collect_torino:
    input:
        'results/torino/tables/Cattle_filtered/EL.K10.bed.gz',
        expand('results/torino/isoforms/Cattle.uncorrected_{n}.gtf', n = range(1, n_slice + 1))





rule TorinoKTest:
    input:
        'resources/count_matrices_filtered/Human/{gene}.csv.gz'
    output:
        "results/torino/k_tests/Human/{gene}.rds"
    log:
        "logs/torino/factorization_filtered/k_tests/{gene}.log"
    resources:
        mem_mb = 82000
    params:
        min_counts = 100
    shell:
        """
        module load gsl
        module load R/4.1.0
        module load gcc/13.2.0
        {config[Rscript]} scripts/run_torino_k_test.R {input} {params} {output} &> {log}
        """

rule collect_torino_ktest:
    input:
        expand("results/torino/k_tests/Human/{gene}.rds", gene=test_genes)