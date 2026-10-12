rule FirstExons:
    input:
        "resources/annotation/{species}/bed/exons.all.bed.gz",
        "resources/annotation/{species}/bed/genes.protein_coding.bed.gz"
    output:
        tmp_first = temp('resources/annotation/{species}/bed_counts/first_exons.bed'),
        tmp_genes = temp('resources/annotation/{species}/bed_counts/genes.bed'),
        first = 'resources/annotation/{species}/bed_counts/first_exons.sorted.bed.gz',
        genes = 'resources/annotation/{species}/bed_counts/genes.sorted.bed.gz'
    wildcard_constraints:
        species = 'Cattle|Sheep|Mouse|Macaque|Rat|Rabbit|Chicken'
    log:
        "logs/get_annot/{species}.first_exons.log"
    resources:
        mem_mb = 24000
    shell:
        """
        (python scripts/get_first_exons.py {wildcards.species}) &> {log};
        (bedtools sort -header -i {output.tmp_first} | bgzip -c - > {output.first}) &>> {log}
        (bedtools sort -header -i {output.tmp_genes} | bgzip -c - > {output.genes}) &>> {log}
        """
        
        
def GetSamplesForQuant(wildcards):
    if wildcards.species == 'Cattle':
        return f'resources/FarmGTEx/cattleGTEx/bed/{wildcards.tissue}/{wildcards.sample}.bed.gz'
    elif wildcards.species == 'Sheep':
        return f'resources/FarmGTEx/sheepGTEx/bed/{wildcards.tissue}/{wildcards.sample}.bed.gz'

#def GetSamplesForQuantCardoso(wildcards):
#    return f'resources/Cardoso/{species}/bed/{wildcards.sample}'
        
rule QuantAnnotFromBedGraph:
    input:
        first_exons = 'resources/annotation/{species}/bed_counts/first_exons.sorted.bed.gz',
        genes = 'resources/annotation/{species}/bed_counts/genes.sorted.bed.gz',
        sample_bed = GetSamplesForQuant
    output:
        first_exons = 'resources/annotation/{species}/bed_counts/sample_counts/{tissue}/{sample}.first_exons.bed.gz',
        genes = 'resources/annotation/{species}/bed_counts/sample_counts/{tissue}/{sample}.genes.bed.gz',
    wildcard_constraints:
        species = 'Cattle|Sheep',
    log:
        "logs/get_quant/{species}/{tissue}/{sample}.log"
    resources:
        mem_mb = 48000
    shell:
        """
        (bedtools intersect -wo -a {input.sample_bed} -b {input.first_exons} | bgzip -c > {output.first_exons}) &> {log};
        (bedtools intersect -wo -a {input.sample_bed} -b {input.genes} | bgzip -c > {output.genes}) &> {log};
        """


rule collectQuantFarmGTEx:
    input:
        expand('resources/annotation/Cattle/bed_counts/sample_counts/{tissue}/{sample}.first_exons.bed.gz', zip, 
        tissue=list(cattleGTEx.Tissue), sample=list(cattleGTEx.Sample)),
        expand('resources/annotation/Sheep/bed_counts/sample_counts/{tissue}/{sample}.first_exons.bed.gz', zip, 
                tissue=list(sheepGTEx.Tissue), sample=list(sheepGTEx.Sample))


#rule FeatureCountsCardoso
# Note: quant_tables (bed_counts/sample_counts) were deleted for space; easy to rerun

rule FirstExonMeta:
    output:
        'resources/annotation/{species}/tables/{tissue}/gene.counts.txt.gz',
        'resources/annotation/{species}/tables/{tissue}/first_exon.counts.txt.gz',
        'resources/annotation/{species}/tables/{tissue}/gene.coverage.txt.gz',
        'resources/annotation/{species}/tables/{tissue}/first_exon.coverage.txt.gz'
    log:
        "logs/meta/{tissue}.{species}.first_exons.log"
    resources:
        mem_mb = 80000
    wildcard_constraints:
        species = 'Cattle|Sheep'
    shell:
        """
        python scripts/merge_first_count_tables.py {wildcards.species} {wildcards.tissue}
        """
        
        
rule CollectFirstExonMeta:
    input:
        expand('resources/annotation/Cattle/tables/{tissue}/gene.counts.txt.gz', tissue = cattle_tissues),
        expand('resources/annotation/Sheep/tables/{tissue}/gene.counts.txt.gz', tissue = sheep_tissues),


        


rule GetTotalMetadataPerTissue:
    output:
        'resources/metadata/{species}/tissue_tables/metadata.{tissue}.txt'
    log:
        "logs/summary_metadata/{species}/{tissue}.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (python scripts/first_exon_metadata.py {wildcards.species} {wildcards.tissue}) &> {log}
        """

rule collectFarmMetadata:
    input:
        expand('resources/metadata/Sheep/tissue_tables/metadata.{tissue}.txt',
                tissue = sheep_tissues),
        #expand('resources/metadata/Cattle/tissue_tables/metadata.{tissue}.txt',
        #        tissue = cattle_tissues)

def getExonCoverageInput(wildcards):
    if wildcards.species == 'Sheep':
        return expand('resources/annotation/Sheep/tables/{tissue}/gene.counts.txt.gz', tissue = sheep_tissues)
    elif wildcards.species == 'Cattle':
        return expand('resources/annotation/Cattle/tables/{tissue}/gene.counts.txt.gz', tissue = cattle_tissues)

rule collectExonCoverageMetadata:
    input:
        getExonCoverageInput
    output:
        'resources/metadata/{species}/first_exon.coverage.tab.gz',
        'resources/metadata/{species}/gene.coverage.tab.gz',
        'resources/metadata/{species}/first_exon.counts.tab.gz',
        'resources/metadata/{species}/gene.counts.tab.gz'
    log:
        'logs/exon_cov/{species}.log'
    resources:
        mem_mb = 24000
    shell:
        """
        (python scripts/collect_first_exon_metadata.py {wildcards.species}) &> {log}
        """
        

def GetLibSizeInput(wildcards):
    if wildcards.species == 'Sheep':
        return expand('resources/metadata/Sheep/tissue_tables/metadata.{tissue}.txt', tissue = sheep_tissues)
    elif wildcards.species == 'Cattle':
        expand('resources/metadata/Cattle/tissue_tables/metadata.{tissue}.txt', tissue = cattle_tissues)


rule CollectLibSize:
    input:
        GetLibSizeInput
    output:
        'resources/metadata/{species}/library_size.txt'
    log:
        "logs/libsize/{species}.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (cat {input} > {output}) & 
        """

rule DownloadCattle:
    output:
        "resources/FarmGTEx/cattleGTEx/download_token.txt"
    log:
        "logs/resources/download_cattle.log"
    shell:
        """
        scripts/DownloadCattleGTEx.sh &> {log}
        """

rule tabixCattle:
    input:
        "/project2/yangili1/cfbuenabadn/torino_paper/code/resources/FarmGTEx/cattleGTEx/bed/{tissue}/{sample}.bed.gz"
    output:
        "/project2/yangili1/cfbuenabadn/torino_paper/code/resources/FarmGTEx/cattleGTEx/bed/{tissue}/{sample}.bed.gz.tbi"
    log:
        "logs/farmgtex/cattle/tabix.{tissue}.{sample}.log"
    resources:
        mem_mb = 8000
    shell:
        """
        (tabix -p bed {input}) &> {log}
        """

rule collectTabixCattle:
    input:
        expand("/project2/yangili1/cfbuenabadn/torino_paper/code/resources/FarmGTEx/cattleGTEx/bed/{tissue}/{sample}.bed.gz.tbi", 
               zip, tissue=list(cattleGTEx.Tissue), sample=list(cattleGTEx.Sample))


rule GetGeneCountsERVK31:
    output:
        'resources/count_matrices_ervk31/Human/ERVK3-1.csv.gz'
    log:
        'logs/get_counts/Human.ERVK3-1.log'
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/prepare_counts_ERVK3-1.py &> {log}
        """


rule GetGeneCounts:
    output:
        'resources/count_matrices/{species}/{gene}.csv.gz'
    log:
        'logs/get_counts/{species}.{gene}.log'
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/prepare_count_tables.py {wildcards.species} {wildcards.gene} &> {log}
        """

rule GetGeneCountsCardosoExtended:
    output:
        'resources/count_matrices_extended/{species}/{gene}.csv.gz'
    log:
        'logs/get_counts/{species}.{gene}.log'
    wildcard_constraints:
        species = 'Macaque|Mouse|Rat|Rabbit|Cattle|Chicken'
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/prepare_count_tables_extended.py {wildcards.species} {wildcards.gene} &> {log}
        """

rule collectCardosoExtended:
    input:
        'resources/count_matrices_extended/Macaque/CABLES1.csv.gz',
        'resources/count_matrices_extended/Macaque/FAM98B.csv.gz',


rule collectGTEX_counts:
    input:
        expand('resources/count_matrices_filtered/Human/{gene}.csv.gz', gene = humanGenes)# ['SRSF3', 'SYNGR1', 'SRSF7', 'PFKP'])

#rule collectCardoso_torino:
#    input:
#        expand('resources/count_matrices/Macaque/{gene}.csv.gz', gene=macaqueGenes),
#        expand('resources/count_matrices/Mouse/{gene}.csv.gz', gene=mouseGenes),
#        expand('resources/count_matrices/Rat/{gene}.csv.gz', gene=ratGenes),
#        expand('resources/count_matrices/Chicken/{gene}.csv.gz', gene=chickenGenes),
#        expand('resources/count_matrices/Rabbit/{gene}.csv.gz', gene=rabbitGenes),
        
rule FilterGeneCounts:
    input:
        'resources/count_matrices/{species}/{gene}.csv.gz',
        'Annotations/gencode.v44.primary_assembly.exons.bed.gz',
        'Annotations/gencode.v44.primary_assembly.appris_principal_exons.bed.gz'
    output:
        'resources/count_matrices_filtered/{species}/{gene}.csv.gz',
        'resources/filter_stats/{species}/{gene}.stats'
    resources:
        mem_mb = 12000
    wildcard_constraints:
        species = 'Human'
    log:
        'logs/filter_counts/{species}/{gene}.log'
    shell:
        """
        (python scripts/filter_counts.py {wildcards.gene} {wildcards.species}) &> {log}
        """

use rule FilterGeneCounts as FilterGeneCounts_Animals with:
    input:
        'resources/count_matrices/{species}/{gene}.csv.gz',
        'resources/annotation/{species}/bed/exons.all.bed.gz',
        'resources/annotation/{species}/bed/exons.protein_coding.bed.gz'
    wildcard_constraints:
        species = 'Cattle|Sheep|Mouse|Macaque|Rabbit|Chicken'


rule collectSheep_Torino:
    input:
        #expand('resources/count_matrices_filtered/Cattle/{gene}.csv.gz', gene=sheepGenes)
        expand("results/torino/factorization_filtered/Sheep/{gene}.rds", gene=sheepGenes)


        
        #['SRSF3', 'SYNGR1', 'SRSF7', 'PFKP']

rule cardoso_annotations:
    input:
        expand('resources/annotation/{species}/bed/genes.protein_coding.bed.gz', species = ['Chicken']),#['Macaque', 'Mouse', 'Rat', 'Chicken', 'Rabbit']),
        expand("resources/annotation/{species}/bed/exons.protein_coding.bed.gz", species = ['Chicken']),#['Macaque', 'Rat', 'Chicken', 'Rabbit']),
        expand("resources/annotation/{species}/bed/exons.all.bed.gz", species = ['Chicken']),# ['Macaque', 'Rat', 'Chicken', 'Rabbit', 'Mouse']),
        #"resources/annotation/Mouse/bed/exons.apris_principal.bed.gz"

rule cardoso_test:
    input:
        expand('resources/count_matrices/{species}/{gene}.csv.gz', 
            species = ['Macaque', 'Mouse', 'Rat', 'Chicken', 'Rabbit'], gene = ['NTKMT', 'MRPL28', 'PFKP', 'PKM', 'PRBP1', 
'RTN2', 'S100A6', 'SRSF3', 'SRSF7', 'SYNGR1', 'TUBB2B', 'DCTN5', 'TSC22D4', 'CPEB3']
)


#rule collectRDS_animals:
#    input:
#        expand("results/torino/factorization/{species}/{gene}.rds", 
#               zip, species=list(gene_sepecies.species), gene=list(gene_sepecies.gene))


AnimalChunks = range(1, 1+400) 
rule CollectJSDChunk:
    output:
        'results/animal_comparisons/tmp/JSD.{chunk}.txt'
    resources:
        mem_mb = 36000
    wildcard_constraints:
        n = "|".join(str(i) for i in AnimalChunks),
    log:
        "logs/animals_jsd.{chunk}.log"
    shell:
        """
        python scripts/animal_jsd.py {wildcards.chunk} {output} &> {log}
        """

rule CollectJSD:
    input:
        expand('results/animal_comparisons/tmp/JSD.{chunk}.txt', chunk = AnimalChunks)
    output:
        tab = "results/animal_comparisons/JSD.tab.gz",
    resources:
        mem_mb = 52000
    log:
        'logs/collect_jsd.log'
    shell:
        """
        for I in {input}
        do
           (cat ${{I}} >> results/animal_comparisons/JSD.tab) &>> {log}
        done;
        (gzip results/animal_comparisons/JSD.tab) &>> {log}
        """

def GetMetaDataForSpecies(wildcards):
    if wildcards.species == 'Human':
        return '/project2/mstephens/cfbuenabadn/gtex-stm/data/sample.tsv'
    elif wildcards.species == 'Cattle':
        return 'config/cattleGTEx_metadata.txt'
    else:
        return 'config/cardoso_samples.tsv'

def GetAnnotForSpecies(wildcards):
    if wildcards.species == 'Human':
        return "results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz"
    else:
        return 'results/torino/{species}/isoforms/annotated.isoforms.tab.gz'


rule GetCorrectedELForSpecies:
    input:
        EL = 'results/torino/tables/{species}/isoforms.EL.sorted.bed.gz',
        annot = GetAnnotForSpecies,
    output:
        'results/torino/tables/{species}/SSMD/EL_corrected.tsv',
    log:
        'logs/ssmd/EL_{species}.log'
    resources:
        mem_mb = 36000
    shell:
        """
        (python scripts/get_corrected_EL.py --species {wildcards.species} --el_path {input.EL} --annot_path {input.annot} --out_dir results/torino/tables/{wildcards.species}/SSMD) &> {log}
        """

rule GetSSMDForSpecies:
    input:
        EL = 'results/torino/tables/{species}/SSMD/EL_corrected.tsv',
        meta = GetMetaDataForSpecies,
    output:
        'results/torino/tables/{species}/SSMD/ssmd.tsv.gz',
        'results/torino/tables/{species}/SSMD/median_diff.tsv.gz',
    log:
        'logs/ssmd/{species}.log'
    resources:
        mem_mb = 36000
    shell:
        """
        (python scripts/get_ssmd_per_isoforms.py --species {wildcards.species} --el_path {input.EL} --meta_path {input.meta} --out_dir results/torino/tables/{wildcards.species}/SSMD) &> {log}
        """
        

rule collect_species_ssmd:
    input:
        expand('results/torino/tables/{species}/SSMD/ssmd.tsv.gz', species = ['Human', 'Cattle', 'Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque'])

#####

rule compute_pval:
    input:
        ssmd  = "results/torino/tables/{species}/SSMD/ssmd.tsv.gz",
        mdiff = "results/torino/tables/{species}/SSMD/median_diff.tsv.gz",
    output:
        null    = "results/torino/tables/{species}/SSMD/PVal/null_fit.tsv",
        ziso    = "results/torino/tables/{species}/SSMD/PVal/zscore_isoform.tsv.gz",
        obs     = "results/torino/tables/{species}/SSMD/PVal/obs_max_ssmd.tsv.gz",
        obsz    = "results/torino/tables/{species}/SSMD/PVal/obs_max_z.tsv.gz",
        mdiff   = "results/torino/tables/{species}/SSMD/PVal/obs_median_diff.tsv.gz",
        argmax  = "results/torino/tables/{species}/SSMD/PVal/argmax_isoform.tsv.gz",
        niso    = "results/torino/tables/{species}/SSMD/PVal/n_isoforms.tsv.gz",
        pval    = "results/torino/tables/{species}/SSMD/PVal/pval.tsv.gz",
        fdr     = "results/torino/tables/{species}/SSMD/PVal/fdr.tsv.gz",
        sig01   = "results/torino/tables/{species}/SSMD/PVal/sig_0p01.tsv.gz",
        sig05   = "results/torino/tables/{species}/SSMD/PVal/sig_0p05.tsv.gz",
        sig10   = "results/torino/tables/{species}/SSMD/PVal/sig_0p1.tsv.gz",
    params:
        out_dir = "results/torino/tables/{species}/SSMD/PVal",
    threads: 1
    resources:
        mem_mb = 32000,
    log:
        "logs/ssmd_pval/{species}.log"
    shell:
        """
        (python scripts/get_pval_per_gene.py \
            --ssmd_path        {input.ssmd} \
            --median_diff_path {input.mdiff} \
            --out_dir          {params.out_dir}) &> {log}
        """


####

rule GetSSMDForSpecies_test:
    input:
        EL = 'results/torino/tables/{species}/SSMD/EL_corrected.tsv',
        meta = GetMetaDataForSpecies,
    output:
        'results/torino/tables/{species}/SSMD_test/ssmd.tsv.gz',
        'results/torino/tables/{species}/SSMD_test/median_diff.tsv.gz',
    log:
        'logs/SSMD_test/{species}.log'
    resources:
        mem_mb = 36000
    shell:
        """
        (python scripts/get_ssmd_per_isoforms.py --species {wildcards.species} --el_path {input.EL} --meta_path {input.meta} --out_dir results/torino/tables/{wildcards.species}/SSMD_test) &> {log}
        """

rule compute_pval_test:
    input:
        ssmd  = "results/torino/tables/{species}/SSMD_test/ssmd.tsv.gz",
        mdiff = "results/torino/tables/{species}/SSMD_test/median_diff.tsv.gz",
    output:
        null    = "results/torino/tables/{species}/SSMD_test/PVal/null_fit.tsv",
        ziso    = "results/torino/tables/{species}/SSMD_test/PVal/zscore_isoform.tsv.gz",
        obs     = "results/torino/tables/{species}/SSMD_test/PVal/obs_max_ssmd.tsv.gz",
        obsz    = "results/torino/tables/{species}/SSMD_test/PVal/obs_max_z.tsv.gz",
        mdiff   = "results/torino/tables/{species}/SSMD_test/PVal/obs_median_diff.tsv.gz",
        argmax  = "results/torino/tables/{species}/SSMD_test/PVal/argmax_isoform.tsv.gz",
        niso    = "results/torino/tables/{species}/SSMD_test/PVal/n_isoforms.tsv.gz",
        pval    = "results/torino/tables/{species}/SSMD_test/PVal/pval.tsv.gz",
        fdr     = "results/torino/tables/{species}/SSMD_test/PVal/fdr.tsv.gz",
        sig01   = "results/torino/tables/{species}/SSMD_test/PVal/sig_0p01.tsv.gz",
        sig05   = "results/torino/tables/{species}/SSMD_test/PVal/sig_0p05.tsv.gz",
        sig10   = "results/torino/tables/{species}/SSMD_test/PVal/sig_0p1.tsv.gz",
    params:
        out_dir = "results/torino/tables/{species}/SSMD_test/PVal",
    threads: 1
    resources:
        mem_mb = 32000,
    log:
        "logs/ssmd_pval/{species}.log"
    shell:
        """
        (python scripts/get_pval_per_gene.py \
            --ssmd_path        {input.ssmd} \
            --median_diff_path {input.mdiff} \
            --out_dir          {params.out_dir}) &> {log}
        """



rule collect_species_fdr:
    input:
        expand("results/torino/tables/{species}/SSMD/PVal/fdr.tsv.gz", species = ['Human', 'Cattle', 'Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque'])

rule collect_species_fdr_test:
    input:
        expand("results/torino/tables/{species}/SSMD_test/PVal/fdr.tsv.gz", species = ['Human', 'Cattle', 'Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque'])


##########################################################################

rule compute_fdr:
    input:
        el    = "results/torino/tables/{species}/SSMD/EL_corrected.tsv",
        meta  = GetMetaDataForSpecies,
    output:
        fdr   = "results/torino/tables/{species}/SSMD/FDR/fdr.tsv.gz",
        obs   = "results/torino/tables/{species}/SSMD/FDR/obs_max_ssmd.tsv.gz",
        sig01 = "results/torino/tables/{species}/SSMD/FDR/sig_0p01.tsv.gz",
        sig05 = "results/torino/tables/{species}/SSMD/FDR/sig_0p05.tsv.gz",
        sig10 = "results/torino/tables/{species}/SSMD/FDR/sig_0p1.tsv.gz",
    params:
        out_dir = "results/torino/tables/{species}/SSMD/FDR",
        k       = 1000,
    threads: 4  # one per tissue for Human; set to len(tissues) for your species
    resources:
        mem_mb = 52000,
    log:
        'logs/ssmd_fdr/{species}.log'
    shell:
        """
        (python scripts/get_fdr_per_gene.py \
            --species   {wildcards.species} \
            --el_path   {input.el} \
            --meta_path {input.meta} \
            --out_dir   {params.out_dir} \
            --k         {params.k} \
            --workers   {threads}) &> {log}
        """


rule compute_zscore:
    input:
        el   = "results/torino/tables/{species}/SSMD/EL_corrected.tsv",
        meta = GetMetaDataForSpecies,
    output:
        obs       = "results/torino/tables/{species}/SSMD/ZScore/obs_max_ssmd.tsv.gz",
        null_mean = "results/torino/tables/{species}/SSMD/ZScore/null_mean.tsv.gz",
        null_std  = "results/torino/tables/{species}/SSMD/ZScore/null_std.tsv.gz",
        zscore    = "results/torino/tables/{species}/SSMD/ZScore/zscore.tsv.gz",
    params:
        out_dir = "results/torino/tables/{species}/SSMD/ZScore",
        k       = 1000,
    threads: 4
    resources:
        mem_mb = 52000,
    log:
        "logs/ssmd_zscore/{species}.log"
    shell:
        """
        (python scripts/get_zscore_per_gene.py \
            --species   {wildcards.species} \
            --el_path   {input.el} \
            --meta_path {input.meta} \
            --out_dir   {params.out_dir} \
            --k         {params.k} \
            --workers   {threads}) &> {log}
        """


rule compute_storey_fdr:
    input:
        obs = "results/torino/tables/{species}/SSMD/ZScore/obs_max_ssmd.tsv.gz",
    output:
        fdr   = "results/torino/tables/{species}/SSMD/StoreyFDR/fdr_storey.tsv.gz",
        pi0   = "results/torino/tables/{species}/SSMD/StoreyFDR/pi0.tsv",
        sig01 = "results/torino/tables/{species}/SSMD/StoreyFDR/sig_0p01.tsv.gz",
        sig05 = "results/torino/tables/{species}/SSMD/StoreyFDR/sig_0p05.tsv.gz",
        sig10 = "results/torino/tables/{species}/SSMD/StoreyFDR/sig_0p1.tsv.gz",
    params:
        out_dir        = "results/torino/tables/{species}/SSMD/StoreyFDR",
        null_threshold = 1.0,
    threads: 1
    resources:
        mem_mb = 16000,
    log:
        "logs/ssmd_storey_fdr/{species}.log"
    shell:
        """
        (python scripts/get_storey_fdr_per_gene.py \
            --obs_path       {input.obs} \
            --null_threshold {params.null_threshold} \
            --out_dir        {params.out_dir}) &> {log}
        """


#rule GetFDRSpecies:
#    input:
#        EL = 'results/torino/tables/{species}/SSMD/EL_corrected.tsv',
#        meta = GetMetaDataForSpecies,
#        ssmd = 'results/torino/tables/{species}/SSMD/ssmd.tsv.gz',
#    output:
#        expand('results/torino/tables/{{species}}/SSMD/FDR/thresholds_fdr_{fdr}.tsv.gz', fdr = ['0p01', '0p05', '0p1'])
#    log:
#        'logs/ssmd_fdr/{species}.log'
#    resources:
#        mem_mb = 36000
#    shell:
#        """
#        (python scripts/run_thresholds.py --species {wildcards.species} --el_path {input.EL} --meta_path {input.meta} --ssmd_path {input.ssmd} --k 100 --out_dir results/torino/tables/{wildcards.species}/SSMD/FDR) &> {log}
#        """

#rule collect_species_fdr:
#    input:
#        expand('results/torino/tables/{species}/SSMD/StoreyFDR/fdr_storey.tsv.gz', species = ['Human', 'Cattle', 'Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque'])
