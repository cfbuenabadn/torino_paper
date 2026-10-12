rule GetIntronBeds:
    input:
        'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        'results/torino/Human/isoforms/isoforms.bed.gz',
        'results/torino/Human/isoforms/annotated.isoforms.tab.gz'
    output:
        'results/CoveragePlots/bed_files/introns/torino.retained_introns.bed.gz',
        'results/CoveragePlots/bed_files/introns/gencode.retained_introns.bed.gz',
        'results/CoveragePlots/bed_files/introns/vastdb.retained_introns.bed.gz',
        'results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed.gz',
        'results/CoveragePlots/bed_files/introns/torino.flanking_exons_1.bed',
        'results/CoveragePlots/bed_files/introns/gencode.flanking_exons_1.bed',
        'results/CoveragePlots/bed_files/introns/vastdb.flanking_exons_1.bed',
        'results/CoveragePlots/bed_files/introns/appris_flanking_exons_1.bed',
        'results/CoveragePlots/bed_files/introns/torino.flanking_exons_2.bed',
        'results/CoveragePlots/bed_files/introns/gencode.flanking_exons_2.bed',
        'results/CoveragePlots/bed_files/introns/vastdb.flanking_exons_2.bed',
        'results/CoveragePlots/bed_files/introns/appris_flanking_exons_2.bed'
    log:
        "/scratch/midway3/cnajar/logs/bigwigs/get_introns_bed.log"
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/get_retained_introns.py &> {log}
        """


###############################################

rule SortBEDForIntronMetaplotsRI:
    input:
        'results/CoveragePlots/bed_files/introns/{annot}.{region}.bed.gz'
    output:
        'results/CoveragePlots/bed_files/IntronsDeNovo/bed/{annot}.{region}.sorted.bed'
    log:
        'logs/sort_bed/{annot}.{region}.log'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        annot = 'torino|gencode|vastdb|appris',
        region = 'retained_introns'
    shell:
        """
        (zcat {input} | bedtools sort -i - > {output}) &> {log}
        """

rule SortBEDForIntronMetaplotsFE:
    input:
        'results/CoveragePlots/bed_files/introns/{annot}.{region}.bed'
    output:
        'results/CoveragePlots/bed_files/IntronsDeNovo/bed/{annot}.{region}.sorted.bed'
    log:
        'logs/sort_bed/{annot}.{region}.log'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        annot = 'torino|gencode|vastdb|appris',
        region = 'flanking_exons_1|flanking_exons_2'
    shell:
        """
        (bedtools sort -i {input} > {output}) &> {log}
        """

def getComputeMatrixParams2(wildcards):
    if wildcards.region == 'retained_introns':
        return '--regionBodyLength 1000'
    else:
        return '--regionBodyLength 200'

rule ComputeMatrixForCoverageMetaplots_Introns:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = "results/CoveragePlots/bed_files/IntronsDeNovo/bed/{annot}.{region}.sorted.bed" 
    output:
        mat = "results/CoveragePlots/bed_files/IntronsDeNovo/matrices/{Tissue}.{annot}.{region}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        annot = 'torino|gencode|vastdb|appris',
        region = 'retained_introns|flanking_exons_1|flanking_exons_2'
    log:
        "logs/Metaplots/ComputeMatrix.{Tissue}.{annot}.{region}.log"
    resources:
        mem_mb = 24000
    params:
        getComputeMatrixParams2
    shell:
        """
        computeMatrix scale-regions -S {input.bigwigs} -R {input.bed} {params} -b 50 -a 50 --startLabel "5' splice site" --endLabel "3' splice site" -o {output} &> {log}
        """


rule CollectIntronDeNovo:
    input:
        expand("results/CoveragePlots/bed_files/IntronsDeNovo/matrices/{Tissue}.{annot}.{region}.mat.gz",
               Tissue = tissue_sub_list, annot = ['torino', 'gencode', 'vastdb', 'appris'], 
               region = ['retained_introns', 'flanking_exons_1', 'flanking_exons_2'])


###############################################


rule GetCoverageGaps:
    output:
        'results/CoveragePlots/other_files/GTEx_overlapping_regions.bed.gz'
    log:
        'logs/coverage_gaps.log'
    resources:
        mem_mb = 8000
    shell:
        """
        python scripts/get_gaps_in_gtex_models.py &> {log}
        """

rule GetCassetteExonsBeds:
    input:
        'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        'results/torino/Human/isoforms/isoforms.bed.gz',
        'results/torino/Human/isoforms/annotated.isoforms.tab.gz',
        'results/CoveragePlots/other_files/GTEx_overlapping_regions.bed.gz'
    output:
        expand('results/CoveragePlots/bed_files/cassette_exons/{annotation}/{capture}.{region}.bed', 
                annotation = ['gencode', 'vastdb'], capture = ['exon_only', 'cassette_exon', 'intron_only'], 
                region = ['exon', 'flanking1', 'flanking2']),
        expand('results/CoveragePlots/bed_files/cassette_exons/torino/{annotation_match}.{region}.bed', 
                annotation_match = ['gencode', 'vastdb', 'junctions', 'unmatched'], region = ['exon', 'flanking1', 'flanking2']),
        'results/torino/Human/isoforms/cassette_exons.bed.gz'
    log:
        "/scratch/midway3/cnajar/logs/bigwigs/get_cassette_exons_bed.log"
    resources:
        mem_mb = 100000
    shell:
        """
        python scripts/get_cassette_exons.py &> {log}
        """


rule CollectCassetteExons:
    input:
        expand('results/CoveragePlots/bed_files/cassette_exons/{annotation}/{capture}.{region}.bed', 
                annotation = ['gencode', 'vastdb'], capture = ['exon_only', 'cassette_exon', 'intron_only'], 
                region = ['exon', 'flanking1', 'flanking2']),
        expand('results/CoveragePlots/bed_files/cassette_exons/torino/{annotation_match}.{region}.bed', 
                annotation_match = ['gencode', 'vastdb', 'junctions', 'unmatched'], region = ['exon', 'flanking1', 'flanking2']),
        'results/torino/Human/isoforms/cassette_exons.bed.gz'








rule GetIntronBeds_all:
    input:
        'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        'results/torino/Human/isoforms/isoforms.bed.gz',
        'results/torino/Human/isoforms/annotated.isoforms.tab.gz'
    output:
        ri = 'results/CoveragePlots/bed_files/introns/torino.all_retained_introns.bed',
        f1 = 'results/CoveragePlots/bed_files/introns/torino.all_flanking_exons_1.bed',
        f2 = 'results/CoveragePlots/bed_files/introns/torino.all_flanking_exons_2.bed',
    log:
        "/scratch/midway3/cnajar/logs/bigwigs/get_introns_bed_all.log"
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/get_all_retained_introns.py &> {log};
        """



















rule GetInputForIntronMetaplot:
    input:
        torino = 'results/CoveragePlots/bed_files/introns/torino.retained_introns.bed.gz',
        gencode = 'results/CoveragePlots/bed_files/introns/gencode.retained_introns.bed.gz',
        gencode_exons = 'resources/annotation/Human/bed/exons.all.bed.gz', 
        torino_exons = 'results/torino/Human/isoforms/isoforms.bed.gz',
        appris = 'results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed.gz'
    output:
        torino_only_temp = temp('results/CoveragePlots/bed_files/introns/torino_only_temp.retained_introns.bed'),
        torino_only = 'results/CoveragePlots/bed_files/introns/torino_only.retained_introns.bed',
        gencode_only_temp = temp('results/CoveragePlots/bed_files/introns/gencode_only_temp.retained_introns.bed'),
        gencode_only = 'results/CoveragePlots/bed_files/introns/gencode_only.retained_introns.bed',
        torino_gencode = 'results/CoveragePlots/bed_files/introns/torino_and_gencode.retained_introns.bed',
        appris = 'results/CoveragePlots/bed_files/introns/appris_introns.bed'
    log:
        "/scratch/midway3/cnajar/logs/bigwigs/get_introns_bed_for_metaplots.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (bedtools intersect -v -a {input.torino} -b {input.gencode} | bedtools sort -i - > {output.torino_only_temp}) &>> {log};
        (bedtools intersect -v -f 0.01 -a {output.torino_only_temp} -b {input.gencode_exons} > {output.torino_only}) &>> {log};
        (bedtools intersect -v -a {input.gencode} -b {input.torino} | bedtools sort -i - > {output.gencode_only_temp}) &>> {log};
        (bedtools intersect -v -f 0.01 -a {output.gencode_only_temp} -b {input.torino_exons} > {output.gencode_only}) &>> {log};
        (bedtools intersect -s -F 1 -f 1 -wa -a {input.torino} -b {input.gencode} | bedtools sort -i - > {output.torino_gencode}) &>> {log};
        (bedtools intersect -v -f 0.01 -F 0.01 -a {input.appris} -b {input.torino_exons} | bedtools intersect -v -f 0.01 -F 0.01 -a - -b {input.gencode_exons} | bedtools sort -i - > {output.appris}) &>> {log}
        """

rule GetApprisIntronsForMetaplotsAll:
    input:
        "results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed.gz"
    output:
        "results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed"
    resources:
        mem_mb = 12000
    shell:
        """
        zcat {input} > {output}
        """

rule get_bed_for_all_torino_metaplot:
    input:
        'results/CoveragePlots/bed_files/introns/torino.retained_introns.bed.gz'
    output:
        'results/CoveragePlots/bed_files/introns/torino.retained_introns.bed'
    log:
        "/scratch/midway3/cnajar/logs/get_unzipped_bed_torino_introns.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (zcat {input} > {output}) &> {log}
        """

rule GatherIntronMetaplotInput:
    input:
        'results/CoveragePlots/bed_files/introns/torino_only.retained_introns.bed',
        'results/CoveragePlots/bed_files/introns/gencode_only.retained_introns.bed',
        'results/CoveragePlots/bed_files/introns/torino_and_gencode.retained_introns.bed',
        'results/CoveragePlots/bed_files/introns/appris_introns.bed'


def getComputeMatrixParams(wildcards):
    if 'introns' in wildcards.ri_group:
        return '--regionBodyLength 1000'
    else:
        return '--regionBodyLength 200'


rule ComputeMatrixForCoverageMetaplots_sep:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = "results/CoveragePlots/bed_files/introns/{ri_group}.bed" 
    output:
        mat = "results/CoveragePlots/matrices/{Tissue}.{ri_group}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        ri_group = '|'.join(['torino_and_gencode.retained_introns', 'torino_only.retained_introns', 'gencode_only.retained_introns', 'appris_introns',
                             'torino.flanking_exons_1', 'gencode.flanking_exons_1', 'appris_flanking_exons_1',
                             'torino.flanking_exons_2', 'gencode.flanking_exons_2', 'appris_flanking_exons_2',
                             'vastdb.flanking_exons_1', 'vastdb.flanking_exons_2', 'torino.retained_introns', 'gencode.appris_introns', 
'torino.all_retained_introns', 'torino.all_flanking_exons_2', 'torino.all_flanking_exons_1'
])
    log:
        "logs/Metaplots/ComputeMatrix.{Tissue}.{ri_group}.log"
    resources:
        mem_mb = 12000
    params:
        getComputeMatrixParams
    shell:
        """
        computeMatrix scale-regions -S {input.bigwigs} -R {input.bed} {params} -b 50 -a 50 --startLabel "5' splice site" --endLabel "3' splice site" -o {output} &> {log}
        """




rule collect_metaplot_matrices:
    input:
        expand('results/CoveragePlots/matrices/{Tissue}.{ri_group}.mat.gz', Tissue=tissue_sub_list, ri_group = ['torino_and_gencode.retained_introns',
               'torino_only.retained_introns', 'gencode_only.retained_introns', 'appris_introns', 'torino.flanking_exons_1', 'gencode.flanking_exons_1',
               'appris_flanking_exons_1', 'torino.flanking_exons_2', 'gencode.flanking_exons_2', 'appris_flanking_exons_2', 'torino.retained_introns', 
               'gencode.appris_introns', 'torino.all_retained_introns', 'torino.all_flanking_exons_2', 'torino.all_flanking_exons_1'])







rule GetInputForIntronMetaplot_split:
    input:
        torino = 'results/CoveragePlots/bed_files/introns/torino.retained_introns.bed.gz',
        gencode = 'results/CoveragePlots/bed_files/introns/gencode.retained_introns.bed.gz',
        gencode_exons = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        torino_exons = 'results/torino/Human/isoforms/isoforms.bed.gz',
        appris = 'results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed.gz',
        vastdb = 'results/CoveragePlots/bed_files/introns/vastdb.retained_introns.bed.gz'
    output:
        torino_only_temp = temp('results/CoveragePlots/bed_files/introns_split/torino_only_temp.retained_introns.bed'),
        torino_only = 'results/CoveragePlots/bed_files/introns_split/torino_only.retained_introns.bed',
        gencode_only_temp = temp('results/CoveragePlots/bed_files/introns_split/gencode_only_temp.retained_introns.bed'),
        gencode_only = 'results/CoveragePlots/bed_files/introns_split/gencode_only.retained_introns.bed',
        torino_gencode = 'results/CoveragePlots/bed_files/introns_split/torino_and_gencode.retained_introns.bed',
        appris = 'results/CoveragePlots/bed_files/introns_split/appris_introns.bed',
        vastdb_only_temp = temp('results/CoveragePlots/bed_files/introns_split/vastdb_only_temp.retained_introns.bed'),
        vastdb_only = 'results/CoveragePlots/bed_files/introns_split/vastdb_only.retained_introns.bed',
        torino_vastdb = 'results/CoveragePlots/bed_files/introns_split/torino_and_vastdb.retained_introns.bed',
        gencode_vastdb = 'results/CoveragePlots/bed_files/introns_split/gencode_and_vastdb.retained_introns.bed',
        torino_gencode_vastdb = 'results/CoveragePlots/bed_files/introns_split/torino_and_gencode_and_vastdb.retained_introns.bed'
    log:
        "/scratch/midway3/cnajar/logs/bigwigs/get_introns_bed_for_metaplots_split.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (bedtools intersect -v -a {input.torino} -b {input.gencode} | bedtools intersect -v -a - -b {input.vastdb} | bedtools sort -i - > {output.torino_only_temp}) &>> {log};
        (bedtools intersect -v -f 0.01 -a {output.torino_only_temp} -b {input.gencode_exons} > {output.torino_only}) &>> {log};
        (bedtools intersect -v -a {input.gencode} -b {input.torino} | bedtools intersect -v -a - -b {input.vastdb}  | bedtools sort -i - > {output.gencode_only_temp}) &>> {log};
        (bedtools intersect -v -f 0.01 -a {output.gencode_only_temp} -b {input.torino_exons} > {output.gencode_only}) &>> {log};
        (bedtools intersect -v -a {input.vastdb} -b {input.torino} | bedtools intersect -v -a - -b {input.gencode}  | bedtools sort -i - > {output.vastdb_only_temp}) &>> {log};
        (bedtools intersect -v -f 0.01 -a {output.vastdb_only_temp} -b {input.torino_exons} | bedtools intersect -v -f 0.01 -a - -b {input.gencode_exons} > {output.vastdb_only}) &>> {log};
        (bedtools intersect -s -F 1 -f 1 -u -a {input.torino} -b {input.gencode} | bedtools intersect -v -a - -b {input.vastdb} | bedtools sort -i - > {output.torino_gencode}) &>> {log};
        (bedtools intersect -s -F 1 -f 1 -u -a {input.torino} -b {input.vastdb} | bedtools intersect -v -f 0.01 -a - -b {input.gencode_exons} | bedtools sort -i - > {output.torino_vastdb}) &>> {log};
        (bedtools intersect -s -F 1 -f 1 -u -a {input.gencode} -b {input.vastdb} | bedtools intersect -v -f 0.01 -a - -b {input.torino_exons} | bedtools sort -i - > {output.gencode_vastdb}) &>> {log};
        (bedtools intersect -s -F 1 -f 1 -u -a {input.torino} -b {input.gencode} | bedtools intersect -s -F 1 -f 1 -u -a - -b {input.vastdb} | bedtools sort -i - > {output.torino_gencode_vastdb}) &>> {log};
        (bedtools intersect -v -f 0.01 -F 0.01 -a {input.appris} -b {input.torino_exons} | bedtools intersect -v -f 0.01 -F 0.01 -a - -b {input.gencode_exons} | bedtools intersect -v -f 0.01 -F 0.01 -a - -b {input.vastdb} | bedtools sort -i - > {output.appris}) &>> {log}
        """

use rule ComputeMatrixForCoverageMetaplots_sep as ComputeMatrixForCoverageMetaplots_sep_split with:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = "results/CoveragePlots/bed_files/introns_split/{ri_group}.bed"
    output:
        mat = "results/CoveragePlots/matrices_split/{Tissue}.{ri_group}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        ri_group = '|'.join(['torino_and_gencode.retained_introns', 'torino_only.retained_introns', 'gencode_only.retained_introns', 'appris_introns',
'vastdb_only.retained_introns', 'torino_and_vastdb.retained_introns', 'gencode_and_vastdb.retained_introns', 'torino_and_gencode_and_vastdb.retained_introns'])
    log:
        "logs/Metaplots/ComputeMatrix_sep_split.{Tissue}.{ri_group}.log"


rule collect_metaplot_matrices_split:
    input:
        expand('results/CoveragePlots/matrices_split/{Tissue}.{ri_group}.mat.gz', Tissue=tissue_sub_list, ri_group = ['torino_and_gencode.retained_introns', 'torino_only.retained_introns', 'gencode_only.retained_introns', 'appris_introns', 'vastdb_only.retained_introns', 'torino_and_vastdb.retained_introns', 'gencode_and_vastdb.retained_introns', 'torino_and_gencode_and_vastdb.retained_introns'])




rule ComputeMatrixForCoverageMetaplots_CassetteExons:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = 'results/CoveragePlots/bed_files/cassette_exons/{annotation}/{capture}.{region}.bed',
    output:
        mat = "results/CoveragePlots/matrices/cassette_exons/{Tissue}.{annotation}.{capture}.{region}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        annotation = 'gencode|vastdb',
        capture = 'exon_only|cassette_exon|intron_only',
        region = 'exon|flanking1|flanking2'
    log:
        "logs/Metaplots/ComputeMatrix.CE.{Tissue}.{annotation}.{capture}.{region}.log"
    resources:
        mem_mb = 12000
    params:
        '--regionBodyLength 300 --binSize 5'
    shell:
        """
        computeMatrix scale-regions -S {input.bigwigs} -R {input.bed} {params} -b 150 -a 150 --startLabel "5' splice site" --endLabel "3' splice site" -o {output} &> {log}
        """

use rule ComputeMatrixForCoverageMetaplots_CassetteExons as ComputeMatrixForCoverageMetaplots_CassetteExons_torino with:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = 'results/CoveragePlots/bed_files/cassette_exons/torino/{annotation_match}.{region}.bed'
    output:
        mat = "results/CoveragePlots/matrices/cassette_exons/{Tissue}.torino.{annotation_match}.{region}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        annotation_match = 'gencode|vastdb|junctions|unmatched',
        region = 'exon|flanking1|flanking2'
    log:
        "logs/Metaplots/ComputeMatrix.CE.{Tissue}.torino.{annotation_match}.{region}.log"

rule CollectCassetteExons_metaplots:
    input:
        expand("results/CoveragePlots/matrices/cassette_exons/{Tissue}.{annotation}.{capture}.{region}.mat.gz", Tissue = tissue_sub_list, 
                annotation = ['gencode', 'vastdb'], capture = ['exon_only', 'cassette_exon', 'intron_only'], 
                region = ['exon', 'flanking1', 'flanking2']),
        expand("results/CoveragePlots/matrices/cassette_exons/{Tissue}.torino.{annotation_match}.{region}.mat.gz", Tissue = tissue_sub_list,
                annotation_match = ['gencode', 'vastdb', 'junctions', 'unmatched'], region = ['exon', 'flanking1', 'flanking2']),





rule SortTabixApprisIntrons:
    input:
        "results/CoveragePlots/bed_files/introns/gencode.appris_introns.bed"
    output:
        bed = "results/CoveragePlots/bed_files/introns/gencode.appris_introns.sorted.bed.gz",
        tbi = "results/CoveragePlots/bed_files/introns/gencode.appris_introns.sorted.bed.gz.tbi",
    log:
        "logs/Metaplots/sort_and_tabix_appris.log"
    resources:
        mem_mb = 24000
    shell:
        """
        (bedtools sort -i {input} | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """

rule GetCountsForApprisIntrons:
    input:
        "results/CoveragePlots/bed_files/introns/gencode.appris_introns.sorted.bed.gz",
    output:
        'results/CoveragePlots/bed_files/introns/tmp/gencode.appris_introns.{n}.bed.gz'
    log:
        "logs/Metaplots/GetIntronCounts.{n}.log"
    resources:
        mem_mb = 12000
    wildcard_constraints:
        n = "|".join([str(x) for x in range(1, 301)]),
    shell:
        """
        (python scripts/get_appris_intron_counts.py {wildcards.n} 300) &> {log}
        """


rule CollectIntronCounts:
    input:
        expand('results/CoveragePlots/bed_files/introns/tmp/gencode.appris_introns.{n}.bed.gz', n=range(1, 301))
    output:
        tmp = temp("results/CoveragePlots/bed_files/introns/gencode.appris_introns.counts.bed"),
        bed = "results/CoveragePlots/bed_files/introns/gencode.appris_introns.counts.sorted.bed.gz",
        tbi = "results/CoveragePlots/bed_files/introns/gencode.appris_introns.counts.sorted.bed.gz.tbi",
    log:
        "logs/collect_intron_counts.log"
    resources:
        mem_mb = 48000
    shell:
        """
        for I in {input}; do
            zcat $I >> {output.tmp}
        done;
        (bedtools sort -i {output.tmp} | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """




rule ComputeMatrixForCoverageMetaplots_CassetteExons_New:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = 'results/CoveragePlots/cassette_exons/bed/{annotation}.{region}.bed'
    output:
        mat = "results/CoveragePlots/cassette_exons/matrices/{Tissue}.{annotation}.{region}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        annotation = 'gencode|vastdb|torino',
        region = 'cassette_exon|flanking1|flanking2'
    log:
        "logs/Metaplots/ComputeMatrix.CE_new.{Tissue}.{annotation}.{region}.log"
    resources:
        mem_mb = 12000
    params:
        '--regionBodyLength 300 --binSize 5'
    shell:
        """
        computeMatrix scale-regions -S {input.bigwigs} -R {input.bed} {params} -b 150 -a 150 --startLabel "5' splice site" --endLabel "3' splice site" -o {output} &> {log}
        """

rule CollectCassetteExons_metaplots_new:
    input:
        expand("results/CoveragePlots/cassette_exons/matrices/{Tissue}.{annotation}.{region}.mat.gz", Tissue = tissue_sub_list, 
                annotation = ['gencode', 'vastdb', 'torino'], 
                region = ['cassette_exon', 'flanking1', 'flanking2']),



########################################################################

rule SortCassetteExonsBED:
    input:
        'results/torino/CoveragePlots/CassetteExons/bed/{annotation}.{exon}.bed'
    output:
        'results/torino/CoveragePlots/CassetteExons/bed/{annotation}.{exon}.sorted.bed'
    log:
        'logs/coverageplots/{annotation}.{exon}.log'
    resources:
        mem_mb = 12000
    params:
        exon = 'cassette_exons|flanking1_exons|flanking2_exons',
        annotation = 'torino|gencode|vastdb'
    shell:
        """
        (bedtools sort -i {input} > {output}) &> {log}
        """
    

rule ComputeMatrixForCoverageMetaplots_CassetteExons_Revision:
    input:
        bigwigs = "/project2/mstephens/cfbuenabadn/gtex-stm/code/CoveragePlots/bigwigs/{Tissue}.bw",
        bed = 'results/torino/CoveragePlots/CassetteExons/bed/{annotation}.{exon}.sorted.bed',
    output:
        mat = "results/torino/CoveragePlots/CassetteExons/matrices/{annotation}/{Tissue}.{exon}.mat.gz"
    wildcard_constraints:
        Tissue = "|".join(tissue_sub_list),
        exon = 'cassette_exons|flanking1_exons|flanking2_exons',
        annotation = 'torino|gencode|vastdb'
    log:
        "logs/Metaplots/ComputeMatrix.CE.{annotation}.{Tissue}.{exon}.log"
    resources:
        mem_mb = 12000
    params:
        '--regionBodyLength 300 --binSize 5'
    shell:
        """
        computeMatrix scale-regions -S {input.bigwigs} -R {input.bed} {params} -b 150 -a 150 --startLabel "5' splice site" --endLabel "3' splice site" -o {output} &> {log}
        """



rule CollectTorinoCoverageCE:
    input:
        expand("results/torino/CoveragePlots/CassetteExons/matrices/{annotation}/{Tissue}.{exon}.mat.gz", Tissue = tissue_sub_list, 
                exon = ['cassette_exons', 'flanking1_exons', 'flanking2_exons'],
annotation = ['torino', 'gencode', 'vastdb']
),
        

