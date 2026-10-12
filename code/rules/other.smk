# An example collection of Snakemake rules imported in the main Snakefile.


rule GetCattleExons:
    input:
        'resources/annotation/gtf/Bos_taurus.ARS-UCD2.0.114.chr.gtf.gz'
    output:
        bed = "resources/annotation/bed/Cattle.protein_coding_exons.bed.gz",
        tbi = "resources/annotation/bed/Cattle.protein_coding_exons.bed.gz.tbi"
    log:
        "logs/cattle_pc_exons.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (zcat {input} | grep 'transcript_biotype "protein_coding"' - | awk -F'[\\t"]' '$3 == "exon" {{print "chr"$1, $4, $5, $10, $20, $7, $24, $14, $30}}' OFS='\\t' - | bedtools sort -i -  | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand\\tgene_type\\ttranscript_id\\ttranscript_type"}} {{print}}' FS="\\t" OFS="\\t" - | bgzip -c - > {output.bed}) &> {log}
        (tabix -p bed {output.bed}) &>> {log};
        """
        
rule GetApprisPrincipalGenes:
    input:
        "resources/annotation/bed/Cattle.protein_coding_exons.bed.gz"
    output:
        bed = "resources/annotation/bed/Cattle.protein_coding_genes.bed.gz",
        tbi = "resources/annotation/bed/Cattle.protein_coding_genes.bed.gz.tbi"
    log:
        "logs/leafcutter/appris_principal_genes.primary_assembly.log"
    resources:
        mem_mb = 48000
    shell:
        """
        (zcat {input} | tail -n+2 - | awk '$7=="protein_coding"' OFS='\\t' - | sort -k4 - | bedtools groupby -g 1,4,5,6 -c 2,3 -o min,max -i - | awk '{{print $1, $5, $6, $2, $3, $4}}' OFS='\\t' - | bedtools sort -i - | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand"}} {{print}}' FS='\\t' OFS='\\t' - | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log};
        """


rule GetAllAnnotatedExons:
    input:
        'resources/annotation/gtf/Bos_taurus.ARS-UCD2.0.114.chr.gtf.gz'
    output:
        bed = "resources/annotation/bed/Cattle.exons.bed.gz",
        tbi = "resources/annotation/bed/Cattle.exons.bed.gz.tbi"
    log:
        "logs/leafcutter/exon_annotation.primary_assembly.log"
    resources:
        mem_mb = 12000
    shell:
        """
        (zcat {input} | awk -F'[\\t"]' '$3 == "exon" {{print "chr"$1, $4, $5, $10, $20, $7, $24, $14, $30}}' OFS='\\t' - | bedtools sort -i -  | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand\\tgene_type\\ttranscript_id\\ttranscript_type"}} {{print}}' FS="\\t" OFS="\\t" - | bgzip -c - > {output.bed}) &> {log}
        (tabix -p bed {output.bed}) &>> {log};
        """

rule collect_exon_bed_cattle:
    input:
        "resources/annotation/bed/Cattle.exons.bed.gz",
        "resources/annotation/bed/Cattle.protein_coding_genes.bed.gz",
        "resources/annotation/bed/Cattle.protein_coding_exons.bed.gz"



########################


rule GetApprisPrincipalGenes:
    input:
        'resources/annotation/Human/bed/exons.apris_principal.bed.gz'
    output:
        bed = 'resources/annotation/Human/bed/genes.apris_principal.bed.gz',
        tbi = 'resources/annotation/Human/bed/genes.apris_principal.bed.gz.tbi'
    log:
        "logs/annotation/appris_principal_genes.primary_assembly.log"
    shell:
        """
        (zcat {input} | tail -n+2 - | awk '$7=="protein_coding"' OFS='\\t' - | sort -k4 - | bedtools groupby -g 1,4,5,6 -c 2,3 -o min,max -i - | awk '{{print $1, $5, $6, $2, $3, $4}}' OFS='\\t' - | bedtools sort -i - | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand"}} {{print}}' FS='\\t' OFS='\\t' - | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log};
        """


rule GetAnimalProteinCodingGenes:
    input:
        "resources/annotation/{species}/bed/exons.protein_coding.bed.gz"
    output:
        bed = "resources/annotation/{species}/bed/genes.protein_coding.bed.gz",
        tbi = "resources/annotation/{species}/bed/genes.protein_coding.bed.gz.tbi"
    log:
        "logs/annotation/{species}/protein_coding_genes.log"
    wildcard_constraints:
        species = 'Cattle|Sheep|Mouse|Chicken|Opossum|Macaque|Rabbit|Rat'
    resources:
        mem_mb = 48000
    shell:
        """
        (zcat {input} | tail -n+2 - | awk '$7=="protein_coding"' OFS='\\t' - | sort -k4 - | bedtools groupby -g 1,4,5,6 -c 2,3 -o min,max -i - | awk '{{print $1, $5, $6, $2, $3, $4}}' OFS='\\t' - | bedtools sort -i - | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand"}} {{print}}' FS='\\t' OFS='\\t' - | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log};
        """


rule GetCovEigTables:
    output:
        'results/torino/top_factors/human/GTEx.CovEig.txt.gz',
        'results/torino/top_factors/human/GTEx.CovEig.transcripts.txt.gz'
    log:
        'logs/get_coveig.log'
    resources:
        mem_mb = 12000
    shell:
        """
        python scripts/get_coveig.py &> {log}
        """









































rule get_snmf_isoforms_slice:
    input:
        getJuncFile
    output:
        'results/torino/{species}/isoforms/{correction_tag}_{i}.gtf'
    resources:
        mem_mb = 12000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/collect_isoforms_{species}_{correction_tag}_{i}.log'
    params:
        n = n_slice,
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
        correction_tag = '3prime_corrected|uncorrected',
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    shell:
        """
        python scripts/collect_isoforms_torino.py {wildcards.i} {params.n} {wildcards.correction_tag} {wildcards.species} {input} {wildcards.species} &> {log}
        """


rule collect_torino_gtf_slices:
    input:
        expand('results/torino/{{species}}/isoforms/{{correction_tag}}_{i}.gtf', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/torino/{species}/isoforms/gtf/{correction_tag}.gtf'
    wildcard_constraints:
        correction_tag = '3prime_corrected|uncorrected',
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    shell:
        """
        for I in {input}
        do
           cat ${{I}} >> {output}
        done
        """

rule get_snmf_isoform_exons:
    input:
        'results/torino/{species}/isoforms/gtf/{correction_tag}.gtf'
    output:
        tmp = temp('results/torino/{species}/isoforms/bed/{correction_tag}.exons.bed.gz'),
        bed = 'results/torino/{species}/isoforms/bed/{correction_tag}.exons.sorted.bed.gz'
    wildcard_constraints:
        correction_tag = '3prime_corrected|uncorrected',
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    log:
        '/scratch/midway3/cnajar/logs/torino.gtf_to_bed.{species}.{correction_tag}.log'
    shell:
        """
        python scripts/gtf2bed.py {input} {output.tmp} &> {log};
        (zcat {output.tmp} | bedtools sort -i - | bgzip -c > {output.bed}) &> {log}
        """
        
rule get_merged_isoforms:
    input:
        'results/torino/{species}/isoforms/bed/uncorrected.exons.sorted.bed.gz'
    output:
        tmp = temp('results/torino/{species}/isoforms/bed/merged_isoforms.exons.sorted.bed'),
        bed = 'results/torino/{species}/isoforms/bed/merged_isoforms.exons.sorted.bed.gz'
    wildcard_constraints:
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    resources:
        mem_mb = 12000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/{species}.merge_isoforms.log'
    shell:
        """
        python scripts/merge_isoforms.py {input} {output.tmp} &> {log};
        (bedtools sort -i {output.tmp} | bgzip -c > {output.bed}) &> {log}
        """













rule GetLongReadCountsGencode:
    output:
        temp('results/LongReads/tmp/long_read_gencode.counts.{i}.tab.gz'),
        #temp('results/LongReads/tmp/long_read_gencode.missing.{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/torino/longreadslong_read_gencode_{i}.log'
    params:
        n = 500
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, 500 + 1)]),
    shell:
        """
        python scripts/GetLongReadGencode.py {wildcards.i} {params} &> {log}
        """

rule CollectLongReadCountsGencode:
    input:
        expand('results/LongReads/tmp/long_read_gencode.counts.{i}.tab.gz', i = [str(j) for j in range(1, 500 + 1)])
    output:
        'results/LongReads/long_read_gencode.counts.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/LongReads/long_read_gencode.counts.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/LongReads/long_read_gencode.counts.tab
            fi
        done
        gzip results/LongReads/long_read_gencode.counts.tab
        """

rule CollectLongReadMissingGencode:
    input:
        expand('results/LongReads/tmp/long_read_gencode.missing.{i}.tab.gz', i = [str(j) for j in range(1, 400 + 1)])
    output:
        'results/LongReads/long_read_gencode.missing.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/LongReads/long_read_gencode.missing.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/LongReads/long_read_gencode.missing.tab
            fi
        done
        gzip results/LongReads/long_read_gencode.missing.tab
        """

rule CollectAllLongReadsGencode:
    input:
        'results/LongReads/long_read_gencode.counts.tab.gz',
        #'results/LongReads/long_read_gencode.missing.tab.gz'


















rule GetLongReadCountsGencodeAll:
    output:
        temp('results/LongReads/tmp/long_read_gencode_all.counts.{i}.tab.gz'),
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/torino/longreadslong_read_gencode_all_{i}.log'
    params:
        n = 500
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, 500 + 1)]),
    shell:
        """
        python scripts/GetLongReadsAllGencode.py {wildcards.i} {params} &> {log}
        """

rule CollectLongReadCountsGencodeAll:
    input:
        expand('results/LongReads/tmp/long_read_gencode_all.counts.{i}.tab.gz', i = [str(j) for j in range(1, 500 + 1)])
    output:
        'results/LongReads/long_read_gencode_all.counts.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/LongReads/long_read_gencode_all.counts.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/LongReads/long_read_gencode_all.counts.tab
            fi
        done
        gzip results/LongReads/long_read_gencode_all.counts.tab
        """



rule CollectAllLongReadsGencodeAll:
    input:
        'results/LongReads/long_read_gencode_all.counts.tab.gz',

