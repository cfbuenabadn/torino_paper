rule get_torino_isoforms_slice_corrected:
    input:
        getJuncFile
    output:
        'results/torino/{species}/corrected_isoforms/tmp/factors_{i}.bed',
        'results/torino/{species}/corrected_isoforms/tmp/isoforms_{i}.bed',
        'results/torino/{species}/corrected_isoforms/tmp/missing_genes_{i}.txt'
    resources:
        mem_mb = 24000
    log:
        '/scratch/midway3/cnajar/logs/corrected_isoforms/collect_isoforms_{species}_{i}.log'
    params:
        n = n_slice,
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    shell:
        """
        python scripts/collect_isoforms_torino_corrected.py {wildcards.i} {params.n} {wildcards.species} {input} {wildcards.species} &> {log}
        """




rule collect_torino_isoform_slices_corrected_isoforms:
    input:
        expand('results/torino/{{species}}/corrected_isoforms/tmp/{{structure}}_{i}.bed', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        tmp = 'results/torino/{species}/corrected_isoforms/{structure}.bed',
        bed = 'results/torino/{species}/corrected_isoforms/{structure}.bed.gz',
        tbi = 'results/torino/{species}/corrected_isoforms/{structure}.bed.gz.tbi',
    wildcard_constraints:
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human',
        structure = 'factors|isoforms'
    log:
        'logs/torino/isoforms/{structure}.{species}.log'
    shell:
        """
        for I in {input}
        do
           cat ${{I}} >> {output.tmp}
        done
        (awk 'BEGIN {{OFS=FS="\\t"}} {{for (i=1; i<=NF; i++) if ($i == "") $i = "."; print}}' {output.tmp} | bedtools sort -i - | bgzip -c > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """


rule GetTranscriptEL_corrected:
    input:
        exons_bed = 'results/torino/{species}/corrected_isoforms/isoforms.bed.gz',
        el_bed = 'results/torino/tables/{species}{filter_level}/EL.K10.bed.gz'
    output:
        'results/torino/tables/{species}/corrected_isoforms{filter_level}.merged_isoforms.EL.bed.gz'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        filter_level = '|_filtered|_extended',
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    log:
        '/scratch/midway3/cnajar/logs/torino/corrected_isoforms_transcript_EL.{species}{filter_level}.log'
    shell:
        """
        python scripts/get_transcript_EL.py {input.exons_bed} {input.el_bed} {output} {wildcards.species} &> {log}
        """





rule annotate_snmf_isoforms_corrected2:
    input:
        gencode = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        snmf = 'results/torino/Human/corrected_isoforms/isoforms.bed.gz',
    output:
        temp('results/torino/Human/corrected_isoforms/tmp/corrected_annotated.isoforms_{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/corrected_isoforms/corrected_annotate_isoforms.snmf_{i}.log'
    params:
        n = n_slice
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
    shell:
        """
        python scripts/annotate_isoforms_corrected.py {input} {wildcards.i} {params} &> {log}
        """


rule collect_annotated_snmf_isoforms_corrected2:
    input:
        expand('results/torino/Human/corrected_isoforms/tmp/corrected_annotated.isoforms_{i}.tab.gz', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/torino/Human/corrected_isoforms/corrected_annotated.isoforms.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/torino/Human/corrected_isoforms/corrected_annotated.isoforms.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/torino/Human/corrected_isoforms/corrected_annotated.isoforms.tab
            fi
        done
        gzip results/torino/Human/corrected_isoforms/corrected_annotated.isoforms.tab
        """
