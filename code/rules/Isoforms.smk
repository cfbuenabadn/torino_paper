rule runLeafCutter2:
    input:
        junc_files = "resources/Cardoso/{species}/junc_files.txt"
    output:
        'results/leafcutter2/{species}/leafcutter2.junction_counts.unclassified.gz'
    log:
        "logs/leafcutter2/{species}/lc2.log"
    resources:
        mem_mb = 24000
    params:
        'results/leafcutter2/{species}/'
    wildcard_constraints:
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum'
    shell:
        """
        (python /project/yangili1/cfbuenabadn/leafcutter2/scripts/leafcutter2.py -j {input} -C -r {params}) &> {log}
        """

rule LeafCutter2_to_bed:
    input:
        'results/leafcutter2/{species}/leafcutter2.junction_counts.unclassified.gz'
    output:
        temp('results/leafcutter2/{species}/junctions.tab')
    log:
        'logs/leafcutter2/{species}/get_bed.log'
    wildcard_constraints:
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum'
    resources:
        mem_mb = 8000
    shell:
         """
        (python scripts/leafcutter2_to_bed.py {wildcards.species}) &> {log}
        """

rule bgzip_lc2:
    input:
        'results/leafcutter2/{species}/junctions.tab'
    output:
        sorted_bed = temp('results/leafcutter2/{species}/junctions.sorted.tab'),
        bed = 'results/leafcutter2/{species}/junctions.sorted.tab.gz',
        tbi = 'results/leafcutter2/{species}/junctions.sorted.tab.gz.tbi'
    log:
        'logs/leafcutter2/{species}/bgzip.log'
    wildcard_constraints:
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum'
    resources:
        mem_mb = 12000
    shell:
        """
        (bedtools sort -i {input} > {output.sorted_bed}) &> {log};
        (bgzip {output.sorted_bed} -c > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """

rule collectjunctions_species:
    input:
        expand('results/torino/tables/{species}/isoforms.merged_isoforms.EL.bed.gz', species = ['Mouse', 'Rat', 'Macaque', 'Rabbit', 'Chicken'])




def getJuncFile(wildcards):
    if wildcards.species == 'Cattle':
        return 'results/leafcutter2/Cattle/junctions.bed.gz'
    if wildcards.species == 'Mouse':
        return 'results/leafcutter2/Mouse/junctions.sorted.tab.gz'
    if wildcards.species == 'Macaque':
        return 'results/leafcutter2/Macaque/junctions.sorted.tab.gz'
    if wildcards.species == 'Rat':
        return 'results/leafcutter2/Rat/junctions.sorted.tab.gz'
    if wildcards.species == 'Chicken':
        return 'results/leafcutter2/Chicken/junctions.sorted.tab.gz'
    if wildcards.species == 'Rabbit':
        return 'results/leafcutter2/Rabbit/junctions.sorted.tab.gz'
    #if wildcards.species in ['Mouse', 'Rat', 'Macaque', 'Rabbit', 'Chicken', 'Opossum']:
    #    return f'results/leafcutter2/{wildcards.species}/junctions.sorted.tab.gz'
    if wildcards.species == 'Human':
        return '/project2/mstephens/cfbuenabadn/gtex-stm/code/junctions.tab.gz'


n_slice = 200






rule get_torino_isoforms_slice:
    input:
        getJuncFile
    output:
        'results/torino/{species}/isoforms/tmp/factors_{i}.bed',
        'results/torino/{species}/isoforms/tmp/isoforms_{i}.bed',
        'results/torino/{species}/isoforms/tmp/missing_genes_{i}.txt'
    resources:
        mem_mb = 24000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/collect_isoforms_{species}_{i}.log'
    params:
        n = n_slice,
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    shell:
        """
        python scripts/collect_isoforms_torino.py {wildcards.i} {params.n} {wildcards.species} {input} {wildcards.species} &> {log}
        """

rule collect_torino_missing_slices:
    input:
        expand('results/torino/{{species}}/isoforms/tmp/missing_genes_{i}.txt', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/torino/{species}/isoforms/missing_genes.txt',
    wildcard_constraints:
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human',
    log:
        'logs/torino/isoforms/missing.{species}.log'
    shell:
        """
        for I in {input}
        do
           cat ${{I}} >> {output}
        done
        """


rule collect_torino_isoform_slices:
    input:
        expand('results/torino/{{species}}/isoforms/tmp/{{structure}}_{i}.bed', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        tmp = 'results/torino/{species}/isoforms/{structure}.bed',
        bed = 'results/torino/{species}/isoforms/{structure}.bed.gz',
        tbi = 'results/torino/{species}/isoforms/{structure}.bed.gz.tbi',
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

#rule collect_species_isoforms:
#    input:
#        expand('results/torino/{species}/isoforms/isoforms.bed.gz', species = ['Cattle', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Human']),
#        expand('results/torino/{species}/isoforms/factors.bed.gz', species = ['Cattle', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Human']),
#        expand('results/torino/{species}/isoforms/missing_genes.txt', species = ['Cattle', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Human'])



rule collect_species_isoforms_extended:
    input:
        expand('results/torino/{species}/isoforms/isoforms.bed.gz', species = ['Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque']),
        expand('results/torino/{species}/isoforms/factors.bed.gz', species = ['Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque']),
        expand('results/torino/{species}/isoforms/missing_genes.txt', species = ['Chicken', 'Mouse', 'Rat', 'Rabbit', 'Macaque'])



#rule collect_species_isoforms:
#    input:
#        expand('results/torino/{species}/isoforms/isoforms.bed.gz', species = ['Cattle', 'Mouse', 'Rat', 'Rabbit', 'Chicken', 'Macaque', 'Human']),
#        expand('results/torino/{species}/isoforms/factors.bed.gz', species = ['Cattle', 'Mouse', 'Rat', 'Rabbit', 'Chicken', 'Macaque', 'Human']),
#        expand('results/torino/{species}/isoforms/missing_genes.txt', species = ['Cattle', 'Mouse', 'Rat', 'Rabbit', 'Chicken', 'Macaque', 'Human'])




rule GetTranscriptEL:
    input:
        exons_bed = 'results/torino/{species}/isoforms/isoforms.bed.gz',
        el_bed = 'results/torino/tables/{species}{filter_level}/EL.K10.bed.gz'
    output:
        'results/torino/tables/{species}/isoforms{filter_level}.merged_isoforms.EL.bed.gz'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        filter_level = '|_filtered|_extended',
        species = 'Cattle|Mouse|Rat|Rabbit|Chicken|Macaque|Opossum|Human'
    log:
        '/scratch/midway3/cnajar/logs/torino/transcript_EL.{species}{filter_level}.log'
    shell:
        """
        python scripts/get_transcript_EL.py {input.exons_bed} {input.el_bed} {output} {wildcards.species} &> {log}
        """


rule collect_iso_EL:
    input:
        #expand('results/torino/tables/{species}/isoforms_filtered.merged_isoforms.EL.bed.gz', species=['Human', 'Cattle']),
        expand('results/torino/tables/{species}/isoforms_extended.merged_isoforms.EL.bed.gz', species=['Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken']),
        #expand('results/torino/tables/{species}/isoforms.merged_isoforms.EL.bed.gz', species=['Mouse', 'Rat', 'Rabbit', 'Chicken', 'Macaque']),






rule get_gencode_isoform_exons:
    input:
        'resources/annotation/Human/gtf/gencode.v47.annotation.gtf.gz',
    output:
        tmp = temp('resources/annotation/Human/bed/gencode.exons_annotation.bed'),
        tmp_sort = temp('resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed'),
        bed = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz'
    resources:
        mem_mb = 12000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/gencode_gtf_to_bed.log'
    shell:
        """
        python scripts/get_gencode_annotated_exons.py &> {log};
        (grep '^#' {output.tmp} > {output.tmp_sort}) &>> {log};
        (grep -v '^#' {output.tmp} | bedtools sort -i - >> {output.tmp_sort})  &>> {log}
        (bedtools sort -i {output.tmp_sort} | gzip - > {output.bed}) &>> {log}
        """



rule annotate_snmf_isoforms:
    input:
        gencode = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        snmf = 'results/torino/Human/isoforms/isoforms.bed.gz',
    output:
        temp('results/torino/Human/isoforms/tmp/annotated.isoforms_{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/annotate_isoforms.snmf_{i}.log'
    params:
        n = n_slice
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
    shell:
        """
        python scripts/annotate_isoforms.py {input} {wildcards.i} {params} &> {log}
        """


rule collect_annotated_snmf_isoforms:
    input:
        expand('results/torino/Human/isoforms/tmp/annotated.isoforms_{i}.tab.gz', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/torino/Human/isoforms/annotated.isoforms.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/torino/Human/isoforms/annotated.isoforms.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/torino/Human/isoforms/annotated.isoforms.tab
            fi
        done
        gzip results/torino/Human/isoforms/annotated.isoforms.tab
        """


rule isoform_second_annot:
    input:
        gencode = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        snmf = 'results/torino/Human/isoforms/isoforms.bed.gz',
        annotation = 'results/torino/Human/isoforms/annotated.isoforms.tab.gz'
    output:
        'results/torino/Human/isoforms/second_annotation.isoforms.tab.gz'
    log:
         'logs/second_annot.log'
    resources:
        mem_mb = 12000
    shell:
        """
        python scripts/second_annotation.py {input} {output} &> {log}
        """

#rule CollectTorinoIsoformAnnotations:
#    input:
#        'results/torino/Human/isoforms/annotated.isoforms.tab.gz'



def GetGTFPerSpecies(wildcards):
    gtf_file_suffix = gtf_annotations.loc[gtf_annotations.Species == wildcards.species].File_name.iloc[0]
    gtf_file = f'/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/{wildcards.species}/gtf/{gtf_file_suffix}'
    return gtf_file



rule get_gencode_isoform_exons_animals:
    input:
        'resources/annotation/Mouse/gtf/gencode.vM36.annotation.gtf.gz' 
    output:
        tmp = temp('resources/annotation/Mouse/bed/gencode.exons_annotation.bed'),
        tmp_sort = temp('resources/annotation/Mouse/bed/gencode.exons_annotation.sorted.bed'),
        bed = 'resources/annotation/Mouse/bed/gencode.exons_annotation.sorted.bed.gz'
    resources:
        mem_mb = 12000
    wildcard_constraints:
        species = 'Mouse'
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/gencode_gtf_to_bed.log'
    shell:
        """
        python scripts/get_gencode_annotated_mouse.py &> {log};
        (grep '^#' {output.tmp} > {output.tmp_sort}) &>> {log};
        (grep -v '^#' {output.tmp} | bedtools sort -i - >> {output.tmp_sort})  &>> {log}
        (bedtools sort -i {output.tmp_sort} | gzip - > {output.bed}) &>> {log}
        """



def get_exon_annot_species(wildcards):
    if wildcards.species == 'Mouse':
        return 'resources/annotation/Mouse/bed/gencode.exons_annotation.sorted.bed.gz',
    else:
        return 'resources/annotation/{species}/bed/exons.all.bed.gz'



rule annotate_snmf_isoforms_animals:
    input:
        gencode = get_exon_annot_species,
        snmf = 'results/torino/{species}/isoforms/isoforms.bed.gz',
    output:
        temp('results/torino/{species}/isoforms/tmp/annotated.isoforms_{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/{species}.annotate_isoforms.snmf_{i}.log'
    params:
        n = n_slice
    wildcard_constraints:
        species = 'Mouse|Cattle|Macaque|Rat|Rabbit|Chicken',
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
    shell:
        """
        python scripts/annotate_isoforms_animals.py {wildcards.species} {input} {wildcards.i} {params} &> {log}
        """


rule collect_annotated_snmf_isoforms_animals:
    input:
        expand('results/torino/{{species}}/isoforms/tmp/annotated.isoforms_{i}.tab.gz', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/torino/{species}/isoforms/annotated.isoforms.tab.gz'
    resources:
        mem_mb = 24000
    wildcard_constraints:
        species = 'Mouse|Cattle|Macaque|Rat|Rabbit|Chicken',
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/torino/{wildcards.species}/isoforms/annotated.isoforms.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/torino/{wildcards.species}/isoforms/annotated.isoforms.tab
            fi
        done
        gzip results/torino/{wildcards.species}/isoforms/annotated.isoforms.tab
        """


rule CollectAnnotSpecies:
    input:
        expand('results/torino/{species}/isoforms/annotated.isoforms.tab.gz', species = ['Mouse', 'Macaque', 'Chicken', 'Rat', 'Rabbit'])

n_slice = 100
rule getUTRMetadata:
    output:
        'results/metaplots/tmp/metaplot_utr3_{i}.csv'
    resources:
        mem_mb = 36000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/utr3.snmf_{i}.log'
    params:
        n = n_slice
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
    shell:
        """
        python scripts/get_utr_metaplot.py {wildcards.i} {params} &> {log}
        """

rule collectUTRAll:
    input:
        expand('results/metaplots/tmp/metaplot_utr3_{i}.csv', i = range(1, n_slice + 1))
    output:
        'results/metaplots/metaplot_utr3.csv.gz'
    log:
        'logs/torino/isoforms/utr.log'
    resources:
        mem_mb = 36000
    shell:
        """
        for I in {input}
        do
           cat ${{I}} >> results/metaplots/metaplot_utr3.csv
        done
        gzip results/metaplots/metaplot_utr3.csv
        """



####################



rule annotate_snmf_isoforms_corrected:
    input:
        gencode = 'resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        snmf = 'results/torino/Human/isoforms/isoforms.bed.gz',
    output:
        temp('results/torino/Human/isoforms/tmp/corrected_annotated.isoforms_{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/corrected_annotate_isoforms.snmf_{i}.log'
    params:
        n = n_slice
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
    shell:
        """
        python scripts/annotate_isoforms_corrected.py {input} {wildcards.i} {params} &> {log}
        """


rule collect_annotated_snmf_isoforms_corrected:
    input:
        expand('results/torino/Human/isoforms/tmp/corrected_annotated.isoforms_{i}.tab.gz', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/torino/Human/isoforms/corrected_annotated.isoforms.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/torino/Human/isoforms/corrected_annotated.isoforms.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/torino/Human/isoforms/corrected_annotated.isoforms.tab
            fi
        done
        gzip results/torino/Human/isoforms/corrected_annotated.isoforms.tab
        """


rule isoforms_bed_to_gtf:
    input:
        bed = 'results/torino/Human/corrected_isoforms/isoforms.bed.gz',
        eff_tab = 'results/torino/Human/isoforms/corrected_annotated.isoforms.eff.tab.gz'
    output:
        all_gtf = 'results/torino/Human/isoforms/all_isoforms.gtf.gz',
        filtered_gtf = 'results/torino/Human/isoforms/filtered_isoforms.gtf.gz'
    resources:
        mem_mb = 12000
    log:
        'logs/torino/isoforms/bed_to_gtf.log'
    shell:
        """
        python scripts/isoforms_bed_to_gtf.py {input.bed} {input.eff_tab} {output.all_gtf} {output.filtered_gtf} &> {log}
        """
