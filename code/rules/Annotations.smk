def GetGTFPerSpecies(wildcards):
    gtf_file_suffix = gtf_annotations.loc[gtf_annotations.Species == wildcards.species].File_name.iloc[0]
    gtf_file = f'/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/{wildcards.species}/gtf/{gtf_file_suffix}'
    return gtf_file


def GetAwkCommand(wildcards):
    annot_source = gtf_annotations.loc[gtf_annotations.Species == wildcards.species].Source.iloc[0]
    if annot_source == 'Gencode':
        return """awk -F'[\t".]' '$3=="gene" {{ print $1, $4, $5, $12, $17, $8}}'  OFS='\t' -"""
    else:
        return """awk -F'[\t"]' '$3=="gene" { print "chr"$1, $4, $5, $10, $14, $7}' OFS='\t' - """

def GetGeneType(wildcards):
    annot_source = gtf_annotations.loc[gtf_annotations.Species == wildcards.species].Source.iloc[0]
    if annot_source == 'Gencode':
        return 'gene_type "protein_coding";'
    else:
        return 'gene_biotype "protein_coding";'

rule GetGeneBeds:
    input:
        GetGTFPerSpecies
    output:
        temp('resources/annotation/{species}/bed/genes.protein_coding.uncorrected.bed')
    log:
        'logs/gene_beds/{species}.log'
    params:
        awk_cmd = GetAwkCommand,
        gene_type = GetGeneType
    resources:
        mem_mb = 12000
    shell:
        """
        (zcat {input} | grep '{params.gene_type}' - | {params.awk_cmd} | bedtools sort -i - > {output}) &> {log}
        """


from contextlib import redirect_stdout

rule CorrectGeneSymbolsBeds:
    input:
        'resources/annotation/{species}/bed/genes.protein_coding.uncorrected.bed'
    output:
        'resources/annotation/{species}/bed/genes.protein_coding.bed.gz'
    log:
        'logs/correctgenes/{species}.log'
    resources:
        mem_mb = 8000
    run:
        with open(log[0], "w") as log_file:
            log_file.write(f"Init annot")
            with redirect_stdout(log_file):
                log_file.write(f"Is problem")
                import gzip
                log_file.write(f"with module?")
                with open(input[0], 'r') as fh:
                    with gzip.open(output[0], 'wb') as fh_out:
                        header = ('\t'.join(['#chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand']) + '\n').encode()
                        fh_out.write(header)
                        for line in fh:
                            
                            row = line.rstrip().split('\t')
                            log_file.write(f"{row[4]}")
                            if row[4] == 'ensembl':
                                row[4] = row[3]
                            out_line = ('\t'.join(row) + '\n').encode()
                            fh_out.write(out_line)


rule IndexGenesBed:
    input:
        'resources/annotation/{species}/bed/genes.protein_coding.bed.gz',
    output:
        bed = 'resources/annotation/{species}/bed/genes.protein_coding.sorted.bed.gz',
        tbi = 'resources/annotation/{species}/bed/genes.protein_coding.sorted.bed.gz.tbi'
    resources:
        mem_mb = 8000
    log:
       'logs/index_bed/{species}.genes.log' 
    shell:
        """
        (zcat {input} | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """

rule GetApprisPrincipalExon:
    input:
        GetGTFPerSpecies
        #"resources/annotation/Human/gtf/gencode.v47.annotation.gtf.gz"
    output:
        bed = 'resources/annotation/{species}/bed/exons.apris_principal.bed.gz',
        tbi = 'resources/annotation/{species}/bed/exons.apris_principal.bed.gz.tbi'
    log:
        "logs/annotation/{species}.appris_principal_exons.primary_assembly.log"
    wildcard_constraints:
        species = 'Human|Mouse'
    resources:
        mem_mb = 48000
    shell:
        """
        (zcat {input} | awk '$3=="exon" && ($1 ~ /^chr/)'  OFS='\\t' - | grep appris_principal - | awk -F'[\\t"]' '{{print $1, $4, $5, $10, $16, $7, $14, $12, $18}}' OFS='\\t' - | awk 'BEGIN {{OFS="\\t"}} {{gsub(/\..*/, "", $4); gsub(/\..*/, "", $8); print}}' - | bedtools sort -i -  | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand\\tgene_type\\ttranscript_id\\ttranscript_type"}} {{print}}' FS="\\t" OFS="\\t" - | bgzip -c - > {output.bed}) &> {log}
        (tabix -p bed {output.bed}) &>> {log};
        """


rule GetAllAnnotatedExons:
    input:
        GetGTFPerSpecies
        #"annotation/Human/gtf/gencode.v47.annotation.gtf.gz"
    output:
        bed = 'resources/annotation/{species}/bed/exons.all.bed.gz',
        tbi = 'resources/annotation/{species}/bed/exons.all.bed.gz.tbi'
    log:
        "logs/annotation/{species}.exon_annotation.primary_assembly.log"
    wildcard_constraints:
        species = 'Human|Mouse'
    resources:
        mem_mb = 48000
    shell:
        """
        (zcat {input} | awk '$3=="exon" && ($1 ~ /^chr/)'  OFS='\\t' - | awk -F'[\\t"]' '{{print $1, $4, $5, $10, $16, $7, $14, $12, $18}}' OFS='\\t' - | awk 'BEGIN {{OFS="\\t"}} {{gsub(/\..*/, "", $4); gsub(/\..*/, "", $8); print}}' - | bedtools sort -i - | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand\\tgene_type\\ttranscript_id\\ttranscript_type"}} {{print}}'  FS="\\t" OFS="\\t" - | bgzip -c - > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log};
        """











rule GetAnimalProteinCodingExons:
    input:
        GetGTFPerSpecies
    output:
        bed = "resources/annotation/{species}/bed/exons.protein_coding.bed.gz",
        tbi = "resources/annotation/{species}/bed/exons.protein_coding.bed.gz.tbi"
    log:
        "logs/annotation/{species}/protein_coding_exons.log"
    resources:
        mem_mb = 12000
    wildcard_constraints:
        species = 'Cattle|Sheep|Chicken|Opossum|Macaque|Rabbit|Rat'
    shell:
        """
        (zcat {input} | grep 'transcript_biotype "protein_coding"' - | awk -F'[\\t"]' '$3 == "exon" {{print "chr"$1, $4, $5, $10, $20, $7, $24, $14, $30}}' OFS='\\t' - | bedtools sort -i -  | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand\\tgene_type\\ttranscript_id\\ttranscript_type"}} {{print}}' FS="\\t" OFS="\\t" - | bgzip -c - > {output.bed}) &> {log}
        (tabix -p bed {output.bed}) &>> {log};
        """
        


rule GetAnimalAllAnnotatedExons:
    input:
        GetGTFPerSpecies
    output:
        bed = "resources/annotation/{species}/bed/exons.all.bed.gz",
        tbi = "resources/annotation/{species}/bed/exons.all.bed.gz.tbi"
    log:
        "logs/annotation/{species}/all_exons.log"
    resources:
        mem_mb = 12000
    wildcard_constraints:
        species = 'Cattle|Sheep|Chicken|Opossum|Macaque|Rabbit|Rat'
    shell:
        """
        (zcat {input} | awk -F'[\\t"]' '$3 == "exon" {{print "chr"$1, $4, $5, $10, $20, $7, $24, $14, $30}}' OFS='\\t' - | bedtools sort -i -  | awk 'BEGIN {{print "#chrom\\tstart\\tend\\tgene_id\\tgene_name\\tstrand\\tgene_type\\ttranscript_id\\ttranscript_type"}} {{print}}' FS="\\t" OFS="\\t" - | bgzip -c - > {output.bed}) &> {log}
        (tabix -p bed {output.bed}) &>> {log};
        """


























rule CollectGeneBeds:
    input:
        expand('resources/annotation/{species}/bed/genes.bed.gz', species = ['Human', 'Mouse', 'Macaque', 'Rabbit', 'Mouse', 'Rat', 'Opossum', 
'Sheep', 'Chicken', 'Cattle'])