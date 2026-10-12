rule IndexBAM:
    input:
        'resources/GTEx/bam_files/{Tissue}/{IndID}.Aligned.sortedByCoord.out.patched.md.bam',
    output:
        'resources/GTEx/bam_files/{Tissue}/{IndID}.Aligned.sortedByCoord.out.patched.md.bam.bai'
    resources:
        mem_mb = 4000
    log:
        "logs/index_bam/GTEx/{Tissue}/{IndID}.log"
    shell:
        """
        (samtools index {input} {output}) &> {log}
        """

rule GetBedsAndTabix:
    input:
        bam = 'resources/GTEx/bam_files/{Tissue}/{IndID}.Aligned.sortedByCoord.out.patched.md.bam',
        bai = 'resources/GTEx/bam_files/{Tissue}/{IndID}.Aligned.sortedByCoord.out.patched.md.bam.bai'
    output:
        bed = "resources/GTEx/bed/{Tissue}/{IndID}.bed.gz",
        tabix = "resources/GTEx/bed/{Tissue}/{IndID}.bed.gz.tbi"
    resources:
        mem_mb = 8000
    log:
        "logs/bed_and_tabix/GTEx/{Tissue}/{IndID}.log"
    shell:
        """
        (bedtools genomecov -5 -bga -ibam {input.bam} | bgzip > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """
        
rule GetBedsAndTabix_Primary:
    input:
        bam = 'resources/GTEx/bam_files/{Tissue}/{IndID}.Aligned.sortedByCoord.out.patched.md.bam',
        bai = 'resources/GTEx/bam_files/{Tissue}/{IndID}.Aligned.sortedByCoord.out.patched.md.bam.bai'
    output:
        bam = temp("resources/GTEx/bam_tmp/{Tissue}/{IndID}.primary_alignment.bam"),
        bai = temp("resources/GTEx/bam_tmp/{Tissue}/{IndID}.primary_alignment.bam.bai"),
        bed = "resources/GTEx/bed_primary/{Tissue}/{IndID}.bed.gz",
        tabix = "resources/GTEx/bed_primary/{Tissue}/{IndID}.bed.gz.tbi"
    resources:
        mem_mb = 8000
    log:
        "logs/bed_and_tabix/GTEx/{Tissue}/{IndID}.log"
    shell:
        """
        (samtools view -bh -F 256 {input.bam} > {output.bam}) &> {log};
        (samtools index {output.bam} {output.bai}) &>> {log};
        (bedtools genomecov -5 -bga -ibam {output.bam} | bgzip > {output.bed}) &>> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """
        
#rule collect_download_test:
#    input:
#        expand("resources/GTEx/bed/{Tissue}/{IndID}.bed.gz", zip, Tissue = download_test.Tissue, IndID = download_test.IndID),
#        expand("resources/GTEx/bed_primary/{Tissue}/{IndID}.bed.gz", zip, Tissue = download_test.Tissue, IndID = download_test.IndID)