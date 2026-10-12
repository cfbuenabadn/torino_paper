def GetCardosoBam(wildcards):
    bam = f'/project2/yangili1/bjf79/2024_comparativesplicing/code/rna-seq/Alignments/STAR_Align/{wildcards.IndID}/Aligned.sortedByCoord.out.bam'
    return bam

def GetCardosoBai(wildcards):
    bai = f'/project2/yangili1/bjf79/2024_comparativesplicing/code/rna-seq/Alignments/STAR_Align/{wildcards.IndID}/Aligned.sortedByCoord.out.bam.bai'
    return bai

rule GetBedsAndTabixCardoso:
    input:
        bam = GetCardosoBam
    output:
        forward_bam = temp("resources/Cardoso/{animal}/bam/{IndID}_NN_Forward.bam"),
        reverse_bam = temp("resources/Cardoso/{animal}/bam/{IndID}_NN_Reverse.bam"),
        bed_f5 = "resources/Cardoso/{animal}/bed/{IndID}.forward5.bed.gz",
        bed_f3 = "resources/Cardoso/{animal}/bed/{IndID}.forward3.bed.gz",
        bed_r5 = "resources/Cardoso/{animal}/bed/{IndID}.reverse5.bed.gz",
        bed_r3 = "resources/Cardoso/{animal}/bed/{IndID}.reverse3.bed.gz",
        bed_f_all = "resources/Cardoso/{animal}/bed/{IndID}.forward_all.bed.gz",
        bed_r_all = "resources/Cardoso/{animal}/bed/{IndID}.reverse_all.bed.gz",
        tbi_f5 = "resources/Cardoso/{animal}/bed/{IndID}.forward5.bed.gz.tbi",
        tbi_f3 = "resources/Cardoso/{animal}/bed/{IndID}.forward3.bed.gz.tbi",
        tbi_r5 = "resources/Cardoso/{animal}/bed/{IndID}.reverse5.bed.gz.tbi",
        tbi_r3 = "resources/Cardoso/{animal}/bed/{IndID}.reverse3.bed.gz.tbi",
        tbi_f_all = "resources/Cardoso/{animal}/bed/{IndID}.forward_all.bed.gz.tbi",
        tbi_r_all = "resources/Cardoso/{animal}/bed/{IndID}.reverse_all.bed.gz.tbi",
    log:
        "logs/filter_bam/Cardoso/{animal}.{IndID}.log"
    resources:
        mem_mb = 12000
    wildcard_constraints: animal = 'Chicken|Macaque|Mouse|Rabbit|Rat'
    shell:
        """
        module load samtools;
        (samtools view -bh -F 272 {input.bam} > {output.forward_bam}) &> {log};
        (samtools view -bh -F 256 -f 16 {input.bam} > {output.reverse_bam}) &> {log};
        (bedtools genomecov -5 -bga -ibam {output.forward_bam} | bgzip -c > {output.bed_f5}) &> {log};
        (tabix -p bed {output.bed_f5}) &>> {log}
        (bedtools genomecov -3 -bga -ibam {output.forward_bam} | bgzip -c > {output.bed_f3}) &> {log};
        (tabix -p bed {output.bed_f3}) &>> {log}
        (bedtools genomecov -5 -bga -ibam {output.reverse_bam} | bgzip -c > {output.bed_r5}) &> {log};
        (tabix -p bed {output.bed_r5}) &>> {log}
        (bedtools genomecov -3 -bga -ibam {output.reverse_bam} | bgzip -c > {output.bed_r3}) &> {log};
        (tabix -p bed {output.bed_r3}) &>> {log}
        (bedtools genomecov -bga -split -ibam {output.reverse_bam} | bgzip -c > {output.bed_f_all}) &> {log};
        (tabix -p bed {output.bed_f_all}) &>> {log}
        (bedtools genomecov -bga -split -ibam {output.reverse_bam} | bgzip -c > {output.bed_r_all}) &> {log};
        (tabix -p bed {output.bed_r_all}) &>> {log}
        """



rule GetBedsAndTabixCardosoOpposum:
    input:
        bam = GetCardosoBam
    output:
        forward_bam = temp("resources/Cardoso/{animal}/bam/{IndID}_NN_Forward.bam"),
        reverse_bam = temp("resources/Cardoso/{animal}/bam/{IndID}_NN_Reverse.bam"),
        bed_f5 = "resources/Cardoso/{animal}/bed/{IndID}.forward5.bed.gz",
        bed_f3 = "resources/Cardoso/{animal}/bed/{IndID}.forward3.bed.gz",
        bed_r5 = "resources/Cardoso/{animal}/bed/{IndID}.reverse5.bed.gz",
        bed_r3 = "resources/Cardoso/{animal}/bed/{IndID}.reverse3.bed.gz",
        bed_f_all = "resources/Cardoso/{animal}/bed/{IndID}.forward_all.bed.gz",
        bed_r_all = "resources/Cardoso/{animal}/bed/{IndID}.reverse_all.bed.gz",
        tbi_f5 = "resources/Cardoso/{animal}/bed/{IndID}.forward5.bed.gz.csi",
        tbi_f3 = "resources/Cardoso/{animal}/bed/{IndID}.forward3.bed.gz.csi",
        tbi_r5 = "resources/Cardoso/{animal}/bed/{IndID}.reverse5.bed.gz.csi",
        tbi_r3 = "resources/Cardoso/{animal}/bed/{IndID}.reverse3.bed.gz.csi",
        tbi_f_all = "resources/Cardoso/{animal}/bed/{IndID}.forward_all.bed.gz.csi",
        tbi_r_all = "resources/Cardoso/{animal}/bed/{IndID}.reverse_all.bed.gz.csi",
    log:
        "logs/filter_bam/Cardoso/{animal}.{IndID}.log"
    resources:
        mem_mb = 12000
    wildcard_constraints: animal = 'Opossum'
    shell:
        """
        module load samtools;
        (samtools view -bh -F 272 {input.bam} > {output.forward_bam}) &> {log};
        (samtools view -bh -F 256 -f 16 {input.bam} > {output.reverse_bam}) &> {log};
        (bedtools genomecov -5 -bga -ibam {output.forward_bam} | bgzip -c > {output.bed_f5}) &> {log};
        (tabix -p bed --csi {output.bed_f5}) &>> {log}
        (bedtools genomecov -3 -bga -ibam {output.forward_bam} | bgzip -c > {output.bed_f3}) &> {log};
        (tabix -p bed --csi {output.bed_f3}) &>> {log}
        (bedtools genomecov -5 -bga -ibam {output.reverse_bam} | bgzip -c > {output.bed_r5}) &> {log};
        (tabix -p bed --csi {output.bed_r5}) &>> {log}
        (bedtools genomecov -3 -bga -ibam {output.reverse_bam} | bgzip -c > {output.bed_r3}) &> {log};
        (tabix -p bed --csi {output.bed_r3}) &>> {log}
        (bedtools genomecov -bga -split -ibam {output.reverse_bam} | bgzip -c > {output.bed_f_all}) &> {log};
        (tabix -p bed --csi {output.bed_f_all}) &>> {log}
        (bedtools genomecov -bga -split -ibam {output.reverse_bam} | bgzip -c > {output.bed_r_all}) &> {log};
        (tabix -p bed --csi {output.bed_r_all}) &>> {log}
        """

rule collectCardoso:
    input:
        expand("resources/Cardoso/Mouse/bed/{IndID}.forward5.bed.gz", IndID = list(cardoso.loc[cardoso.Species=='Mouse'].sampleID)),
        expand("resources/Cardoso/Rat/bed/{IndID}.forward5.bed.gz", IndID = list(cardoso.loc[cardoso.Species=='Rat'].sampleID)),
        expand("resources/Cardoso/Macaque/bed/{IndID}.forward5.bed.gz", IndID = list(cardoso.loc[cardoso.Species=='Macaque'].sampleID)),
        expand("resources/Cardoso/Opossum/bed/{IndID}.forward5.bed.gz", IndID = list(cardoso.loc[cardoso.Species=='Opossum'].sampleID)),
        expand("resources/Cardoso/Rabbit/bed/{IndID}.forward5.bed.gz", IndID = list(cardoso.loc[cardoso.Species=='Rabbit'].sampleID)),
        expand("resources/Cardoso/Chicken/bed/{IndID}.forward5.bed.gz", IndID = list(cardoso.loc[cardoso.Species=='Chicken'].sampleID)),

rule index_sheepGTEx:
    input:
        'resources/FarmGTEx/sheepGTEx/bed/{tissue}/{sample}.bed.gz'
    output:
        'resources/FarmGTEx/sheepGTEx/bed/{tissue}/{sample}.bed.gz.tbi'
    log:
        'logs/sheepGTEx/preprocess/tbi/{tissue}.{sample}.log'
    shell:
        """
        (tabix -p bed {input}) &> {log}
        """

def GetSheepGTExTbiSamples(wildcards):
    file_list = []
    for idx, row in sheepGTEx.iterrows():
        file_name = f'resources/FarmGTEx/sheepGTEx/bed/{row.Tissue}/{row.Sample}.bed.gz.tbi'
        file_list.append(file_name)
    return file_list

rule CollectSheepGTExTbi:
    input:
        GetSheepGTExTbiSamples