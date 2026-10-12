f5_tissues = ['Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 
              'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Lung', 'Liver', 'Testis', 'Testis', 
              'Testis', 'Testis', 'Skin', 'Skin', 'Subcutaneous_Adipose', 'Subcutaneous_Adipose']
              
f5_samples = ['SAMEA110329029', 'SAMEA110329035', 'SAMEA110328993', 'SAMEA110329031', 'SAMEA110329020', 'SAMEA110329028', 
              'SAMEA110329042', 'SAMEA110329019', 'SAMN16130455', 'SAMEA110329017', 'SAMEA110329026', 'SAMEA110329022', 
              'SAMEA110329006', 'SAMEA110329018', 'SAMEA110329025', 'SAMEA110329045', 'SAMEA110329037', 'SAMEA110329041', 
              'SAMEA110329021', 'SAMEA110329024', 'SAMEA110329059', 'SAMEA110329049', 'SAMEA110329030', 'SAMEA110329033', 
              'SAMN34293067', 'SAMN29068903', 'SAMN17043320', 'SAMN29068906', 'SAMN29068910', 'SAMN32316069', 'SAMN32316077', 
              'SAMN30429496', 'SAMN30429500']
              
rule GetTempBed:
    input:
        'resources/FarmGTEx/cattleGTEx/bed/{tissue}/{sample}.bed.gz'
    output:
        temp('resources/FarmGTEx/cattleGTEx/bed_filtered/{tissue}/{sample}.bed')
    log:
        'logs/filter_bed/{tissue}/{sample}.log'
    resources:
        mem_mb = 12000
    shell:
        """
        (zcat {input} | awk '$1 == "X" || ($1 ~ /^[0-9]+$/ && $1 >= 1 && $1 <= 30)' - > {output}) &> {log};
        """
    
              
rule bedGraphToBigWig_cattle:
    input:
        bed = 'resources/FarmGTEx/cattleGTEx/bed_filtered/{tissue}/{sample}.bed',
        chroms = 'resources/annotation/chrom_sizes/bosTau9_ensembl.chrom.sizes'
    output:
        'resources/FarmGTEx/cattleGTEx/bigwig/{tissue}/{sample}.bw'
    log:
        'logs/bed2bw/{tissue}/{sample}.log'
    resources:
        mem_mb = 12000
    shell:
        """
        (bedGraphToBigWig {input.bed} {input.chroms} {output}) &>> {log}
        """
        
        
rule MergeOutliers:
    input:
        expand('resources/FarmGTEx/cattleGTEx/bed_filtered/{tissue}/{sample}.bed', zip, tissue=f5_tissues, sample=f5_samples)
    output:
        'resources/FarmGTEx/cattleGTEx/bed_merged/f5_outliers.bed'
    log:
        'logs/f5_outliers.log'
    resources:
        mem_mb = 48000
    shell:
        """
        (bedtools unionbedg -i {input} | bedtools sort -i - > {output}) &> {log}
        """

use rule bedGraphToBigWig_cattle as bedGraphToBigWig_f5 with:
    input:
        bed = 'resources/FarmGTEx/cattleGTEx/bed_merged/f5_outliers.bed',
        chroms = 'resources/annotation/chrom_sizes/bosTau9_ensembl.chrom.sizes'
    output:
        'resources/FarmGTEx/cattleGTEx/bigwig/merged/f5_outliers.bw'
    log:
        'logs/f5_outliers_bw.log'
        
rule GatherOutliers:
    input:
        'resources/FarmGTEx/cattleGTEx/bigwig/merged/f5_outliers.bw',
        expand('resources/FarmGTEx/cattleGTEx/bigwig/{tissue}/{sample}.bw', zip, tissue=f5_tissues, sample=f5_samples)