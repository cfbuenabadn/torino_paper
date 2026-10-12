rule AD_differential_expression:
    input:
        counts = '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/AD_data/ROSMAP_bulk_count_data/{region}_samples_list.rnaseqc.gene_readsCount.subset.gct.gz',
        metadata = '/project2/mstephens/cfbuenabadn/gtex-stm/code/gao_data/metadata/merged_metadata.tab.gz'
    output:
        '/project2/yangili1/cfbuenabadn/torino_paper/code/results/AD_DGE/{region}.AD_DE.tab.gz'
    wildcard_constraints:
        region = 'DLPFC|AC|PCC'
    log:
        "logs/DGE/{region}.log"
    shell:
        """
        Rscript scripts/runDE_AD.R {input.counts} {input.metadata} {output} &> {log}
        """

rule collectAD_DGE:
    input:
        expand('/project2/yangili1/cfbuenabadn/torino_paper/code/results/AD_DGE/{region}.AD_DE.tab.gz',
               region=['DLPFC', 'AC', 'PCC'])

#############################################


rule get_torino_isoforms_gao_chunksK3:
    output:
        'results/gao_k3/isoforms/tmp/factors_{chunk}.bed',
        'results/gao_k3/isoforms/tmp/isoforms_{chunk}.bed',
        'results/gao_k3/isoforms/tmp/missing_genes_{chunk}.txt'
    resources:
        mem_mb = 12000
    log:
        '/scratch/midway3/cnajar/logs/ebpmf_run/collect_gao_isoforms.{chunk}.log'
    wildcard_constraints:
        chunk = '|'.join([str(x) for x in range(0,2001)]),
    shell:
        """
        python scripts/collect_gao_isoforms_K3.py {wildcards.chunk} &> {log}
        """


rule collect_torino_missing_slicesK3:
    input:
        expand('results/gao_k3/isoforms/tmp/missing_genes_{chunk}.txt', chunk = [str(x) for x in range(0,2001)])
    output:
        'results/gao_k3/isoforms/missing_genes.txt',
    log:
        'logs/torino_k3/isoforms/missing.gao.log'
    shell:
        """
        for I in {input}
        do
           cat ${{I}} >> {output}
        done
        """


rule collect_torino_isoform_slicesK3:
    input:
        expand('results/gao_k3/isoforms/tmp/{{structure}}_{chunk}.bed', chunk = [str(x) for x in range(0,2001)])
    output:
        tmp = 'results/gao_k3/isoforms/{structure}.bed',
        bed = 'results/gao_k3/isoforms/{structure}.bed.gz',
        tbi = 'results/gao_k3/isoforms/{structure}.bed.gz.tbi',
    wildcard_constraints:
        structure = 'factors|isoforms'
    log:
        'logs/torino/isoforms/{structure}.log'
    shell:
        """
        for I in {input}
        do
           cat ${{I}} >> {output.tmp}
        done
        (awk 'BEGIN {{OFS=FS="\\t"}} {{for (i=1; i<=NF; i++) if ($i == "") $i = "."; print}}' {output.tmp} | bedtools sort -i - | bgzip -c > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """

rule rename_snmf3K3:
    input:
        '/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/tables/snmf3_EL.tab.gz'
    output:
        'results/gao_k3/tables/snmf3_EL.all.bed.gz',
        'results/gao_k3/tables/snmf3_EL.AC.bed.gz',
        'results/gao_k3/tables/snmf3_EL.PCC.bed.gz',
        'results/gao_k3/tables/snmf3_EL.DLPFC.bed.gz'
    log:
        'logs/torino_ad/EL3.log'
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/rename_snmf3_EL.py &> {log}
        """



rule sortGaoK3:
    input:
        'results/gao_k3/tables/snmf3_EL.unnormalized_selected.{tissue}.bed.gz',
    output:
        bed = 'results/gao_k3/tables/snmf3_EL.unnormalized_selected.{tissue}.sorted.bed.gz',
        tbi = 'results/gao_k3/tables/snmf3_EL.unnormalized_selected.{tissue}.sorted.bed.gz.tbi',
    resources:
        mem_mb = 24000
    log:
        'logs/sortgao.{tissue}.log'
    wildcard_constraints:
        tissue = 'AC|PCC|DLPFC'
    shell:
        """
        (zcat {input} | bedtools sort -i - -header | bgzip /dev/stdin -c > {output.bed}) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """


rule CollectSortedGaoK3:
    input:
        expand('results/gao_k3/tables/snmf3_EL.unnormalized_selected.{tissue}.sorted.bed.gz.tbi', tissue = ['AC', 'PCC', 'DLPFC'])






rule annotate_snmf_isoforms_gaoK3:
    input:
        gencode = '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/annotation/Human/bed/gencode.exons_annotation.sorted.bed.gz',
        snmf = 'results/gao_k3/isoforms/isoforms.bed.gz',
    output:
        temp('results/gao_k3/isoforms/tmp/annotated.isoforms_{i}.tab.gz')
    resources:
        mem_mb = 48000
    log:
        '/scratch/midway3/cnajar/logs/gao_k3/annotate_isoforms.snmf_{i}.log'
    params:
        n = n_slice
    wildcard_constraints:
        i = '|'.join([str(j) for j in range(1, n_slice + 1)]),
    shell:
        """
        python scripts/annotate_isoforms_gao.py {input} {wildcards.i} {params} &> {log}
        """





rule collect_annotated_snmf_isoforms_gaoK3:
    input:
        expand('results/gao_k3/isoforms/tmp/annotated.isoforms_{i}.tab.gz', i = [str(j) for j in range(1, n_slice + 1)])
    output:
        'results/gao_k3/isoforms/annotated.isoforms.tab.gz'
    resources:
        mem_mb = 24000
    shell:
        """
        FIRST=true
        for I in {input}
        do
            if [ "$FIRST" = true ]; then
                zcat "${{I}}" >> results/gao_k3/isoforms/annotated.isoforms.tab
                FIRST=false
            else
                zcat "${{I}}" | tail -n +2 >> results/gao_k3/isoforms/annotated.isoforms.tab
            fi
        done
        gzip results/gao_k3/isoforms/annotated.isoforms.tab
        """


##############







rule QQNormDataForADK3:
    input:
        'results/gao_k3/tables/snmf3_EL.{tissue}.bed.gz'
    output:
        'results/gao_k3/tables/snmf3_EL.{tissue}.qqnorm.bed'
    log:
        'logs/Gao_norm/{tissue}.log'
    wildcard_constraints:
        tissue = 'all|AC|PCC|DLPFC'
    resources:
        mam_mb = 72000
    shell:
        """
        python scripts/QQNorm_factors.py {input} {output} &> {log}
        """



rule SortAndTabixGaoK3:
    input:
        'results/gao_k3/tables/snmf3_EL.{tissue}.qqnorm.bed'
    output:
        bed = 'results/gao_k3/tables/snmf3_EL.{tissue}.sorted.qqnorm.bed.gz',
        tbi = 'results/gao_k3/tables/snmf3_EL.{tissue}.sorted.qqnorm.bed.gz.tbi',
    log:
        'logs/tabix_Gao_{tissue}.log'
    wildcard_constraints:
        tissue = 'AC|PCC|DLPFC|all'
    resources:
        mam_mb = 72000
    shell:
        """
        (bedtools sort -i {input} -header | bgzip /dev/stdin -c > {output.bed} ) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """


rule pca_gaoK3:
    """
    QTLtools format expression PCs as covariates
    including the number of PCs that explain more
    variance then when the phenotype table is
    permuted
    """
    input:
        'results/gao_k3/tables/snmf3_EL.{tissue}.sorted.qqnorm.bed.gz',
    output:
        'results/gao_k3/tables/snmf3_EL.{tissue}.sorted.qqnorm.bed.pca',
    log:
        "logs/QTLs/PhenotypePCs/gao{tissue}.log"
    resources:
        mem_mb = 24000
    wildcard_constraints:
        tissue = 'all|AC|PCC|DLPFC'
    conda:
        "../envs/r_essentials.yml"
    shell:
        """
        Rscript scripts/PermuteAndPCA.R {input} {output} &> {log}
        """



rule CollectGaoQQnormK3:
    input:
        expand('results/gao_k3/tables/snmf3_EL.{tissue}.sorted.qqnorm.bed.gz', tissue = ['AC', 'PCC', 'DLPFC', 'all']),
        expand('results/gao_k3/tables/snmf3_EL.{tissue}.sorted.qqnorm.bed.pca', tissue = ['AC', 'PCC', 'DLPFC', 'all']),
        'results/gao_k3/isoforms/annotated.isoforms.tab.gz'
        
        





rule AD_K3_LR:
    input:
        'results/gao_k3/tables/snmf3_EL.all.sorted.qqnorm.bed.gz',
        'results/gao_k3/isoforms/annotated.isoforms.tab.gz',
        'results/gao_k3/isoforms/isoforms.bed.gz'
    output:
        "results/gao_k3/tables/LR.output.tab.gz"
    log:
        "logs/QTLs/PhenotypePCs/gao_k3_logReg.log"
    resources:
        mem_mb = 24000
    shell:
        """
        : > {log}
        python scripts/prepare_ad_torino_k3.py {log} &>> {log}
        """


rule rename_snmf3K5:
    input:
        '/project2/mstephens/cfbuenabadn/gtex-stm/code/ebpmf_models/gao_models/tables/snmf5_EL.tab.gz'
    output:
        'results/gao_k5/tables/snmf5_EL.all.bed.gz',
        'results/gao_k5/tables/snmf5_EL.AC.bed.gz',
        'results/gao_k5/tables/snmf5_EL.PCC.bed.gz',
        'results/gao_k5/tables/snmf5_EL.DLPFC.bed.gz'
    log:
        'logs/torino_ad/EL5.log'
    resources:
        mem_mb = 24000
    shell:
        """
        python scripts/rename_snmf5_EL.py &> {log}
        """


rule QQNormDataForADK5:
    input:
        'results/gao_k5/tables/snmf5_EL.{tissue}.bed.gz'
    output:
        'results/gao_k5/tables/snmf5_EL.{tissue}.qqnorm.bed'
    log:
        'logs/Gao_norm/{tissue}.log'
    wildcard_constraints:
        tissue = 'all|AC|PCC|DLPFC'
    resources:
        mam_mb = 72000
    shell:
        """
        python scripts/QQNorm_factors.py {input} {output} &> {log}
        """

rule SortAndTabixGaoK5:
    input:
        'results/gao_k5/tables/snmf5_EL.{tissue}.qqnorm.bed'
    output:
        bed = 'results/gao_k5/tables/snmf5_EL.{tissue}.sorted.qqnorm.bed.gz',
        tbi = 'results/gao_k5/tables/snmf5_EL.{tissue}.sorted.qqnorm.bed.gz.tbi',
    log:
        'logs/tabix_Gao_{tissue}k5.log'
    wildcard_constraints:
        tissue = 'AC|PCC|DLPFC|all'
    resources:
        mam_mb = 72000
    shell:
        """
        (bedtools sort -i {input} -header | bgzip /dev/stdin -c > {output.bed} ) &> {log};
        (tabix -p bed {output.bed}) &>> {log}
        """

rule AD_braak_OLS:
    """
    OLS of braak stage on isoform usage, the continuous counterpart to
    AD_K3_LR's logistic regression of AD status. Usage comes from the
    qq-normalised bed and is not transformed again.
    """
    input:
        'results/gao_k3/tables/snmf3_EL.all.sorted.qqnorm.bed.gz',
        'results/gao_k3/isoforms/annotated.isoforms.tab.gz',
        'results/gao_k3/isoforms/isoforms.bed.gz'
    output:
        "results/gao_k3/tables/OLS.output.tab.gz"
    log:
        "logs/QTLs/PhenotypePCs/gao_braak_OLS.log"
    resources:
        mem_mb = 24000
    shell:
        """
        : > {log}
        python scripts/prepare_ad_torino_braak_ols.py {log} &>> {log}
        """


rule AD_UTR3_logreg:
    """
    The 3' UTR logistic regression of AD_UTR.ipynb, with the covariate set
    AD_K3_LR uses. K=5 throughout (snmf5_EL + the gao isoform annotation).
    """
    output:
        "results/gao_k5/tables/UTR3.logreg.output.tab.gz",
        # the usage matrix the regression was fitted on, kept so a finished run
        # can be traced back to its input
        "results/gao_k5/tables/UTR3.usage.tab.gz"
    log:
        "logs/QTLs/PhenotypePCs/gao_UTR3_logreg.log"
    resources:
        mem_mb = 24000
    shell:
        """
        : > {log}
        python scripts/prepare_ad_utr3_logreg.py {log} &>> {log}
        """


rule AD_K5_LR:
    output:
        "results/gao_k5/tables/LR.output.tab.gz"
    log:
        "logs/QTLs/PhenotypePCs/gao_k5_logReg.log"
    resources:
        mem_mb = 24000
    shell:
        """
        : > {log}
        python scripts/prepare_ad_torino_k5.py {log} &>> {log}
        """



rule get_introns_for_ad_plot:
    input:
        "results/gao_k3/tables/LR.output.tab.gz"
    output:
        'results/gao_k3/tables/tmp/AD_IR_plots.{chunk}.bed.gz'
    resources:
        mem_mb = 12000
    log:
        'logs/print_ad_introns_bed.{chunk}.log'
    wildcard_constraints:
        chunk = '|'.join([str(x) for x in range(0,101)]),
    shell:
        """
        python scripts/prepare_ad_introns_plots.py {wildcards.chunk} &> {log}
        """

rule collect_ad_for_plot:
    input:
        expand('results/gao_k3/tables/tmp/AD_IR_plots.{chunk}.bed.gz', chunk = range(0,101))
    output:
        'results/gao_k3/tables/AD_IR_plots.bed.gz'
    resources:
        mem_mb = 12000
    shell:
        """
        for I in {input}
        do
           zcat ${{I}} >> results/gao_k3/tables/AD_IR_plots.bed
        done;
        gzip results/gao_k3/tables/AD_IR_plots.bed
        """














rule get_introns_for_ad_plot_IR_only:
    input:
        "results/gao_k3/tables/LR.output.tab.gz"
    output:
        'results/gao_k3/tables/tmp/AD_IR_only_plots.{chunk}.bed.gz'
    resources:
        mem_mb = 12000
    log:
        'logs/print_ad_introns_bed.{chunk}.log'
    wildcard_constraints:
        chunk = '|'.join([str(x) for x in range(0,101)]),
    shell:
        """
        python scripts/prepare_ad_introns_plots_no_efficiency.py {wildcards.chunk} &> {log}
        """

rule collect_ad_for_plot_IR_only:
    input:
        expand('results/gao_k3/tables/tmp/AD_IR_only_plots.{chunk}.bed.gz', chunk = range(0,101))
    output:
        'results/gao_k3/tables/AD_IR_only_plots.bed.gz'
    resources:
        mem_mb = 12000
    shell:
        """
        for I in {input}
        do
           zcat ${{I}} >> results/gao_k3/tables/AD_IR_only_plots.bed
        done;
        gzip results/gao_k3/tables/AD_IR_only_plots.bed
        """




############
############
############
############
############










#############################################
# Relaxed IR-only plots
#
# Same pipeline as get_introns_for_ad_plot_IR_only / collect_ad_for_plot_IR_only,
# with one change inside get_IR_bed(): a torino exon has to fall within the body
# of the gene (every gencode exon of it) rather than within the span of the
# APPRIS-principal transcript. When that transcript is a short isoform which
# does not cover the torino model, the strict `f=1` containment test discards
# every torino exon outside it and the gene can end up with no IR interval at
# all -- MRPL43 (ENSG00000055950) is the case this was written for: its APPRIS
# span is 1.2 kb against a 17.8 kb torino isoform, so the gene never reached the
# bed even though its IR factor is significant and is not an efficiency factor.
#
# Everything downstream is unchanged: still IR_factor_only, still subtracting
# the APPRIS exons, still keeping only intervals flanked by exactly two of them.
#
# Writes its own output; AD_IR_only_plots.bed.gz is not touched.
#############################################

rule get_introns_for_ad_plot_IR_only_relaxed:
    input:
        "results/gao_k3/tables/LR.output.tab.gz"
    output:
        'results/gao_k3/tables/tmp_relaxed_ir_only/AD_IR_only_plots.ir_only.{chunk}.bed.gz'
    resources:
        mem_mb = 52000
    log:
        'logs/print_ad_introns_bed.relaxed_ir_only.{chunk}.log'
    wildcard_constraints:
        chunk = '|'.join([str(x) for x in range(0,101)]),
    shell:
        """
        python scripts/prepare_ad_introns_plots_no_efficiency_relaxed.py {wildcards.chunk} ir_only &> {log}
        """

rule get_introns_for_ad_plot_IR_only_relaxed_sig:
    """
    The relaxed bed, restricted to features significant in the logistic
    regression at FDR <= 0.1. Without the filter the gene set is an annotation
    filter rather than a statistical one: 3,908 genes carry an IR-only factor
    but only 1,082 of them reach FDR 0.1.
    """
    input:
        "results/gao_k3/tables/LR.output.tab.gz"
    output:
        'results/gao_k3/tables/tmp_relaxed_ir_only_fdr0p1/AD_IR_only_plots.ir_only_fdr0p1.{chunk}.bed.gz'
    resources:
        mem_mb = 12000
    log:
        'logs/print_ad_introns_bed.relaxed_ir_only_fdr0p1.{chunk}.log'
    wildcard_constraints:
        chunk = '|'.join([str(x) for x in range(0,101)]),
    shell:
        """
        python scripts/prepare_ad_introns_plots_no_efficiency_relaxed.py {wildcards.chunk} ir_only 0.1 &> {log}
        """

rule collect_ad_for_plot_IR_only_relaxed_sig:
    input:
        expand('results/gao_k3/tables/tmp_relaxed_ir_only_fdr0p1/AD_IR_only_plots.ir_only_fdr0p1.{chunk}.bed.gz',
               chunk = range(0,101))
    output:
        'results/gao_k3/tables/AD_IR_only_plots.relaxed.fdr0p1.bed.gz'
    resources:
        mem_mb = 12000
    shell:
        """
        rm -f results/gao_k3/tables/AD_IR_only_plots.relaxed.fdr0p1.bed
        for I in {input}
        do
           zcat ${{I}} >> results/gao_k3/tables/AD_IR_only_plots.relaxed.fdr0p1.bed
        done;
        gzip -f results/gao_k3/tables/AD_IR_only_plots.relaxed.fdr0p1.bed
        """


rule collect_ad_for_plot_IR_only_relaxed:
    input:
        expand('results/gao_k3/tables/tmp_relaxed_ir_only/AD_IR_only_plots.ir_only.{chunk}.bed.gz',
               chunk = range(0,101))
    output:
        'results/gao_k3/tables/AD_IR_only_plots.relaxed.bed.gz'
    resources:
        mem_mb = 12000
    shell:
        """
        rm -f results/gao_k3/tables/AD_IR_only_plots.relaxed.bed
        for I in {input}
        do
           zcat ${{I}} >> results/gao_k3/tables/AD_IR_only_plots.relaxed.bed
        done;
        gzip -f results/gao_k3/tables/AD_IR_only_plots.relaxed.bed
        """
